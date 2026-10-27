#!/usr/bin/env python3
"""Measure peak RSS for real-data fits in fresh processes.

Each (library, dataset, repetition) pair runs `scripts/measure_memory.py` in a
new interpreter.  Peak RSS is process-wide and includes the interpreter, the
loaded dataset, the model object, and temporary training buffers.  The delta is
taken over a post-load/post-model-construction baseline, so treat it as a
coarse training-memory estimate rather than an allocator profile.

This module only collects and summarizes data.  `bench/plot_real.py` produces
the parity memory figure from `memory_real.csv`.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATASETS = (
    "breast_cancer",
    "wine",
    "digits",
    "diabetes",
    "california_housing",
)
LIBRARIES = (
    "gbdt",
    "xgboost",
    "lightgbm",
    "sklearn-hist",
    "sklearn-gb",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", default=",".join(DATASETS))
    parser.add_argument("--libraries", default=",".join(LIBRARIES))
    parser.add_argument("--repeat", type=int, default=2)
    parser.add_argument("--n-estimators", type=int, default=60)
    parser.add_argument("--max-depth", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument(
        "--outdir",
        type=Path,
        default=ROOT / "bench" / "results",
    )
    return parser.parse_args()


def cpu_name() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown CPU"


def run_one(
    dataset: str,
    library: str,
    args: argparse.Namespace,
    env: dict[str, str],
) -> dict:
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "measure_memory.py"),
        "--dataset",
        dataset,
        "--library",
        library,
        "--n-estimators",
        str(args.n_estimators),
        "--max-depth",
        str(args.max_depth),
        "--seed",
        str(args.seed),
        "--n-jobs",
        str(args.n_jobs),
        "--json",
    ]
    proc = subprocess.run(
        cmd,
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise SystemExit(
            f"memory run failed: {' '.join(cmd)}\n"
            f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
        )
    text = proc.stdout.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        if start < 0:
            raise
        return json.loads(text[start:])


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, restval="")
        writer.writeheader()
        writer.writerows(rows)


def aggregate(raw: list[dict]) -> list[dict]:
    grouped: dict[tuple[str, str], list[dict]] = {}
    for row in raw:
        grouped.setdefault((row["dataset"], row["library"]), []).append(row)
    summary = []
    for (dataset, library), rows in grouped.items():
        first = rows[0]
        summary.append(
            {
                "dataset": dataset,
                "library": library,
                "task": first["task"],
                "n_samples": first["n_samples"],
                "n_features": first["n_features"],
                "n_classes": first["n_classes"],
                "repeats": len(rows),
                "peak_rss_mb": float(
                    np.median([row["peak_rss_mb"] for row in rows])
                ),
                "delta_peak_rss_mb": float(
                    np.median([row["delta_peak_rss_mb"] for row in rows])
                ),
                "after_fit_rss_mb": float(
                    np.median([row["after_fit_rss_mb"] for row in rows])
                ),
                "fit_seconds": float(
                    np.median([row["fit_seconds"] for row in rows])
                ),
            }
        )
    order_ds = {name: i for i, name in enumerate(DATASETS)}
    order_lib = {name: i for i, name in enumerate(LIBRARIES)}
    summary.sort(
        key=lambda row: (
            order_ds.get(row["dataset"], 99),
            order_lib.get(row["library"], 99),
        )
    )
    return summary


def markdown(summary: list[dict], args: argparse.Namespace, cpu: str) -> str:
    lines = [
        "# Memory benchmark",
        "",
        f"- Machine: {cpu}",
        f"- Python {platform.python_version()}",
        f"- {args.repeat} fresh-process repetitions per point; median shown",
        f"- {args.n_estimators} trees, depth {args.max_depth}, n_jobs={args.n_jobs}",
        "- One warmup fit is performed before the measured fit. Total peak RSS "
        "includes that warmup and library import/runtime overhead.",
        "- Delta is measured against a post-warmup/post-data/post-model baseline; "
        "it is a coarse training-memory estimate, not pure model size.",
        "",
    ]
    datasets = list(dict.fromkeys(row["dataset"] for row in summary))
    for dataset in datasets:
        rows = [row for row in summary if row["dataset"] == dataset]
        if not rows:
            continue
        first = rows[0]
        lines += [
            "",
            f"### {dataset} ({first['task']}, n={first['n_samples']}, "
            f"k={first['n_features']})",
            "",
            "| library | peak RSS (MB) | delta peak (MB) | RSS after fit (MB) | fit (s) |",
            "|---|---:|---:|---:|---:|",
        ]
        for row in rows:
            lines.append(
                f"| {row['library']} | {row['peak_rss_mb']:.1f} "
                f"| {row['delta_peak_rss_mb']:.1f} "
                f"| {row['after_fit_rss_mb']:.1f} "
                f"| {row['fit_seconds']:.4f} |"
            )
    lines += [
        "",
        "## Reading the numbers honestly",
        "",
        "- Peak RSS includes the Python interpreter, NumPy/scikit-learn, the "
        "input data, and library import overhead. Compare both total peak and "
        "the delta rather than treating either as pure model size.",
        "- `ru_maxrss` is a high-water mark, so a transient allocation during "
        "data loading can hide a later training allocation. Repeated fresh "
        "processes reduce, but do not eliminate, that noise.",
        "- This is not a substitute for a memory profiler such as "
        "`memray`/`tracemalloc`; it is the end-to-end number a user sees.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    if args.repeat < 1:
        raise SystemExit("--repeat must be at least 1")
    datasets = [item.strip() for item in args.datasets.split(",") if item.strip()]
    libraries = [item.strip() for item in args.libraries.split(",") if item.strip()]
    env = os.environ.copy()
    env.update(
        {
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
        }
    )
    cpu = cpu_name()
    print(f"machine: {cpu}")
    print(
        f"datasets={datasets} libraries={libraries} repeats={args.repeat} "
        f"trees={args.n_estimators} depth={args.max_depth}"
    )

    raw: list[dict] = []
    for dataset in datasets:
        for library in libraries:
            for repeat in range(args.repeat):
                row = run_one(dataset, library, args, env)
                row["repeat"] = repeat
                raw.append(row)
                print(
                    f"  {dataset:<18} {library:<13} "
                    f"peak={row['peak_rss_mb']:7.1f} MB "
                    f"delta={row['delta_peak_rss_mb']:7.1f} MB "
                    f"fit={row['fit_seconds']:.4f}s",
                    flush=True,
                )

    summary = aggregate(raw)
    args.outdir.mkdir(parents=True, exist_ok=True)
    write_csv(args.outdir / "memory_runs.csv", raw)
    write_csv(args.outdir / "memory_real.csv", summary)
    (args.outdir / "memory_summary.md").write_text(markdown(summary, args, cpu))
    print(f"wrote {args.outdir}/memory_runs.csv")
    print(f"wrote {args.outdir}/memory_real.csv")
    print(f"wrote {args.outdir}/memory_summary.md")


if __name__ == "__main__":
    main()
