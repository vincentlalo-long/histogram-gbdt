#!/usr/bin/env python3
"""Generate a synthetic tabular dataset for the benchmark harness."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gbdt.datasets import make_dataset  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--task",
        choices=("regression", "classification", "multiclass"),
        default="regression",
    )
    parser.add_argument("--n-samples", type=int, default=20_000)
    parser.add_argument("--n-features", type=int, default=20)
    parser.add_argument("--n-informative", type=int, default=8)
    parser.add_argument("--noise", type=float, default=1.0)
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--n-classes", type=int, default=3)
    parser.add_argument("--missing-fraction", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output .npz path (default: data/synthetic_<task>.npz)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = args.output or ROOT / "data" / f"synthetic_{args.task}.npz"
    output.parent.mkdir(parents=True, exist_ok=True)

    X_train, X_test, y_train, y_test = make_dataset(
        task=args.task,
        n_samples=args.n_samples,
        n_features=args.n_features,
        n_informative=args.n_informative,
        noise=args.noise,
        test_fraction=args.test_fraction,
        n_classes=args.n_classes,
        missing_fraction=args.missing_fraction,
        random_state=args.seed,
    )
    np.savez_compressed(
        output,
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
    )

    print(f"wrote {output}")
    print(f"train: X={X_train.shape} y={y_train.shape}")
    print(f"test:  X={X_test.shape} y={y_test.shape}")
    if args.task == "classification":
        print(f"positive rate (train): {float(np.mean(y_train)):.4f}")
    elif args.task == "multiclass":
        counts = np.bincount(y_train.astype(np.int64), minlength=args.n_classes)
        print(f"class counts (train): {counts.tolist()}")
    else:
        print(
            "target mean/std: "
            f"{float(np.mean(y_train)):.4f}/{float(np.std(y_train)):.4f}"
        )
    if args.missing_fraction > 0.0:
        print(
            "missing fraction (train/test): "
            f"{float(np.mean(np.isnan(X_train))):.4f}/"
            f"{float(np.mean(np.isnan(X_test))):.4f}"
        )


if __name__ == "__main__":
    main()
