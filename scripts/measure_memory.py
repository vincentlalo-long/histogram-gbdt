#!/usr/bin/env python3
"""Measure peak RSS of one fit in a fresh process.

Run this script once per library and dataset.  RSS is process-wide, so it
includes the interpreter, the loaded data, the model object, and temporary
buffers.  A small warmup fit initializes lazy library state before the
measured fit.  The delta is taken over the post-warmup/post-load baseline;
`ru_maxrss` is a high-water mark, so the total peak still includes warmup and
import overhead.  This is a coarse end-to-end estimate, not an allocator
profile.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np

try:
    import resource
except ImportError:  # pragma: no cover - Windows
    resource = None

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

DATASETS = (
    "synthetic",
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
    "random-forest",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", choices=LIBRARIES, default="gbdt")
    parser.add_argument("--dataset", choices=DATASETS, default="synthetic")
    parser.add_argument(
        "--task",
        choices=("regression", "classification", "multiclass"),
        default="classification",
        help="Ignored for real datasets (task is inferred).",
    )
    parser.add_argument("--n-samples", type=int, default=20_000)
    parser.add_argument("--n-features", type=int, default=20)
    parser.add_argument("--n-informative", type=int, default=8)
    parser.add_argument("--noise", type=float, default=1.0)
    parser.add_argument("--n-classes", type=int, default=3)
    parser.add_argument("--missing-fraction", type=float, default=0.0)
    parser.add_argument("--n-estimators", type=int, default=60)
    parser.add_argument("--max-depth", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=0.1)
    parser.add_argument("--max-bin", type=int, default=256)
    parser.add_argument("--backend", default="auto")
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--no-warmup",
        action="store_true",
        help="Do not initialize the library before the baseline snapshot.",
    )
    parser.add_argument("--json", action="store_true")
    return parser.parse_args()


def peak_rss_mb() -> float:
    if resource is None:
        raise SystemExit("peak-RSS measurement requires the Unix resource module")
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    divisor = 1024 * 1024 if sys.platform == "darwin" else 1024
    return float(usage) / divisor


def current_rss_mb() -> float:
    if os.name != "posix" or not Path("/proc/self/statm").exists():
        return float("nan")
    pages = int(Path("/proc/self/statm").read_text().split()[1])
    return pages * os.sysconf("SC_PAGE_SIZE") / (1024 * 1024)


def load_data(args: argparse.Namespace):
    if args.dataset == "synthetic":
        from gbdt.datasets import make_dataset

        X_train, _, y_train, _ = make_dataset(
            task=args.task,
            n_samples=args.n_samples,
            n_features=args.n_features,
            n_informative=args.n_informative,
            noise=args.noise,
            n_classes=args.n_classes,
            missing_fraction=args.missing_fraction,
            random_state=args.seed,
        )
        task = args.task
        n_classes = args.n_classes if task != "regression" else None
        return X_train, y_train, task, n_classes

    from sklearn import datasets
    from sklearn.preprocessing import LabelEncoder

    loaders = {
        "breast_cancer": datasets.load_breast_cancer,
        "wine": datasets.load_wine,
        "digits": datasets.load_digits,
        "diabetes": datasets.load_diabetes,
        "california_housing": datasets.fetch_california_housing,
    }
    bunch = loaders[args.dataset]()
    X = np.asarray(bunch.data, dtype=np.float64)
    y = np.asarray(bunch.target)
    task = (
        "regression"
        if args.dataset in {"diabetes", "california_housing"}
        else "classification"
    )
    if task == "classification":
        y = LabelEncoder().fit_transform(y).astype(np.int64)
        n_classes = int(np.unique(y).size)
        task = "multiclass" if n_classes > 2 else "classification"
    else:
        y = y.astype(np.float64)
        n_classes = None
    return X, y, task, n_classes


def build_model(
    args: argparse.Namespace,
    task: str,
    n_classes: int | None,
    base_score: float,
):
    if args.library == "gbdt":
        from gbdt import HistGBDTClassifier, HistGBDTRegressor

        common = dict(
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            learning_rate=args.learning_rate,
            max_bin=args.max_bin,
            random_state=args.seed,
            backend=args.backend,
            n_jobs=args.n_jobs,
        )
        if task == "regression":
            return HistGBDTRegressor(**common)
        return HistGBDTClassifier(**common)

    if args.library == "xgboost":
        try:
            import xgboost as xgb
        except ImportError as exc:  # pragma: no cover
            raise SystemExit("xgboost is not installed") from exc
        common = dict(
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            learning_rate=args.learning_rate,
            max_bin=args.max_bin,
            tree_method="hist",
            n_jobs=args.n_jobs,
            random_state=args.seed,
            verbosity=0,
        )
        if task == "regression":
            return xgb.XGBRegressor(objective="reg:squarederror", **common)
        if task == "multiclass":
            return xgb.XGBClassifier(
                objective="multi:softprob",
                num_class=n_classes,
                eval_metric="mlogloss",
                **common,
            )
        return xgb.XGBClassifier(
            objective="binary:logistic",
            base_score=base_score,
            eval_metric="logloss",
            **common,
        )

    if args.library == "lightgbm":
        try:
            import lightgbm as lgb
        except ImportError as exc:  # pragma: no cover
            raise SystemExit("lightgbm is not installed") from exc
        common = dict(
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            learning_rate=args.learning_rate,
            max_bin=min(args.max_bin, 255),
            n_jobs=args.n_jobs,
            random_state=args.seed,
            verbosity=-1,
            deterministic=True,
            force_col_wise=True,
        )
        if task == "regression":
            return lgb.LGBMRegressor(objective="regression", **common)
        if task == "multiclass":
            return lgb.LGBMClassifier(
                objective="multiclass", num_class=n_classes, **common
            )
        return lgb.LGBMClassifier(objective="binary", **common)

    if args.library == "sklearn-hist":
        from sklearn.ensemble import (
            HistGradientBoostingClassifier,
            HistGradientBoostingRegressor,
        )

        common = dict(
            max_iter=args.n_estimators,
            max_depth=args.max_depth,
            learning_rate=args.learning_rate,
            max_bins=min(args.max_bin, 255),
            l2_regularization=1.0,
            min_samples_leaf=1,
            early_stopping=False,
            random_state=args.seed,
        )
        if task == "regression":
            return HistGradientBoostingRegressor(**common)
        return HistGradientBoostingClassifier(**common)

    if args.library == "sklearn-gb":
        from sklearn.ensemble import (
            GradientBoostingClassifier,
            GradientBoostingRegressor,
        )

        common = dict(
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            learning_rate=args.learning_rate,
            random_state=args.seed,
        )
        if task == "regression":
            return GradientBoostingRegressor(loss="squared_error", **common)
        return GradientBoostingClassifier(loss="log_loss", **common)

    if args.library == "random-forest":
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

        common = dict(
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            random_state=args.seed,
            n_jobs=args.n_jobs,
        )
        if task == "regression":
            return RandomForestRegressor(**common)
        return RandomForestClassifier(**common)

    raise ValueError(args.library)


def warmup_indices(y, task: str, seed: int, limit: int = 128) -> np.ndarray:
    if task == "regression":
        return np.arange(min(limit, len(y)))
    rng = np.random.default_rng(seed)
    classes = np.unique(y)
    per_class = max(1, limit // len(classes))
    chosen = []
    for label in classes:
        indices = np.flatnonzero(y == label)
        chosen.extend(
            rng.choice(indices, size=min(per_class, len(indices)), replace=False)
        )
    return np.asarray(chosen, dtype=np.int64)


def main() -> None:
    args = parse_args()
    warnings.filterwarnings("ignore")
    X, y, task, n_classes = load_data(args)
    base_score = float(np.mean(y))
    if not args.no_warmup:
        warm_idx = warmup_indices(y, task, args.seed)
        warm_model = build_model(args, task, n_classes, base_score=base_score)
        warm_model.fit(X[warm_idx], y[warm_idx])
        del warm_model
        gc.collect()

    model = build_model(args, task, n_classes, base_score=base_score)

    baseline_peak = peak_rss_mb()
    baseline_current = current_rss_mb()
    start = time.perf_counter()
    model.fit(X, y)
    fit_seconds = time.perf_counter() - start
    after_current = current_rss_mb()
    after_peak = peak_rss_mb()

    result = {
        "library": args.library,
        "dataset": args.dataset,
        "task": task,
        "n_samples": int(X.shape[0]),
        "n_features": int(X.shape[1]),
        "n_classes": n_classes,
        "n_estimators": args.n_estimators,
        "max_depth": args.max_depth,
        "n_jobs": args.n_jobs,
        "warmup": not args.no_warmup,
        "baseline_rss_mb": round(baseline_current, 3),
        "peak_rss_mb": round(after_peak, 3),
        "delta_peak_rss_mb": round(after_peak - baseline_peak, 3),
        "after_fit_rss_mb": round(after_current, 3),
        "delta_after_fit_rss_mb": round(after_current - baseline_current, 3),
        "fit_seconds": round(fit_seconds, 4),
    }
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        for key, value in result.items():
            print(f"{key}: {value}")


if __name__ == "__main__":
    main()
