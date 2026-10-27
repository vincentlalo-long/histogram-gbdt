#!/usr/bin/env python3
"""Time the NumPy/C++ engines against XGBoost Hist on synthetic tabular data."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gbdt import HistGBDTClassifier, HistGBDTRegressor  # noqa: E402
from gbdt.datasets import make_dataset  # noqa: E402

try:
    from sklearn.metrics import log_loss, roc_auc_score
except ImportError:  # pragma: no cover - optional dependency
    log_loss = None
    roc_auc_score = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--task",
        choices=(
            "regression",
            "classification",
            "multiclass",
            "both",
            "all",
        ),
        default="both",
    )
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--n-samples", type=int, default=None)
    parser.add_argument("--n-features", type=int, default=None)
    parser.add_argument("--n-informative", type=int, default=None)
    parser.add_argument("--noise", type=float, default=1.0)
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--n-classes", type=int, default=3)
    parser.add_argument("--missing-fraction", type=float, default=0.0)
    parser.add_argument("--n-estimators", type=int, default=None)
    parser.add_argument("--max-depth", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=0.1)
    parser.add_argument("--max-bin", type=int, default=256)
    parser.add_argument("--reg-lambda", type=float, default=1.0)
    parser.add_argument("--min-child-weight", type=float, default=1.0)
    parser.add_argument("--repeat", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--backend", default="auto")
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--no-xgboost", action="store_true")
    return parser.parse_args()


def _settings(args: argparse.Namespace) -> dict:
    return {
        "n_samples": args.n_samples or (3_000 if args.quick else 20_000),
        "n_features": args.n_features or (10 if args.quick else 20),
        "n_informative": args.n_informative or (5 if args.quick else 8),
        "n_estimators": args.n_estimators or (30 if args.quick else 100),
        "max_depth": args.max_depth or (4 if args.quick else 6),
        "repeat": args.repeat or (1 if args.quick else 3),
    }


def _make_model(task: str, args: argparse.Namespace):
    common = dict(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        learning_rate=args.learning_rate,
        max_bin=args.max_bin,
        reg_lambda=args.reg_lambda,
        min_child_weight=args.min_child_weight,
        random_state=args.seed,
        backend=args.backend,
        n_jobs=args.n_jobs,
    )
    if task == "regression":
        return HistGBDTRegressor(**common)
    return HistGBDTClassifier(**common)


def _make_xgboost(
    task: str, args: argparse.Namespace, base_score: float = 0.5
):
    try:
        import xgboost as xgb
    except ImportError:
        return None

    common = dict(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        learning_rate=args.learning_rate,
        reg_lambda=args.reg_lambda,
        min_child_weight=args.min_child_weight,
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
            num_class=args.n_classes,
            eval_metric="mlogloss",
            **common,
        )
    return xgb.XGBClassifier(
        objective="binary:logistic",
        base_score=base_score,
        eval_metric="logloss",
        **common,
    )


def _predict(model, X):
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(X)
        if probabilities.shape[1] == 2:
            return probabilities[:, 1]
        return probabilities
    return model.predict(X)


def _metric(task: str, y, prediction) -> str:
    if task == "regression":
        rmse = float(np.sqrt(np.mean((prediction - y) ** 2)))
        return f"rmse={rmse:.5f}"
    if task == "multiclass":
        if log_loss is None:
            return "mlogloss/acc=n/a (install scikit-learn)"
        labels = np.arange(prediction.shape[1])
        logloss = float(log_loss(y, prediction, labels=labels))
        accuracy = float(np.mean(np.argmax(prediction, axis=1) == y))
        return f"mlogloss={logloss:.5f} acc={accuracy:.5f}"
    if log_loss is None or roc_auc_score is None:
        return "logloss/auc=n/a (install scikit-learn)"
    logloss = float(log_loss(y, prediction))
    auc = float(roc_auc_score(y, prediction))
    return f"logloss={logloss:.5f} auc={auc:.5f}"


def _run_library(task, name, factory, args, data):
    X_train, X_test, y_train, y_test = data
    fit_times = []
    predict_times = []
    last_prediction = None

    for _ in range(args.repeat):
        model = factory()
        if model is None:
            return None
        start = time.perf_counter()
        model.fit(X_train, y_train)
        fit_times.append(time.perf_counter() - start)
        start = time.perf_counter()
        last_prediction = _predict(model, X_test)
        predict_times.append(time.perf_counter() - start)

    return {
        "model": name,
        "fit_mean": float(np.mean(fit_times)),
        "fit_std": float(np.std(fit_times)),
        "predict_mean": float(np.mean(predict_times)),
        "metric": _metric(task, y_test, last_prediction),
    }


def _print_results(task: str, results: list[dict]) -> None:
    print(f"\n{task}")
    print(f"{'model':<22} {'fit (s)':>16} {'predict (s)':>14}  metric")
    print("-" * 82)
    for result in results:
        fit = f"{result['fit_mean']:.4f} ± {result['fit_std']:.4f}"
        print(
            f"{result['model']:<22} {fit:>16} "
            f"{result['predict_mean']:>14.4f}  {result['metric']}"
        )


def main() -> None:
    args = parse_args()
    settings = _settings(args)
    args.n_samples = settings["n_samples"]
    args.n_features = settings["n_features"]
    args.n_informative = settings["n_informative"]
    args.n_estimators = settings["n_estimators"]
    args.max_depth = settings["max_depth"]
    args.repeat = settings["repeat"]
    if args.repeat < 1:
        raise SystemExit("--repeat must be at least 1")

    if args.task == "both":
        tasks = ("regression", "classification")
    elif args.task == "all":
        tasks = ("regression", "classification", "multiclass")
    else:
        tasks = (args.task,)

    print(
        "dataset: "
        f"n={args.n_samples} k={args.n_features} "
        f"informative={args.n_informative} noise={args.noise} "
        f"classes={args.n_classes} "
        f"missing={args.missing_fraction:.2f}"
    )
    print(
        "model: "
        f"trees={args.n_estimators} depth={args.max_depth} "
        f"lr={args.learning_rate} max_bin={args.max_bin} "
        f"backend={args.backend} repeat={args.repeat}"
    )

    for task in tasks:
        data = make_dataset(
            task=task,
            n_samples=args.n_samples,
            n_features=args.n_features,
            n_informative=args.n_informative,
            noise=args.noise,
            test_fraction=args.test_fraction,
            n_classes=args.n_classes,
            missing_fraction=args.missing_fraction,
            random_state=args.seed,
        )
        results = []

        def our_factory(task=task):
            return _make_model(task, args)

        our_result = _run_library(task, "gbdt", our_factory, args, data)
        if our_result is not None:
            results.append(our_result)

        if not args.no_xgboost:

            def xgb_factory(task=task, data=data):
                return _make_xgboost(
                    task, args, base_score=float(np.mean(data[2]))
                )

            xgb_result = _run_library(
                task, "xgboost-hist", xgb_factory, args, data
            )
            if xgb_result is not None:
                results.append(xgb_result)
            else:
                print("\nxgboost is not installed; reference run skipped")

        _print_results(task, results)

    print(
        "\nMemory is measured separately: "
        "`python scripts/measure_memory.py --help`"
    )


if __name__ == "__main__":
    main()
