#!/usr/bin/env python3
"""Benchmark hist-gbdt on real datasets against classical and modern learners.

The suite is intentionally a breadth-first comparison, not a tuned
leaderboard.  Every tree model gets the same nominal tree budget and depth,
the same folds are used for every model, and the first call is warmed up so
library initialization is not charged to the first fold.
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import sys
import time
import warnings
from pathlib import Path

import numpy as np
from sklearn import datasets
from sklearn.ensemble import (
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    log_loss,
    mean_absolute_error,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gbdt import HistGBDTClassifier, HistGBDTRegressor  # noqa: E402

try:
    import lightgbm as lgb
except ImportError:  # pragma: no cover - optional dependency
    lgb = None

try:
    import xgboost as xgb
except ImportError:  # pragma: no cover - optional dependency
    xgb = None


DATASETS = (
    "breast_cancer",
    "wine",
    "digits",
    "diabetes",
    "california_housing",
)

SHORT = {
    "hist-gbdt (ours, C++)": "hist-gbdt",
    "xgboost-hist (3.x)": "xgb-hist",
    "xgboost-exact (3.x)": "xgb-exact",
    "lightgbm (4.x)": "lightgbm",
    "sklearn-hist": "sk-hist",
    "sklearn-gb (classic)": "sk-gb",
    "random-forest": "rf",
    "extra-trees": "et",
    "logistic": "logistic",
    "ridge": "ridge",
}

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--datasets",
        default=",".join(DATASETS),
        help="Comma-separated dataset names.",
    )
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--n-estimators", type=int, default=60)
    parser.add_argument("--max-depth", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=0.1)
    parser.add_argument("--max-bin", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument(
        "--no-warmup",
        action="store_true",
        help="Include each library's first-call initialization in the fit time.",
    )
    parser.add_argument("--no-figures", action="store_true")
    parser.add_argument(
        "--outdir",
        type=Path,
        default=ROOT / "bench" / "results",
    )
    parser.add_argument(
        "--figdir",
        type=Path,
        default=ROOT / "docs" / "figures",
    )
    return parser.parse_args()


def load_dataset(name: str):
    loaders = {
        "breast_cancer": datasets.load_breast_cancer,
        "wine": datasets.load_wine,
        "digits": datasets.load_digits,
        "diabetes": datasets.load_diabetes,
        "california_housing": datasets.fetch_california_housing,
    }
    if name not in loaders:
        raise ValueError(f"unknown dataset {name!r}; choose from {sorted(loaders)}")
    bunch = loaders[name]()
    X = np.asarray(bunch.data, dtype=np.float64)
    y = np.asarray(bunch.target)
    task = "regression" if name in {"diabetes", "california_housing"} else "classification"
    if task == "classification":
        y = LabelEncoder().fit_transform(y).astype(np.int64)
    else:
        y = y.astype(np.float64)
    return X, y, task


def make_models(
    task: str,
    n_classes: int,
    args: argparse.Namespace,
    n_samples: int,
    n_features: int,
):
    common = dict(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        learning_rate=args.learning_rate,
        random_state=args.seed,
    )
    forest_common = {
        key: value for key, value in common.items() if key != "learning_rate"
    }
    models = {}

    if task == "regression":
        models["hist-gbdt (ours, C++)"] = lambda: HistGBDTRegressor(
            backend="cpp",
            n_jobs=args.n_jobs,
            max_bin=args.max_bin,
            reg_lambda=1.0,
            min_child_weight=1.0,
            **common,
        )
        models["sklearn-hist"] = lambda: HistGradientBoostingRegressor(
            max_iter=args.n_estimators,
            max_depth=args.max_depth,
            learning_rate=args.learning_rate,
            max_bins=min(args.max_bin, 255),
            l2_regularization=1.0,
            min_samples_leaf=1,
            early_stopping=False,
            random_state=args.seed,
        )
        models["sklearn-gb (classic)"] = lambda: GradientBoostingRegressor(
            loss="squared_error", **common
        )
        models["random-forest"] = lambda: RandomForestRegressor(
            n_jobs=args.n_jobs, **forest_common
        )
        models["extra-trees"] = lambda: ExtraTreesRegressor(
            n_jobs=args.n_jobs, **forest_common
        )
        models["ridge"] = lambda: make_pipeline(StandardScaler(), Ridge(alpha=1.0))
    else:
        models["hist-gbdt (ours, C++)"] = lambda: HistGBDTClassifier(
            backend="cpp",
            n_jobs=args.n_jobs,
            max_bin=args.max_bin,
            reg_lambda=1.0,
            min_child_weight=1.0,
            **common,
        )
        models["sklearn-hist"] = lambda: HistGradientBoostingClassifier(
            max_iter=args.n_estimators,
            max_depth=args.max_depth,
            learning_rate=args.learning_rate,
            max_bins=min(args.max_bin, 255),
            l2_regularization=1.0,
            min_samples_leaf=1,
            early_stopping=False,
            random_state=args.seed,
        )
        models["sklearn-gb (classic)"] = lambda: GradientBoostingClassifier(
            loss="log_loss", **common
        )
        models["random-forest"] = lambda: RandomForestClassifier(
            n_jobs=args.n_jobs, **forest_common
        )
        models["extra-trees"] = lambda: ExtraTreesClassifier(
            n_jobs=args.n_jobs, **forest_common
        )
        models["logistic"] = lambda: make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=2000, random_state=args.seed),
        )

    if xgb is not None:
        common_xgb = dict(
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            learning_rate=args.learning_rate,
            reg_lambda=1.0,
            min_child_weight=1.0,
            n_jobs=args.n_jobs,
            random_state=args.seed,
            verbosity=0,
        )
        if task == "regression":
            models["xgboost-hist (3.x)"] = lambda: xgb.XGBRegressor(
                objective="reg:squarederror",
                tree_method="hist",
                max_bin=args.max_bin,
                **common_xgb,
            )
        else:
            objective = "binary:logistic" if n_classes == 2 else "multi:softprob"
            extra = {} if n_classes == 2 else {"num_class": n_classes}
            models["xgboost-hist (3.x)"] = (
                lambda objective=objective, extra=extra: xgb.XGBClassifier(
                    objective=objective,
                    tree_method="hist",
                    max_bin=args.max_bin,
                    **extra,
                    **common_xgb,
                )
            )

        if n_samples <= 2000 and n_features <= 32:
            if task == "regression":
                models["xgboost-exact (3.x)"] = lambda: xgb.XGBRegressor(
                    objective="reg:squarederror",
                    tree_method="exact",
                    **common_xgb,
                )
            else:
                objective = (
                    "binary:logistic" if n_classes == 2 else "multi:softprob"
                )
                extra = {} if n_classes == 2 else {"num_class": n_classes}
                models["xgboost-exact (3.x)"] = (
                    lambda objective=objective, extra=extra: xgb.XGBClassifier(
                        objective=objective,
                        tree_method="exact",
                        **extra,
                        **common_xgb,
                    )
                )

    if lgb is not None:
        common_lgb = dict(
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            learning_rate=args.learning_rate,
            reg_lambda=1.0,
            min_child_weight=1.0,
            min_child_samples=1,
            max_bin=min(args.max_bin, 255),
            n_jobs=args.n_jobs,
            random_state=args.seed,
            verbosity=-1,
            deterministic=True,
            force_col_wise=True,
        )
        if task == "regression":
            models["lightgbm (4.x)"] = lambda: lgb.LGBMRegressor(
                objective="regression", **common_lgb
            )
        else:
            objective = "binary" if n_classes == 2 else "multiclass"
            extra = {} if n_classes == 2 else {"num_class": n_classes}
            models["lightgbm (4.x)"] = (
                lambda objective=objective, extra=extra: lgb.LGBMClassifier(
                    objective=objective, **extra, **common_lgb
                )
            )

    return models


def predict_scores(model, X):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)
    return model.predict(X)


def metrics_for(task: str, y_true, prediction, scores, n_classes: int):
    if task == "regression":
        return {
            "rmse": float(np.sqrt(np.mean((prediction - y_true) ** 2))),
            "mae": float(mean_absolute_error(y_true, prediction)),
            "r2": float(r2_score(y_true, prediction)),
        }
    labels = np.arange(n_classes)
    out = {
        "accuracy": float(accuracy_score(y_true, prediction)),
        "f1_macro": float(
            f1_score(y_true, prediction, average="macro", zero_division=0)
        ),
        "logloss": float(log_loss(y_true, scores, labels=labels)),
    }
    if n_classes == 2:
        out["auc"] = float(roc_auc_score(y_true, scores[:, 1]))
    return out


def run_dataset(name: str, args: argparse.Namespace):
    X, y, task = load_dataset(name)
    n_classes = int(np.unique(y).size) if task == "classification" else 1
    info = {
        "dataset": name,
        "task": task,
        "n_samples": int(X.shape[0]),
        "n_features": int(X.shape[1]),
        "n_classes": n_classes if task == "classification" else None,
    }
    print(
        f"\n{name}: task={task} n={X.shape[0]} k={X.shape[1]}"
        + (f" classes={n_classes}" if task == "classification" else "")
    )

    splitter = (
        StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
        if task == "classification"
        else KFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    )
    splits = list(splitter.split(X, y))
    models = make_models(task, n_classes, args, X.shape[0], X.shape[1])
    records = []
    raw_records = []

    for model_name, factory in models.items():
        fold_metrics = {}
        fit_times = []
        predict_times = []
        error = None
        if not args.no_warmup:
            warm_idx = splits[0][0][: min(128, len(splits[0][0]))]
            warm_model = factory()
            warm_model.fit(X[warm_idx], y[warm_idx])
        for fold, (train_idx, test_idx) in enumerate(splits):
            model = factory()
            start = time.perf_counter()
            model.fit(X[train_idx], y[train_idx])
            fit_time = time.perf_counter() - start
            start = time.perf_counter()
            prediction = model.predict(X[test_idx])
            predict_time = time.perf_counter() - start
            scores = predict_scores(model, X[test_idx])
            if task == "classification" and np.asarray(scores).ndim == 1:
                error = "predict_proba missing"
                break
            metrics = metrics_for(task, y[test_idx], prediction, scores, n_classes)
            fit_times.append(fit_time)
            predict_times.append(predict_time)
            raw_row = {
                **info,
                "fold": fold,
                "model": model_name,
                "fit_seconds": fit_time,
                "predict_seconds": predict_time,
                **metrics,
            }
            raw_records.append(raw_row)
            for key, value in metrics.items():
                fold_metrics.setdefault(key, []).append(value)
        if error is not None:
            print(f"  {model_name:<24} {error}")
            continue

        row = {
            **info,
            "model": model_name,
            "fit_seconds_mean": float(np.mean(fit_times)),
            "fit_seconds_std": float(np.std(fit_times)),
            "predict_seconds_mean": float(np.mean(predict_times)),
            "predict_seconds_std": float(np.std(predict_times)),
        }
        for key, values in fold_metrics.items():
            row[f"{key}_mean"] = float(np.mean(values))
            row[f"{key}_std"] = float(np.std(values))
        records.append(row)

    primary = "rmse" if task == "regression" else "accuracy"
    reverse = task != "regression"
    records.sort(key=lambda r: r[f"{primary}_mean"], reverse=reverse)

    if task == "regression":
        headers = ("rmse", "mae", "r2")
    else:
        headers = ("accuracy", "f1_macro", "logloss", "auc")
        headers = tuple(h for h in headers if any(f"{h}_mean" in r for r in records))

    print(
        f"  {'model':<24} "
        + " ".join(f"{h:>16}" for h in headers)
        + f" {'fit_s':>10} {'pred_ms':>10}"
    )
    print("  " + "-" * (24 + 17 * len(headers) + 23))
    for row in records:
        metric_text = " ".join(
            f"{row[f'{h}_mean']:.4f}±{row[f'{h}_std']:.3f}".rjust(16)
            if f"{h}_mean" in row
            else "-".rjust(16)
            for h in headers
        )
        print(
            f"  {row['model']:<24} {metric_text} "
            f"{row['fit_seconds_mean']:>10.4f} "
            f"{1000 * row['predict_seconds_mean']:>10.2f}"
        )

    return records, raw_records, info


def cpu_name() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown CPU"


def environment() -> dict:
    return {
        "cpu": cpu_name(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "sklearn": __import__("sklearn").__version__,
        "xgboost": getattr(xgb, "__version__", "missing"),
        "lightgbm": getattr(lgb, "__version__", "missing"),
        "gbdt": __import__("gbdt").__version__,
    }


def fmt(value: float, digits: int = 4) -> str:
    return f"{value:.{digits}f}"


def markdown_tables(records: list[dict], infos: list[dict]) -> list[str]:
    lines: list[str] = []
    for info in infos:
        subset = [row for row in records if row["dataset"] == info["dataset"]]
        if not subset:
            continue
        task = info["task"]
        lines += [
            "",
            f"### {info['dataset']} ({task}, n={info['n_samples']}, "
            f"k={info['n_features']}"
            + (f", classes={info['n_classes']}" if task == "classification" else "")
            + ")",
            "",
        ]
        if task == "classification":
            lines += [
                "| model | accuracy | macro-F1 | logloss | AUC | fit (s) | predict (ms) |",
                "|---|---:|---:|---:|---:|---:|---:|",
            ]
            for row in subset:
                auc = (
                    f"{row['auc_mean']:.4f} ± {row['auc_std']:.3f}"
                    if "auc_mean" in row
                    else "—"
                )
                lines.append(
                    f"| {SHORT.get(row['model'], row['model'])} "
                    f"| {row['accuracy_mean']:.4f} ± {row['accuracy_std']:.3f} "
                    f"| {row['f1_macro_mean']:.4f} ± {row['f1_macro_std']:.3f} "
                    f"| {row['logloss_mean']:.4f} ± {row['logloss_std']:.3f} "
                    f"| {auc} "
                    f"| {row['fit_seconds_mean']:.4f} "
                    f"| {1000 * row['predict_seconds_mean']:.2f} |"
                )
        else:
            lines += [
                "| model | RMSE | MAE | R² | fit (s) | predict (ms) |",
                "|---|---:|---:|---:|---:|---:|",
            ]
            for row in subset:
                lines.append(
                    f"| {SHORT.get(row['model'], row['model'])} "
                    f"| {row['rmse_mean']:.4f} ± {row['rmse_std']:.3f} "
                    f"| {row['mae_mean']:.4f} ± {row['mae_std']:.3f} "
                    f"| {row['r2_mean']:.4f} ± {row['r2_std']:.3f} "
                    f"| {row['fit_seconds_mean']:.4f} "
                    f"| {1000 * row['predict_seconds_mean']:.2f} |"
                )
    return lines


def markdown_report(
    records: list[dict],
    infos: list[dict],
    raw_records: list[dict],
    args: argparse.Namespace,
    env: dict,
) -> str:
    lines = [
        "# Real-data benchmark",
        "",
        "- This is a learning benchmark, not a tuned leaderboard.",
        f"- Machine: {env['cpu']}",
        f"- Python {env['python']}, NumPy {env['numpy']}, "
        f"scikit-learn {env['sklearn']}",
        f"- hist-gbdt {env['gbdt']}, XGBoost {env['xgboost']}, "
        f"LightGBM {env['lightgbm']}",
        f"- {args.folds}-fold CV, {args.n_estimators} trees, depth "
        f"{args.max_depth}, learning rate {args.learning_rate}, "
        f"max_bin {args.max_bin}, n_jobs={args.n_jobs}",
        "- Same folds and nominal tree budget for every model; one warmup "
        "call per model; no early stopping.",
        "",
        "## Datasets",
        "",
        "| dataset | task | n | k | classes |",
        "|---|---|---:|---:|---:|",
    ]
    for info in infos:
        lines.append(
            f"| {info['dataset']} | {info['task']} | {info['n_samples']} "
            f"| {info['n_features']} "
            f"| {info['n_classes'] if info['n_classes'] is not None else '—'} |"
        )

    lines += markdown_tables(records, infos)

    lines += [
        "",
        "## Figures",
        "",
        "The same saved CSV is shown through two lenses: a like-for-like",
        "histogram comparison and a broader model-family landscape.",
        "",
        "### Histogram parity",
        "",
        "![Parity: quality](../../docs/figures/parity_quality.png)",
        "",
        "![Parity: latency](../../docs/figures/parity_latency.png)",
        "",
        "![Parity: memory](../../docs/figures/parity_memory.png)",
        "",
        "### ML landscape",
        "",
        "![Landscape: quality](../../docs/figures/landscape_quality.png)",
        "",
        "![Landscape: fit time vs quality](../../docs/figures/landscape_pareto.png)",
        "",
        "## Reading the numbers honestly",
        "",
        "- The linear baselines are strongest on the small tabular datasets "
        "(`breast_cancer`, `wine`, `digits`, `diabetes`). This is expected: "
        "the sample count is small, the signal is close to linear, and the "
        "tree models are deliberately not tuned.",
        "- On `california_housing`, `hist-gbdt` is competitive with XGBoost "
        "and LightGBM in RMSE, but it is slower to fit than both.",
        "- Prediction is still pure Python tree traversal. Multiclass "
        "prediction on `digits` is therefore much slower than the libraries "
        "even when fit time is comparable.",
        "- XGBoost Exact is only run when `n_samples <= 2000` and "
        "`n_features <= 32`; it is the older exact-greedy split strategy, "
        "not an old XGBoost package version.",
        "- The real datasets here have no missing values. Missing-value "
        "handling is exercised by the synthetic and parity tests, not by "
        "this table.",
        "- Treat sub-millisecond differences as noise. Treat the real-data "
        "result as a sanity check on engineering, not a claim of "
        "state-of-the-art accuracy.",
        "",
        f"Raw rows: `{len(raw_records)}`; aggregate rows: `{len(records)}`.",
        "",
    ]
    return "\n".join(lines)


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


def main() -> None:
    args = parse_args()
    if args.folds < 2:
        raise SystemExit("--folds must be at least 2")
    names = [name.strip() for name in args.datasets.split(",") if name.strip()]
    warnings.filterwarnings(
        "ignore", message="X does not have valid feature names", category=UserWarning
    )
    env = environment()
    print(
        f"machine: {env['cpu']} | Python {env['python']} | "
        f"numpy {env['numpy']} | sklearn {env['sklearn']} | "
        f"gbdt {env['gbdt']} | xgboost {env['xgboost']} | "
        f"lightgbm {env['lightgbm']}"
    )
    print(
        f"settings: folds={args.folds} trees={args.n_estimators} "
        f"depth={args.max_depth} lr={args.learning_rate} "
        f"max_bin={args.max_bin} n_jobs={args.n_jobs}"
    )

    all_records: list[dict] = []
    raw_records: list[dict] = []
    infos: list[dict] = []
    for name in names:
        records, raw, info = run_dataset(name, args)
        all_records.extend(records)
        raw_records.extend(raw)
        infos.append(info)

    args.outdir.mkdir(parents=True, exist_ok=True)
    (args.outdir / "real_comparison.json").write_text(
        json.dumps(all_records, indent=2)
    )
    write_csv(args.outdir / "real_comparison.csv", all_records)
    write_csv(args.outdir / "real_runs.csv", raw_records)
    (args.outdir / "real_summary.md").write_text(
        markdown_report(all_records, infos, raw_records, args, env)
    )

    if not args.no_figures:
        from plot_real import make_figures

        for figure in make_figures(args.outdir, args.figdir):
            print(f"wrote {figure}")

    print(f"\nwrote {args.outdir}/real_comparison.json")
    print(f"wrote {args.outdir}/real_comparison.csv")
    print(f"wrote {args.outdir}/real_runs.csv")
    print(f"wrote {args.outdir}/real_summary.md")


if __name__ == "__main__":
    main()
