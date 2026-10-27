#!/usr/bin/env python3
"""Plot benchmark views from already-saved CSV files.

The report separates two questions:

* histogram parity: hist-gbdt vs XGBoost Hist, LightGBM, sklearn Hist;
* ML landscape: hist-gbdt vs exact GBDT, classic GBDT, bagging and linear.

Nothing is refitted here.  The script reads `bench/results/real_comparison.csv`
and, if present, `bench/results/memory_real.csv`.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATASET_ORDER = (
    "breast_cancer",
    "wine",
    "digits",
    "diabetes",
    "california_housing",
)
HIST_MODELS = (
    "hist-gbdt (ours, C++)",
    "xgboost-hist (3.x)",
    "lightgbm (4.x)",
    "sklearn-hist",
)
LANDSCAPE_MODELS = (
    "hist-gbdt (ours, C++)",
    "xgboost-exact (3.x)",
    "sklearn-gb (classic)",
    "random-forest",
    "extra-trees",
)
LINEAR_MODEL = {"classification": "logistic", "regression": "ridge"}

SHORT = {
    "hist-gbdt (ours, C++)": "hist-gbdt",
    "xgboost-hist (3.x)": "xgb-hist",
    "lightgbm (4.x)": "lightgbm",
    "sklearn-hist": "sk-hist",
    "xgboost-exact (3.x)": "xgb-exact",
    "sklearn-gb (classic)": "sk-gb",
    "random-forest": "rf",
    "extra-trees": "et",
    "logistic": "logistic",
    "ridge": "ridge",
}
COLORS = {
    "hist-gbdt (ours, C++)": "#1f77b4",
    "xgboost-hist (3.x)": "#ff7f0e",
    "lightgbm (4.x)": "#2ca02c",
    "sklearn-hist": "#9467bd",
    "xgboost-exact (3.x)": "#d62728",
    "sklearn-gb (classic)": "#8c564b",
    "random-forest": "#7f7f7f",
    "extra-trees": "#17becf",
    "logistic": "#bcbd22",
    "ridge": "#bcbd22",
}
MARKERS = {
    "hist-gbdt (ours, C++)": "*",
    "xgboost-exact (3.x)": "X",
    "sklearn-gb (classic)": "P",
    "random-forest": "o",
    "extra-trees": "^",
    "logistic": "D",
    "ridge": "D",
}
LINEAR_COLOR = "#bcbd22"


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def value(row: dict, key: str) -> float:
    return float(row[key])


def find(rows: list[dict], dataset: str, model: str) -> dict | None:
    return next(
        (
            row
            for row in rows
            if row["dataset"] == dataset and row["model"] == model
        ),
        None,
    )


def datasets(rows: list[dict], task: str | None = None) -> list[str]:
    present = {row["dataset"] for row in rows}
    return [
        name
        for name in DATASET_ORDER
        if name in present
        and (task is None or next(row for row in rows if row["dataset"] == name)["task"] == task)
    ]


def setup():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import NullFormatter

    return plt, NullFormatter


def _hist_color(model: str) -> str:
    return COLORS[model]


def plot_parity_quality(rows: list[dict], path: Path) -> None:
    plt, _ = setup()
    competitors = HIST_MODELS[1:]
    cls_ds = datasets(rows, "classification")
    reg_ds = datasets(rows, "regression")

    fig, (ax_acc, ax_rmse) = plt.subplots(1, 2, figsize=(13.0, 4.9))
    for ax, ds_list, mode in (
        (ax_acc, cls_ds, "accuracy"),
        (ax_rmse, reg_ds, "rmse"),
    ):
        x = np.arange(len(ds_list), dtype=float)
        width = 0.8 / len(competitors)
        all_values = []
        for i, model in enumerate(competitors):
            values = []
            for dataset in ds_list:
                ours = find(rows, dataset, HIST_MODELS[0])
                other = find(rows, dataset, model)
                if ours is None or other is None:
                    values.append(np.nan)
                elif mode == "accuracy":
                    values.append(
                        100.0
                        * (
                            value(other, "accuracy_mean")
                            - value(ours, "accuracy_mean")
                        )
                    )
                else:
                    values.append(
                        100.0
                        * (
                            value(other, "rmse_mean")
                            / value(ours, "rmse_mean")
                            - 1.0
                        )
                    )
            values = np.asarray(values)
            all_values.extend(values[~np.isnan(values)])
            bars = ax.bar(
                x + (i - (len(competitors) - 1) / 2) * width,
                values,
                width,
                label=SHORT[model],
                color=_hist_color(model),
            )
            for bar, item in zip(bars, values):
                if np.isnan(item):
                    continue
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    item + (0.02 if item >= 0 else -0.04),
                    f"{item:+.2f}",
                    ha="center",
                    va="bottom" if item >= 0 else "top",
                    fontsize=7,
                )
        limit = max(0.25, (max(abs(item) for item in all_values) * 1.35) if all_values else 0.25)
        ax.axhline(0.0, color="black", linewidth=0.9)
        ax.set_ylim(-limit, limit)
        ax.set_xticks(x)
        ax.set_xticklabels(ds_list, rotation=15, ha="right")
        ax.grid(True, axis="y", alpha=0.25)
        if mode == "accuracy":
            ax.set_title("Classification accuracy delta")
            ax.set_ylabel("percentage points vs hist-gbdt\n(> 0 = better)")
        else:
            ax.set_title("Regression RMSE relative difference")
            ax.set_ylabel("% vs hist-gbdt\n(> 0 = worse)")
    handles = [
        plt.Line2D(
            [0],
            [0],
            marker="s",
            color="w",
            markerfacecolor=COLORS[model],
            markersize=8,
            label=SHORT[model],
        )
        for model in competitors
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=len(handles),
        fontsize=8,
        frameon=False,
    )
    fig.suptitle("Histogram parity: difference from hist-gbdt (zero = same)")
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_parity_latency(rows: list[dict], path: Path) -> None:
    plt, _ = setup()
    ds_list = datasets(rows)
    x = np.arange(len(ds_list), dtype=float)
    width = 0.8 / len(HIST_MODELS)
    fig, (ax_fit, ax_pred) = plt.subplots(1, 2, figsize=(13.0, 4.8))
    for i, model in enumerate(HIST_MODELS):
        fit = []
        pred = []
        for dataset in ds_list:
            row = find(rows, dataset, model)
            fit.append(value(row, "fit_seconds_mean") if row else np.nan)
            pred.append(
                1000.0 * value(row, "predict_seconds_mean") if row else np.nan
            )
        offset = (i - (len(HIST_MODELS) - 1) / 2) * width
        ax_fit.bar(x + offset, fit, width, label=SHORT[model], color=COLORS[model])
        ax_pred.bar(x + offset, pred, width, label=SHORT[model], color=COLORS[model])
    for ax, title, ylabel in (
        (ax_fit, "Fit time", "seconds"),
        (ax_pred, "Predict time", "milliseconds"),
    ):
        ax.set_yscale("log")
        ax.set_xticks(x)
        ax.set_xticklabels(ds_list, rotation=15, ha="right")
        ax.set_title(title)
        ax.set_ylabel(f"{ylabel} (log scale)")
        ax.grid(True, axis="y", which="both", alpha=0.25)
    ax_fit.legend(fontsize=8, frameon=False)
    fig.suptitle("Histogram family latency (single thread)")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_parity_memory(memory_rows: list[dict], path: Path) -> None:
    plt, _ = setup()
    libraries = ["gbdt", "xgboost", "lightgbm", "sklearn-hist"]
    labels = {
        "gbdt": "hist-gbdt",
        "xgboost": "xgb-hist",
        "lightgbm": "lightgbm",
        "sklearn-hist": "sk-hist",
    }
    colors = {
        "gbdt": COLORS["hist-gbdt (ours, C++)"],
        "xgboost": COLORS["xgboost-hist (3.x)"],
        "lightgbm": COLORS["lightgbm (4.x)"],
        "sklearn-hist": COLORS["sklearn-hist"],
    }
    ds_list = [
        name
        for name in DATASET_ORDER
        if any(row["dataset"] == name for row in memory_rows)
    ]
    x = np.arange(len(ds_list), dtype=float)
    width = 0.8 / len(libraries)
    fig, (ax_peak, ax_delta) = plt.subplots(1, 2, figsize=(13.0, 4.8))
    for i, library in enumerate(libraries):
        peaks = []
        deltas = []
        for dataset in ds_list:
            row = next(
                (
                    item
                    for item in memory_rows
                    if item["dataset"] == dataset and item["library"] == library
                ),
                None,
            )
            peaks.append(float(row["peak_rss_mb"]) if row else np.nan)
            deltas.append(float(row["delta_peak_rss_mb"]) if row else np.nan)
        offset = (i - (len(libraries) - 1) / 2) * width
        ax_peak.bar(
            x + offset,
            peaks,
            width,
            label=labels[library],
            color=colors[library],
        )
        ax_delta.bar(
            x + offset,
            deltas,
            width,
            label=labels[library],
            color=colors[library],
        )
    for ax, title, ylabel in (
        (ax_peak, "Total peak RSS", "MB"),
        (ax_delta, "Peak RSS over post-warmup baseline", "MB"),
    ):
        ax.set_xticks(x)
        ax.set_xticklabels(ds_list, rotation=15, ha="right")
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.grid(True, axis="y", alpha=0.25)
    ax_peak.legend(fontsize=8, frameon=False)
    fig.suptitle("Histogram family training memory (fresh process)")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _landscape_models(task: str) -> list[str]:
    return list(LANDSCAPE_MODELS) + [LINEAR_MODEL[task]]


def _landscape_color(model: str) -> str:
    if model in {"logistic", "ridge"}:
        return LINEAR_COLOR
    return COLORS[model]


def plot_landscape_quality(rows: list[dict], path: Path) -> None:
    plt, _ = setup()
    fig, (ax_acc, ax_rmse) = plt.subplots(
        1,
        2,
        figsize=(13.5, 5.0),
        gridspec_kw={"width_ratios": [1.5, 1.0]},
    )

    cls_ds = datasets(rows, "classification")
    cls_models = _landscape_models("classification")
    x = np.arange(len(cls_ds), dtype=float)
    width = 0.8 / len(cls_models)
    for i, model in enumerate(cls_models):
        values = []
        for dataset in cls_ds:
            row = find(rows, dataset, model)
            values.append(value(row, "accuracy_mean") if row else np.nan)
        ax_acc.bar(
            x + (i - (len(cls_models) - 1) / 2) * width,
            values,
            width,
            label=SHORT.get(model, model),
            color=_landscape_color(model),
        )
    ax_acc.set_xticks(x)
    ax_acc.set_xticklabels(cls_ds, rotation=15, ha="right")
    ax_acc.set_ylabel("accuracy (higher is better)")
    ax_acc.set_title("Classification accuracy")
    ax_acc.grid(True, axis="y", alpha=0.25)

    reg_ds = datasets(rows, "regression")
    reg_models = _landscape_models("regression")[1:]
    x = np.arange(len(reg_ds), dtype=float)
    width = 0.8 / len(reg_models)
    for i, model in enumerate(reg_models):
        values = []
        for dataset in reg_ds:
            ours = find(rows, dataset, LANDSCAPE_MODELS[0])
            row = find(rows, dataset, model)
            if ours is None or row is None:
                values.append(np.nan)
            else:
                values.append(
                    value(row, "rmse_mean") / value(ours, "rmse_mean")
                )
        ax_rmse.bar(
            x + (i - (len(reg_models) - 1) / 2) * width,
            values,
            width,
            label=SHORT.get(model, model),
            color=_landscape_color(model),
        )
    ax_rmse.axhline(1.0, color="black", linewidth=1.0)
    ax_rmse.set_xticks(x)
    ax_rmse.set_xticklabels(reg_ds, rotation=15, ha="right")
    ax_rmse.set_ylabel("RMSE / hist-gbdt (lower is better)")
    ax_rmse.set_title("Regression RMSE relative to hist-gbdt")
    ax_rmse.grid(True, axis="y", alpha=0.25)

    handles = [
        plt.Line2D(
            [0],
            [0],
            marker="s",
            color="w",
            markerfacecolor=_landscape_color(model),
            markersize=8,
            label=SHORT.get(model, model),
        )
        for model in cls_models[:-1]
    ]
    handles.append(
        plt.Line2D(
            [0],
            [0],
            marker="s",
            color="w",
            markerfacecolor=LINEAR_COLOR,
            markersize=8,
            label="linear (logistic / ridge)",
        )
    )
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=6,
        fontsize=8,
        frameon=False,
    )
    fig.suptitle("ML landscape: histogram GBDT against other model families")
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_landscape_pareto(rows: list[dict], path: Path) -> None:
    plt, NullFormatter = setup()
    ds_list = datasets(rows)
    fig, axes = plt.subplots(2, 3, figsize=(13.0, 7.6))
    axes = axes.ravel()
    for ax, dataset in zip(axes, ds_list):
        task = next(row for row in rows if row["dataset"] == dataset)["task"]
        models = _landscape_models(task)
        metric = "rmse_mean" if task == "regression" else "accuracy_mean"
        for model in models:
            row = find(rows, dataset, model)
            if row is None:
                continue
            ax.scatter(
                value(row, "fit_seconds_mean"),
                value(row, metric),
                s=150 if model.startswith("hist-gbdt") else 50,
                marker=MARKERS.get(model, "o"),
                color=_landscape_color(model),
                edgecolor="black" if model.startswith("hist-gbdt") else "none",
                linewidth=0.7,
                zorder=3,
            )
        ax.set_xscale("log")
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.set_title(dataset)
        ax.set_xlabel("fit time (s, log scale)")
        ax.set_ylabel(
            "RMSE (lower is better)" if task == "regression"
            else "accuracy (higher is better)"
        )
        ax.grid(True, which="both", alpha=0.25)
    for ax in axes[len(ds_list):]:
        ax.set_visible(False)

    handles = []
    for model in LANDSCAPE_MODELS:
        handles.append(
            plt.Line2D(
                [0],
                [0],
                marker=MARKERS[model],
                color="w",
                markerfacecolor=COLORS[model],
                markersize=10 if model.startswith("hist-gbdt") else 7,
                label=SHORT[model],
            )
        )
    handles.append(
        plt.Line2D(
            [0],
            [0],
            marker="D",
            color="w",
            markerfacecolor=LINEAR_COLOR,
            markersize=7,
            label="linear (logistic / ridge)",
        )
    )
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=6,
        fontsize=8,
        frameon=False,
    )
    fig.suptitle("Fit time vs quality: hist-gbdt against other families")
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def make_figures(outdir: Path, figdir: Path) -> list[Path]:
    rows = read_csv(outdir / "real_comparison.csv")
    figures = [
        figdir / "parity_quality.png",
        figdir / "parity_latency.png",
        figdir / "landscape_quality.png",
        figdir / "landscape_pareto.png",
    ]
    plot_parity_quality(rows, figures[0])
    plot_parity_latency(rows, figures[1])
    plot_landscape_quality(rows, figures[2])
    plot_landscape_pareto(rows, figures[3])
    memory_path = outdir / "memory_real.csv"
    if memory_path.exists():
        memory_rows = read_csv(memory_path)
        figure = figdir / "parity_memory.png"
        plot_parity_memory(memory_rows, figure)
        figures.append(figure)
    return figures


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
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
    args = parser.parse_args()
    figures = make_figures(args.outdir, args.figdir)
    for figure in figures:
        print(f"wrote {figure}")


if __name__ == "__main__":
    main()
