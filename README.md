# hist-gbdt

A **Histogram Gradient Boosted Decision Tree (GBDT)** engine written from
scratch: a readable NumPy reference implementation, an optional
multi-threaded C++/OpenMP tree builder, and a scikit-learn-style API.

This is a learning project, built for fun and as a way to get started with
XGBoost and tree-based algorithms. It exists to make the mathematics of
2nd-order Taylor boosting and the systems ideas behind histogram split
finding explicit, testable, and comparable with XGBoost. It is **not** a
production replacement for XGBoost, LightGBM, or scikit-learn.

---

## What is implemented

- **2nd-order Taylor boosting** with per-leaf Newton weights
  `w* = -G / (H + λ)` and split gain
  `½[G_L²/(H_L+λ) + G_R²/(H_R+λ) - G²/(H+λ)] - γ`.
- **Objectives:** least-squares regression, binary logistic loss, and
  multiclass softmax.
- **Missing values:** NaN samples get a dedicated bin per feature and are
  routed left or right by the learned `default_left` direction.
- **Histogram binning:** quantile edges, at most 256 bins, `uint8` codes,
  one bin per value for low-cardinality features.
- **Histogram split finding:** prefix sums over bins, L2 regularization,
  minimum gain (`gamma`), minimum child Hessian, and minimum leaf size.
- **Tree growth:** level-wise greedy trees with configurable depth.
- **Histogram subtraction:** the C++ tree builder reuses the parent
  histogram and computes the larger child by subtraction.
- **Boosting controls:** learning rate, row subsampling, column
  subsampling, early stopping, deterministic seeds.
- **Two backends:** pure NumPy reference and a C++/OpenMP tree builder
  that parallelizes histogram construction and split scans across
  features.
- **Scikit-learn-style estimators:** `HistGBDTRegressor` and
  `HistGBDTClassifier`.
- **Tests** for hand-computed gains, exact-vs-histogram splits, backend
  parity, API behavior, edge cases, and optional XGBoost parity.
- **Benchmarks** for synthetic fit/predict time, real-data cross-validated
  quality, and fresh-process peak RSS.

The boosting loop and the NumPy reference tree builder are intentionally
written in Python so the algorithm stays inspectable. The C++ backend
grows a complete level-wise tree, including histogram construction,
split search, missing-value routing, and histogram subtraction.

---

## Documentation

Full derivations and the systems design are in
[`docs/theory.pdf`](docs/theory.pdf), compiled from
[`docs/theory.tex`](docs/theory.tex). The benchmark report is
[`docs/benchmark.md`](docs/benchmark.md):

```
Tabular Limits → Decision Trees → Ensemble Variance/Bias
    → 2nd-Order Taylor Expansion → Optimal Leaf Weights w*
    → Split Gain Formula (with L2 λ & Pruning γ)
    → Histogram Binning (uint8) → Missing-Value Bins
    → Histogram Subtraction Trick → Multiclass Softmax
    → Multi-threaded OpenMP Architecture
```

---

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[test,bench]"

pytest -q
python bench/bench.py --quick
```

The C++ extension is built by `pip install -e .` and needs an OpenMP
compiler. If it is not available, the package automatically falls back
to the NumPy backend. Select a backend explicitly with
`backend="numpy"` or `backend="cpp"`.

```python
import numpy as np
from gbdt import HistGBDTClassifier, HistGBDTRegressor

rng = np.random.default_rng(0)
X = rng.normal(size=(500, 5))
y = 2.0 * X[:, 0] - X[:, 1] + 0.3 * X[:, 0] * X[:, 2]

reg = HistGBDTRegressor(
    n_estimators=100,
    max_depth=4,
    learning_rate=0.1,
    max_bin=256,
    backend="auto",
).fit(X, y)
print(reg.predict(X[:3]))
print(reg.feature_importances_)

labels = (X[:, 0] + X[:, 1] > 0).astype(int)
clf = HistGBDTClassifier(
    n_estimators=100,
    max_depth=3,
    learning_rate=0.1,
).fit(X, labels)
print(clf.predict_proba(X[:3]))

multi_labels = (X[:, 0] > 0).astype(int) + (X[:, 1] > 0).astype(int)
X_missing = X.copy()
X_missing[rng.random(X_missing.shape) < 0.1] = np.nan
multi = HistGBDTClassifier(
    n_estimators=80,
    max_depth=3,
    learning_rate=0.1,
).fit(X_missing, multi_labels)
print(multi.predict_proba(X_missing[:3]))
```

Early stopping uses the last `(X, y)` pair in `eval_set`:

```python
reg = HistGBDTRegressor(n_estimators=1000, max_depth=4)
reg.fit(
    X_train,
    y_train,
    eval_set=[(X_valid, y_valid)],
    early_stopping_rounds=30,
)
print(reg.n_estimators_, reg.best_iteration_)
```

---

## Code layout

| Path | Description |
|---|---|
| `src/gbdt/binning.py` | Quantile binner; `(n_features, n_samples)` `uint8` layout |
| `src/gbdt/splitter.py` | NumPy histogram split finder |
| `src/gbdt/reference.py` | Exact greedy splitter used as a test oracle |
| `src/gbdt/tree.py` | Flat-array tree + level-wise tree builder |
| `src/gbdt/objectives.py` | MSE, logistic, and softmax gradients/Hessians |
| `src/gbdt/booster.py` | Boosting loop, subsampling, early stopping |
| `src/gbdt/backend.py` | Backend selection, splitter and C++ tree wrappers |
| `src/gbdt/sklearn_api.py` | `HistGBDTRegressor`, `HistGBDTClassifier` |
| `src/gbdt/_core.cpp` | OpenMP split kernel and full tree builder (pybind11) |
| `src/gbdt/datasets.py` | Synthetic regression/binary/multiclass data |
| `tests/` | Math, parity, API, edge-case, optional XGBoost tests |
| `bench/bench.py` | Synthetic fit/predict timing harness vs XGBoost Hist |
| `bench/compare_real.py` | Repeated-CV comparison on real sklearn datasets |
| `bench/memory_real.py` | Fresh-process memory data collection on real datasets |
| `bench/plot_real.py` | Regenerate parity/landscape figures from saved CSV |
| `scripts/gen_data.py` | Save a synthetic dataset to `.npz` |
| `scripts/measure_memory.py` | Peak-RSS measurement in a fresh process |
| `docs/theory.pdf` | Mathematical foundations monograph |
| `docs/benchmark.md` | Full benchmark protocol, tables, figures and caveats |

---

## Design notes

### Split finding

For a node with gradient sums `G`, Hessian sums `H`, and `B` histogram
bins, a feature histogram takes `O(N_node)` work. Scanning all `B - 1`
thresholds and evaluating the gain takes `O(B)`. Across `K` features,
the node costs `O(K · N_node + K · B)`. For a complete level-wise tree
with `D` levels, histogram builds sum to `O(D · K · N)`, while split
scans add `O(K · B · 2^D)`; `B ≤ 256`.

The implementation uses inclusive left partitions:
`Xbin[feature, row] <= threshold_bin`. A split is accepted only when
its regularized gain is strictly positive and both children satisfy
`min_child_weight` and `min_samples_leaf`. Every feature reserves its
last bin for NaN; for each threshold both missing-left and missing-right
routings are evaluated, with deterministic tie-breaking.

### Backends

`backend="auto"` chooses C++ when the extension is importable, otherwise
NumPy. `backend="numpy"` is the portable reference and is useful for
debugging. `backend="cpp"` raises if the extension was not built.

The C++ kernel receives the binned matrix, node row indices, gradients,
Hessians, candidate features, and missing-bin indices. It builds one
histogram per feature in parallel (`#pragma omp parallel for`) and then
performs a deterministic serial reduction over features. For each split
it builds the histogram of the smaller child and obtains the larger
child's histogram by subtraction; the Python layer owns the boosting
loop and the NumPy reference tree builder.

### Memory

Feature matrices are stored as `uint8` rather than `float64`, which is an
8× reduction for that matrix alone. Histograms are `O(K · B)` doubles
per thread. Total process memory also includes the original input,
gradient/Hessian arrays, tree rows, and Python/NumPy overhead, so compare
end-to-end RSS with `scripts/measure_memory.py` rather than assuming the
feature-matrix ratio applies to the whole training process.

---

## Tests

```bash
pytest -q
```

The test suite includes:

- hand-computed histogram gain and leaf weights;
- exact greedy vs histogram split agreement on low-cardinality data;
- NumPy vs C++ split, tree, and prediction parity, including missing
  values and multiclass softmax (skipped if the extension is not built);
- missing-value routing, all-missing features, and NaN input handling;
- multiclass probabilities, string labels, early stopping, and
  reproducibility;
- input validation and optional AUC parity against XGBoost Hist.

---

## Benchmarks

Quick smoke benchmark:

```bash
python bench/bench.py --quick
```

Full synthetic benchmark (default 20k × 20, 100 trees):

```bash
python bench/bench.py --task all --repeat 3
```

Multiclass with missing values:

```bash
python bench/bench.py \
  --task multiclass --n-classes 4 --missing-fraction 0.1 --quick
```

### Real-data comparison

The report separates two questions instead of putting every model on one
crowded chart:

- **Histogram parity:** `hist-gbdt` vs XGBoost Hist, LightGBM, and
  scikit-learn Hist — same algorithm family, different engineering.
- **ML landscape:** `hist-gbdt` as the histogram-GBDT representative vs exact
  GBDT, classic GBDT, bagging, and linear models.

Protocol: 3-fold CV, 60 trees, depth 4, learning rate 0.1, `max_bin=256`,
single-thread (`n_jobs=1`), one warmup per model, no early stopping, no
per-model tuning. This is a fixed-budget breadth comparison, not a
leaderboard. XGBoost Exact is only run on small feature counts.

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python bench/compare_real.py \
  --folds 3 --n-estimators 60 --max-depth 4 --n-jobs 1
```

#### Part 1 — Histogram parity

| dataset | metric | hist-gbdt | xgb-hist | lightgbm | sklearn-hist |
|---|---|---:|---:|---:|---:|
| breast_cancer | accuracy | 0.9649 | 0.9649 | 0.9649 | 0.9561 |
| wine | accuracy | 0.9496 | 0.9496 | 0.9496 | 0.9497 |
| digits | accuracy | 0.9577 | 0.9510 | 0.9572 | 0.9605 |
| diabetes | RMSE | 60.1940 | 60.0499 | 59.8554 | 60.2261 |
| california_housing | RMSE | **0.5254** | 0.5257 | 0.5280 | 0.5275 |

![Histogram parity: quality deltas](./docs/figures/parity_quality.png)

XGBoost Hist is identical on `breast_cancer` and `wine` and 0.67 pp lower on
`digits`. LightGBM is within 0.06 pp everywhere. scikit-learn Hist is 0.88 pp
worse on `breast_cancer` but 0.28 pp better on `digits`. Regression gaps are
at most 0.56% RMSE. The differences are small relative to fold noise; this is
parity, not a ranking.

![Histogram parity: fit and predict latency](./docs/figures/parity_latency.png)

Fit is competitive on small data but slower on the larger sets: on `digits`,
0.40 s vs 0.22 s (XGBoost), 0.18 s (LightGBM), and 0.71 s (sklearn Hist).
Prediction is still Python tree traversal: on `digits`, 77.3 ms vs 4.2–19.7 ms
for the libraries. A compiled prediction path is the obvious next
optimization.

![Histogram parity: training memory](./docs/figures/parity_memory.png)

| dataset | hist-gbdt | xgb-hist | lightgbm | sklearn-hist |
|---|---:|---:|---:|---:|
| breast_cancer | 130.2 | 323.7 | 156.5 | 143.1 |
| wine | 127.9 | 321.2 | 153.8 | 141.6 |
| digits | 131.1 | 327.2 | 160.0 | 146.6 |
| diabetes | 128.6 | 320.7 | 152.9 | 142.2 |
| california_housing | 130.4 | 322.6 | 155.3 | 144.2 |

Total peak RSS in MB, fresh process, median of two runs. Fit-time growth over
the post-warmup baseline is only a few MB for every library. XGBoost's total
is dominated by its native runtime and import path.

#### Part 2 — ML landscape

| dataset | metric | hist-gbdt | best non-hist model | value | difference vs ours |
|---|---|---:|---|---:|---:|
| breast_cancer | accuracy | 0.9649 | logistic | 0.9772 | +1.23 pp |
| wine | accuracy | 0.9496 | ExtraTrees | 0.9831 | +3.35 pp |
| digits | accuracy | 0.9577 | logistic | 0.9705 | +1.28 pp |
| diabetes | RMSE | 60.1940 | ridge | 54.6891 | −9.14% |
| california_housing | RMSE | **0.5254** | sklearn-gb | 0.5277 | +0.44% |

![ML landscape: quality across model families](./docs/figures/landscape_quality.png)

![ML landscape: fit time versus quality](./docs/figures/landscape_pareto.png)

Small data favors simple models: logistic wins `breast_cancer` and `digits`,
ridge wins `diabetes`, and ExtraTrees has the best `wine` accuracy. The
`wine` win is not a clean probability win: ExtraTrees has log-loss 0.3356 vs
about 0.12–0.13 for the boosting models. On `california_housing`, `hist-gbdt`
is best in this landscape set, but the 0.44% RMSE gap to classic
GradientBoosting is a tie band, not a victory.

#### Reproduce the numbers and figures

After the quickstart environment is active, run the whole pipeline from the
repository root:

```bash
# 1. Real-data quality and latency rows, plus figures.
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python bench/compare_real.py \
  --folds 3 --n-estimators 60 --max-depth 4 --n-jobs 1

# 2. Fresh-process memory rows (quality rows are not refitted).
python bench/memory_real.py \
  --repeat 2 --n-estimators 60 --max-depth 4 --n-jobs 1

# 3. Rebuild every figure from the saved CSV only. No model fitting.
python bench/plot_real.py
```

All image links above are repo-relative (`./docs/figures/...`), so they work
on any machine after cloning. The scripts write to these defaults:

- `bench/results/real_comparison.csv`, `real_runs.csv`, `real_summary.md`
- `bench/results/memory_real.csv`, `memory_runs.csv`, `memory_summary.md`
- `docs/figures/parity_*.png` and `docs/figures/landscape_*.png`

Use `--outdir` and `--figdir` on the scripts if you want different output
locations. The full report with protocol, all-model tables, and caveats is
[`docs/benchmark.md`](docs/benchmark.md); raw outputs are in
[`bench/results/real_summary.md`](bench/results/real_summary.md) and
[`bench/results/memory_summary.md`](bench/results/memory_summary.md).

Generate and save a dataset:

```bash
python scripts/gen_data.py --task regression --n-samples 50000
python scripts/gen_data.py \
  --task multiclass --n-classes 4 --missing-fraction 0.1
```

Use the same hyperparameters when comparing. XGBoost is run with
`tree_method="hist"` and the same `max_bin`, depth, learning rate, `λ`,
and `min_child_weight`.

---

## Current limitations

- No ranking objective or public custom-objective API.
- No categorical-feature handling.
- Histogram splits only; there is no exact-greedy training mode in the
  public API.
- The NumPy reference tree builder rebuilds each node histogram; only
  the C++ tree builder uses histogram subtraction.
- Prediction is pure Python tree traversal, which is visible in
  multiclass benchmark latency.
- No model serialization format yet.
- The sklearn-style wrappers are lightweight and do not implement the
  full estimator API (e.g. `pickle` stability or all metadata routes).

---

## References

- Chen, T., & Guestrin, C. (2016). *XGBoost: A Scalable Tree Boosting
  System.* KDD.
- Ke, G., et al. (2017). *LightGBM: A Highly Efficient Gradient Boosting
  Decision Tree.* NeurIPS.
- Friedman, J. (2001). *Greedy Function Approximation: A Gradient
  Boosting Machine.* Annals of Statistics.
- Breiman, L., et al. (1984). *Classification and Regression Trees.*
