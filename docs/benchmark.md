# Real-data benchmark report

This is the benchmark companion to `hist-gbdt`. It is a **learning
project**: the point is to make the machinery inspectable and to compare it
honestly with mature libraries, not to claim a production win.

The report deliberately asks two different questions instead of putting every
model on one crowded chart:

1. **Histogram parity** — is `hist-gbdt` in the same quality band as the other
   histogram implementations, and what do speed and memory cost?
2. **ML landscape** — where does a histogram GBDT sit against exact GBDT,
   classic GBDT, bagging, and linear models?

Part 1 compares like with like. Part 2 uses `hist-gbdt` as the single
histogram-GBDT representative and changes the model family. The full per-fold
numbers are in [`bench/results/real_summary.md`](../bench/results/real_summary.md)
and [`bench/results/memory_summary.md`](../bench/results/memory_summary.md).

## Protocol

- Machine: AMD Ryzen 7 7735HS with Radeon Graphics (16 logical CPUs).
- Python 3.14.7, NumPy 2.5.4, scikit-learn 1.9.1, XGBoost 3.4.1,
  LightGBM 4.6.0, hist-gbdt 0.1.0.
- **Single-thread** comparison: `n_jobs=1` and
  `OMP_NUM_THREADS=OPENBLAS_NUM_THREADS=MKL_NUM_THREADS=1`.
  This isolates implementation choices from thread scheduling. The C++
  OpenMP backend is still used; it simply runs with one worker.
- 3-fold stratified K-fold (classification) or K-fold (regression),
  `random_state=0`. Every model sees the same folds.
- 60 trees, depth 4, learning rate 0.1, `max_bin=256`, L2
  regularization 1.0, `min_child_weight=1.0`. No early stopping and no
  per-model tuning. This is a fixed-budget breadth comparison, not a
  leaderboard.
- One warmup fit per model, outside the timed folds, so library
  initialization is not charged to the first fold.
- XGBoost Exact is only run when `n_samples <= 2000` and
  `n_features <= 32`. It represents the older exact-greedy split strategy;
  it is not an old XGBoost package version.

## Datasets

| dataset | task | n | k | classes |
|---|---|---:|---:|---:|
| `breast_cancer` | binary classification | 569 | 30 | 2 |
| `wine` | multiclass classification | 178 | 13 | 3 |
| `digits` | multiclass classification | 1,797 | 64 | 10 |
| `diabetes` | regression | 442 | 10 | — |
| `california_housing` | regression | 20,640 | 8 | — |

All are real datasets bundled with scikit-learn (the California housing
data is downloaded and cached). They have no missing values; missing-value
handling is covered by the synthetic and parity tests instead.

## Part 1 — Histogram parity

The first view keeps the algorithm family fixed. The only changes are the
implementation and the engineering around it: XGBoost Hist, LightGBM,
scikit-learn HistGradientBoosting, and this package.

### Predictive parity

Mean over three folds. `hist-gbdt` is the reference row for the figures; the
table shows absolute values so the comparison does not hide small differences.

| dataset | metric | hist-gbdt | xgb-hist | lightgbm | sklearn-hist |
|---|---|---:|---:|---:|---:|
| `breast_cancer` | accuracy | 0.9649 | 0.9649 | 0.9649 | 0.9561 |
| `wine` | accuracy | 0.9496 | 0.9496 | 0.9496 | 0.9497 |
| `digits` | accuracy | 0.9577 | 0.9510 | 0.9572 | 0.9605 |
| `diabetes` | RMSE | 60.1940 | 60.0499 | 59.8554 | 60.2261 |
| `california_housing` | RMSE | 0.5254 | 0.5257 | 0.5280 | 0.5275 |

![Histogram parity: quality deltas](./figures/parity_quality.png)

The left panel is accuracy in percentage points relative to `hist-gbdt`; the
right panel is RMSE as a percentage of `hist-gbdt`. Zero means identical.

The story is parity, with one honest exception: scikit-learn Hist is 0.88 pp
worse on `breast_cancer` but 0.28 pp better on `digits`. XGBoost Hist is
identical on `breast_cancer` and `wine`, and 0.67 pp lower on `digits`.
LightGBM is within 0.06 pp everywhere. Regression differences are at most
0.56% RMSE. These gaps are small relative to fold-to-fold variation; none of
this establishes a ranking among histogram implementations.

### Latency

Single-thread fit time in seconds and prediction time in milliseconds.

| dataset | hist-gbdt fit | xgb-hist fit | lightgbm fit | sklearn-hist fit |
|---|---:|---:|---:|---:|
| `breast_cancer` | 0.0523 | 0.0530 | 0.0172 | 0.0738 |
| `wine` | 0.0200 | 0.0240 | 0.0111 | 0.0734 |
| `digits` | 0.3988 | 0.2212 | 0.1753 | 0.7066 |
| `diabetes` | 0.0222 | 0.0187 | 0.0120 | 0.0433 |
| `california_housing` | 0.1462 | 0.0695 | 0.0603 | 0.0848 |

| dataset | hist-gbdt predict | xgb-hist predict | lightgbm predict | sklearn-hist predict |
|---|---:|---:|---:|---:|
| `breast_cancer` | 6.61 | 0.75 | 1.34 | 1.15 |
| `wine` | 6.47 | 0.68 | 1.18 | 1.89 |
| `digits` | 77.28 | 4.16 | 13.08 | 19.70 |
| `diabetes` | 6.39 | 0.62 | 1.08 | 1.13 |
| `california_housing` | 44.43 | 4.20 | 11.82 | 14.64 |

![Histogram parity: fit and predict latency](./figures/parity_latency.png)

Fit time is respectable on small data: on `breast_cancer`, `wine`, and
`diabetes`, `hist-gbdt` is in the same range as XGBoost Hist and faster than
scikit-learn Hist. It loses on the larger problems: on `digits` and
`california_housing` it is about 2.3–2.4× slower than LightGBM and about
1.8–2.1× slower than XGBoost Hist. Prediction is the clear bottleneck: pure
Python tree traversal is roughly 3–19× slower than the compiled libraries and
hurts most on multiclass `digits` (10 classes × 60 trees).

### Memory

Fresh process per (library, dataset), two repetitions, median shown. One
warmup fit happens before the measured fit.

| dataset | hist-gbdt | xgb-hist | lightgbm | sklearn-hist |
|---|---:|---:|---:|---:|
| `breast_cancer` | 130.2 | 323.7 | 156.5 | 143.1 |
| `wine` | 127.9 | 321.2 | 153.8 | 141.6 |
| `digits` | 131.1 | 327.2 | 160.0 | 146.6 |
| `diabetes` | 128.6 | 320.7 | 152.9 | 142.2 |
| `california_housing` | 130.4 | 322.6 | 155.3 | 144.2 |

Total peak RSS in MB. Fit-time peak growth over the post-warmup baseline is
only 0.1–1.5 MB for `hist-gbdt`, 0.2–2.0 MB for XGBoost Hist, 0.6–2.3 MB for
LightGBM, and 0.1–2.5 MB for scikit-learn Hist.

![Histogram parity: training memory](./figures/parity_memory.png)

The total-RSS result mostly reflects runtime footprint: XGBoost's native
runtime and import path cost far more than this small NumPy/C++ package.
The fit deltas are all only a few megabytes at these sizes, so do not
over-interpret them. `ru_maxrss` is a high-water mark; use a profiler if
you need allocation-level attribution.

### Parity takeaway

At this fixed budget, `hist-gbdt` is in the same quality band as the other
histogram implementations. The differences are not a ranking. Its fit is
competitive on small data, slower on the larger problems, and its prediction
path is the main engineering weakness. Memory is low end to end.

## Part 2 — ML landscape

The second view keeps `hist-gbdt` as the histogram-GBDT representative and
changes the model family:

- **exact GBDT:** XGBoost Exact, when the size cap allows it;
- **classic GBDT:** scikit-learn's sequential GradientBoosting;
- **bagging:** RandomForest and ExtraTrees;
- **linear:** logistic regression (classification) or ridge (regression).

This is not a tuned tournament. It is a map of where a histogram GBDT sits
when the assumptions change.

### Quality across families

| dataset | metric | hist-gbdt | best non-hist model | value | difference vs ours |
|---|---|---:|---|---:|---:|
| `breast_cancer` | accuracy | 0.9649 | logistic | 0.9772 | +1.23 pp |
| `wine` | accuracy | 0.9496 | ExtraTrees | 0.9831 | +3.35 pp |
| `digits` | accuracy | 0.9577 | logistic | 0.9705 | +1.28 pp |
| `diabetes` | RMSE | 60.1940 | ridge | 54.6891 | −9.14% |
| `california_housing` | RMSE | 0.5254 | sklearn-gb | 0.5277 | +0.44% |

![ML landscape: quality across model families](./figures/landscape_quality.png)

The small-data story is consistent: linear models win `breast_cancer` and
`digits`, ridge wins `diabetes`, and ExtraTrees has the best `wine` accuracy.
The `wine` result needs a caveat: ExtraTrees has much worse log-loss
(0.3356) than the boosting models (about 0.12–0.13), so its accuracy win is
not a clean probability win. Logistic regression is close on accuracy and
much better calibrated (log-loss 0.0660).

On `california_housing`, `hist-gbdt` is the best model in this landscape set,
but the margin over classic GradientBoosting is only 0.44% RMSE, and XGBoost
Hist is closer still in Part 1. Treat it as a tie band, not a victory lap.
XGBoost Exact is absent on `digits` (64 features, above the feature cap) and
on `california_housing` (above the sample cap).

### Fit time versus quality

![ML landscape: fit time versus quality](./figures/landscape_pareto.png)

The star is `hist-gbdt`. Each panel has its own metric scale; the x-axis is
log-scaled fit time. Linear models are extremely cheap and win several small
datasets. Bagging is expensive for mixed returns. Exact and classic GBDT do
not buy accuracy here at this tree budget. `hist-gbdt` is near the good corner
on `california_housing` but still slower than the lightweight histogram
libraries compared in Part 1.

### Landscape takeaway

There is no universal winner. Histogram GBDT is the right tool when the
signal is nonlinear and interactions matter, but with small tabular data a
linear model or ExtraTrees can be better and cheaper. The honest claim is
**competitive in the GBDT band**, not state of the art.

## Reading the numbers honestly

- **Small data favors simple models.** Logistic regression wins
  `breast_cancer` and `digits`; ridge wins `diabetes`; ExtraTrees has the
  best `wine` accuracy. The tree ensembles are not tuned, and the signal in
  these small tabular sets is close to linear.
- **`hist-gbdt` is in the same quality band as the other GBDTs** and is
  numerically best on `california_housing` (a statistical tie with XGBoost).
  It is not uniformly better than XGBoost, LightGBM, or scikit-learn.
- **Fit time is respectable, prediction is not.** The multiclass prediction
  gap is the clearest bottleneck; a compiled prediction path is the obvious
  next optimization.
- **Memory does not explain the quality result.** At these sizes, training
  growth is only a few MB; total RSS is dominated by runtime and imports.
- **This is not a claim of state-of-the-art accuracy.** The protocol is
  deliberately fixed and untuned. A fair accuracy comparison would tune
  each family, use early stopping, and repeat over more seeds.
- **Missing data and categorical features are not benchmarked here.**
  Missing-value routing has dedicated tests; categorical support is not
  implemented.

## Reproduce

After activating the virtual environment from the quickstart, collect raw folds
(this is the only step that fits models):

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python bench/compare_real.py \
  --folds 3 --n-estimators 60 --max-depth 4 --n-jobs 1

python bench/memory_real.py \
  --repeat 2 --n-estimators 60 --max-depth 4 --n-jobs 1
```

Regenerate every figure from the saved CSV **without refitting**:

```bash
python bench/plot_real.py
```

`compare_real.py` also calls the plotting module automatically unless
`--no-figures` is passed. `memory_real.py` only collects and summarizes
memory data; `plot_real.py` builds the parity memory figure.

Outputs:

- `bench/results/real_comparison.csv`, `real_runs.csv`, `real_summary.md`
- `bench/results/memory_real.csv`, `memory_runs.csv`, `memory_summary.md`
- `docs/figures/parity_quality.png`, `parity_latency.png`, `parity_memory.png`
- `docs/figures/landscape_quality.png`, `landscape_pareto.png`
