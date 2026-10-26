# hist-gbdt

A high-performance **Histogram Gradient Boosted Decision Tree (GBDT)** engine written from scratch: a NumPy reference implementation plus a multi-threaded C++ (pybind11 + OpenMP) core.

> **This is a learning & portfolio project.** It exists to understand the mathematics behind 2nd-order Taylor loss optimization (XGBoost) and the systems architecture behind histogram-based split finding (LightGBM / XGBoost Hist). It is designed to be mathematically rigorous, transparent, and tested directly against XGBoost.

> **Status (docs-first):** the mathematical monograph and repository scaffold are complete. The NumPy reference engine, C++/OpenMP core, test suite, and benchmarks are in progress. The architecture below describes the target design, not shipped code yet.

---

## Why this exists

- **Understand 2nd-Order Taylor Optimization**: How computing gradients $g_i$ and Hessians $h_i$ turns split-finding and leaf-weight derivation into an exact Newton-Raphson step on tree structures.
- **Implement Histogram-Based Splitting**: Replace $O(K \cdot N \log N)$ exact greedy sorting with an $O(K \cdot N_{\text{node}})$ histogram build plus an $O(K \cdot B)$ split scan using `uint8` quantization. This cuts raw feature-matrix storage by up to 87.5% versus `float64`; total training-memory savings versus XGBoost are measured separately.
- **Histogram Subtraction Trick**: Roughly halve histogram construction cost below the root by computing the smaller child and getting the sibling via $H_{\text{sibling}} = H_{\text{parent}} - H_{\text{child}}$.
- **Multi-threaded C++ with OpenMP**: Write cache-friendly C++ kernels with `#pragma omp parallel for` across features and samples.
- **Fair Comparisons with XGBoost**: Compare training speed, memory consumption, and AUC/RMSE under identical hyperparameter conditions.

---

## Documentation

Full mathematical derivations and systems design are documented in [`docs/theory.pdf`](docs/theory.pdf) (compiled from [`docs/theory.tex`](docs/theory.tex)):

```
Tabular Limits → Decision Trees → Ensemble Variance/Bias 
    → 2nd-Order Taylor Expansion → Optimal Leaf Weights w* 
    → Split Gain Formula (with L2 λ & Pruning γ)
    → Histogram Binning (uint8) → Histogram Subtraction Trick
    → Multi-threaded OpenMP Architecture
```

---

## Planned Architecture & Code Layout

| Path | Description |
|---|---|
| `docs/theory.pdf` | Comprehensive mathematical foundations and proofs monograph (LaTeX) |
| `docs/theory.tex` | LaTeX source code for the theory paper |
| `src/gbdt/reference.py` | Pure Python / NumPy reference implementation (correctness oracle) |
| `src/gbdt/_core.cpp` | Multi-threaded C++ core (pybind11 + OpenMP histogram engine) |
| `src/gbdt/backend.py` | Scikit-learn compatible wrappers (`HistGBDTClassifier`, `HistGBDTRegressor`) |
| `tests/` | Unit tests, numerical parity vs reference, and correctness vs XGBoost |
| `bench/bench.py` | Timing and memory benchmarking harness vs XGBoost |
| `scripts/gen_data.py` | Synthetic tabular dataset generator for classification & regression |

---

## Planned Core Design

### 1. Objective & Split Gain
For any twice-differentiable loss function $l(y, \hat{y})$:
$$\text{Gain} = \frac{1}{2} \left[ \frac{G_L^2}{H_L + \lambda} + \frac{G_R^2}{H_R + \lambda} - \frac{(G_L + G_R)^2}{H_L + H_R + \lambda} \right] - \gamma$$
$$w^* = -\frac{G}{H + \lambda}$$

### 2. Planned Objectives
- **Regression**: Squared Error Loss ($g = \hat{y} - y$, $h = 1$).
- **Binary Classification**: Binary Cross-Entropy / Log-Loss ($g = p - y$, $h = p(1 - p)$).

---

## Quickstart

```bash
# Set up virtual environment
python -m venv .venv
source .venv/bin/activate
pip install -e ".[test,bench]"

# After implementation: run test suite
pytest -v

# After implementation: run benchmark vs XGBoost
python bench/bench.py
```
