# Memory benchmark

- Machine: AMD Ryzen 7 7735HS with Radeon Graphics
- Python 3.14.7
- 2 fresh-process repetitions per point; median shown
- 60 trees, depth 4, n_jobs=1
- One warmup fit is performed before the measured fit. Total peak RSS includes that warmup and library import/runtime overhead.
- Delta is measured against a post-warmup/post-data/post-model baseline; it is a coarse training-memory estimate, not pure model size.


### breast_cancer (classification, n=569, k=30)

| library | peak RSS (MB) | delta peak (MB) | RSS after fit (MB) | fit (s) |
|---|---:|---:|---:|---:|
| gbdt | 130.2 | 1.5 | 130.4 | 0.0799 |
| xgboost | 323.7 | 2.0 | 324.3 | 0.0627 |
| lightgbm | 156.5 | 2.3 | 154.4 | 0.0297 |
| sklearn-hist | 143.1 | 0.6 | 143.8 | 0.0860 |
| sklearn-gb | 142.0 | 0.1 | 142.6 | 0.3154 |

### wine (multiclass, n=178, k=13)

| library | peak RSS (MB) | delta peak (MB) | RSS after fit (MB) | fit (s) |
|---|---:|---:|---:|---:|
| gbdt | 127.9 | 0.1 | 128.6 | 0.0251 |
| xgboost | 321.2 | 0.2 | 321.8 | 0.0311 |
| lightgbm | 153.8 | 0.6 | 154.6 | 0.0193 |
| sklearn-hist | 141.6 | 0.1 | 142.6 | 0.0856 |
| sklearn-gb | 141.6 | 0.0 | 142.1 | 0.2177 |

### digits (multiclass, n=1797, k=64)

| library | peak RSS (MB) | delta peak (MB) | RSS after fit (MB) | fit (s) |
|---|---:|---:|---:|---:|
| gbdt | 131.1 | 1.4 | 132.0 | 0.5628 |
| xgboost | 327.2 | 1.8 | 327.8 | 0.2856 |
| lightgbm | 160.0 | 1.8 | 157.6 | 0.3073 |
| sklearn-hist | 146.6 | 2.5 | 146.6 | 0.9130 |
| sklearn-gb | 145.2 | 1.1 | 145.7 | 6.0925 |

### diabetes (regression, n=442, k=10)

| library | peak RSS (MB) | delta peak (MB) | RSS after fit (MB) | fit (s) |
|---|---:|---:|---:|---:|
| gbdt | 128.6 | 0.6 | 129.0 | 0.0252 |
| xgboost | 320.7 | 0.5 | 321.3 | 0.0216 |
| lightgbm | 152.9 | 0.9 | 153.4 | 0.0112 |
| sklearn-hist | 142.2 | 0.1 | 142.9 | 0.0476 |
| sklearn-gb | 140.7 | 0.0 | 141.2 | 0.0744 |

### california_housing (regression, n=20640, k=8)

| library | peak RSS (MB) | delta peak (MB) | RSS after fit (MB) | fit (s) |
|---|---:|---:|---:|---:|
| gbdt | 130.4 | 1.2 | 131.4 | 0.2007 |
| xgboost | 322.6 | 1.8 | 323.7 | 0.0857 |
| lightgbm | 155.3 | 1.4 | 156.1 | 0.0828 |
| sklearn-hist | 144.2 | 1.1 | 144.9 | 0.1032 |
| sklearn-gb | 143.9 | 1.6 | 144.4 | 3.3033 |

## Reading the numbers honestly

- Peak RSS includes the Python interpreter, NumPy/scikit-learn, the input data, and library import overhead. Compare both total peak and the delta rather than treating either as pure model size.
- `ru_maxrss` is a high-water mark, so a transient allocation during data loading can hide a later training allocation. Repeated fresh processes reduce, but do not eliminate, that noise.
- This is not a substitute for a memory profiler such as `memray`/`tracemalloc`; it is the end-to-end number a user sees.
