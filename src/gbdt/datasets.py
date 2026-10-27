"""Small synthetic tabular datasets used by the scripts and benchmarks."""

from __future__ import annotations

import numpy as np

from .objectives import sigmoid, softmax


def _latent_signal(
    X: np.ndarray, n_informative: int, rng: np.random.Generator
) -> np.ndarray:
    n_informative = min(n_informative, X.shape[1])
    weights = rng.normal(size=n_informative)
    weights /= np.sqrt(n_informative)
    signal = X[:, :n_informative] @ weights
    if n_informative >= 2:
        signal = signal + 0.5 * X[:, 0] * X[:, 1]
    std = float(signal.std())
    if std > 0.0:
        signal = (signal - signal.mean()) / std
    return signal


def make_dataset(
    task: str = "regression",
    n_samples: int = 10_000,
    n_features: int = 20,
    n_informative: int = 8,
    noise: float = 1.0,
    test_fraction: float = 0.2,
    n_classes: int = 3,
    missing_fraction: float = 0.0,
    random_state: int | None = 0,
):
    """Return ``X_train, X_test, y_train, y_test``.

    The first ``n_informative`` columns carry signal; the rest are noise.
    Regression targets are continuous, binary targets are 0/1, and
    multiclass targets are class ids in ``[0, n_classes)``.  When
    ``missing_fraction`` is positive, independent NaN entries are injected
    into both feature splits.
    """
    if n_samples < 2:
        raise ValueError("n_samples must be at least 2")
    if n_features < 1:
        raise ValueError("n_features must be at least 1")
    if n_informative < 1:
        raise ValueError("n_informative must be at least 1")
    if not 0.0 < test_fraction < 1.0:
        raise ValueError("test_fraction must be in (0, 1)")
    if n_classes < 2:
        raise ValueError("n_classes must be at least 2")
    if not 0.0 <= missing_fraction < 1.0:
        raise ValueError("missing_fraction must be in [0, 1)")

    task = task.lower()
    if task not in {"regression", "classification", "multiclass"}:
        raise ValueError(
            "task must be 'regression', 'classification', or 'multiclass'"
        )

    rng = np.random.default_rng(random_state)
    X = rng.normal(size=(n_samples, n_features)).astype(np.float64)
    signal = _latent_signal(X, n_informative, rng)

    if task == "regression":
        y = 3.0 * signal + noise * rng.normal(size=n_samples)
    elif task == "classification":
        logits = 2.0 * signal + noise * rng.normal(size=n_samples)
        y = rng.binomial(1, sigmoid(logits)).astype(np.float64)
    else:
        n_informative = min(n_informative, n_features)
        weights = rng.normal(size=(n_informative, n_classes))
        weights /= np.sqrt(n_informative)
        logits = 2.0 * X[:, :n_informative] @ weights
        logits += noise * rng.normal(size=logits.shape)
        probabilities = softmax(logits)
        cumulative = np.cumsum(probabilities, axis=1)
        y = (rng.random((n_samples, 1)) > cumulative).sum(axis=1)
        y = np.clip(y, 0, n_classes - 1).astype(np.float64)

    if missing_fraction > 0.0:
        missing = rng.random(X.shape) < missing_fraction
        X[missing] = np.nan

    order = rng.permutation(n_samples)
    n_test = int(round(test_fraction * n_samples))
    n_test = min(max(n_test, 1), n_samples - 1)
    test_idx = order[:n_test]
    train_idx = order[n_test:]
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]
