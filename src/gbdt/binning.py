"""Quantile binning for histogram-based tree learning.

The binner maps every feature to at most ``max_bin`` unsigned 8-bit bins.
It is fitted on the training set only and then reused for validation,
test, and inference data.  Keeping the binned matrix as ``(n_features,
n_samples)`` makes each feature contiguous, which is the layout used by
both the NumPy reference splitter and the C++/OpenMP kernel.

Every feature reserves its last bin for missing values.  This lets NaN
values be routed separately at split time while keeping the ordinary
threshold scan unchanged.  With ``max_bin=256`` a feature therefore has
at most 255 non-missing bins and one missing bin.
"""

from __future__ import annotations

import numpy as np

_MAX_BINS = 256


def _as_2d_float(
    X, *, name: str = "X", allow_nan: bool = True
) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError(f"{name} must be a 2-D array, got shape {X.shape}")
    if X.shape[0] == 0:
        raise ValueError(f"{name} must contain at least one sample")
    if X.shape[1] == 0:
        raise ValueError(f"{name} must contain at least one feature")
    if np.isinf(X).any():
        raise ValueError(f"{name} contains infinity")
    if not allow_nan and np.isnan(X).any():
        raise ValueError(f"{name} contains NaN")
    return X


class Binner:
    """Fit quantile bin edges and transform features to ``uint8`` bins.

    Parameters
    ----------
    max_bin:
        Maximum number of bins per feature, including the missing bin.
        Must be between 2 and 256.

    Attributes
    ----------
    bin_edges_:
        List of internal split edges for each feature.  A feature with
        ``b`` non-missing bins has ``b - 1`` edges.
    missing_bin_:
        Bin index used for NaN values, shape ``(n_features,)``.  The
        missing bin is always the last bin of a feature.
    n_bins_:
        Total number of bins per feature, including the missing bin,
        shape ``(n_features,)``.
    n_bins_max_:
        Maximum number of bins across features.  Histogram kernels use
        this as the allocated histogram width.
    """

    def __init__(self, max_bin: int = _MAX_BINS) -> None:
        max_bin = int(max_bin)
        if not 2 <= max_bin <= _MAX_BINS:
            raise ValueError(f"max_bin must be in [2, {_MAX_BINS}], got {max_bin}")
        self.max_bin = max_bin
        self.bin_edges_: list[np.ndarray] | None = None
        self.missing_bin_: np.ndarray | None = None
        self.n_bins_: np.ndarray | None = None
        self.n_bins_max_: int | None = None
        self.n_features_in_: int | None = None

    def fit(self, X) -> "Binner":
        X = _as_2d_float(X)
        n_samples, n_features = X.shape
        edges: list[np.ndarray] = []
        missing_bins = np.empty(n_features, dtype=np.int32)
        n_bins = np.empty(n_features, dtype=np.int32)

        max_non_missing_bins = self.max_bin - 1
        for feature in range(n_features):
            column = X[:, feature]
            observed = column[~np.isnan(column)]
            unique = np.unique(observed)

            if unique.size == 0:
                feature_edges = np.empty(0, dtype=np.float64)
                n_non_missing = 1
            elif unique.size <= max_non_missing_bins:
                if unique.size == 1:
                    feature_edges = np.empty(0, dtype=np.float64)
                else:
                    feature_edges = 0.5 * (unique[:-1] + unique[1:])
                n_non_missing = int(unique.size)
            else:
                quantiles = np.linspace(
                    0.0, 1.0, max_non_missing_bins + 1
                )[1:-1]
                feature_edges = np.quantile(
                    observed, quantiles, method="linear"
                )
                feature_edges = np.unique(feature_edges)
                n_non_missing = int(feature_edges.size + 1)

            edges.append(np.asarray(feature_edges, dtype=np.float64))
            missing_bins[feature] = n_non_missing
            n_bins[feature] = n_non_missing + 1

        self.bin_edges_ = edges
        self.missing_bin_ = missing_bins
        self.n_bins_ = n_bins
        self.n_bins_max_ = int(n_bins.max())
        self.n_features_in_ = n_features
        return self

    def transform(self, X) -> np.ndarray:
        """Return a C-contiguous ``uint8`` matrix with shape ``(K, N)``."""
        if (
            self.bin_edges_ is None
            or self.n_bins_ is None
            or self.missing_bin_ is None
        ):
            raise RuntimeError("Binner must be fitted before transform()")
        X = _as_2d_float(X)
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X.shape[1]} features, expected {self.n_features_in_}"
            )

        n_features, n_samples = X.shape[1], X.shape[0]
        binned = np.empty((n_features, n_samples), dtype=np.uint8)
        for feature in range(n_features):
            column = X[:, feature]
            missing = np.isnan(column)
            n_non_missing = int(self.n_bins_[feature] - 1)
            codes = np.searchsorted(
                self.bin_edges_[feature], column, side="right"
            )
            np.minimum(codes, n_non_missing - 1, out=codes)
            codes[missing] = self.missing_bin_[feature]
            binned[feature] = codes.astype(np.uint8, copy=False)
        return binned

    def fit_transform(self, X) -> np.ndarray:
        return self.fit(X).transform(X)
