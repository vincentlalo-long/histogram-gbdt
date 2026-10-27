"""Histogram split finding in pure NumPy.

This is deliberately written as a direct translation of the math in
``docs/theory.tex``.  The C++ backend implements the same algorithm and
is checked against this function in the test suite.

Missing values live in a dedicated bin per feature.  For every candidate
threshold we evaluate routing that missing bin to the left child and to
the right child, and keep the direction with the larger gain.  Ties are
resolved in favour of the right child (``default_left=False``).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

_GAIN_EPS = 1e-12


@dataclass(frozen=True)
class Split:
    """Best split found at a node (binned threshold, inclusive left side)."""

    feature: int
    threshold_bin: int
    gain: float
    left_count: int
    left_grad: float
    left_hess: float
    default_left: bool = False


def _gain(
    left_grad: np.ndarray,
    left_hess: np.ndarray,
    right_grad: np.ndarray,
    right_hess: np.ndarray,
    parent_term: float,
    reg_lambda: float,
    gamma: float,
) -> np.ndarray:
    left_denom = left_hess + reg_lambda
    right_denom = right_hess + reg_lambda
    gain = np.full(left_grad.shape, -np.inf, dtype=np.float64)
    valid = (left_denom > 0.0) & (right_denom > 0.0)
    gain[valid] = (
        0.5
        * (
            left_grad[valid] ** 2 / left_denom[valid]
            + right_grad[valid] ** 2 / right_denom[valid]
            - parent_term
        )
        - gamma
    )
    return gain


def find_best_split_numpy(
    Xbin: np.ndarray,
    rows: np.ndarray,
    grad: np.ndarray,
    hess: np.ndarray,
    features: np.ndarray,
    n_bins: int,
    reg_lambda: float,
    gamma: float,
    min_child_weight: float,
    min_samples_leaf: int,
    missing_bins: np.ndarray | None = None,
) -> Split | None:
    """Find the highest-gain histogram split for one node.

    Parameters
    ----------
    Xbin:
        ``uint8`` array with shape ``(n_features, n_samples)``.
    rows:
        Row indices belonging to this node.
    grad, hess:
        Per-sample first and second derivatives, length ``n_samples``.
    features:
        Candidate feature indices for this tree/node.
    n_bins:
        Allocated histogram width (``B``), at least the widest feature.
    missing_bins:
        Bin index reserved for missing values per feature.  If omitted,
        no missing bin is assumed and every bin is a normal threshold bin.
    """
    rows = np.asarray(rows, dtype=np.int64)
    features = np.asarray(features, dtype=np.int64)
    if rows.size == 0 or features.size == 0:
        return None

    Xbin = np.asarray(Xbin)
    n_features = int(Xbin.shape[0])
    if missing_bins is None:
        missing_bins = np.full(n_features, n_bins, dtype=np.int64)
    else:
        missing_bins = np.asarray(missing_bins, dtype=np.int64)
        if missing_bins.size != n_features:
            raise ValueError(
                f"missing_bins has length {missing_bins.size}, "
                f"expected {n_features}"
            )

    node_grad = grad[rows]
    node_hess = hess[rows]
    total_grad = float(node_grad.sum())
    total_hess = float(node_hess.sum())
    total_count = int(rows.size)
    parent_denom = total_hess + reg_lambda
    if parent_denom <= 0.0:
        return None
    parent_term = total_grad * total_grad / parent_denom

    best: Split | None = None
    best_gain = 0.0

    for feature in features:
        feature = int(feature)
        bins = Xbin[feature, rows]
        hist_grad = np.bincount(bins, weights=node_grad, minlength=n_bins)
        hist_hess = np.bincount(bins, weights=node_hess, minlength=n_bins)
        hist_count = np.bincount(bins, minlength=n_bins)

        missing_bin = int(missing_bins[feature])
        if missing_bin < 0 or missing_bin > n_bins:
            raise ValueError("missing bin index is out of bounds")

        if missing_bin < n_bins:
            missing_grad = float(hist_grad[missing_bin])
            missing_hess = float(hist_hess[missing_bin])
            missing_count = int(hist_count[missing_bin])
            non_missing_limit = missing_bin
        else:
            missing_grad = 0.0
            missing_hess = 0.0
            missing_count = 0
            non_missing_limit = n_bins

        left_grad = np.cumsum(hist_grad[:non_missing_limit])[:-1]
        left_hess = np.cumsum(hist_hess[:non_missing_limit])[:-1]
        left_count = np.cumsum(hist_count[:non_missing_limit])[:-1]

        non_missing_grad = total_grad - missing_grad
        non_missing_hess = total_hess - missing_hess
        non_missing_count = total_count - missing_count
        right_grad = non_missing_grad - left_grad
        right_hess = non_missing_hess - left_hess
        right_count = non_missing_count - left_count

        base_valid = (left_count > 0) & (right_count > 0)
        if not np.any(base_valid):
            continue

        # Missing values go right: left stays non-missing; right gains
        # the missing bin.  Missing left is the mirror image.
        count_valid = (
            base_valid
            & (left_count >= min_samples_leaf)
            & (right_count + missing_count >= min_samples_leaf)
            & (left_hess >= min_child_weight)
            & (right_hess + missing_hess >= min_child_weight)
        )
        gains_right = _gain(
            left_grad,
            left_hess,
            right_grad + missing_grad,
            right_hess + missing_hess,
            parent_term,
            reg_lambda,
            gamma,
        )
        gains_right[~count_valid] = -np.inf

        count_valid = (
            base_valid
            & (left_count + missing_count >= min_samples_leaf)
            & (right_count >= min_samples_leaf)
            & (left_hess + missing_hess >= min_child_weight)
            & (right_hess >= min_child_weight)
        )
        gains_left = _gain(
            left_grad + missing_grad,
            left_hess + missing_hess,
            right_grad,
            right_hess,
            parent_term,
            reg_lambda,
            gamma,
        )
        gains_left[~count_valid] = -np.inf

        right_max = float(np.max(gains_right))
        left_max = float(np.max(gains_left))
        if not np.isfinite(right_max) and not np.isfinite(left_max):
            continue

        finite_scale = max(
            1.0,
            abs(right_max) if np.isfinite(right_max) else 1.0,
            abs(left_max) if np.isfinite(left_max) else 1.0,
        )
        eps = _GAIN_EPS * finite_scale
        if right_max > left_max + eps:
            default_left = False
        elif left_max > right_max + eps:
            default_left = True
        else:
            right_bin = (
                int(np.flatnonzero(gains_right >= right_max - eps)[0])
                if np.isfinite(right_max)
                else np.iinfo(np.int64).max
            )
            left_bin = (
                int(np.flatnonzero(gains_left >= left_max - eps)[0])
                if np.isfinite(left_max)
                else np.iinfo(np.int64).max
            )
            default_left = left_bin < right_bin

        gains = gains_left if default_left else gains_right
        direction_max = left_max if default_left else right_max
        local_bin = int(
            np.flatnonzero(gains >= direction_max - eps)[0]
        )
        local_gain = float(gains[local_bin])
        if local_gain <= best_gain + _GAIN_EPS * max(1.0, abs(best_gain)):
            continue

        best_gain = local_gain
        if default_left:
            best = Split(
                feature=feature,
                threshold_bin=local_bin,
                gain=local_gain,
                left_count=int(left_count[local_bin] + missing_count),
                left_grad=float(left_grad[local_bin] + missing_grad),
                left_hess=float(left_hess[local_bin] + missing_hess),
                default_left=True,
            )
        else:
            best = Split(
                feature=feature,
                threshold_bin=local_bin,
                gain=local_gain,
                left_count=int(left_count[local_bin]),
                left_grad=float(left_grad[local_bin]),
                left_hess=float(left_hess[local_bin]),
                default_left=False,
            )

    return best
