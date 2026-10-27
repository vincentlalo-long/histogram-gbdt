"""Exact greedy split finding on raw (un-binned) features.

This is the correctness oracle used by the tests.  It is intentionally
slow: every distinct value boundary of every candidate feature is
evaluated.  On low-cardinality data the histogram splitter should match
its gain exactly because the binner assigns one bin per distinct value.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ExactSplit:
    feature: int
    threshold: float
    gain: float
    left_count: int
    left_grad: float
    left_hess: float


def find_best_split_exact(
    X: np.ndarray,
    rows: np.ndarray,
    grad: np.ndarray,
    hess: np.ndarray,
    features: np.ndarray,
    reg_lambda: float,
    gamma: float,
    min_child_weight: float,
    min_samples_leaf: int,
) -> ExactSplit | None:
    """Return the best raw-threshold split for one node."""
    rows = np.asarray(rows, dtype=np.int64)
    features = np.asarray(features, dtype=np.int64)
    if rows.size < 2 or features.size == 0:
        return None

    node_grad = grad[rows]
    node_hess = hess[rows]
    total_grad = float(node_grad.sum())
    total_hess = float(node_hess.sum())
    total_count = int(rows.size)
    parent_denom = total_hess + reg_lambda
    if parent_denom <= 0.0:
        return None
    parent_term = total_grad * total_grad / parent_denom

    best: ExactSplit | None = None
    best_gain = 0.0

    for feature in features:
        values = X[rows, int(feature)]
        order = np.argsort(values, kind="mergesort")
        sorted_values = values[order]
        sorted_grad = node_grad[order]
        sorted_hess = node_hess[order]

        split_positions = np.flatnonzero(sorted_values[:-1] < sorted_values[1:])
        if split_positions.size == 0:
            continue

        left_grad = np.cumsum(sorted_grad)[split_positions]
        left_hess = np.cumsum(sorted_hess)[split_positions]
        left_count = split_positions.astype(np.int64) + 1

        right_grad = total_grad - left_grad
        right_hess = total_hess - left_hess
        right_count = total_count - left_count
        left_denom = left_hess + reg_lambda
        right_denom = right_hess + reg_lambda

        valid = (
            (left_count > 0)
            & (right_count > 0)
            & (left_count >= min_samples_leaf)
            & (right_count >= min_samples_leaf)
            & (left_hess >= min_child_weight)
            & (right_hess >= min_child_weight)
            & (left_denom > 0.0)
            & (right_denom > 0.0)
        )
        if not np.any(valid):
            continue

        gains = np.full(split_positions.shape, -np.inf, dtype=np.float64)
        gains[valid] = (
            0.5
            * (
                left_grad[valid] ** 2 / left_denom[valid]
                + right_grad[valid] ** 2 / right_denom[valid]
                - parent_term
            )
            - gamma
        )

        local_index = int(np.argmax(gains))
        local_gain = float(gains[local_index])
        if local_gain > best_gain:
            position = int(split_positions[local_index])
            lower = float(sorted_values[position])
            upper = float(sorted_values[position + 1])
            best_gain = local_gain
            best = ExactSplit(
                feature=int(feature),
                threshold=lower + 0.5 * (upper - lower),
                gain=local_gain,
                left_count=int(left_count[local_index]),
                left_grad=float(left_grad[local_index]),
                left_hess=float(left_hess[local_index]),
            )

    return best
