"""Regression tree structure and level-wise histogram tree builder."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from .splitter import Split

Splitter = Callable[
    [
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        int,
        np.ndarray,
    ],
    Split | None,
]


@dataclass
class Tree:
    """A binary tree stored in flat arrays.

    ``feature[node] == -1`` marks a leaf.  Internal nodes route non-missing
    samples with ``Xbin[feature, sample] <= threshold[node]`` to the left
    child.  Samples in the feature's missing bin follow ``default_left``
    instead.  Leaf ``value`` is the raw Newton weight, before the learning
    rate is applied by the booster.
    """

    feature: np.ndarray
    threshold: np.ndarray
    left: np.ndarray
    right: np.ndarray
    value: np.ndarray
    gain: np.ndarray
    default_left: np.ndarray
    missing_bins: np.ndarray
    n_features: int
    max_depth: int

    @property
    def n_nodes(self) -> int:
        return int(self.feature.size)

    @property
    def n_leaves(self) -> int:
        return int(np.count_nonzero(self.feature < 0))

    def predict(self, Xbin: np.ndarray) -> np.ndarray:
        Xbin = np.asarray(Xbin)
        if Xbin.ndim != 2 or Xbin.shape[0] != self.n_features:
            raise ValueError(
                f"Xbin must have shape ({self.n_features}, n_samples), "
                f"got {Xbin.shape}"
            )
        n_samples = Xbin.shape[1]
        if n_samples == 0:
            return np.empty(0, dtype=np.float64)

        node = np.zeros(n_samples, dtype=np.int32)
        row_index = np.arange(n_samples)
        for _ in range(self.max_depth + 2):
            feature = self.feature[node]
            active = feature >= 0
            if not np.any(active):
                break
            safe_feature = np.where(active, feature, 0)
            value = Xbin[safe_feature, row_index]
            missing = value == self.missing_bins[safe_feature]
            go_left = np.where(
                missing,
                self.default_left[node],
                value <= self.threshold[node],
            )
            child = np.where(go_left, self.left[node], self.right[node])
            node = np.where(active, child, node)
        return self.value[node].astype(np.float64, copy=False)


@dataclass
class _Node:
    rows: np.ndarray
    grad_sum: float
    hess_sum: float
    count: int
    depth: int
    feature: int = -1
    threshold_bin: int = 0
    left: int = -1
    right: int = -1
    value: float = 0.0
    gain: float = 0.0
    default_left: bool = False
    is_leaf: bool = True


class TreeBuilder:
    """Grow one tree with a greedy, level-wise breadth-first schedule."""

    def __init__(
        self,
        splitter: Splitter,
        n_bins: int,
        max_depth: int,
        reg_lambda: float,
        gamma: float,
        min_child_weight: float,
        min_samples_leaf: int,
        missing_bins: np.ndarray | None = None,
    ) -> None:
        if max_depth < 0:
            raise ValueError("max_depth must be non-negative")
        if reg_lambda < 0.0:
            raise ValueError("reg_lambda must be non-negative")
        if min_samples_leaf < 1:
            raise ValueError("min_samples_leaf must be at least 1")
        self.splitter = splitter
        self.n_bins = int(n_bins)
        self.max_depth = int(max_depth)
        self.reg_lambda = float(reg_lambda)
        self.gamma = float(gamma)
        self.min_child_weight = float(min_child_weight)
        self.min_samples_leaf = int(min_samples_leaf)
        self.missing_bins = (
            None
            if missing_bins is None
            else np.asarray(missing_bins, dtype=np.int64)
        )

    def _leaf_value(self, grad_sum: float, hess_sum: float) -> float:
        denominator = hess_sum + self.reg_lambda
        return -grad_sum / denominator if denominator > 0.0 else 0.0

    def build(
        self,
        Xbin: np.ndarray,
        grad: np.ndarray,
        hess: np.ndarray,
        rows: np.ndarray,
        features: np.ndarray,
    ) -> Tree:
        rows = np.asarray(rows, dtype=np.int64)
        features = np.asarray(features, dtype=np.int64)
        if rows.size == 0:
            raise ValueError("cannot build a tree with no rows")

        n_features = int(Xbin.shape[0])
        if self.missing_bins is None:
            missing_bins = np.full(n_features, self.n_bins, dtype=np.int64)
        else:
            if self.missing_bins.size != n_features:
                raise ValueError(
                    f"missing_bins has length {self.missing_bins.size}, "
                    f"expected {n_features}"
                )
            missing_bins = self.missing_bins

        root = _Node(
            rows=rows,
            grad_sum=float(grad[rows].sum()),
            hess_sum=float(hess[rows].sum()),
            count=int(rows.size),
            depth=0,
        )
        nodes: list[_Node] = [root]
        index = 0

        while index < len(nodes):
            node = nodes[index]
            index += 1

            if (
                node.depth >= self.max_depth
                or node.count < self.min_samples_leaf
                or node.hess_sum < self.min_child_weight
            ):
                node.value = self._leaf_value(node.grad_sum, node.hess_sum)
                continue

            split = self.splitter(
                Xbin,
                node.rows,
                grad,
                hess,
                features,
                self.n_bins,
                missing_bins,
            )
            if split is None or split.gain <= 0.0:
                node.value = self._leaf_value(node.grad_sum, node.hess_sum)
                continue

            values = Xbin[split.feature, node.rows]
            is_missing = values == missing_bins[split.feature]
            go_left = np.where(
                is_missing,
                split.default_left,
                values <= split.threshold_bin,
            )
            left_count = int(np.count_nonzero(go_left))
            right_count = node.count - left_count
            if (
                left_count < self.min_samples_leaf
                or right_count < self.min_samples_leaf
            ):
                node.value = self._leaf_value(node.grad_sum, node.hess_sum)
                continue

            left_hess = split.left_hess
            right_hess = node.hess_sum - left_hess
            if (
                left_hess < self.min_child_weight
                or right_hess < self.min_child_weight
            ):
                node.value = self._leaf_value(node.grad_sum, node.hess_sum)
                continue

            left_grad = split.left_grad
            right_grad = node.grad_sum - left_grad
            left_rows = node.rows[go_left]
            right_rows = node.rows[~go_left]

            node.is_leaf = False
            node.feature = split.feature
            node.threshold_bin = split.threshold_bin
            node.default_left = bool(split.default_left)
            node.gain = split.gain
            node.left = len(nodes)
            nodes.append(
                _Node(
                    rows=left_rows,
                    grad_sum=left_grad,
                    hess_sum=left_hess,
                    count=left_count,
                    depth=node.depth + 1,
                )
            )
            node.right = len(nodes)
            nodes.append(
                _Node(
                    rows=right_rows,
                    grad_sum=right_grad,
                    hess_sum=right_hess,
                    count=right_count,
                    depth=node.depth + 1,
                )
            )

        n_nodes = len(nodes)
        feature = np.empty(n_nodes, dtype=np.int32)
        threshold = np.empty(n_nodes, dtype=np.int32)
        left = np.empty(n_nodes, dtype=np.int32)
        right = np.empty(n_nodes, dtype=np.int32)
        value = np.empty(n_nodes, dtype=np.float64)
        gain = np.empty(n_nodes, dtype=np.float64)
        default_left = np.empty(n_nodes, dtype=bool)

        for i, node in enumerate(nodes):
            feature[i] = node.feature
            threshold[i] = node.threshold_bin
            left[i] = node.left
            right[i] = node.right
            value[i] = node.value
            gain[i] = node.gain
            default_left[i] = node.default_left

        return Tree(
            feature=feature,
            threshold=threshold,
            left=left,
            right=right,
            value=value,
            gain=gain,
            default_left=default_left,
            missing_bins=missing_bins,
            n_features=n_features,
            max_depth=self.max_depth,
        )
