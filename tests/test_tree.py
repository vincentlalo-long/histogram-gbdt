import numpy as np
import pytest

from gbdt.backend import get_splitter
from gbdt.tree import TreeBuilder


def _builder(max_depth=1, reg_lambda=1.0, gamma=0.0, min_child_weight=1.0):
    splitter, _ = get_splitter(
        "numpy", -1, reg_lambda, gamma, min_child_weight, 1
    )
    return TreeBuilder(
        splitter=splitter,
        n_bins=2,
        max_depth=max_depth,
        reg_lambda=reg_lambda,
        gamma=gamma,
        min_child_weight=min_child_weight,
        min_samples_leaf=1,
    )


def test_tree_structure_and_leaf_values():
    Xbin = np.array([[0, 0, 1, 1]], dtype=np.uint8)
    rows = np.arange(4, dtype=np.int64)
    grad = np.array([1.0, 1.0, -1.0, -1.0])
    hess = np.ones(4)
    tree = _builder(max_depth=1).build(
        Xbin, grad, hess, rows, np.array([0], dtype=np.int64)
    )

    assert tree.n_nodes == 3
    assert tree.n_leaves == 2
    assert tree.feature.tolist() == [0, -1, -1]
    assert tree.threshold.tolist() == [0, 0, 0]
    assert np.allclose(tree.value, [0.0, -2.0 / 3.0, 2.0 / 3.0])
    assert np.allclose(
        tree.predict(Xbin), [-2.0 / 3.0, -2.0 / 3.0, 2.0 / 3.0, 2.0 / 3.0]
    )


def test_depth_zero_is_a_single_leaf():
    Xbin = np.array([[0, 1]], dtype=np.uint8)
    rows = np.arange(2, dtype=np.int64)
    grad = np.ones(2)
    hess = np.ones(2)
    tree = _builder(max_depth=0).build(
        Xbin, grad, hess, rows, np.array([0], dtype=np.int64)
    )
    assert tree.n_nodes == 1
    assert tree.feature.tolist() == [-1]
    assert np.isclose(tree.value[0], -2.0 / 3.0)


def test_feature_subset_restricts_splits():
    # Feature 0 separates the gradients; feature 1 is constant.
    Xbin = np.array(
        [[0, 0, 1, 1], [0, 0, 0, 0]], dtype=np.uint8
    )
    rows = np.arange(4, dtype=np.int64)
    grad = np.array([1.0, 1.0, -1.0, -1.0])
    hess = np.ones(4)

    restricted = _builder(max_depth=1).build(
        Xbin, grad, hess, rows, np.array([1], dtype=np.int64)
    )
    assert restricted.n_nodes == 1

    allowed = _builder(max_depth=1).build(
        Xbin, grad, hess, rows, np.array([0, 1], dtype=np.int64)
    )
    assert allowed.n_nodes == 3
    assert allowed.feature[0] == 0


def test_predict_rejects_wrong_feature_count():
    Xbin = np.array([[0, 1]], dtype=np.uint8)
    tree = _builder(max_depth=0).build(
        Xbin, np.ones(2), np.ones(2), np.arange(2), np.array([0])
    )
    with pytest.raises(ValueError):
        tree.predict(np.zeros((2, 2), dtype=np.uint8))
