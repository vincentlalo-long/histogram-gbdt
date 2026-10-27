import numpy as np

from gbdt.binning import Binner
from gbdt.reference import find_best_split_exact
from gbdt.splitter import find_best_split_numpy


def _hand_data():
    Xbin = np.array([[0, 0, 1, 1]], dtype=np.uint8)
    rows = np.arange(4, dtype=np.int64)
    grad = np.array([1.0, 1.0, -1.0, -1.0])
    hess = np.ones(4)
    features = np.array([0], dtype=np.int64)
    return Xbin, rows, grad, hess, features


def test_histogram_gain_matches_hand_calculation():
    Xbin, rows, grad, hess, features = _hand_data()
    split = find_best_split_numpy(
        Xbin, rows, grad, hess, features, 2, 1.0, 0.0, 1.0, 1
    )
    assert split is not None
    assert split.feature == 0
    assert split.threshold_bin == 0
    assert np.isclose(split.gain, 4.0 / 3.0)
    assert split.left_count == 2
    assert np.isclose(split.left_grad, 2.0)
    assert np.isclose(split.left_hess, 2.0)


def test_lambda_shrinks_gain():
    Xbin, rows, grad, hess, features = _hand_data()
    no_reg = find_best_split_numpy(
        Xbin, rows, grad, hess, features, 2, 0.0, 0.0, 1.0, 1
    )
    regularised = find_best_split_numpy(
        Xbin, rows, grad, hess, features, 2, 1.0, 0.0, 1.0, 1
    )
    assert no_reg is not None and regularised is not None
    assert no_reg.gain > regularised.gain
    assert np.isclose(no_reg.gain, 2.0)


def test_gamma_prunes_zero_and_low_gain_splits():
    Xbin, rows, grad, hess, features = _hand_data()
    split = find_best_split_numpy(
        Xbin, rows, grad, hess, features, 2, 1.0, 2.0, 1.0, 1
    )
    assert split is None


def test_min_child_weight_and_leaf_size():
    Xbin, rows, grad, hess, features = _hand_data()
    assert (
        find_best_split_numpy(
            Xbin, rows, grad, hess, features, 2, 1.0, 0.0, 2.5, 1
        )
        is None
    )
    assert (
        find_best_split_numpy(
            Xbin, rows, grad, hess, features, 2, 1.0, 0.0, 1.0, 3
        )
        is None
    )


def test_constant_feature_has_no_split():
    Xbin = np.zeros((1, 8), dtype=np.uint8)
    rows = np.arange(8, dtype=np.int64)
    grad = np.linspace(-1.0, 1.0, 8)
    hess = np.ones(8)
    split = find_best_split_numpy(
        Xbin, rows, grad, hess, np.array([0]), 1, 1.0, 0.0, 1.0, 1
    )
    assert split is None


def test_exact_and_histogram_splits_agree_on_low_cardinality_data():
    rng = np.random.default_rng(23)
    for _ in range(10):
        X = rng.integers(0, 9, size=(60, 4)).astype(np.float64)
        grad = rng.normal(size=60)
        hess = rng.uniform(0.1, 2.0, size=60)
        rows = np.arange(60, dtype=np.int64)
        binner = Binner(max_bin=16).fit(X)
        Xbin = binner.transform(X)
        features = np.array([0, 1, 2, 3], dtype=np.int64)

        hist = find_best_split_numpy(
            Xbin,
            rows,
            grad,
            hess,
            features,
            binner.n_bins_max_,
            1.0,
            0.0,
            1.0,
            1,
        )
        exact = find_best_split_exact(
            X, rows, grad, hess, features, 1.0, 0.0, 1.0, 1
        )
        assert hist is not None and exact is not None
        assert hist.feature == exact.feature
        assert np.isclose(hist.gain, exact.gain)
        assert hist.left_count == exact.left_count

        hist_left = Xbin[hist.feature] <= hist.threshold_bin
        exact_left = X[:, exact.feature] <= exact.threshold
        assert np.array_equal(hist_left, exact_left)


def test_exact_split_handles_duplicate_values():
    X = np.array([[1.0], [1.0], [2.0], [2.0], [3.0]])
    rows = np.arange(5, dtype=np.int64)
    grad = np.array([1.0, 1.0, -1.0, -1.0, 0.0])
    hess = np.ones(5)
    split = find_best_split_exact(
        X, rows, grad, hess, np.array([0]), 1.0, 0.0, 1.0, 1
    )
    assert split is not None
    assert split.threshold == 1.5
    assert split.left_count == 2
