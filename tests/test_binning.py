import numpy as np
import pytest

from gbdt.binning import Binner


def test_low_cardinality_one_bin_per_value_and_missing_bin():
    X = np.array(
        [[0.0, 10.0], [1.0, 20.0], [2.0, 20.0], [3.0, 30.0]]
    )
    binner = Binner(max_bin=8).fit(X)
    assert binner.n_bins_.tolist() == [5, 4]
    assert binner.missing_bin_.tolist() == [4, 3]
    assert binner.n_bins_max_ == 5

    binned = binner.transform(X)
    assert binned.shape == (2, 4)
    assert binned[0].tolist() == [0, 1, 2, 3]
    assert binned[1].tolist() == [0, 1, 1, 2]


def test_nan_is_mapped_to_the_reserved_missing_bin():
    X = np.array([[1.0], [2.0], [np.nan], [4.0]])
    binner = Binner(max_bin=8).fit(X)
    binned = binner.transform(X)[0]
    assert binner.n_bins_[0] == binner.missing_bin_[0] + 1
    assert binned[2] == binner.missing_bin_[0]
    assert np.all(binned[[0, 1, 3]] < binner.missing_bin_[0])


def test_missing_bin_is_available_when_fit_has_no_nan():
    binner = Binner(max_bin=8).fit(np.array([[0.0], [1.0], [2.0]]))
    binned = binner.transform(np.array([[np.nan], [1.0]]))[0]
    assert binned[0] == binner.missing_bin_[0]
    assert binned[1] < binner.missing_bin_[0]


def test_quantile_bins_are_monotone_and_bounded():
    X = np.arange(1000, dtype=np.float64).reshape(-1, 1)
    binner = Binner(max_bin=16).fit(X)
    assert binner.n_bins_[0] <= 16

    binned = binner.transform(X)[0]
    assert binned.min() >= 0
    assert binned.max() < binner.n_bins_[0]
    assert np.all(np.diff(binned) >= 0)


def test_constant_feature_has_one_non_missing_and_one_missing_bin():
    X = np.full((10, 2), 5.0)
    binner = Binner(max_bin=8).fit(X)
    assert binner.n_bins_.tolist() == [2, 2]
    assert binner.missing_bin_.tolist() == [1, 1]
    assert np.all(binner.transform(X) == 0)


def test_all_missing_feature():
    X = np.full((6, 1), np.nan)
    binner = Binner(max_bin=8).fit(X)
    assert binner.n_bins_.tolist() == [2]
    assert binner.transform(X)[0].tolist() == [1] * 6
    assert binner.transform(np.array([[3.0]]))[0, 0] == 0


def test_transform_after_fit_rejects_feature_mismatch():
    binner = Binner(max_bin=8).fit(np.zeros((5, 3)))
    with pytest.raises(ValueError):
        binner.transform(np.zeros((5, 2)))


def test_rejects_infinity_but_accepts_nan():
    with pytest.raises(ValueError):
        Binner().fit(np.array([[0.0], [np.inf]]))
    Binner().fit(np.array([[0.0], [np.nan]]))


def test_invalid_max_bin():
    with pytest.raises(ValueError):
        Binner(max_bin=1)
    with pytest.raises(ValueError):
        Binner(max_bin=257)


def test_transform_before_fit_raises():
    with pytest.raises(RuntimeError):
        Binner().transform(np.zeros((3, 1)))
