import numpy as np
import pytest

from gbdt import HistGBDTClassifier, HistGBDTRegressor
from gbdt.backend import CPP_AVAILABLE, get_splitter
from gbdt.binning import Binner

pytestmark = pytest.mark.skipif(
    not CPP_AVAILABLE, reason="C++ extension is not built"
)


def test_cpp_and_numpy_splitters_agree():
    rng = np.random.default_rng(123)
    numpy_splitter, _ = get_splitter("numpy", -1, 1.0, 0.0, 1.0, 1)
    cpp_splitter, name = get_splitter("cpp", 2, 1.0, 0.0, 1.0, 1)
    assert name == "cpp"

    for _ in range(20):
        n_features, n_samples, n_bins = 5, 80, 16
        Xbin = rng.integers(
            0, n_bins, size=(n_features, n_samples), dtype=np.uint8
        )
        rows = np.sort(
            rng.choice(n_samples, size=50, replace=False)
        ).astype(np.int64)
        grad = rng.normal(size=n_samples)
        hess = rng.uniform(0.1, 2.0, size=n_samples)
        features = np.arange(n_features, dtype=np.int64)
        missing_bins = np.full(n_features, n_bins, dtype=np.int64)

        reference = numpy_splitter(
            Xbin, rows, grad, hess, features, n_bins, missing_bins
        )
        compiled = cpp_splitter(
            Xbin, rows, grad, hess, features, n_bins, missing_bins
        )
        assert (reference is None) == (compiled is None)
        if reference is None:
            continue
        assert reference.feature == compiled.feature
        assert reference.threshold_bin == compiled.threshold_bin
        assert reference.left_count == compiled.left_count
        assert reference.default_left == compiled.default_left
        assert np.isclose(reference.gain, compiled.gain, atol=1e-10)
        assert np.isclose(
            reference.left_grad, compiled.left_grad, atol=1e-10
        )
        assert np.isclose(
            reference.left_hess, compiled.left_hess, atol=1e-10
        )


def test_cpp_and_numpy_splitters_agree_with_missing_bin():
    rng = np.random.default_rng(321)
    numpy_splitter, _ = get_splitter("numpy", -1, 1.0, 0.0, 1.0, 1)
    cpp_splitter, _ = get_splitter("cpp", 2, 1.0, 0.0, 1.0, 1)

    for _ in range(20):
        n_features, n_samples, n_bins = 4, 100, 8
        Xbin = rng.integers(
            0, n_bins - 1, size=(n_features, n_samples), dtype=np.uint8
        )
        missing = rng.random((n_features, n_samples)) < 0.2
        Xbin[missing] = n_bins - 1
        rows = np.sort(
            rng.choice(n_samples, size=70, replace=False)
        ).astype(np.int64)
        grad = rng.normal(size=n_samples)
        hess = rng.uniform(0.1, 2.0, size=n_samples)
        features = np.arange(n_features, dtype=np.int64)
        missing_bins = np.full(n_features, n_bins - 1, dtype=np.int64)

        reference = numpy_splitter(
            Xbin, rows, grad, hess, features, n_bins, missing_bins
        )
        compiled = cpp_splitter(
            Xbin, rows, grad, hess, features, n_bins, missing_bins
        )
        assert (reference is None) == (compiled is None)
        if reference is None:
            continue
        assert reference.feature == compiled.feature
        assert reference.threshold_bin == compiled.threshold_bin
        assert reference.left_count == compiled.left_count
        assert reference.default_left == compiled.default_left
        assert np.isclose(reference.gain, compiled.gain, atol=1e-10)


def test_splitters_handle_one_invalid_missing_direction():
    numpy_splitter, _ = get_splitter("numpy", -1, 1.0, 0.0, 1.0, 1)
    cpp_splitter, _ = get_splitter("cpp", 2, 1.0, 0.0, 1.0, 1)
    Xbin = np.array([[0, 0, 0, 0, 2, 2, 4, 4]], dtype=np.uint8)
    rows = np.arange(8, dtype=np.int64)
    grad = np.array([0.1, 0.1, 0.1, 0.2, -0.1, -0.1, 0.05, 0.05])
    hess = np.array([0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.35, 0.35])
    features = np.array([0], dtype=np.int64)
    missing_bins = np.array([4], dtype=np.int64)

    reference = numpy_splitter(
        Xbin, rows, grad, hess, features, 5, missing_bins
    )
    compiled = cpp_splitter(
        Xbin, rows, grad, hess, features, 5, missing_bins
    )
    assert reference is not None
    assert compiled is not None
    assert reference.feature == compiled.feature == 0
    assert reference.threshold_bin == compiled.threshold_bin == 0
    assert reference.default_left is False
    assert compiled.default_left is False
    assert np.isclose(reference.gain, compiled.gain, atol=1e-10)


def test_cpp_and_numpy_estimators_predict_the_same():
    rng = np.random.default_rng(5)
    X = rng.normal(size=(250, 4))
    y = 2.0 * X[:, 0] - X[:, 1] + rng.normal(scale=0.1, size=250)
    params = dict(
        n_estimators=30,
        max_depth=3,
        learning_rate=0.2,
        random_state=0,
    )
    numpy_model = HistGBDTRegressor(backend="numpy", **params).fit(X, y)
    cpp_model = HistGBDTRegressor(backend="cpp", n_jobs=2, **params).fit(X, y)
    assert cpp_model.backend_ == "cpp"
    assert np.allclose(
        numpy_model.predict(X), cpp_model.predict(X), atol=1e-8
    )


def test_cpp_and_numpy_estimators_agree_with_missing_values():
    rng = np.random.default_rng(55)
    X = rng.normal(size=(350, 5))
    X[rng.random(X.shape) < 0.2] = np.nan
    y = (
        2.0 * np.nan_to_num(X[:, 0], nan=-1.0)
        - np.nan_to_num(X[:, 1], nan=2.0)
        + rng.normal(scale=0.1, size=350)
    )
    params = dict(
        n_estimators=30,
        max_depth=3,
        learning_rate=0.2,
        random_state=0,
    )
    numpy_model = HistGBDTRegressor(backend="numpy", **params).fit(X, y)
    cpp_model = HistGBDTRegressor(backend="cpp", n_jobs=2, **params).fit(X, y)
    assert np.allclose(
        numpy_model.predict(X), cpp_model.predict(X), atol=1e-8
    )


def test_cpp_and_numpy_multiclass_predictions_agree():
    rng = np.random.default_rng(77)
    X = rng.normal(size=(300, 5))
    logits = np.column_stack(
        [X[:, 0], X[:, 1], -0.5 * X[:, 0] - X[:, 1] + X[:, 2]]
    )
    y = np.argmax(logits + rng.normal(scale=0.2, size=logits.shape), axis=1)
    params = dict(
        n_estimators=20,
        max_depth=3,
        learning_rate=0.2,
        random_state=0,
    )
    numpy_model = HistGBDTClassifier(backend="numpy", **params).fit(X, y)
    cpp_model = HistGBDTClassifier(backend="cpp", n_jobs=2, **params).fit(X, y)
    assert np.array_equal(
        numpy_model.predict(X), cpp_model.predict(X)
    )
    assert np.allclose(
        numpy_model.predict_proba(X),
        cpp_model.predict_proba(X),
        atol=1e-8,
    )


def test_cpp_and_numpy_multiclass_agree_with_missing_values():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(300, 5))
    weights = rng.normal(size=(5, 3))
    logits = X @ weights
    y = np.argmax(logits + rng.normal(scale=0.3, size=logits.shape), axis=1)
    X[rng.random(X.shape) < 0.2] = np.nan
    params = dict(
        n_estimators=20,
        max_depth=3,
        learning_rate=0.2,
        random_state=0,
    )
    numpy_model = HistGBDTClassifier(backend="numpy", **params).fit(X, y)
    cpp_model = HistGBDTClassifier(backend="cpp", n_jobs=2, **params).fit(X, y)
    assert np.array_equal(
        numpy_model.predict(X), cpp_model.predict(X)
    )
    assert np.allclose(
        numpy_model.predict_proba(X),
        cpp_model.predict_proba(X),
        atol=1e-8,
    )


def test_cpp_rejects_out_of_range_feature():
    cpp_splitter, _ = get_splitter("cpp", 1, 1.0, 0.0, 1.0, 1)
    Xbin = np.zeros((2, 4), dtype=np.uint8)
    rows = np.arange(4, dtype=np.int64)
    grad = np.ones(4)
    hess = np.ones(4)
    missing_bins = np.full(2, 2, dtype=np.int64)
    with pytest.raises(ValueError):
        cpp_splitter(
            Xbin, rows, grad, hess, np.array([2], dtype=np.int64), 2,
            missing_bins,
        )
