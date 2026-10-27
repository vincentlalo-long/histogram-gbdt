import numpy as np
import pytest

from gbdt import HistGBDTClassifier, HistGBDTRegressor


def test_constant_target_predicts_mean():
    X = np.arange(20, dtype=np.float64).reshape(10, 2)
    y = np.full(10, 3.5)
    model = HistGBDTRegressor(
        n_estimators=20, max_depth=3, backend="numpy"
    ).fit(X, y)
    assert np.allclose(model.predict(X), 3.5, atol=1e-12)
    assert np.allclose(model.feature_importances_, 0.0)


def test_single_sample_regression():
    X = np.array([[1.0, 2.0]])
    y = np.array([3.0])
    model = HistGBDTRegressor(
        n_estimators=5, max_depth=2, backend="numpy"
    ).fit(X, y)
    assert np.allclose(model.predict(X), [3.0])


def test_max_depth_zero_uses_only_base_score_plus_leaf():
    X = np.array([[0.0], [1.0], [2.0], [3.0]])
    y = np.array([0.0, 0.0, 1.0, 1.0])
    model = HistGBDTRegressor(
        n_estimators=5,
        max_depth=0,
        learning_rate=0.1,
        reg_lambda=0.0,
        backend="numpy",
    ).fit(X, y)
    # Base score is the mean and the root leaf receives no gradient.
    assert np.allclose(model.predict(X), 0.5)


def test_large_gamma_prevents_splits():
    X = np.array([[0.0], [1.0], [2.0], [3.0]])
    y = np.array([0.0, 0.0, 1.0, 1.0])
    model = HistGBDTRegressor(
        n_estimators=5,
        max_depth=3,
        gamma=1e9,
        backend="numpy",
    ).fit(X, y)
    assert np.allclose(model.predict(X), 0.5)
    assert np.allclose(model.feature_importances_, 0.0)


def test_invalid_parameters_fail_at_fit():
    X = np.zeros((4, 2))
    y = np.zeros(4)
    invalid = [
        dict(n_estimators=0),
        dict(learning_rate=0.0),
        dict(subsample=0.0),
        dict(colsample_bytree=0.0),
        dict(max_bin=1),
        dict(max_depth=-1),
        dict(min_samples_leaf=0),
    ]
    for params in invalid:
        with pytest.raises(ValueError):
            HistGBDTRegressor(backend="numpy", **params).fit(X, y)


def test_invalid_inputs():
    model = HistGBDTRegressor(backend="numpy")
    with pytest.raises(ValueError):
        model.fit(np.zeros((4, 2)), np.zeros(3))
    with pytest.raises(ValueError):
        model.fit(np.array([[np.inf], [1.0]]), np.zeros(2))
    with pytest.raises(ValueError):
        model.fit(np.zeros((4, 2)), np.array([0.0, np.nan, 0.0, 0.0]))

    model.fit(np.zeros((4, 2)), np.zeros(4))
    with pytest.raises(ValueError):
        model.predict(np.zeros((4, 3)))


def test_early_stopping_requires_eval_set():
    X = np.zeros((4, 2))
    y = np.zeros(4)
    with pytest.raises(ValueError):
        HistGBDTRegressor(backend="numpy").fit(
            X, y, early_stopping_rounds=5
        )


def test_empty_eval_set_is_rejected():
    X = np.zeros((4, 2))
    y = np.zeros(4)
    with pytest.raises(ValueError):
        HistGBDTRegressor(backend="numpy").fit(X, y, eval_set=[])


def test_unknown_backend_is_rejected():
    with pytest.raises(ValueError):
        HistGBDTRegressor(backend="fortran").fit(
            np.zeros((4, 2)), np.zeros(4)
        )


def test_classifier_ignores_unknown_eval_labels():
    X = np.array([[0.0], [1.0], [2.0], [3.0]])
    y = np.array([0, 0, 1, 1])
    model = HistGBDTClassifier(
        n_estimators=5, backend="numpy", random_state=0
    )
    with pytest.raises(ValueError):
        model.fit(
            X,
            y,
            eval_set=[(X, np.array([0, 0, 1, 2]))],
        )
