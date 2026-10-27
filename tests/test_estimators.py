import numpy as np
import pytest

from gbdt import HistGBDTClassifier, HistGBDTRegressor
from gbdt.objectives import BinaryLogistic, MeanSquaredError


@pytest.fixture(scope="module")
def regression_data():
    rng = np.random.default_rng(7)
    X = rng.normal(size=(300, 5))
    y = (
        2.0 * X[:, 0]
        - 3.0 * X[:, 1]
        + 0.5 * X[:, 0] * X[:, 2]
        + rng.normal(scale=0.1, size=300)
    )
    return X, y


@pytest.fixture(scope="module")
def classification_data():
    rng = np.random.default_rng(11)
    X = rng.normal(size=(400, 5))
    logits = 1.5 * X[:, 0] - 2.0 * X[:, 1] + 0.8 * X[:, 2]
    y = (logits + rng.normal(scale=0.3, size=400) > 0.0).astype(int)
    return X, y


def test_regressor_learns_nonlinear_signal(regression_data):
    X, y = regression_data
    model = HistGBDTRegressor(
        n_estimators=80,
        max_depth=3,
        learning_rate=0.2,
        backend="numpy",
        random_state=0,
    ).fit(X, y)
    rmse = np.sqrt(np.mean((model.predict(X) - y) ** 2))
    assert rmse < 0.3
    assert model.backend_ == "numpy"


def test_classifier_probabilities_and_labels(classification_data):
    X, y = classification_data
    model = HistGBDTClassifier(
        n_estimators=80,
        max_depth=3,
        learning_rate=0.2,
        backend="numpy",
        random_state=0,
    ).fit(X, y)
    probabilities = model.predict_proba(X)
    assert probabilities.shape == (X.shape[0], 2)
    assert np.allclose(probabilities.sum(axis=1), 1.0)
    assert np.all(probabilities >= 0.0)
    assert np.all(probabilities <= 1.0)
    assert model.score(X, y) > 0.9
    assert set(np.unique(model.predict(X))) <= {0, 1}


def test_classifier_maps_arbitrary_binary_labels(classification_data):
    X, y = classification_data
    labels = np.where(y == 1, "yes", "no")
    model = HistGBDTClassifier(
        n_estimators=30, max_depth=2, backend="numpy", random_state=0
    ).fit(X, labels)
    assert model.classes_.tolist() == ["no", "yes"]
    prediction = model.predict(X)
    assert set(np.unique(prediction)) <= {"no", "yes"}
    assert model.score(X, labels) > 0.9


def test_feature_importances_are_normalised(regression_data):
    X, y = regression_data
    model = HistGBDTRegressor(
        n_estimators=30, max_depth=3, backend="numpy", random_state=0
    ).fit(X, y)
    assert model.feature_importances_.shape == (X.shape[1],)
    assert np.isclose(model.feature_importances_.sum(), 1.0)


def test_get_and_set_params_roundtrip():
    model = HistGBDTRegressor(n_estimators=12, max_depth=2)
    params = model.get_params()
    assert params["n_estimators"] == 12
    assert params["max_depth"] == 2
    model.set_params(n_estimators=20, learning_rate=0.05)
    assert model.n_estimators == 20
    assert model.learning_rate == 0.05
    with pytest.raises(ValueError):
        model.set_params(not_a_parameter=1)


def test_early_stopping_keeps_best_iteration(regression_data):
    X, y = regression_data
    X_val, y_val = X[:100], y[:100]
    model = HistGBDTRegressor(
        n_estimators=500,
        max_depth=3,
        learning_rate=0.1,
        backend="numpy",
        random_state=0,
    ).fit(
        X,
        y,
        eval_set=[(X_val, y_val)],
        early_stopping_rounds=10,
    )
    assert model.n_estimators_ == model.best_iteration_ + 1
    assert model.n_estimators_ <= 500
    assert len(model.evals_result_["rmse"]) >= model.n_estimators_


def test_metric_direction_for_early_stopping():
    logistic = BinaryLogistic()
    assert logistic.metric_greater_is_better("auc")
    assert not logistic.metric_greater_is_better("logloss")
    assert not MeanSquaredError().metric_greater_is_better()


def test_early_stopping_with_auc_metric(classification_data):
    X, y = classification_data
    model = HistGBDTClassifier(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.1,
        eval_metric="auc",
        backend="numpy",
        random_state=0,
    ).fit(
        X,
        y,
        eval_set=[(X[:100], y[:100])],
        early_stopping_rounds=10,
    )
    assert model.n_estimators_ == model.best_iteration_ + 1
    assert 0.0 <= model.best_score_ <= 1.0


def test_predict_before_fit_raises():
    with pytest.raises(RuntimeError):
        HistGBDTRegressor().predict(np.zeros((2, 2)))
    with pytest.raises(RuntimeError):
        HistGBDTClassifier().predict_proba(np.zeros((2, 2)))


def test_same_random_state_is_reproducible(regression_data):
    X, y = regression_data
    kwargs = dict(
        n_estimators=25,
        max_depth=3,
        subsample=0.8,
        colsample_bytree=0.8,
        backend="numpy",
        random_state=42,
    )
    first = HistGBDTRegressor(**kwargs).fit(X, y)
    second = HistGBDTRegressor(**kwargs).fit(X, y)
    assert np.array_equal(first.predict(X), second.predict(X))


def test_classifier_requires_two_classes():
    X = np.zeros((4, 2))
    y = np.zeros(4)
    with pytest.raises(ValueError):
        HistGBDTClassifier(backend="numpy").fit(X, y)
