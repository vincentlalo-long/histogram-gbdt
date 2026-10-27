import numpy as np

from gbdt import HistGBDTClassifier
from gbdt.objectives import MulticlassSoftmax, softmax


def _multiclass_data(n=600, seed=42):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 5))
    logits = np.column_stack(
        [
            1.8 * X[:, 0] + 0.5 * X[:, 2],
            -1.5 * X[:, 1] + 0.4 * X[:, 3],
            0.8 * X[:, 0] + 1.2 * X[:, 1] - X[:, 2],
        ]
    )
    y = np.argmax(logits + rng.normal(scale=0.3, size=logits.shape), axis=1)
    return X, y


def test_softmax_gradients_and_base_score():
    rng = np.random.default_rng(3)
    y = rng.integers(0, 3, size=20)
    prediction = rng.normal(size=(20, 3))
    probabilities = softmax(prediction)
    assert np.allclose(probabilities.sum(axis=1), 1.0)

    objective = MulticlassSoftmax(3)
    gradient, hessian = objective.grad_hess(y, prediction)
    assert gradient.shape == prediction.shape
    assert hessian.shape == prediction.shape
    assert np.all(hessian > 0.0)
    assert objective.base_score(y).shape == (3,)
    loss = objective.metric(y, prediction, "mlogloss")
    assert np.isfinite(loss)
    assert 0.0 <= objective.metric(y, prediction, "merror") <= 1.0


def test_multiclass_probabilities_labels_and_importances():
    X, y = _multiclass_data()
    model = HistGBDTClassifier(
        n_estimators=40,
        max_depth=3,
        learning_rate=0.2,
        backend="numpy",
        random_state=0,
    ).fit(X, y)
    probabilities = model.predict_proba(X)
    assert model.n_classes_ == 3
    assert probabilities.shape == (X.shape[0], 3)
    assert np.allclose(probabilities.sum(axis=1), 1.0)
    assert np.all(probabilities >= 0.0)
    assert np.all(probabilities <= 1.0)
    assert model.score(X, y) > 0.9
    assert model.feature_importances_.shape == (X.shape[1],)
    assert np.isclose(model.feature_importances_.sum(), 1.0)
    assert np.array_equal(
        model.predict(X), model.classes_[np.argmax(probabilities, axis=1)]
    )


def test_multiclass_string_labels():
    X, y = _multiclass_data(n=400)
    labels = np.array(["ant", "bee", "cat"])[y]
    model = HistGBDTClassifier(
        n_estimators=30,
        max_depth=3,
        learning_rate=0.2,
        backend="numpy",
        random_state=0,
    ).fit(X, labels)
    assert model.classes_.tolist() == ["ant", "bee", "cat"]
    assert set(np.unique(model.predict(X))) <= {"ant", "bee", "cat"}
    assert model.score(X, labels) > 0.85


def test_multiclass_early_stopping_and_eval_mapping():
    X, y = _multiclass_data()
    model = HistGBDTClassifier(
        n_estimators=300,
        max_depth=3,
        learning_rate=0.1,
        eval_metric="mlogloss",
        backend="numpy",
        random_state=0,
    ).fit(
        X[:450],
        y[:450],
        eval_set=[(X[450:], y[450:])],
        early_stopping_rounds=10,
    )
    assert model.n_estimators_ == model.best_iteration_ + 1
    assert model.n_estimators_ <= 300
    assert len(model.evals_result_["mlogloss"]) >= model.n_estimators_
    assert np.isfinite(model.best_score_)


def test_multiclass_unknown_eval_label_is_rejected():
    X, y = _multiclass_data(n=200)
    model = HistGBDTClassifier(n_estimators=3, backend="numpy")
    bad = y[150:].copy()
    bad[0] = 99
    try:
        model.fit(X[:150], y[:150], eval_set=[(X[150:], bad)])
    except ValueError:
        pass
    else:
        raise AssertionError("unknown multiclass eval label was accepted")
