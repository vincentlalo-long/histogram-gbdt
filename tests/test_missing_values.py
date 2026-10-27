import numpy as np

from gbdt import HistGBDTClassifier, HistGBDTRegressor
from gbdt.binning import Binner


def test_missing_values_are_learned_as_a_separate_branch():
    X = np.array(
        [[0.0], [1.0], [np.nan], [np.nan], [0.0], [1.0], [np.nan], [np.nan]]
    )
    y = np.array([0.0, 0.0, 10.0, 10.0, 0.0, 0.0, 10.0, 10.0])
    for backend in ("numpy", "cpp"):
        model = HistGBDTRegressor(
            n_estimators=1,
            max_depth=1,
            learning_rate=1.0,
            reg_lambda=0.0,
            min_child_weight=0.0,
            backend=backend,
            random_state=0,
        ).fit(X, y)
        prediction = model.predict(X)
        tree = model._booster.trees[0][0]
        assert tree.feature[0] >= 0
        assert not np.isclose(prediction[0], prediction[1])
        missing_prediction = prediction[2]
        if tree.default_left[0]:
            assert np.allclose(prediction[0], missing_prediction)
            assert not np.allclose(prediction[1], missing_prediction)
        else:
            assert np.allclose(prediction[1], missing_prediction)
            assert not np.allclose(prediction[0], missing_prediction)
        assert np.allclose(prediction[:4], prediction[4:8])


def test_missing_values_are_accepted_in_eval_set():
    rng = np.random.default_rng(9)
    X = rng.normal(size=(200, 3))
    X[rng.random(X.shape) < 0.1] = np.nan
    y = np.nan_to_num(X[:, 0], nan=0.5) - 0.5 * np.nan_to_num(
        X[:, 1], nan=-0.5
    )
    model = HistGBDTRegressor(
        n_estimators=50,
        max_depth=3,
        learning_rate=0.2,
        backend="numpy",
        random_state=0,
    ).fit(
        X[:150],
        y[:150],
        eval_set=[(X[150:], y[150:])],
        early_stopping_rounds=5,
    )
    assert model.n_estimators_ == model.best_iteration_ + 1
    assert np.all(np.isfinite(model.predict(X)))


def test_missing_values_with_classifier():
    rng = np.random.default_rng(19)
    X = rng.normal(size=(300, 4))
    X[rng.random(X.shape) < 0.15] = np.nan
    logits = np.nan_to_num(X[:, 0], nan=1.0) - np.nan_to_num(
        X[:, 1], nan=-1.0
    )
    y = (logits + rng.normal(scale=0.2, size=300) > 0).astype(int)
    model = HistGBDTClassifier(
        n_estimators=50,
        max_depth=3,
        learning_rate=0.2,
        backend="numpy",
        random_state=0,
    ).fit(X, y)
    assert np.all(np.isfinite(model.predict_proba(X)))
    assert model.score(X, y) > 0.85


def test_binner_missing_bin_is_last():
    X = np.array([[1.0], [2.0], [np.nan], [3.0]])
    binner = Binner(max_bin=16).fit(X)
    assert binner.missing_bin_.tolist() == [binner.n_bins_[0] - 1]
    binned = binner.transform(X)
    assert binned[0, 2] == binner.missing_bin_[0]
