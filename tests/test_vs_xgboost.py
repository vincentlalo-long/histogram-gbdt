import numpy as np
import pytest

from gbdt import HistGBDTClassifier


def test_binary_auc_is_close_to_xgboost_hist():
    xgboost = pytest.importorskip("xgboost")
    metrics = pytest.importorskip("sklearn.metrics")

    rng = np.random.default_rng(31)
    X = rng.normal(size=(500, 6))
    logits = (
        1.8 * X[:, 0]
        - 1.4 * X[:, 1]
        + 0.9 * X[:, 2] * X[:, 3]
        + 0.4 * X[:, 4]
    )
    y = (logits + rng.normal(scale=0.4, size=500) > 0.0).astype(int)

    ours = HistGBDTClassifier(
        n_estimators=60,
        max_depth=3,
        learning_rate=0.1,
        backend="numpy",
        random_state=0,
    ).fit(X, y)

    reference = xgboost.XGBClassifier(
        n_estimators=60,
        max_depth=3,
        learning_rate=0.1,
        reg_lambda=1.0,
        min_child_weight=1.0,
        max_bin=256,
        tree_method="hist",
        base_score=float(np.mean(y)),
        objective="binary:logistic",
        eval_metric="logloss",
        n_jobs=2,
        random_state=0,
        verbosity=0,
    ).fit(X, y)

    ours_probability = ours.predict_proba(X)[:, 1]
    reference_probability = reference.predict_proba(X)[:, 1]
    ours_auc = metrics.roc_auc_score(y, ours_probability)
    reference_auc = metrics.roc_auc_score(y, reference_probability)

    assert ours_auc > 0.9
    assert reference_auc > 0.9
    assert abs(ours_auc - reference_auc) < 0.03
