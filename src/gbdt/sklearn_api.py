"""Scikit-learn-style estimators backed by the histogram GBDT booster."""

from __future__ import annotations

import numpy as np

from .binning import _as_2d_float
from .booster import Booster
from .objectives import (
    BinaryLogistic,
    MeanSquaredError,
    MulticlassSoftmax,
    sigmoid,
    softmax,
)

_PARAM_NAMES = (
    "n_estimators",
    "max_depth",
    "learning_rate",
    "max_bin",
    "reg_lambda",
    "gamma",
    "min_child_weight",
    "min_samples_leaf",
    "subsample",
    "colsample_bytree",
    "random_state",
    "backend",
    "n_jobs",
    "eval_metric",
)


class _BaseHistGBDT:
    _objective = None

    def _init_params(
        self,
        n_estimators: int,
        max_depth: int,
        learning_rate: float,
        max_bin: int,
        reg_lambda: float,
        gamma: float,
        min_child_weight: float,
        min_samples_leaf: int,
        subsample: float,
        colsample_bytree: float,
        random_state: int | None,
        backend: str,
        n_jobs: int,
        eval_metric: str | None,
    ) -> None:
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.max_bin = max_bin
        self.reg_lambda = reg_lambda
        self.gamma = gamma
        self.min_child_weight = min_child_weight
        self.min_samples_leaf = min_samples_leaf
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.random_state = random_state
        self.backend = backend
        self.n_jobs = n_jobs
        self.eval_metric = eval_metric
        self._booster: Booster | None = None

    def get_params(self, deep: bool = True) -> dict:
        return {name: getattr(self, name) for name in _PARAM_NAMES}

    def set_params(self, **params) -> "_BaseHistGBDT":
        for name, value in params.items():
            if name not in _PARAM_NAMES:
                raise ValueError(f"invalid parameter {name!r}")
            setattr(self, name, value)
        return self

    def _new_booster(self, objective=None) -> Booster:
        if objective is None:
            objective = self._objective
        return Booster(
            objective,
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            max_bin=self.max_bin,
            reg_lambda=self.reg_lambda,
            gamma=self.gamma,
            min_child_weight=self.min_child_weight,
            min_samples_leaf=self.min_samples_leaf,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            random_state=self.random_state,
            backend=self.backend,
            n_jobs=self.n_jobs,
            eval_metric=self.eval_metric,
        )

    def _copy_fitted_state(self, booster: Booster) -> None:
        self._booster = booster
        self.n_features_in_ = booster.n_features_in_
        self.n_estimators_ = booster.n_estimators_
        self.best_iteration_ = booster.best_iteration_
        self.best_score_ = booster.best_score_
        self.evals_result_ = booster.evals_result_
        self.feature_importances_ = booster.feature_importances_
        self.backend_ = booster.backend_
        self.base_score_ = booster.base_score_

    def _check_fitted(self) -> Booster:
        if self._booster is None:
            raise RuntimeError(
                f"{type(self).__name__} is not fitted; call fit() first"
            )
        return self._booster


class HistGBDTRegressor(_BaseHistGBDT):
    """Histogram gradient boosted trees for least-squares regression."""

    _objective = MeanSquaredError()

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 6,
        learning_rate: float = 0.1,
        max_bin: int = 256,
        reg_lambda: float = 1.0,
        gamma: float = 0.0,
        min_child_weight: float = 1.0,
        min_samples_leaf: int = 1,
        subsample: float = 1.0,
        colsample_bytree: float = 1.0,
        random_state: int | None = 0,
        backend: str = "auto",
        n_jobs: int = -1,
        eval_metric: str = "rmse",
    ) -> None:
        self._init_params(
            n_estimators,
            max_depth,
            learning_rate,
            max_bin,
            reg_lambda,
            gamma,
            min_child_weight,
            min_samples_leaf,
            subsample,
            colsample_bytree,
            random_state,
            backend,
            n_jobs,
            eval_metric,
        )

    def fit(
        self,
        X,
        y,
        *,
        eval_set=None,
        early_stopping_rounds: int | None = None,
        verbose: bool = False,
    ) -> "HistGBDTRegressor":
        booster = self._new_booster()
        booster.fit(
            X,
            y,
            eval_set=eval_set,
            early_stopping_rounds=early_stopping_rounds,
            verbose=verbose,
        )
        self._copy_fitted_state(booster)
        return self

    def predict(self, X) -> np.ndarray:
        return self._check_fitted().predict_raw(X)

    def score(self, X, y) -> float:
        y = np.asarray(y, dtype=np.float64)
        prediction = self.predict(X)
        residual = float(np.sum((y - prediction) ** 2))
        total = float(np.sum((y - np.mean(y)) ** 2))
        if total == 0.0:
            return 1.0 if residual == 0.0 else 0.0
        return 1.0 - residual / total


class HistGBDTClassifier(_BaseHistGBDT):
    """Histogram gradient boosted trees for binary and multiclass classification."""

    _objective = BinaryLogistic()

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 6,
        learning_rate: float = 0.1,
        max_bin: int = 256,
        reg_lambda: float = 1.0,
        gamma: float = 0.0,
        min_child_weight: float = 1.0,
        min_samples_leaf: int = 1,
        subsample: float = 1.0,
        colsample_bytree: float = 1.0,
        random_state: int | None = 0,
        backend: str = "auto",
        n_jobs: int = -1,
        eval_metric: str = "logloss",
    ) -> None:
        self._init_params(
            n_estimators,
            max_depth,
            learning_rate,
            max_bin,
            reg_lambda,
            gamma,
            min_child_weight,
            min_samples_leaf,
            subsample,
            colsample_bytree,
            random_state,
            backend,
            n_jobs,
            eval_metric,
        )

    def fit(
        self,
        X,
        y,
        *,
        eval_set=None,
        early_stopping_rounds: int | None = None,
        verbose: bool = False,
    ) -> "HistGBDTClassifier":
        X = _as_2d_float(X)
        y = np.asarray(y)
        if y.ndim != 1:
            raise ValueError(f"y must be 1-D, got shape {y.shape}")
        if y.size != X.shape[0]:
            raise ValueError(
                f"X has {X.shape[0]} rows but y has {y.size} values"
            )
        if y.dtype.kind in "fc" and not np.isfinite(y).all():
            raise ValueError("y contains NaN or infinity")

        classes, y_encoded = np.unique(y, return_inverse=True)
        if classes.size < 2:
            raise ValueError(
                "HistGBDTClassifier requires at least two classes; "
                f"got {classes.size}"
            )
        self.classes_ = classes
        self.n_classes_ = int(classes.size)
        if self.n_classes_ == 2:
            objective = BinaryLogistic()
            y_target = (y_encoded == 1).astype(np.float64)
        else:
            objective = MulticlassSoftmax(self.n_classes_)
            y_target = y_encoded.astype(np.float64)

        prepared_eval = None
        if eval_set is not None:
            prepared_eval = []
            for X_val, y_val in eval_set:
                y_val = np.asarray(y_val)
                unknown = ~np.isin(y_val, classes)
                if np.any(unknown):
                    raise ValueError("eval_set contains labels not seen in fit")
                if self.n_classes_ == 2:
                    encoded = (y_val == classes[1]).astype(np.float64)
                else:
                    encoded = np.searchsorted(classes, y_val).astype(
                        np.float64
                    )
                prepared_eval.append((X_val, encoded))

        booster = self._new_booster(objective)
        booster.fit(
            X,
            y_target,
            eval_set=prepared_eval,
            early_stopping_rounds=early_stopping_rounds,
            verbose=verbose,
        )
        self._copy_fitted_state(booster)
        return self

    def predict_proba(self, X) -> np.ndarray:
        raw = self._check_fitted().predict_raw(X)
        if self.n_classes_ == 2:
            positive = sigmoid(raw)
            return np.column_stack((1.0 - positive, positive))
        return softmax(raw)

    def predict_log_proba(self, X) -> np.ndarray:
        probabilities = self.predict_proba(X)
        return np.log(np.clip(probabilities, 1e-15, 1.0))

    def predict(self, X) -> np.ndarray:
        probabilities = self.predict_proba(X)
        return self.classes_[np.argmax(probabilities, axis=1)]

    def score(self, X, y) -> float:
        y = np.asarray(y)
        return float(np.mean(self.predict(X) == y))
