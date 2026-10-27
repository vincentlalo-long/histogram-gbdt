"""Loss functions and their first/second derivatives.

Every objective exposes the interface required by the 2nd-order boosting
loop: a base score, per-sample gradients/hessians, and an evaluation
metric.  The gradients follow the sign convention used in the theory
notes: ``g_i = d loss / d prediction`` and the Newton step is
``w* = -G / (H + lambda)``.
"""

from __future__ import annotations

import numpy as np

_EPS = 1e-12


def sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable logistic function."""
    x = np.asarray(x, dtype=np.float64)
    out = np.empty_like(x)
    positive = x >= 0
    out[positive] = 1.0 / (1.0 + np.exp(-x[positive]))
    exp_x = np.exp(x[~positive])
    out[~positive] = exp_x / (1.0 + exp_x)
    return out


def softmax(x: np.ndarray) -> np.ndarray:
    """Numerically stable softmax over the last axis."""
    x = np.asarray(x, dtype=np.float64)
    shifted = x - np.max(x, axis=1, keepdims=True)
    exp_x = np.exp(shifted)
    return exp_x / exp_x.sum(axis=1, keepdims=True)


def _roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Rank-based ROC-AUC with average ranks for tied scores."""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_score = np.asarray(y_score, dtype=np.float64)
    n_pos = float(y_true.sum())
    n_neg = float(y_true.size - n_pos)
    if n_pos == 0.0 or n_neg == 0.0:
        return 0.5

    order = np.argsort(y_score, kind="mergesort")
    sorted_scores = y_score[order]
    ranks = np.empty(y_score.size, dtype=np.float64)
    start = 0
    while start < y_score.size:
        end = start + 1
        while end < y_score.size and sorted_scores[end] == sorted_scores[start]:
            end += 1
        # Ranks start+1 ... end, so their average is (start+1+end)/2.
        ranks[order[start:end]] = 0.5 * (start + 1 + end)
        start = end

    rank_sum_pos = ranks[y_true == 1.0].sum()
    return float(
        (rank_sum_pos - n_pos * (n_pos + 1.0) / 2.0) / (n_pos * n_neg)
    )


class Objective:
    name = "objective"
    is_classification = False
    is_multiclass = False
    n_outputs = 1
    default_metric = "rmse"
    allowed_metrics: tuple[str, ...] = ("rmse", "mse")

    def base_score(self, y: np.ndarray) -> float | np.ndarray:
        raise NotImplementedError

    def grad_hess(
        self, y: np.ndarray, prediction: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError

    def metric_greater_is_better(self, name: str | None = None) -> bool:
        name = self.default_metric if name is None else name
        return name == "auc"

    def metric(self, y: np.ndarray, prediction: np.ndarray, name: str | None = None) -> float:
        name = self.default_metric if name is None else name
        if name not in self.allowed_metrics:
            raise ValueError(
                f"unsupported metric {name!r} for {self.name}; "
                f"choose from {self.allowed_metrics}"
            )
        if name == "mse":
            return float(np.mean((prediction - y) ** 2))
        if name == "rmse":
            return float(np.sqrt(np.mean((prediction - y) ** 2)))
        if name in {"logloss", "mlogloss"}:
            return self._logloss(y, prediction)
        if name in {"error", "merror", "accuracy"}:
            if prediction.ndim == 2:
                labels = np.argmax(prediction, axis=1)
            else:
                labels = (prediction >= 0.0).astype(np.float64)
            accuracy = float(np.mean(labels == y))
            return accuracy if name == "accuracy" else 1.0 - accuracy
        if name == "auc":
            return _roc_auc(y, prediction)
        raise AssertionError(f"unhandled metric {name}")

    def _logloss(self, y: np.ndarray, prediction: np.ndarray) -> float:
        raise ValueError(f"metric logloss is not defined for {self.name}")


class MeanSquaredError(Objective):
    name = "regression"
    default_metric = "rmse"

    def base_score(self, y: np.ndarray) -> float:
        return float(np.mean(y))

    def grad_hess(
        self, y: np.ndarray, prediction: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        return prediction - y, np.ones_like(prediction)


class BinaryLogistic(Objective):
    name = "binary_logistic"
    is_classification = True
    default_metric = "logloss"
    allowed_metrics = ("logloss", "error", "auc")

    def base_score(self, y: np.ndarray) -> float:
        p = float(np.mean(y))
        p = min(max(p, 1e-6), 1.0 - 1e-6)
        return float(np.log(p / (1.0 - p)))

    def grad_hess(
        self, y: np.ndarray, prediction: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        p = sigmoid(prediction)
        # A tiny floor keeps the Hessian positive for saturated scores.
        h = np.maximum(p * (1.0 - p), _EPS)
        return p - y, h

    def _logloss(self, y: np.ndarray, prediction: np.ndarray) -> float:
        p = sigmoid(prediction)
        p = np.clip(p, _EPS, 1.0 - _EPS)
        return float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))


class MulticlassSoftmax(Objective):
    name = "multiclass_softmax"
    is_classification = True
    is_multiclass = True
    default_metric = "mlogloss"
    allowed_metrics = ("mlogloss", "logloss", "merror", "error")

    def __init__(self, n_classes: int) -> None:
        n_classes = int(n_classes)
        if n_classes < 2:
            raise ValueError("n_classes must be at least 2")
        self.n_classes = n_classes
        self.n_outputs = n_classes

    def base_score(self, y: np.ndarray) -> np.ndarray:
        counts = np.bincount(y.astype(np.int64), minlength=self.n_classes)
        prior = (counts + 1e-6) / (counts.sum() + 1e-6 * self.n_classes)
        return np.log(prior)

    def grad_hess(
        self, y: np.ndarray, prediction: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        if prediction.ndim != 2 or prediction.shape[1] != self.n_classes:
            raise ValueError(
                f"multiclass prediction must have shape (n, {self.n_classes})"
            )
        probabilities = softmax(prediction)
        labels = y.astype(np.int64)
        one_hot = np.zeros_like(probabilities)
        one_hot[np.arange(labels.size), labels] = 1.0
        gradient = probabilities - one_hot
        hessian = np.maximum(
            probabilities * (1.0 - probabilities), _EPS
        )
        return gradient, hessian

    def _logloss(self, y: np.ndarray, prediction: np.ndarray) -> float:
        probabilities = softmax(prediction)
        labels = y.astype(np.int64)
        true_probability = probabilities[np.arange(labels.size), labels]
        return float(-np.mean(np.log(np.clip(true_probability, _EPS, 1.0))))


def make_objective(name: str, n_classes: int | None = None) -> Objective:
    key = name.lower().replace("-", "_")
    if key in {"regression", "mse", "squared_error", "reg:squarederror"}:
        return MeanSquaredError()
    if key in {"binary", "binary_logistic", "binary:logistic", "logistic"}:
        return BinaryLogistic()
    if key in {"multiclass", "softmax", "multi:softprob"}:
        if n_classes is None:
            raise ValueError("multiclass objective requires n_classes")
        return MulticlassSoftmax(n_classes)
    raise ValueError(
        f"unsupported objective {name!r}; use 'regression', "
        "'binary_logistic', or 'multiclass'"
    )
