"""Gradient boosting loop with 2nd-order Taylor updates."""

from __future__ import annotations

import numpy as np

from .backend import build_tree_cpp, get_splitter
from .binning import Binner, _as_2d_float
from .objectives import Objective
from .tree import Tree, TreeBuilder


def _as_1d_float(y, n_samples: int) -> np.ndarray:
    y = np.asarray(y, dtype=np.float64)
    if y.ndim == 2 and y.shape[1] == 1:
        y = y[:, 0]
    if y.ndim != 1:
        raise ValueError(f"y must be 1-D, got shape {y.shape}")
    if y.size != n_samples:
        raise ValueError(f"X has {n_samples} rows but y has {y.size} values")
    if not np.isfinite(y).all():
        raise ValueError("y contains NaN or infinity")
    return y


class Booster:
    """Fit an additive ensemble of histogram regression trees."""

    def __init__(
        self,
        objective: Objective,
        *,
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
        eval_metric: str | None = None,
    ) -> None:
        if n_estimators < 1:
            raise ValueError("n_estimators must be at least 1")
        if max_depth < 0:
            raise ValueError("max_depth must be non-negative")
        if learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive")
        if reg_lambda < 0.0:
            raise ValueError("reg_lambda must be non-negative")
        if gamma < 0.0:
            raise ValueError("gamma must be non-negative")
        if min_child_weight < 0.0:
            raise ValueError("min_child_weight must be non-negative")
        if min_samples_leaf < 1:
            raise ValueError("min_samples_leaf must be at least 1")
        if not 0.0 < subsample <= 1.0:
            raise ValueError("subsample must be in (0, 1]")
        if not 0.0 < colsample_bytree <= 1.0:
            raise ValueError("colsample_bytree must be in (0, 1]")

        self.objective = objective
        self.n_estimators = int(n_estimators)
        self.max_depth = int(max_depth)
        self.learning_rate = float(learning_rate)
        self.max_bin = int(max_bin)
        self.reg_lambda = float(reg_lambda)
        self.gamma = float(gamma)
        self.min_child_weight = float(min_child_weight)
        self.min_samples_leaf = int(min_samples_leaf)
        self.subsample = float(subsample)
        self.colsample_bytree = float(colsample_bytree)
        self.random_state = random_state
        self.backend = backend
        self.n_jobs = int(n_jobs)
        self.eval_metric = eval_metric

        self.binner = Binner(self.max_bin)
        self.splitter, self.backend_ = get_splitter(
            self.backend,
            self.n_jobs,
            self.reg_lambda,
            self.gamma,
            self.min_child_weight,
            self.min_samples_leaf,
        )
        self.tree_builder: TreeBuilder | None = None

        self.trees: list[list[Tree]] = []
        self.n_estimators_: int | None = None
        self.n_features_in_: int | None = None
        self.base_score_: float | np.ndarray | None = None
        self.best_iteration_: int | None = None
        self.best_score_: float | None = None
        self.evals_result_: dict[str, list[float]] = {}
        self.feature_importances_: np.ndarray | None = None

    @property
    def n_outputs_(self) -> int:
        return int(self.objective.n_outputs)

    def _validate_y(self, y, n_samples: int) -> np.ndarray:
        y = _as_1d_float(y, n_samples)
        if self.objective.is_multiclass:
            labels = y.astype(np.int64)
            if not np.all(y == labels):
                raise ValueError("multiclass y must contain integer class ids")
            if np.any(labels < 0) or np.any(labels >= self.objective.n_outputs):
                raise ValueError(
                    "multiclass y values must be in "
                    f"[0, {self.objective.n_outputs - 1}]"
                )
        elif self.objective.is_classification:
            if not np.all((y == 0.0) | (y == 1.0)):
                raise ValueError(
                    "binary objective requires y values in {0, 1}"
                )
        return y

    def _initial_prediction(self, n_samples: int) -> np.ndarray:
        assert self.base_score_ is not None
        if self.n_outputs_ == 1:
            return np.full(n_samples, float(self.base_score_), dtype=np.float64)
        base = np.asarray(self.base_score_, dtype=np.float64)
        return np.tile(base, (n_samples, 1))

    def _prepare_eval_set(self, eval_set):
        if eval_set is None:
            return []
        if len(eval_set) == 0:
            raise ValueError("eval_set must contain at least one (X, y) pair")
        prepared = []
        for item in eval_set:
            if not isinstance(item, (tuple, list)) or len(item) != 2:
                raise ValueError("eval_set must be a sequence of (X, y) pairs")
            X_val, y_val = item
            X_val = _as_2d_float(X_val, name="eval X")
            if X_val.shape[1] != self.n_features_in_:
                raise ValueError(
                    f"eval X has {X_val.shape[1]} features, "
                    f"expected {self.n_features_in_}"
                )
            y_val = self._validate_y(y_val, X_val.shape[0])
            prepared.append(
                (
                    self.binner.transform(X_val),
                    y_val,
                    self._initial_prediction(X_val.shape[0]),
                )
            )
        return prepared

    def _make_tree_builder(self) -> TreeBuilder | None:
        if self.backend_ == "cpp":
            return None
        assert self.binner.missing_bin_ is not None
        return TreeBuilder(
            splitter=self.splitter,
            n_bins=self.binner.n_bins_max_ or 1,
            max_depth=self.max_depth,
            reg_lambda=self.reg_lambda,
            gamma=self.gamma,
            min_child_weight=self.min_child_weight,
            min_samples_leaf=self.min_samples_leaf,
            missing_bins=self.binner.missing_bin_,
        )

    def _build_tree(
        self,
        Xbin: np.ndarray,
        grad: np.ndarray,
        hess: np.ndarray,
        rows: np.ndarray,
        features: np.ndarray,
    ) -> Tree:
        assert self.binner.missing_bin_ is not None
        if self.backend_ == "cpp":
            return build_tree_cpp(
                Xbin,
                grad,
                hess,
                rows,
                features,
                self.binner.missing_bin_,
                self.binner.n_bins_max_ or 1,
                max_depth=self.max_depth,
                reg_lambda=self.reg_lambda,
                gamma=self.gamma,
                min_child_weight=self.min_child_weight,
                min_samples_leaf=self.min_samples_leaf,
                n_jobs=self.n_jobs,
            )
        assert self.tree_builder is not None
        return self.tree_builder.build(Xbin, grad, hess, rows, features)

    def fit(
        self,
        X,
        y,
        *,
        eval_set=None,
        early_stopping_rounds: int | None = None,
        verbose: bool = False,
    ) -> "Booster":
        X = _as_2d_float(X)
        y = self._validate_y(y, X.shape[0])
        if early_stopping_rounds is not None:
            early_stopping_rounds = int(early_stopping_rounds)
            if early_stopping_rounds < 1:
                raise ValueError("early_stopping_rounds must be at least 1")
            if eval_set is None:
                raise ValueError("early_stopping_rounds requires an eval_set")

        self.trees = [[] for _ in range(self.n_outputs_)]
        self.evals_result_ = {}
        self.n_features_in_ = X.shape[1]
        base_score = self.objective.base_score(y)
        self.base_score_ = (
            float(base_score)
            if self.n_outputs_ == 1
            else np.asarray(base_score, dtype=np.float64)
        )
        self.binner.fit(X)
        Xbin = self.binner.transform(X)
        self.tree_builder = self._make_tree_builder()

        eval_data = self._prepare_eval_set(eval_set)
        metric_name = self.eval_metric or self.objective.default_metric
        if eval_data:
            self.evals_result_[metric_name] = []

        n_samples = X.shape[0]
        n_features = X.shape[1]
        pred = self._initial_prediction(n_samples)
        rng = np.random.default_rng(self.random_state)

        greater_is_better = self.objective.metric_greater_is_better(
            metric_name
        )
        best_score = -np.inf if greater_is_better else np.inf
        best_iteration = 0
        no_improvement = 0
        stopped_early = False

        for iteration in range(self.n_estimators):
            if self.subsample < 1.0:
                n_selected = max(1, int(np.ceil(self.subsample * n_samples)))
                rows = np.sort(
                    rng.choice(n_samples, size=n_selected, replace=False)
                ).astype(np.int64, copy=False)
            else:
                rows = np.arange(n_samples, dtype=np.int64)

            if self.colsample_bytree < 1.0:
                n_selected = max(
                    1, int(np.ceil(self.colsample_bytree * n_features))
                )
                features = np.sort(
                    rng.choice(n_features, size=n_selected, replace=False)
                ).astype(np.int64, copy=False)
            else:
                features = np.arange(n_features, dtype=np.int64)

            grad, hess = self.objective.grad_hess(y, pred)
            for output in range(self.n_outputs_):
                if self.n_outputs_ == 1:
                    grad_output = grad
                    hess_output = hess
                else:
                    grad_output = grad[:, output]
                    hess_output = hess[:, output]

                tree = self._build_tree(
                    Xbin, grad_output, hess_output, rows, features
                )
                self.trees[output].append(tree)
                update = self.learning_rate * tree.predict(Xbin)
                if self.n_outputs_ == 1:
                    pred += update
                else:
                    pred[:, output] += update

            if eval_data:
                for output in range(self.n_outputs_):
                    tree = self.trees[output][-1]
                    for X_val_bin, _, val_pred in eval_data:
                        update = self.learning_rate * tree.predict(X_val_bin)
                        if self.n_outputs_ == 1:
                            val_pred += update
                        else:
                            val_pred[:, output] += update
                _, y_val, val_pred = eval_data[-1]
                score = self.objective.metric(y_val, val_pred, metric_name)
                self.evals_result_[metric_name].append(float(score))

                if verbose:
                    print(
                        f"[{iteration + 1}] validation_0-{metric_name}: "
                        f"{score:.6f}"
                    )

                if early_stopping_rounds is not None:
                    if greater_is_better:
                        improved = score > best_score + 1e-12
                    else:
                        improved = score < best_score - 1e-12
                    if improved:
                        best_score = float(score)
                        best_iteration = iteration
                        no_improvement = 0
                    else:
                        no_improvement += 1
                        if no_improvement >= early_stopping_rounds:
                            stopped_early = True
                            break
            elif verbose:
                print(f"[{iteration + 1}] fitted boosting round")

        if early_stopping_rounds is not None:
            keep = best_iteration + 1
            for output in range(self.n_outputs_):
                if keep < len(self.trees[output]):
                    self.trees[output] = self.trees[output][:keep]
            self.best_iteration_ = best_iteration
            self.best_score_ = best_score
        else:
            self.best_iteration_ = None
            self.best_score_ = None

        self.n_estimators_ = len(self.trees[0])
        self._compute_feature_importances()

        if stopped_early and verbose:
            print(
                f"Early stopping at iteration {self.n_estimators_} "
                f"(best iteration {best_iteration + 1})"
            )
        return self

    def _compute_feature_importances(self) -> None:
        if self.n_features_in_ is None:
            return
        importance = np.zeros(self.n_features_in_, dtype=np.float64)
        for output_trees in self.trees:
            for tree in output_trees:
                internal = tree.feature >= 0
                if np.any(internal):
                    np.add.at(
                        importance,
                        tree.feature[internal],
                        tree.gain[internal],
                    )
        total = float(importance.sum())
        if total > 0.0:
            importance /= total
        self.feature_importances_ = importance

    def _check_fitted(self) -> None:
        if self.n_estimators_ is None or self.base_score_ is None:
            raise RuntimeError("Booster is not fitted; call fit() first")

    def predict_raw(self, X) -> np.ndarray:
        """Return raw additive scores (logits for logistic/softmax loss)."""
        self._check_fitted()
        X = _as_2d_float(X)
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X.shape[1]} features, expected {self.n_features_in_}"
            )
        Xbin = self.binner.transform(X)
        prediction = self._initial_prediction(X.shape[0])
        for output in range(self.n_outputs_):
            for tree in self.trees[output]:
                update = self.learning_rate * tree.predict(Xbin)
                if self.n_outputs_ == 1:
                    prediction += update
                else:
                    prediction[:, output] += update
        return prediction

    def predict(self, X) -> np.ndarray:
        return self.predict_raw(X)
