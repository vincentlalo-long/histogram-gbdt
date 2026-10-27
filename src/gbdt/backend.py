"""Splitter backend selection: NumPy reference or C++/OpenMP kernel."""

from __future__ import annotations

from typing import Callable

import numpy as np

from .splitter import Split, find_best_split_numpy

try:  # The extension is optional: the NumPy reference works on its own.
    from . import _core
except ImportError as exc:  # pragma: no cover - depends on build environment
    _core = None
    _CPP_IMPORT_ERROR: ImportError | None = exc
else:
    _CPP_IMPORT_ERROR = None

CPP_AVAILABLE = _core is not None

Splitter = Callable[
    [
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        int,
        np.ndarray,
    ],
    Split | None,
]


def _normalise_threads(n_jobs: int) -> int:
    n_jobs = int(n_jobs)
    if n_jobs == 0:
        return 1
    return n_jobs


def _numpy_splitter_with_params(
    reg_lambda: float,
    gamma: float,
    min_child_weight: float,
    min_samples_leaf: int,
) -> Splitter:
    def splitter(
        Xbin: np.ndarray,
        rows: np.ndarray,
        grad: np.ndarray,
        hess: np.ndarray,
        features: np.ndarray,
        n_bins: int,
        missing_bins: np.ndarray,
    ) -> Split | None:
        return find_best_split_numpy(
            Xbin,
            rows,
            grad,
            hess,
            features,
            n_bins,
            reg_lambda,
            gamma,
            min_child_weight,
            min_samples_leaf,
            missing_bins=missing_bins,
        )

    return splitter


def _cpp_splitter_with_params(
    n_jobs: int,
    reg_lambda: float,
    gamma: float,
    min_child_weight: float,
    min_samples_leaf: int,
) -> Splitter:
    if _core is None:  # pragma: no cover
        raise ImportError("C++ backend is not built")

    n_threads = _normalise_threads(n_jobs)

    def splitter(
        Xbin: np.ndarray,
        rows: np.ndarray,
        grad: np.ndarray,
        hess: np.ndarray,
        features: np.ndarray,
        n_bins: int,
        missing_bins: np.ndarray,
    ) -> Split | None:
        result = _core.find_best_split(
            np.ascontiguousarray(Xbin, dtype=np.uint8),
            np.ascontiguousarray(rows, dtype=np.int64),
            np.ascontiguousarray(grad, dtype=np.float64),
            np.ascontiguousarray(hess, dtype=np.float64),
            np.ascontiguousarray(features, dtype=np.int64),
            np.ascontiguousarray(missing_bins, dtype=np.int64),
            int(n_bins),
            float(reg_lambda),
            float(gamma),
            float(min_child_weight),
            int(min_samples_leaf),
            n_threads,
        )
        (
            feature,
            threshold,
            gain,
            left_count,
            left_grad,
            left_hess,
            default_left,
        ) = result
        if feature < 0 or gain <= 0.0:
            return None
        return Split(
            feature=int(feature),
            threshold_bin=int(threshold),
            gain=float(gain),
            left_count=int(left_count),
            left_grad=float(left_grad),
            left_hess=float(left_hess),
            default_left=bool(default_left),
        )

    return splitter


def build_tree_cpp(
    Xbin: np.ndarray,
    grad: np.ndarray,
    hess: np.ndarray,
    rows: np.ndarray,
    features: np.ndarray,
    missing_bins: np.ndarray,
    n_bins: int,
    *,
    max_depth: int,
    reg_lambda: float,
    gamma: float,
    min_child_weight: float,
    min_samples_leaf: int,
    n_jobs: int,
):
    """Build one tree in C++ using histogram subtraction."""
    if _core is None:  # pragma: no cover
        raise ImportError("C++ backend is not built")

    from .tree import Tree

    result = _core.build_tree(
        np.ascontiguousarray(Xbin, dtype=np.uint8),
        np.ascontiguousarray(rows, dtype=np.int64),
        np.ascontiguousarray(grad, dtype=np.float64),
        np.ascontiguousarray(hess, dtype=np.float64),
        np.ascontiguousarray(features, dtype=np.int64),
        np.ascontiguousarray(missing_bins, dtype=np.int64),
        int(n_bins),
        int(max_depth),
        float(reg_lambda),
        float(gamma),
        float(min_child_weight),
        int(min_samples_leaf),
        _normalise_threads(n_jobs) if n_jobs != -1 else -1,
    )
    (
        feature,
        threshold,
        left,
        right,
        value,
        gain,
        default_left,
    ) = result
    return Tree(
        feature=np.asarray(feature, dtype=np.int32),
        threshold=np.asarray(threshold, dtype=np.int32),
        left=np.asarray(left, dtype=np.int32),
        right=np.asarray(right, dtype=np.int32),
        value=np.asarray(value, dtype=np.float64),
        gain=np.asarray(gain, dtype=np.float64),
        default_left=np.asarray(default_left, dtype=bool),
        missing_bins=np.asarray(missing_bins, dtype=np.int64),
        n_features=int(Xbin.shape[0]),
        max_depth=int(max_depth),
    )


def get_splitter(
    backend: str,
    n_jobs: int,
    reg_lambda: float,
    gamma: float,
    min_child_weight: float,
    min_samples_leaf: int,
) -> tuple[Splitter, str]:
    """Return ``(splitter, resolved_backend_name)``."""
    name = backend.lower()
    if name == "auto":
        name = "cpp" if CPP_AVAILABLE else "numpy"

    if name == "numpy":
        return (
            _numpy_splitter_with_params(
                reg_lambda,
                gamma,
                min_child_weight,
                min_samples_leaf,
            ),
            "numpy",
        )
    if name == "cpp":
        if not CPP_AVAILABLE:
            message = (
                "C++ backend is not available. Build it with "
                "`pip install -e .` (requires pybind11 and an OpenMP compiler)."
            )
            if _CPP_IMPORT_ERROR is not None:  # pragma: no cover
                message += f" Import error: {_CPP_IMPORT_ERROR}"
            raise ImportError(message)
        return (
            _cpp_splitter_with_params(
                n_jobs,
                reg_lambda,
                gamma,
                min_child_weight,
                min_samples_leaf,
            ),
            "cpp",
        )
    raise ValueError(f"unknown backend {backend!r}; use 'auto', 'numpy', or 'cpp'")
