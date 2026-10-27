"""High-performance histogram gradient boosted decision trees.

The package ships a readable NumPy reference engine and an optional
C++/OpenMP tree builder.  Both implement the same 2nd-order Taylor
boosting algorithm, missing-value routing, and multiclass softmax, and
are kept numerically close by the test suite.
"""

from .backend import CPP_AVAILABLE
from .sklearn_api import HistGBDTClassifier, HistGBDTRegressor

__version__ = "0.1.0"

__all__ = [
    "CPP_AVAILABLE",
    "HistGBDTClassifier",
    "HistGBDTRegressor",
    "__version__",
]
