"""Probability-preserving scikit-learn estimators for System One and Jev."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

from scikit_jev._results import JevBatchResult, JevUsage
from scikit_jev.backends import TypeSafeBackend
from scikit_jev.classification import JevChoiceClassifier, JevNoulClassifier
from scikit_jev.exceptions import (
    JevBatchError,
    JevConfigurationError,
    JevError,
    JevInferenceError,
    JevResponseError,
)
from scikit_jev.regression import JevScoreRegressor
from scikit_jev.transformer import JevTransformer

try:
    __version__ = version("scikit-jev")
except PackageNotFoundError:
    # Source-tree imports do not necessarily have installed distribution
    # metadata.  Keep this PEP 440-compatible and visibly non-release-like.
    __version__ = "0+unknown"

__all__ = [
    "JevBatchError",
    "JevBatchResult",
    "JevChoiceClassifier",
    "JevConfigurationError",
    "JevError",
    "JevInferenceError",
    "JevNoulClassifier",
    "JevResponseError",
    "JevScoreRegressor",
    "JevTransformer",
    "JevUsage",
    "TypeSafeBackend",
    "__version__",
]
