"""The deliberately small supported import surface."""

from __future__ import annotations

import inspect
from pathlib import Path

import scikit_jev
from scikit_jev import (
    JevBatchError,
    JevBatchResult,
    JevChoiceClassifier,
    JevConfigurationError,
    JevError,
    JevInferenceError,
    JevNoulClassifier,
    JevResponseError,
    JevScoreRegressor,
    JevTransformer,
    JevUsage,
    TypeSafeBackend,
)
from scikit_jev._results import JevBatchResult as InternalBatchResult
from scikit_jev._results import JevUsage as InternalUsage
from scikit_jev.backends import TypeSafeBackend as BackendDefinition
from scikit_jev.classification import JevChoiceClassifier as ChoiceDefinition
from scikit_jev.classification import JevNoulClassifier as NoulDefinition
from scikit_jev.regression import JevScoreRegressor as ScoreDefinition
from scikit_jev.transformer import JevTransformer as TransformerDefinition


def test_public_exports_are_explicit_and_stable() -> None:
    assert set(scikit_jev.__all__) == {
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
    }
    assert JevTransformer is TransformerDefinition
    assert JevNoulClassifier is NoulDefinition
    assert JevChoiceClassifier is ChoiceDefinition
    assert JevScoreRegressor is ScoreDefinition
    assert TypeSafeBackend is BackendDefinition
    assert JevBatchResult is InternalBatchResult
    assert JevUsage is InternalUsage


def test_public_exception_hierarchy() -> None:
    assert issubclass(JevConfigurationError, (JevError, ValueError))
    assert issubclass(JevResponseError, (JevError, ValueError))
    assert issubclass(JevInferenceError, JevError)
    assert issubclass(JevBatchError, JevError)


def test_version_is_available_for_installed_and_source_tree_imports() -> None:
    assert isinstance(scikit_jev.__version__, str)
    assert scikit_jev.__version__


def test_public_constructors_never_accept_an_api_key() -> None:
    for public_type in (
        JevTransformer,
        JevNoulClassifier,
        JevChoiceClassifier,
        JevScoreRegressor,
        TypeSafeBackend,
    ):
        assert "api_key" not in inspect.signature(public_type).parameters


def test_distribution_declares_inline_typing_support() -> None:
    package_dir = Path(scikit_jev.__file__).resolve().parent

    assert (package_dir / "py.typed").is_file()
