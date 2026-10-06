"""Shared, network-free fit and inference plumbing for Jev estimators."""

from __future__ import annotations

import copy
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_consistent_length, check_is_fitted
from typesafe_sdk import JSONContent, Question

from scikit_jev._engine import execute_system_one
from scikit_jev._results import CanonicalResponse
from scikit_jev._states import build_states
from scikit_jev._validation import QuestionSpec, validate_questions, validate_response
from scikit_jev.backends import QuestionMapping, SystemOneBackend, TypeSafeBackend
from scikit_jev.exceptions import JevConfigurationError, JevResponseError


@dataclass(frozen=True, slots=True)
class _InputSchema:
    """Safe structural input metadata bound during fit."""

    kind: Literal["samples", "array2d", "dataframe"]
    n_features: int | None = None
    feature_names: tuple[str, ...] | None = None


@dataclass(frozen=True, slots=True)
class _FitBinding:
    """Validated common fitted state awaiting an atomic estimator commit."""

    states: tuple[JSONContent, ...]
    questions: QuestionMapping
    question_specs: tuple[QuestionSpec, ...]
    backend: SystemOneBackend
    input_schema: _InputSchema


class _BaseJevEstimator(BaseEstimator):
    """Common implementation for zero-shot System One estimators.

    Subclass constructors remain explicit and assignment-only.  This class
    assumes they expose ``questions``, ``model``, ``backend``, and
    ``state_builder`` parameters.
    """

    def _prepare_fit(self, X: Any, y: Any = None) -> _FitBinding:
        """Validate fit inputs without contacting a backend or mutating state."""

        self._reset_fitted_state()
        input_schema = _describe_input_schema(X)
        states = build_states(X, state_builder=self.state_builder)
        if not states:
            raise JevConfigurationError("fit requires at least one input sample")
        if y is not None:
            try:
                check_consistent_length(states, y)
            except Exception:
                raise JevConfigurationError("X and y must contain the same number of samples") from None

        if self.model is not None and (not isinstance(self.model, str) or not self.model.strip()):
            raise JevConfigurationError("model must be a non-empty string or None")

        try:
            copied_questions = copy.deepcopy(self._questions_for_fit())
        except Exception:
            raise JevConfigurationError("questions could not be copied into a stable fitted schema") from None
        question_specs = validate_questions(copied_questions)
        # A plain dict freezes the mapping's insertion order independently of a
        # caller-owned mutable mapping.  Nested question data was copied above.
        bound_questions: dict[str, Question] = dict(copied_questions)

        backend = TypeSafeBackend() if self.backend is None else self.backend
        if not callable(getattr(backend, "open_client", None)):
            raise JevConfigurationError("backend must provide an open_client() context manager")

        return _FitBinding(
            states=states,
            questions=bound_questions,
            question_specs=question_specs,
            backend=backend,
            input_schema=input_schema,
        )

    def _questions_for_fit(self) -> Mapping[str, Question]:
        """Return the public question configuration to snapshot during fit.

        ``JevTransformer`` exposes a ``questions`` mapping directly.  Thin
        single-question estimators override this hook to assemble their mapping
        from ``question`` and ``question_id`` without adding derived constructor
        state that would violate sklearn cloning rules.
        """

        return self.questions

    def _commit_fit(self, binding: _FitBinding) -> None:
        """Atomically publish common fitted attributes after all validation."""

        self.questions_ = binding.questions
        self.question_specs_ = binding.question_specs
        self.backend_ = binding.backend
        self.input_schema_ = binding.input_schema
        if binding.input_schema.n_features is not None:
            self.n_features_in_ = binding.input_schema.n_features
        if binding.input_schema.feature_names is not None:
            self.feature_names_in_ = np.asarray(binding.input_schema.feature_names, dtype=object)
        self.is_fitted_ = True

    def _fit_common(self, X: Any, y: Any = None) -> _BaseJevEstimator:
        """Validate and bind common fitted state without remote inference."""

        binding = self._prepare_fit(X, y)
        self._commit_fit(binding)
        return self

    def _evaluate_canonical(self, X: Any) -> tuple[CanonicalResponse, ...]:
        """Execute and validate one ordered response per normalized state."""

        check_is_fitted(self, attributes="is_fitted_")
        input_schema = _describe_input_schema(X)
        _check_input_schema(self.input_schema_, input_schema)
        states: tuple[JSONContent, ...] = build_states(X, state_builder=self.state_builder)
        return self._execute_canonical_states(states)

    def _execute_canonical_states(self, states: tuple[JSONContent, ...]) -> tuple[CanonicalResponse, ...]:
        """Execute states that were already normalized by the common fit path."""

        raw_responses = execute_system_one(
            states,
            self.questions_,
            backend=self.backend_,
            model=self.model,
        )
        canonical_responses: list[CanonicalResponse] = []
        for sample_index, response in enumerate(raw_responses):
            try:
                canonical_responses.append(validate_response(response, self.question_specs_))
            except JevResponseError as error:
                raise JevResponseError(f"Invalid System One response for sample {sample_index}: {error}") from error
        return tuple(canonical_responses)

    def _reset_fitted_state(self) -> None:
        """Remove every sklearn-style fitted attribute before a refit attempt."""

        for name in tuple(vars(self)):
            if name.endswith("_") and not name.startswith("__"):
                delattr(self, name)

    def __sklearn_tags__(self) -> Any:
        """Describe supported JSON-like inputs and remote non-determinism."""

        tags = super().__sklearn_tags__()
        tags.non_deterministic = True
        tags.requires_fit = True
        tags.target_tags.required = False
        tags.input_tags.one_d_array = True
        tags.input_tags.two_d_array = True
        tags.input_tags.three_d_array = False
        tags.input_tags.sparse = False
        tags.input_tags.categorical = True
        tags.input_tags.string = True
        tags.input_tags.dict = True
        tags.input_tags.allow_nan = False
        return tags


def _describe_input_schema(X: Any) -> _InputSchema:
    if _is_dataframe(X):
        names = tuple(X.columns)
        if any(not isinstance(name, str) for name in names):
            raise JevConfigurationError("DataFrame column names must all be strings")
        if len(names) != len(set(names)):
            raise JevConfigurationError("DataFrame column names must be unique")
        return _InputSchema(kind="dataframe", n_features=len(names), feature_names=names)
    if isinstance(X, np.ndarray) and X.ndim == 2:
        return _InputSchema(kind="array2d", n_features=int(X.shape[1]))
    return _InputSchema(kind="samples")


def _check_input_schema(expected: _InputSchema, actual: _InputSchema) -> None:
    if actual.kind != expected.kind:
        raise JevConfigurationError("X has a different container schema from the data used during fit")
    if actual.n_features != expected.n_features:
        raise JevConfigurationError("X has a different number of features from the data used during fit")
    if actual.feature_names != expected.feature_names:
        raise JevConfigurationError("X has different feature names or column order from the data used during fit")


def _is_dataframe(value: Any) -> bool:
    cls = type(value)
    return cls.__name__ == "DataFrame" and cls.__module__.split(".", 1)[0] == "pandas"


__all__ = ["_BaseJevEstimator", "_FitBinding"]
