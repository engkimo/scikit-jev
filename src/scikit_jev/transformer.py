"""scikit-learn transformer for typed System One semantic features."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from numpy.typing import NDArray
from sklearn.base import TransformerMixin
from sklearn.utils.validation import check_is_fitted
from typesafe_sdk import Question

from scikit_jev._base import _BaseJevEstimator, _FitBinding
from scikit_jev._features import ScoreEncoding, compile_feature_schema
from scikit_jev._results import JevBatchResult
from scikit_jev._states import StateBuilder
from scikit_jev.backends import SystemOneBackend


class JevTransformer(TransformerMixin, _BaseJevEstimator):
    """Compile typed System One answers into stable numerical features.

    ``fit`` binds the question and input schemas without making a remote call.
    ``transform`` and ``evaluate`` make exactly one System One request per input
    sample, with every declared question packed into that request.
    """

    def __init__(
        self,
        questions: Mapping[str, Question],
        *,
        model: str | None = None,
        backend: SystemOneBackend | None = None,
        state_builder: StateBuilder = "auto",
        score_encoding: ScoreEncoding = "probabilities",
        include_confidence: bool = False,
    ) -> None:
        self.questions = questions
        self.model = model
        self.backend = backend
        self.state_builder = state_builder
        self.score_encoding = score_encoding
        self.include_confidence = include_confidence

    def fit(self, X: Any, y: Any = None) -> JevTransformer:
        """Bind input and output schemas without contacting System One."""

        binding = self._prepare_fit(X, y)
        self._commit_transformer_fit(binding)
        return self

    def fit_transform(self, X: Any, y: Any = None, **fit_params: Any) -> NDArray[np.float64]:
        """Bind schemas and transform while normalizing each sample only once."""

        if fit_params:
            names = ", ".join(sorted(fit_params))
            raise TypeError(f"JevTransformer.fit_transform() got unexpected fit parameters: {names}")
        binding = self._prepare_fit(X, y)
        self._commit_transformer_fit(binding)
        canonical_responses = self._execute_canonical_states(binding.states)
        return self.feature_schema_.transform(canonical_responses)

    def _commit_transformer_fit(self, binding: _FitBinding) -> None:
        feature_schema = compile_feature_schema(
            binding.question_specs,
            score_encoding=self.score_encoding,
            include_confidence=self.include_confidence,
        )
        self._commit_fit(binding)
        self.feature_schema_ = feature_schema

    def evaluate(self, X: Any) -> JevBatchResult:
        """Return numerical values and immutable metadata from one inference."""

        canonical_responses = self._evaluate_canonical(X)
        values = self.feature_schema_.transform(canonical_responses)
        return JevBatchResult(values=values, responses=canonical_responses)

    def transform(self, X: Any) -> NDArray[np.float64]:
        """Return declaration-ordered ``float64`` semantic features."""

        return self.evaluate(X).values

    def get_feature_names_out(self, input_features: Sequence[str] | None = None) -> NDArray[np.object_]:
        """Return stable feature names compiled from the fitted questions."""

        check_is_fitted(self, attributes="feature_schema_")
        if input_features is not None and hasattr(self, "n_features_in_"):
            provided = np.asarray(input_features, dtype=object)
            if provided.ndim != 1 or len(provided) != self.n_features_in_:
                raise ValueError("input_features must match the fitted input feature count")
            if hasattr(self, "feature_names_in_") and not np.array_equal(provided, self.feature_names_in_):
                raise ValueError("input_features must match feature_names_in_ exactly")
        return self.feature_schema_.get_feature_names_out()


__all__ = ["JevTransformer"]
