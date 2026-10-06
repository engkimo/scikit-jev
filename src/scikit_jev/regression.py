"""Zero-shot scikit-learn regression backed by a System One Score question."""

from __future__ import annotations

import math
from collections.abc import Mapping
from numbers import Real
from typing import Any

import numpy as np
from numpy.typing import NDArray
from sklearn.base import RegressorMixin
from typesafe_sdk import Question

from scikit_jev._base import _BaseJevEstimator, _FitBinding
from scikit_jev._results import CanonicalScoreAnswer, JevBatchResult
from scikit_jev._states import StateBuilder
from scikit_jev.backends import SystemOneBackend
from scikit_jev.exceptions import JevConfigurationError


class JevScoreRegressor(RegressorMixin, _BaseJevEstimator):
    """Regressor returning a Score question's probability-weighted level."""

    def __init__(
        self,
        question: Question,
        *,
        question_id: str = "target",
        model: str | None = None,
        backend: SystemOneBackend | None = None,
        state_builder: StateBuilder = "auto",
    ) -> None:
        self.question = question
        self.question_id = question_id
        self.model = model
        self.backend = backend
        self.state_builder = state_builder

    def _questions_for_fit(self) -> Mapping[str, Question]:
        return {self.question_id: self.question}

    def fit(self, X: Any, y: Any = None) -> JevScoreRegressor:
        """Bind a Score rubric and optional numerical targets without inference."""

        binding = self._prepare_fit(X, y)
        spec = _require_score_question(binding)
        _validate_regression_target(y, spec.levels)
        self._commit_fit(binding)
        return self

    def evaluate(self, X: Any) -> JevBatchResult:
        """Return expected scores and metadata from one inference."""

        responses = self._evaluate_canonical(X)
        values = np.fromiter(
            (_score_answer(response.answer_for(self.question_id)).score for response in responses),
            dtype=np.float64,
            count=len(responses),
        )
        return JevBatchResult(values=values, responses=responses)

    def predict(self, X: Any) -> NDArray[np.float64]:
        """Return validated probability-weighted Score levels."""

        return self.evaluate(X).values

    def __sklearn_tags__(self) -> Any:
        tags = super().__sklearn_tags__()
        tags.target_tags.required = False
        tags.regressor_tags.poor_score = True
        return tags


def _require_score_question(binding: _FitBinding) -> Any:
    if len(binding.question_specs) != 1 or binding.question_specs[0].kind != "score":
        raise JevConfigurationError("JevScoreRegressor requires a Score question")
    return binding.question_specs[0]


def _validate_regression_target(y: Any, levels: tuple[int, ...]) -> None:
    if y is None:
        return
    try:
        target = np.asarray(y, dtype=np.float64)
    except (TypeError, ValueError):
        raise JevConfigurationError("y must be a one-dimensional finite numerical target sequence") from None
    if target.ndim != 1 or not np.all(np.isfinite(target)):
        raise JevConfigurationError("y must be a one-dimensional finite numerical target sequence")
    lower = float(levels[0])
    upper = float(levels[-1])
    if any(not isinstance(value, Real) or not math.isfinite(float(value)) for value in target):
        raise JevConfigurationError("y must be a one-dimensional finite numerical target sequence")
    if np.any(target < lower) or np.any(target > upper):
        raise JevConfigurationError("y contains a value outside the declared Score range")


def _score_answer(answer: Any) -> CanonicalScoreAnswer:
    if not isinstance(answer, CanonicalScoreAnswer):
        raise RuntimeError("validated Score response contained an unexpected canonical answer")
    return answer


__all__ = ["JevScoreRegressor"]
