"""Zero-shot scikit-learn classifiers backed by System One questions."""

from __future__ import annotations

import math
from collections.abc import Mapping
from numbers import Real
from typing import Any

import numpy as np
from numpy.typing import NDArray
from sklearn.base import ClassifierMixin
from typesafe_sdk import Question

from scikit_jev._base import _BaseJevEstimator, _FitBinding
from scikit_jev._results import CanonicalChoiceAnswer, CanonicalNoulAnswer, JevBatchResult
from scikit_jev._states import StateBuilder
from scikit_jev.backends import SystemOneBackend
from scikit_jev.exceptions import JevConfigurationError


class JevNoulClassifier(ClassifierMixin, _BaseJevEstimator):
    """Binary classifier projecting a Noul probability without information loss."""

    def __init__(
        self,
        question: Question,
        *,
        question_id: str = "target",
        model: str | None = None,
        backend: SystemOneBackend | None = None,
        threshold: float = 0.5,
        negative_label: Any = False,
        positive_label: Any = True,
        state_builder: StateBuilder = "auto",
    ) -> None:
        self.question = question
        self.question_id = question_id
        self.model = model
        self.backend = backend
        self.threshold = threshold
        self.negative_label = negative_label
        self.positive_label = positive_label
        self.state_builder = state_builder

    def _questions_for_fit(self) -> Mapping[str, Question]:
        return {self.question_id: self.question}

    def fit(self, X: Any, y: Any = None) -> JevNoulClassifier:
        """Bind a Noul schema and optional target domain without inference."""

        binding = self._prepare_fit(X, y)
        _require_question_kind(binding, "noul", "JevNoulClassifier")
        threshold = _validate_threshold(self.threshold)
        classes = _validate_binary_labels(self.negative_label, self.positive_label)
        _validate_classification_target(y, tuple(classes))
        self._commit_fit(binding)
        self.threshold_ = threshold
        self.classes_ = classes
        return self

    def evaluate(self, X: Any) -> JevBatchResult:
        """Return ``[p_false, p_true]`` values and metadata from one inference."""

        responses = self._evaluate_canonical(X)
        positive = np.fromiter(
            (_noul_answer(response.answer_for(self.question_id)).probability for response in responses),
            dtype=np.float64,
            count=len(responses),
        )
        values = np.empty((len(responses), 2), dtype=np.float64)
        values[:, 0] = 1.0 - positive
        values[:, 1] = positive
        return JevBatchResult(values=values, responses=responses)

    def predict_proba(self, X: Any) -> NDArray[np.float64]:
        """Return negative and positive probabilities in ``classes_`` order."""

        return self.evaluate(X).values

    def predict(self, X: Any) -> NDArray[Any]:
        """Return the positive label when ``p_true >= threshold``."""

        probabilities = self.evaluate(X).values
        indices = (probabilities[:, 1] >= self.threshold_).astype(np.intp, copy=False)
        return self.classes_[indices]

    def __sklearn_tags__(self) -> Any:
        tags = super().__sklearn_tags__()
        tags.target_tags.required = False
        tags.classifier_tags.poor_score = True
        tags.classifier_tags.multi_class = False
        return tags


class JevChoiceClassifier(ClassifierMixin, _BaseJevEstimator):
    """Multiclass classifier preserving a Choice question's declared order."""

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

    def fit(self, X: Any, y: Any = None) -> JevChoiceClassifier:
        """Bind the Choice class domain without inference."""

        binding = self._prepare_fit(X, y)
        spec = _require_question_kind(binding, "choice", "JevChoiceClassifier")
        classes = np.asarray(spec.labels)
        _validate_classification_target(y, spec.labels)
        self._commit_fit(binding)
        self.classes_ = classes
        return self

    def evaluate(self, X: Any) -> JevBatchResult:
        """Return declared-order class probabilities and inference metadata."""

        responses = self._evaluate_canonical(X)
        values = np.empty((len(responses), len(self.classes_)), dtype=np.float64)
        for row_index, response in enumerate(responses):
            answer = _choice_answer(response.answer_for(self.question_id))
            values[row_index] = answer.probabilities
        return JevBatchResult(values=values, responses=responses)

    def predict_proba(self, X: Any) -> NDArray[np.float64]:
        """Return full Choice distributions in ``classes_`` order."""

        return self.evaluate(X).values

    def predict(self, X: Any) -> NDArray[Any]:
        """Return the declaration-first argmax class for every sample."""

        probabilities = self.evaluate(X).values
        indices = np.argmax(probabilities, axis=1)
        return self.classes_[indices]

    def __sklearn_tags__(self) -> Any:
        tags = super().__sklearn_tags__()
        tags.target_tags.required = False
        tags.classifier_tags.poor_score = True
        return tags


def _require_question_kind(binding: _FitBinding, expected: str, estimator_name: str) -> Any:
    if len(binding.question_specs) != 1 or binding.question_specs[0].kind != expected:
        raise JevConfigurationError(f"{estimator_name} requires a {expected.title()} question")
    return binding.question_specs[0]


def _validate_threshold(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise JevConfigurationError("threshold must be a finite number between zero and one")
    threshold = float(value)
    if not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
        raise JevConfigurationError("threshold must be a finite number between zero and one")
    return threshold


def _validate_binary_labels(negative_label: Any, positive_label: Any) -> NDArray[Any]:
    for label in (negative_label, positive_label):
        try:
            hash(label)
        except (TypeError, ValueError):
            raise JevConfigurationError("class labels must be hashable scalar values") from None
    if _labels_equal(negative_label, positive_label):
        raise JevConfigurationError("negative_label and positive_label must be distinct")
    if type(negative_label) is type(positive_label):
        classes = np.asarray((negative_label, positive_label))
        if classes.ndim == 1:
            return classes
    classes = np.empty(2, dtype=object)
    classes[0] = negative_label
    classes[1] = positive_label
    return classes


def _validate_classification_target(y: Any, classes: tuple[Any, ...]) -> None:
    if y is None:
        return
    target = np.asarray(y, dtype=object)
    if target.ndim != 1:
        raise JevConfigurationError("y must be a one-dimensional target sequence")
    for label in target:
        if not any(_labels_equal(label, expected) for expected in classes):
            raise JevConfigurationError("y contains a label outside the declared class domain")


def _labels_equal(left: Any, right: Any) -> bool:
    try:
        equal = left == right
    except Exception:
        raise JevConfigurationError("class labels must be comparable scalar values") from None
    if not isinstance(equal, (bool, np.bool_)):
        raise JevConfigurationError("class labels must be comparable scalar values")
    return bool(equal)


def _noul_answer(answer: Any) -> CanonicalNoulAnswer:
    if not isinstance(answer, CanonicalNoulAnswer):
        raise RuntimeError("validated Noul response contained an unexpected canonical answer")
    return answer


def _choice_answer(answer: Any) -> CanonicalChoiceAnswer:
    if not isinstance(answer, CanonicalChoiceAnswer):
        raise RuntimeError("validated Choice response contained an unexpected canonical answer")
    return answer


__all__ = ["JevChoiceClassifier", "JevNoulClassifier"]
