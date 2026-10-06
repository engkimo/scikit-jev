"""Compile declared System One questions into a stable numerical schema."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal, TypeAlias
from urllib.parse import quote

import numpy as np
from numpy.typing import NDArray

from ._results import CanonicalChoiceAnswer, CanonicalNoulAnswer, CanonicalResponse, CanonicalScoreAnswer
from ._validation import QuestionKind, QuestionSpec
from .exceptions import JevConfigurationError, JevResponseError

ScoreEncoding: TypeAlias = Literal["probabilities", "expected", "expected_and_std"]
FeatureComponent: TypeAlias = Literal["probability", "expected", "std", "confidence"]


@dataclass(frozen=True, slots=True)
class FeatureColumn:
    """One declaration-ordered numerical output column."""

    name: str
    question_id: str
    kind: QuestionKind
    component: FeatureComponent
    key: str | int | None = None


@dataclass(frozen=True, slots=True)
class FeatureSchema:
    """An immutable feature schema compiled without remote inference."""

    question_specs: tuple[QuestionSpec, ...]
    columns: tuple[FeatureColumn, ...]
    score_encoding: ScoreEncoding
    include_confidence: bool

    def __post_init__(self) -> None:
        object.__setattr__(self, "question_specs", tuple(self.question_specs))
        object.__setattr__(self, "columns", tuple(self.columns))

    @property
    def n_features(self) -> int:
        return len(self.columns)

    @property
    def feature_names(self) -> tuple[str, ...]:
        return tuple(column.name for column in self.columns)

    def get_feature_names_out(self, input_features: object = None) -> NDArray[np.object_]:
        """Return a new object array in the exact compiled column order."""

        del input_features
        return np.asarray(self.feature_names, dtype=object)

    def transform(self, responses: Sequence[CanonicalResponse]) -> NDArray[np.float64]:
        """Project canonical responses into a ``float64`` feature matrix."""

        responses = tuple(responses)
        if not responses:
            return np.empty((0, self.n_features), dtype=np.float64)
        rows = [self.encode_response(response) for response in responses]
        return np.asarray(rows, dtype=np.float64)

    def encode_response(self, response: CanonicalResponse) -> tuple[float, ...]:
        """Encode one canonical response, rechecking its bound schema."""

        values: list[float] = []
        for spec in self.question_specs:
            try:
                answer = response.answer_for(spec.question_id)
            except KeyError:
                raise JevResponseError(f"canonical response is missing question {spec.question_id!r}") from None

            if spec.kind == "noul":
                if not isinstance(answer, CanonicalNoulAnswer):
                    raise _canonical_mismatch(spec)
                values.append(answer.probability)
                continue

            if spec.kind == "choice":
                if not isinstance(answer, CanonicalChoiceAnswer) or answer.labels != spec.labels:
                    raise _canonical_mismatch(spec)
                values.extend(answer.probabilities)
                if self.include_confidence:
                    values.append(answer.confidence)
                continue

            if not isinstance(answer, CanonicalScoreAnswer) or answer.levels != spec.levels:
                raise _canonical_mismatch(spec)
            if self.score_encoding == "probabilities":
                values.extend(answer.probabilities)
            else:
                expected = math.fsum(
                    level * probability for level, probability in zip(answer.levels, answer.probabilities, strict=True)
                )
                values.append(expected)
                if self.score_encoding == "expected_and_std":
                    variance = math.fsum(
                        probability * (level - expected) ** 2
                        for level, probability in zip(answer.levels, answer.probabilities, strict=True)
                    )
                    values.append(math.sqrt(max(0.0, variance)))
            if self.include_confidence:
                values.append(answer.confidence)

        if len(values) != self.n_features:
            raise JevResponseError("canonical response does not match the compiled feature count")
        return tuple(values)


def compile_feature_schema(
    question_specs: Sequence[QuestionSpec],
    *,
    score_encoding: ScoreEncoding = "probabilities",
    include_confidence: bool = False,
) -> FeatureSchema:
    """Compile question order and encodings into stable feature columns."""

    specs = tuple(question_specs)
    if not specs:
        raise JevConfigurationError("at least one question is required to compile a feature schema")
    if score_encoding not in ("probabilities", "expected", "expected_and_std"):
        raise JevConfigurationError("score_encoding must be 'probabilities', 'expected', or 'expected_and_std'")
    if not isinstance(include_confidence, bool):
        raise JevConfigurationError("include_confidence must be a boolean")

    columns: list[FeatureColumn] = []
    seen_question_ids: set[str] = set()
    for spec in specs:
        if spec.kind not in ("noul", "choice", "score"):
            raise JevConfigurationError(f"question {spec.question_id!r} has an unsupported type")
        if spec.question_id in seen_question_ids:
            raise JevConfigurationError(f"duplicate question ID {spec.question_id!r}")
        seen_question_ids.add(spec.question_id)
        question_token = escape_feature_token(spec.question_id)

        if spec.kind == "noul":
            columns.append(
                FeatureColumn(
                    name=f"{question_token}__p_true",
                    question_id=spec.question_id,
                    kind=spec.kind,
                    component="probability",
                    key="true",
                )
            )
            continue

        if spec.kind == "choice":
            if not spec.labels:
                raise JevConfigurationError(f"Choice question {spec.question_id!r} has no bound labels")
            for label in spec.labels:
                columns.append(
                    FeatureColumn(
                        name=f"{question_token}__p[{escape_feature_token(label)}]",
                        question_id=spec.question_id,
                        kind=spec.kind,
                        component="probability",
                        key=label,
                    )
                )
            if include_confidence:
                columns.append(
                    FeatureColumn(
                        name=f"{question_token}__confidence",
                        question_id=spec.question_id,
                        kind=spec.kind,
                        component="confidence",
                    )
                )
            continue

        if not spec.levels:
            raise JevConfigurationError(f"Score question {spec.question_id!r} has no bound levels")
        if score_encoding == "probabilities":
            for level in spec.levels:
                columns.append(
                    FeatureColumn(
                        name=f"{question_token}__p[{level}]",
                        question_id=spec.question_id,
                        kind=spec.kind,
                        component="probability",
                        key=level,
                    )
                )
        else:
            columns.append(
                FeatureColumn(
                    name=f"{question_token}__expected",
                    question_id=spec.question_id,
                    kind=spec.kind,
                    component="expected",
                )
            )
            if score_encoding == "expected_and_std":
                columns.append(
                    FeatureColumn(
                        name=f"{question_token}__std",
                        question_id=spec.question_id,
                        kind=spec.kind,
                        component="std",
                    )
                )
        if include_confidence:
            columns.append(
                FeatureColumn(
                    name=f"{question_token}__confidence",
                    question_id=spec.question_id,
                    kind=spec.kind,
                    component="confidence",
                )
            )

    names = tuple(column.name for column in columns)
    if len(names) != len(set(names)):
        # Percent escaping is injective, so this indicates malformed hand-built
        # QuestionSpec instances rather than an ordinary user label collision.
        raise JevConfigurationError("compiled feature names are not unique")
    return FeatureSchema(
        question_specs=specs,
        columns=tuple(columns),
        score_encoding=score_encoding,
        include_confidence=include_confidence,
    )


def escape_feature_token(value: str) -> str:
    """Percent-encode a question ID or label for reversible feature names."""

    if not isinstance(value, str):
        raise JevConfigurationError("feature-name tokens must be strings")
    return quote(value, safe="-._~")


def _canonical_mismatch(spec: QuestionSpec) -> JevResponseError:
    return JevResponseError(
        f"canonical answer for question {spec.question_id!r} does not match its compiled {spec.kind} schema"
    )
