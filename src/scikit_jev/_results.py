"""Immutable canonical answers and rich inference results."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Literal, TypeAlias

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True, slots=True)
class JevUsage:
    """Aggregated request and token usage.

    A token total is ``None`` when at least one contributing response omitted
    that count.  Unknown usage is deliberately not represented as zero.
    """

    request_count: int = 0
    input_tokens: int | None = None
    output_tokens: int | None = None

    def __post_init__(self) -> None:
        if isinstance(self.request_count, bool) or not isinstance(self.request_count, int) or self.request_count < 0:
            raise ValueError("request_count must be a non-negative integer")
        for name in ("input_tokens", "output_tokens"):
            value = getattr(self, name)
            if value is not None and (isinstance(value, bool) or not isinstance(value, int) or value < 0):
                raise ValueError(f"{name} must be a non-negative integer or None")

    @classmethod
    def combine(cls, usages: Iterable[JevUsage]) -> JevUsage:
        """Combine per-request usage without turning unknown counts into zero."""

        materialized = tuple(usages)
        if not materialized:
            return cls(request_count=0, input_tokens=0, output_tokens=0)
        return cls(
            request_count=sum(item.request_count for item in materialized),
            input_tokens=_known_sum(item.input_tokens for item in materialized),
            output_tokens=_known_sum(item.output_tokens for item in materialized),
        )


def _known_sum(values: Iterable[int | None]) -> int | None:
    materialized = tuple(values)
    if any(value is None for value in materialized):
        return None
    return sum(value for value in materialized if value is not None)


@dataclass(frozen=True, slots=True)
class CanonicalNoulAnswer:
    """A validated probability of the positive/true outcome."""

    question_id: str
    probability: float
    kind: Literal["noul"] = field(default="noul", init=False)


@dataclass(frozen=True, slots=True)
class CanonicalChoiceAnswer:
    """A validated declaration-ordered Choice distribution."""

    question_id: str
    choice: str
    confidence: float
    labels: tuple[str, ...]
    probabilities: tuple[float, ...]
    kind: Literal["choice"] = field(default="choice", init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "labels", tuple(self.labels))
        object.__setattr__(self, "probabilities", tuple(self.probabilities))


@dataclass(frozen=True, slots=True)
class CanonicalScoreAnswer:
    """A validated level-ordered Score distribution."""

    question_id: str
    score: float
    confidence: float
    levels: tuple[int, ...]
    probabilities: tuple[float, ...]
    kind: Literal["score"] = field(default="score", init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "levels", tuple(self.levels))
        object.__setattr__(self, "probabilities", tuple(self.probabilities))


CanonicalAnswer: TypeAlias = CanonicalNoulAnswer | CanonicalChoiceAnswer | CanonicalScoreAnswer


@dataclass(frozen=True, slots=True)
class CanonicalResponse:
    """One validated System One response with no retained input state."""

    model: str
    answers: tuple[CanonicalAnswer, ...]
    usage: JevUsage
    warnings: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "answers", tuple(self.answers))
        object.__setattr__(self, "warnings", tuple(self.warnings))
        question_ids = tuple(answer.question_id for answer in self.answers)
        if len(question_ids) != len(set(question_ids)):
            raise ValueError("canonical response contains duplicate question IDs")

    @property
    def resolved_model(self) -> str:
        """Alias emphasizing that the provider, not the request, chose this model."""

        return self.model

    def answer_for(self, question_id: str) -> CanonicalAnswer:
        """Return the answer for *question_id* or raise ``KeyError``."""

        for answer in self.answers:
            if answer.question_id == question_id:
                return answer
        raise KeyError(question_id)


@dataclass(frozen=True, slots=True)
class JevBatchResult:
    """Conventional numerical values plus immutable inference metadata.

    ``values`` is copied and marked read-only.  Its first dimension must match
    the number of responses.  Input states and raw HTTP/provider payloads are
    intentionally absent.
    """

    values: NDArray[np.float64]
    responses: tuple[CanonicalResponse, ...]
    resolved_models: tuple[str, ...] = field(init=False)
    usage: JevUsage = field(init=False)
    warnings: tuple[str, ...] = field(init=False)

    def __post_init__(self) -> None:
        values = np.asarray(self.values, dtype=np.float64).copy()
        if values.ndim not in (1, 2):
            raise ValueError("result values must be a one- or two-dimensional array")
        responses = tuple(self.responses)
        if values.shape[0] != len(responses):
            raise ValueError("result row count must equal response count")
        values.setflags(write=False)
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "responses", responses)
        object.__setattr__(self, "resolved_models", tuple(response.model for response in responses))
        object.__setattr__(self, "usage", JevUsage.combine([response.usage for response in responses]))
        object.__setattr__(
            self,
            "warnings",
            tuple(warning for response in responses for warning in response.warnings),
        )
