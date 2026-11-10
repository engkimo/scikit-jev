"""Question-schema and System One response validation."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from numbers import Integral, Real
from typing import TYPE_CHECKING, Any, Literal, TypeAlias

from typesafe_sdk import TypeSafeError

from ._results import (
    CanonicalAnswer,
    CanonicalChoiceAnswer,
    CanonicalNoulAnswer,
    CanonicalResponse,
    CanonicalScoreAnswer,
    JevUsage,
)
from ._states import _normalize_json_content
from .exceptions import JevConfigurationError, JevResponseError

if TYPE_CHECKING:
    from typesafe_sdk import Question, SystemOneResponse
else:
    Question = Any
    SystemOneResponse = Any

QuestionKind: TypeAlias = Literal["noul", "choice", "score"]

# A provider distribution may be rounded, but only small floating-point error
# is repaired.  Materially invalid probabilities are never silently rescaled.
PROBABILITY_SUM_TOLERANCE = 1e-6
SCALAR_AGREEMENT_TOLERANCE = 1e-6
_NORMALIZATION_NOTICE_TOLERANCE = 1e-12
_MISSING = object()


@dataclass(frozen=True, slots=True)
class QuestionSpec:
    """The immutable part of a question needed after request execution."""

    question_id: str
    kind: QuestionKind
    labels: tuple[str, ...] = ()
    levels: tuple[int, ...] = ()
    criterion_fingerprints: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "labels", tuple(self.labels))
        object.__setattr__(self, "levels", tuple(self.levels))
        object.__setattr__(self, "criterion_fingerprints", tuple(self.criterion_fingerprints))

    @property
    def answer_type(self) -> QuestionKind:
        """Semantic alias used by response-validation call sites."""

        return self.kind


def validate_questions(questions: Mapping[str, Question]) -> tuple[QuestionSpec, ...]:
    """Validate supported question shapes and bind declaration order.

    The returned specs contain no client object and no mutable criteria.  The
    original mapping remains the request payload passed to the official SDK.
    """

    if not isinstance(questions, Mapping):
        raise JevConfigurationError("questions must be a non-empty mapping")
    if not questions:
        raise JevConfigurationError("questions must not be empty")

    specs: list[QuestionSpec] = []
    for question_id, question in questions.items():
        if not isinstance(question_id, str) or not question_id.strip():
            raise JevConfigurationError("every question ID must be a non-empty string")
        kind = _question_field(question, "type")
        if kind not in ("noul", "choice", "score"):
            raise JevConfigurationError(f"question {question_id!r} has an unsupported type")
        _validate_optional_instructions(question_id, question)
        if kind == "noul":
            _validate_noul_question(question_id, question)
            specs.append(QuestionSpec(question_id=question_id, kind="noul"))
        elif kind == "choice":
            labels = _validate_choice_question(question_id, question)
            specs.append(QuestionSpec(question_id=question_id, kind="choice", labels=labels))
        else:
            fingerprints = _validate_score_question(question_id, question)
            levels = tuple(range(len(fingerprints)))
            specs.append(
                QuestionSpec(
                    question_id=question_id,
                    kind="score",
                    levels=levels,
                    criterion_fingerprints=fingerprints,
                )
            )
    return tuple(specs)


def validate_response(
    response: SystemOneResponse,
    specs: Sequence[QuestionSpec],
    *,
    probability_sum_tolerance: float = PROBABILITY_SUM_TOLERANCE,
    scalar_tolerance: float = SCALAR_AGREEMENT_TOLERANCE,
) -> CanonicalResponse:
    """Validate and declaration-order one System One response.

    Parameters controlling tolerances are internal integration seams, not
    estimator hyperparameters.  They must be finite and non-negative.
    """

    probability_sum_tolerance = _validate_tolerance(probability_sum_tolerance, "probability_sum_tolerance")
    scalar_tolerance = _validate_tolerance(scalar_tolerance, "scalar_tolerance")
    specs = tuple(specs)
    expected_ids = tuple(spec.question_id for spec in specs)
    if len(expected_ids) != len(set(expected_ids)):
        raise JevConfigurationError("question specs contain duplicate IDs")

    _validate_raw_answer_ids(response, expected_ids)
    answers = _response_field(response, "answers")
    _validate_answer_ids(answers, expected_ids)

    model = _response_field(response, "model")
    if not isinstance(model, str) or not model:
        raise JevResponseError("System One response model must be a non-empty string")
    usage = _validate_usage(_response_field(response, "usage"))

    canonical_answers: list[CanonicalAnswer] = []
    warnings: list[str] = []
    for spec in specs:
        answer = answers[spec.question_id]
        answer_kind = _response_field(answer, "type")
        if answer_kind != spec.kind:
            raise _answer_error(spec, f"expected answer type {spec.kind!r}")
        if spec.kind == "noul":
            canonical_answers.append(_validate_noul_answer(spec, answer))
        elif spec.kind == "choice":
            choice_answer, warning = _validate_choice_answer(
                spec,
                answer,
                probability_sum_tolerance=probability_sum_tolerance,
                scalar_tolerance=scalar_tolerance,
            )
            canonical_answers.append(choice_answer)
            if warning is not None:
                warnings.append(warning)
        else:
            score_answer, warning = _validate_score_answer(
                spec,
                answer,
                probability_sum_tolerance=probability_sum_tolerance,
                scalar_tolerance=scalar_tolerance,
            )
            canonical_answers.append(score_answer)
            if warning is not None:
                warnings.append(warning)

    return CanonicalResponse(model=model, answers=tuple(canonical_answers), usage=usage, warnings=tuple(warnings))


def _validate_answer_ids(answers: Any, expected_ids: tuple[str, ...]) -> None:
    """Check exact IDs without echoing any provider-controlled keys."""

    if not isinstance(answers, Mapping):
        raise JevResponseError("System One response answers must be a mapping")
    returned_ids = set(answers)
    expected_id_set = set(expected_ids)
    if returned_ids != expected_id_set:
        missing = tuple(question_id for question_id in expected_ids if question_id not in returned_ids)
        extra_count = len(returned_ids - expected_id_set)
        details: list[str] = []
        if missing:
            details.append(f"missing question IDs {missing!r}")
        if extra_count:
            details.append(f"{extra_count} unexpected question ID(s)")
        raise JevResponseError("System One response ID mismatch: " + "; ".join(details))


def _validate_raw_answer_ids(response: Any, expected_ids: tuple[str, ...]) -> None:
    """Catch answers the SDK drops for forward compatibility, when raw data exists.

    Directly constructed SDK models and structural clients need no HTTP object.
    Raw payloads are inspected transiently and never copied to canonical results.
    """

    try:
        raw = getattr(response, "raw_http_response", None)
    except TypeSafeError:
        # Official SDK models constructed in memory have no originating HTTP
        # response and explicitly raise TypeSafeError from this property.
        return
    if raw is None:
        return
    failed = False
    payload: Any = None
    try:
        payload = raw.json()
    except Exception:
        failed = True
    if failed:
        # Raise after leaving the handler so decoder diagnostics are not retained.
        raise JevResponseError("System One raw response could not be inspected")
    if not isinstance(payload, Mapping):
        raise JevResponseError("System One raw response must be a mapping")
    _validate_answer_ids(payload.get("answers"), expected_ids)


def _validate_optional_instructions(question_id: str, question: Any) -> None:
    instructions = _question_field(question, "instructions", default=None)
    if instructions is not None:
        _normalize_json_content(instructions, context=f"instructions for question {question_id!r}")


def _validate_noul_question(question_id: str, question: Any) -> None:
    criteria = _question_field(question, "criteria", default=None)
    if criteria is None:
        return
    if not isinstance(criteria, Mapping):
        raise JevConfigurationError(f"Noul question {question_id!r} criteria must be a mapping")
    if any(key not in ("true", "false") for key in criteria):
        raise JevConfigurationError(f"Noul question {question_id!r} criteria has an unsupported key")
    for description in criteria.values():
        if description is not None:
            _normalize_json_content(description, context=f"criteria for Noul question {question_id!r}")


def _validate_choice_question(question_id: str, question: Any) -> tuple[str, ...]:
    criteria = _question_field(question, "criteria")
    if not isinstance(criteria, Mapping):
        raise JevConfigurationError(f"Choice question {question_id!r} criteria must be a mapping")
    if not 2 <= len(criteria) <= 255:
        raise JevConfigurationError(f"Choice question {question_id!r} must declare between 2 and 255 alternatives")
    labels = tuple(criteria)
    if any(not isinstance(label, str) or not label for label in labels):
        raise JevConfigurationError(f"Choice question {question_id!r} labels must be non-empty strings")
    for description in criteria.values():
        if description is not None:
            _normalize_json_content(description, context=f"criteria for Choice question {question_id!r}")
    return labels


def _validate_score_question(question_id: str, question: Any) -> tuple[str, ...]:
    criteria = _question_field(question, "criteria")
    if isinstance(criteria, (str, bytes, bytearray)) or not isinstance(criteria, Sequence):
        raise JevConfigurationError(f"Score question {question_id!r} criteria must be an ordered sequence")
    if not 2 <= len(criteria) <= 10:
        raise JevConfigurationError(f"Score question {question_id!r} must declare between 2 and 10 levels")
    fingerprints: list[str] = []
    for description in criteria:
        normalized = _normalize_json_content(description, context=f"criteria for Score question {question_id!r}")
        fingerprints.append(_json_fingerprint(normalized))
    return tuple(fingerprints)


def _validate_noul_answer(spec: QuestionSpec, answer: Any) -> CanonicalNoulAnswer:
    probability = _finite_probability(_response_field(answer, "noul"), spec, "noul")
    return CanonicalNoulAnswer(question_id=spec.question_id, probability=probability)


def _validate_choice_answer(
    spec: QuestionSpec,
    answer: Any,
    *,
    probability_sum_tolerance: float,
    scalar_tolerance: float,
) -> tuple[CanonicalChoiceAnswer, str | None]:
    choice = _response_field(answer, "choice")
    if not isinstance(choice, str) or choice not in spec.labels:
        raise _answer_error(spec, "reported choice is not one of the declared labels")
    confidence = _finite_probability(_response_field(answer, "confidence"), spec, "confidence")
    probabilities, warning = _validate_distribution(
        _response_field(answer, "probabilities"),
        expected_keys=spec.labels,
        spec=spec,
        probability_sum_tolerance=probability_sum_tolerance,
    )
    selected_probability = probabilities[spec.labels.index(choice)]
    if selected_probability < max(probabilities) - scalar_tolerance:
        raise _answer_error(spec, "reported choice does not agree with the maximum probability")
    return (
        CanonicalChoiceAnswer(
            question_id=spec.question_id,
            choice=choice,
            confidence=confidence,
            labels=spec.labels,
            probabilities=probabilities,
        ),
        warning,
    )


def _validate_score_answer(
    spec: QuestionSpec,
    answer: Any,
    *,
    probability_sum_tolerance: float,
    scalar_tolerance: float,
) -> tuple[CanonicalScoreAnswer, str | None]:
    score = _finite_number(_response_field(answer, "score"), spec, "score")
    confidence = _finite_probability(_response_field(answer, "confidence"), spec, "confidence")
    legend = _normalize_level_mapping(_response_field(answer, "legend"), spec, "legend")
    if tuple(legend) != spec.levels:
        raise _answer_error(spec, "legend keys do not exactly match the declared score levels")
    fingerprints = tuple(_json_fingerprint(_normalize_legend_value(value, spec)) for value in legend.values())
    if fingerprints != spec.criterion_fingerprints:
        raise _answer_error(spec, "legend values do not match the declared score criteria")
    raw_probabilities = _normalize_level_mapping(_response_field(answer, "probabilities"), spec, "probabilities")
    probabilities, warning = _validate_distribution(
        raw_probabilities,
        expected_keys=spec.levels,
        spec=spec,
        probability_sum_tolerance=probability_sum_tolerance,
    )
    expected_score = sum(level * probability for level, probability in zip(spec.levels, probabilities, strict=True))
    if not math.isclose(score, expected_score, rel_tol=0.0, abs_tol=scalar_tolerance):
        raise _answer_error(spec, "reported score does not agree with the probability-weighted level")
    return (
        CanonicalScoreAnswer(
            question_id=spec.question_id,
            score=score,
            confidence=confidence,
            levels=spec.levels,
            probabilities=probabilities,
        ),
        warning,
    )


def _validate_distribution(
    raw_probabilities: Any,
    *,
    expected_keys: tuple[Any, ...],
    spec: QuestionSpec,
    probability_sum_tolerance: float,
) -> tuple[tuple[float, ...], str | None]:
    if not isinstance(raw_probabilities, Mapping):
        raise _answer_error(spec, "probabilities must be a mapping")
    if set(raw_probabilities) != set(expected_keys):
        raise _answer_error(spec, "probability keys do not exactly match the declared alternatives")
    probabilities = tuple(_finite_probability(raw_probabilities[key], spec, "probability") for key in expected_keys)
    total = math.fsum(probabilities)
    if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=probability_sum_tolerance):
        raise _answer_error(spec, "probabilities do not sum to one")
    if total <= 0.0:
        raise _answer_error(spec, "probabilities have a non-positive sum")
    if math.isclose(total, 1.0, rel_tol=0.0, abs_tol=_NORMALIZATION_NOTICE_TOLERANCE):
        return probabilities, None
    normalized = tuple(probability / total for probability in probabilities)
    return normalized, f"Normalized rounded probabilities for question {spec.question_id!r}."


def _normalize_level_mapping(value: Any, spec: QuestionSpec, field_name: str) -> dict[int, Any]:
    if not isinstance(value, Mapping):
        raise _answer_error(spec, f"{field_name} must be a mapping")
    normalized: dict[int, Any] = {}
    for key, item in value.items():
        if isinstance(key, bool):
            raise _answer_error(spec, f"{field_name} contains an invalid score-level key")
        if isinstance(key, Integral):
            level = int(key)
        elif isinstance(key, str) and key.isdigit() and (key == "0" or not key.startswith("0")):
            level = int(key)
        else:
            raise _answer_error(spec, f"{field_name} contains an invalid score-level key")
        if level in normalized:
            raise _answer_error(spec, f"{field_name} contains duplicate score-level keys")
        normalized[level] = item
    return normalized


def _normalize_legend_value(value: Any, spec: QuestionSpec) -> Any:
    try:
        return _normalize_json_content(value, context=f"legend for Score question {spec.question_id!r}")
    except JevConfigurationError:
        raise _answer_error(spec, "legend contains a value that is not JSON-compatible") from None


def _validate_usage(usage: Any) -> JevUsage:
    if usage is _MISSING or usage is None:
        raise JevResponseError("System One response usage is missing")
    input_tokens = _token_count(_response_field(usage, "input_tokens", default=None), "input_tokens")
    output_tokens = _token_count(_response_field(usage, "output_tokens", default=None), "output_tokens")
    return JevUsage(request_count=1, input_tokens=input_tokens, output_tokens=output_tokens)


def _token_count(value: Any, name: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
        raise JevResponseError(f"System One response {name} must be a non-negative integer or None")
    return int(value)


def _finite_probability(value: Any, spec: QuestionSpec, field_name: str) -> float:
    number = _finite_number(value, spec, field_name)
    if not 0.0 <= number <= 1.0:
        raise _answer_error(spec, f"{field_name} must be between zero and one")
    return number


def _finite_number(value: Any, spec: QuestionSpec, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise _answer_error(spec, f"{field_name} must be a finite number")
    number = float(value)
    if not math.isfinite(number):
        raise _answer_error(spec, f"{field_name} must be a finite number")
    return number


def _validate_tolerance(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise JevConfigurationError(f"{name} must be a finite non-negative number")
    value = float(value)
    if not math.isfinite(value) or value < 0.0:
        raise JevConfigurationError(f"{name} must be a finite non-negative number")
    return value


def _question_field(question: Any, name: str, *, default: Any = _MISSING) -> Any:
    if isinstance(question, Mapping):
        value = question.get(name, _MISSING)
    else:
        value = getattr(question, name, _MISSING)
    if value is _MISSING:
        if default is not _MISSING:
            return default
        raise JevConfigurationError(f"question is missing required field {name!r}")
    return value


def _response_field(value: Any, name: str, *, default: Any = _MISSING) -> Any:
    if isinstance(value, Mapping):
        result = value.get(name, _MISSING)
    else:
        result = getattr(value, name, _MISSING)
    if result is _MISSING:
        if default is not _MISSING:
            return default
        raise JevResponseError(f"System One response is missing required field {name!r}")
    return result


def _json_fingerprint(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":"))


def _answer_error(spec: QuestionSpec, message: str) -> JevResponseError:
    return JevResponseError(f"invalid {spec.kind} answer for question {spec.question_id!r}: {message}")
