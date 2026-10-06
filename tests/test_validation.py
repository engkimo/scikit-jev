from __future__ import annotations

import math

import pytest
from typesafe_sdk import (
    Choice,
    ChoiceAnswer,
    Noul,
    NoulAnswer,
    Score,
    ScoreAnswer,
    SystemOneResponse,
    Usage,
)

from scikit_jev._results import CanonicalChoiceAnswer, CanonicalNoulAnswer, CanonicalScoreAnswer
from scikit_jev._validation import validate_questions, validate_response
from scikit_jev.exceptions import JevConfigurationError, JevResponseError


def mixed_questions() -> dict[str, dict[str, object]]:
    return {
        "spam": {"type": "noul", "instructions": "Spam?"},
        "tone": {
            "type": "choice",
            "instructions": "Tone?",
            "criteria": {"calm": None, "angry": "Hostile"},
        },
        "quality": {
            "type": "score",
            "instructions": "Quality?",
            "criteria": ["bad", {"description": "okay"}, ["great"]],
        },
    }


def valid_response() -> dict[str, object]:
    return {
        "model": "jev-2026-09-15",
        "usage": {"input_tokens": 12, "output_tokens": 3},
        "answers": {
            "quality": {
                "type": "score",
                "score": 1.7,
                "confidence": 0.8,
                "legend": {"0": "bad", "1": {"description": "okay"}, "2": ["great"]},
                "probabilities": {"2": 0.8, "0": 0.1, "1": 0.1},
            },
            "spam": {"type": "noul", "noul": 0.25},
            "tone": {
                "type": "choice",
                "choice": "calm",
                "confidence": 0.75,
                "probabilities": {"angry": 0.25, "calm": 0.75},
            },
        },
    }


def test_question_specs_preserve_declaration_order() -> None:
    specs = validate_questions(mixed_questions())

    assert tuple(spec.question_id for spec in specs) == ("spam", "tone", "quality")
    assert specs[0].kind == "noul"
    assert specs[1].labels == ("calm", "angry")
    assert specs[2].levels == (0, 1, 2)


@pytest.mark.parametrize(
    "questions, message",
    [
        ({}, "must not be empty"),
        ({"": {"type": "noul"}}, "question ID"),
        ({"q": {"type": "unknown"}}, "unsupported type"),
        ({"q": {"type": "choice", "criteria": {"only": None}}}, "between 2 and 255"),
        ({"q": {"type": "score", "criteria": ["only"]}}, "between 2 and 10"),
    ],
)
def test_invalid_question_schemas_are_rejected(questions: object, message: str) -> None:
    with pytest.raises(JevConfigurationError, match=message):
        validate_questions(questions)  # type: ignore[arg-type]


def test_response_is_canonicalized_in_question_and_declaration_order() -> None:
    canonical = validate_response(valid_response(), validate_questions(mixed_questions()))

    assert canonical.model == "jev-2026-09-15"
    assert canonical.usage.request_count == 1
    assert canonical.usage.input_tokens == 12
    assert tuple(answer.question_id for answer in canonical.answers) == ("spam", "tone", "quality")
    noul, choice, score = canonical.answers
    assert isinstance(noul, CanonicalNoulAnswer)
    assert noul.probability == 0.25
    assert isinstance(choice, CanonicalChoiceAnswer)
    assert choice.labels == ("calm", "angry")
    assert choice.probabilities == (0.75, 0.25)
    assert isinstance(score, CanonicalScoreAnswer)
    assert score.levels == (0, 1, 2)
    assert score.probabilities == (0.1, 0.1, 0.8)


def test_official_sdk_question_and_response_models_match_the_canonical_contract() -> None:
    questions = {
        "spam": Noul(instructions="Spam?"),
        "tone": Choice(instructions="Tone?", criteria={"calm": None, "angry": "Hostile"}),
        "quality": Score(instructions="Quality?", criteria=["bad", "okay", "great"]),
    }
    response = SystemOneResponse(
        model="jev-sdk-contract-test",
        usage=Usage(input_tokens=5, output_tokens=3),
        answers={
            "spam": NoulAnswer(noul=0.2),
            "tone": ChoiceAnswer(
                choice="calm",
                confidence=0.8,
                probabilities={"calm": 0.8, "angry": 0.2},
            ),
            "quality": ScoreAnswer(
                score=1.7,
                confidence=0.7,
                legend={0: "bad", 1: "okay", 2: "great"},
                probabilities={0: 0.1, 1: 0.1, 2: 0.8},
            ),
        },
    )

    canonical = validate_response(response, validate_questions(questions))

    assert tuple(answer.question_id for answer in canonical.answers) == ("spam", "tone", "quality")
    assert canonical.model == "jev-sdk-contract-test"
    assert canonical.usage.input_tokens == 5
    assert canonical.usage.output_tokens == 3


@pytest.mark.parametrize(
    "answers",
    [
        {},
        {"spam": {"type": "noul", "noul": 0.5}, "extra": {"type": "noul", "noul": 0.5}},
    ],
)
def test_answer_ids_must_match_exactly(answers: dict[str, object]) -> None:
    response = {"model": "jev", "usage": {}, "answers": answers}

    with pytest.raises(JevResponseError, match="ID mismatch"):
        validate_response(response, validate_questions({"spam": {"type": "noul"}}))


def test_answer_type_must_match_question_type() -> None:
    response = {
        "model": "jev",
        "usage": {},
        "answers": {"spam": {"type": "choice", "choice": "a", "confidence": 1.0, "probabilities": {}}},
    }

    with pytest.raises(JevResponseError, match="expected answer type"):
        validate_response(response, validate_questions({"spam": {"type": "noul"}}))


@pytest.mark.parametrize("probability", [math.nan, math.inf, -0.01, 1.01, True, "0.5"])
def test_noul_must_be_a_finite_probability(probability: object) -> None:
    response = {"model": "jev", "usage": {}, "answers": {"q": {"type": "noul", "noul": probability}}}

    with pytest.raises(JevResponseError):
        validate_response(response, validate_questions({"q": {"type": "noul"}}))


def test_small_distribution_rounding_is_normalized_and_reported() -> None:
    response = {
        "model": "jev",
        "usage": {},
        "answers": {
            "q": {
                "type": "choice",
                "choice": "a",
                "confidence": 0.5,
                "probabilities": {"a": 0.5000002, "b": 0.5000002},
            }
        },
    }
    canonical = validate_response(
        response,
        validate_questions({"q": {"type": "choice", "criteria": {"a": None, "b": None}}}),
    )

    answer = canonical.answers[0]
    assert isinstance(answer, CanonicalChoiceAnswer)
    assert sum(answer.probabilities) == pytest.approx(1.0)
    assert len(canonical.warnings) == 1


@pytest.mark.parametrize(
    "probabilities, choice, message",
    [
        ({"a": 0.9}, "a", "keys"),
        ({"a": 0.4, "b": 0.4}, "a", "sum"),
        ({"a": 0.4, "b": 0.6}, "a", "maximum"),
        ({"a": -0.1, "b": 1.1}, "b", "between"),
    ],
)
def test_invalid_choice_distributions_are_rejected(probabilities: dict[str, float], choice: str, message: str) -> None:
    response = {
        "model": "jev",
        "usage": {},
        "answers": {"q": {"type": "choice", "choice": choice, "confidence": 0.5, "probabilities": probabilities}},
    }
    specs = validate_questions({"q": {"type": "choice", "criteria": {"a": None, "b": None}}})

    with pytest.raises(JevResponseError, match=message):
        validate_response(response, specs)


def test_choice_tie_may_report_either_maximum() -> None:
    response = {
        "model": "jev",
        "usage": {},
        "answers": {
            "q": {
                "type": "choice",
                "choice": "b",
                "confidence": 0.5,
                "probabilities": {"a": 0.5, "b": 0.5},
            }
        },
    }

    validate_response(response, validate_questions({"q": {"type": "choice", "criteria": {"a": None, "b": None}}}))


def test_score_must_match_legend_and_weighted_distribution() -> None:
    specs = validate_questions({"q": {"type": "score", "criteria": ["low", "high"]}})
    response = {
        "model": "jev",
        "usage": {},
        "answers": {
            "q": {
                "type": "score",
                "score": 0.75,
                "confidence": 0.8,
                "legend": {0: "low", 1: "high"},
                "probabilities": {0: 0.25, 1: 0.75},
            }
        },
    }

    canonical = validate_response(response, specs)
    assert isinstance(canonical.answers[0], CanonicalScoreAnswer)

    response["answers"]["q"]["score"] = 0.1  # type: ignore[index]
    with pytest.raises(JevResponseError, match="weighted"):
        validate_response(response, specs)


def test_score_legend_values_must_match_declared_criteria() -> None:
    response = {
        "model": "jev",
        "usage": {},
        "answers": {
            "q": {
                "type": "score",
                "score": 0.5,
                "confidence": 0.8,
                "legend": {0: "different", 1: "high"},
                "probabilities": {0: 0.5, 1: 0.5},
            }
        },
    }

    with pytest.raises(JevResponseError, match="legend values"):
        validate_response(response, validate_questions({"q": {"type": "score", "criteria": ["low", "high"]}}))


def test_unreported_token_counts_remain_unknown() -> None:
    response = {"model": "jev", "usage": {}, "answers": {"q": {"type": "noul", "noul": 0.5}}}

    canonical = validate_response(response, validate_questions({"q": {"type": "noul"}}))

    assert canonical.usage.input_tokens is None
    assert canonical.usage.output_tokens is None
