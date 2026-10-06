from __future__ import annotations

import math

import numpy as np
import pytest

from scikit_jev._features import compile_feature_schema, escape_feature_token
from scikit_jev._results import JevBatchResult
from scikit_jev._validation import validate_questions, validate_response
from scikit_jev.exceptions import JevConfigurationError


def _specs_and_response():
    questions = {
        "is spam": {"type": "noul"},
        "tone/id": {"type": "choice", "criteria": {"calm": None, "very angry": None}},
        "quality": {"type": "score", "criteria": ["bad", "okay", "great"]},
    }
    response = {
        "model": "jev-pinned",
        "usage": {"input_tokens": 10, "output_tokens": 4},
        "answers": {
            "quality": {
                "type": "score",
                "score": 1.7,
                "confidence": 0.7,
                "legend": {0: "bad", 1: "okay", 2: "great"},
                "probabilities": {0: 0.1, 1: 0.1, 2: 0.8},
            },
            "tone/id": {
                "type": "choice",
                "choice": "calm",
                "confidence": 0.75,
                "probabilities": {"very angry": 0.25, "calm": 0.75},
            },
            "is spam": {"type": "noul", "noul": 0.2},
        },
    }
    specs = validate_questions(questions)
    return specs, validate_response(response, specs)


def test_default_schema_preserves_question_and_criterion_order() -> None:
    specs, response = _specs_and_response()
    schema = compile_feature_schema(specs)

    assert schema.feature_names == (
        "is%20spam__p_true",
        "tone%2Fid__p[calm]",
        "tone%2Fid__p[very%20angry]",
        "quality__p[0]",
        "quality__p[1]",
        "quality__p[2]",
    )
    values = schema.transform([response])
    np.testing.assert_allclose(values, [[0.2, 0.75, 0.25, 0.1, 0.1, 0.8]])
    assert values.dtype == np.float64


def test_expected_score_encoding() -> None:
    specs, response = _specs_and_response()
    schema = compile_feature_schema(specs, score_encoding="expected")

    assert schema.feature_names[-1] == "quality__expected"
    assert schema.transform([response])[0, -1] == pytest.approx(1.7)


def test_expected_and_standard_deviation_encoding() -> None:
    specs, response = _specs_and_response()
    schema = compile_feature_schema(specs, score_encoding="expected_and_std")
    row = schema.transform([response])[0]

    expected_std = math.sqrt(0.1 * (0 - 1.7) ** 2 + 0.1 * (1 - 1.7) ** 2 + 0.8 * (2 - 1.7) ** 2)
    assert schema.feature_names[-2:] == ("quality__expected", "quality__std")
    assert row[-2] == pytest.approx(1.7)
    assert row[-1] == pytest.approx(expected_std)


def test_confidence_columns_are_added_only_for_choice_and_score() -> None:
    specs, response = _specs_and_response()
    schema = compile_feature_schema(specs, score_encoding="expected", include_confidence=True)

    assert schema.feature_names == (
        "is%20spam__p_true",
        "tone%2Fid__p[calm]",
        "tone%2Fid__p[very%20angry]",
        "tone%2Fid__confidence",
        "quality__expected",
        "quality__confidence",
    )
    np.testing.assert_allclose(schema.transform([response]), [[0.2, 0.75, 0.25, 0.75, 1.7, 0.7]])


def test_empty_responses_keep_compiled_feature_width() -> None:
    specs, _ = _specs_and_response()
    schema = compile_feature_schema(specs)

    assert schema.transform([]).shape == (0, schema.n_features)


def test_feature_token_escaping_is_injective_for_delimiters_and_percent() -> None:
    assert escape_feature_token("a/b") == "a%2Fb"
    assert escape_feature_token("a%2Fb") == "a%252Fb"


@pytest.mark.parametrize("encoding", ["full", "mean", ""])
def test_invalid_score_encoding_is_rejected(encoding: str) -> None:
    specs, _ = _specs_and_response()

    with pytest.raises(JevConfigurationError, match="score_encoding"):
        compile_feature_schema(specs, score_encoding=encoding)  # type: ignore[arg-type]


def test_batch_result_is_read_only_and_aggregates_known_usage() -> None:
    specs, response = _specs_and_response()
    schema = compile_feature_schema(specs)
    result = JevBatchResult(schema.transform([response, response]), (response, response))

    assert not result.values.flags.writeable
    assert result.resolved_models == ("jev-pinned", "jev-pinned")
    assert result.usage.request_count == 2
    assert result.usage.input_tokens == 20
    assert result.usage.output_tokens == 8
    with pytest.raises(ValueError):
        result.values[0, 0] = 0.9


def test_batch_usage_is_unknown_if_any_response_omits_a_count() -> None:
    specs = validate_questions({"q": {"type": "noul"}})
    known = validate_response(
        {"model": "jev", "usage": {"input_tokens": 2}, "answers": {"q": {"type": "noul", "noul": 0.5}}},
        specs,
    )
    unknown = validate_response(
        {"model": "jev", "usage": {}, "answers": {"q": {"type": "noul", "noul": 0.5}}},
        specs,
    )
    schema = compile_feature_schema(specs)
    result = JevBatchResult(schema.transform([known, unknown]), (known, unknown))

    assert result.usage.input_tokens is None
    assert result.usage.output_tokens is None
