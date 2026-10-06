"""Contracts for the Score regressor."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from sklearn.base import clone, is_regressor
from sklearn.exceptions import NotFittedError
from sklearn.utils import get_tags
from typesafe_sdk import Choice, Noul, Score

from scikit_jev.exceptions import JevConfigurationError
from scikit_jev.regression import JevScoreRegressor
from tests.fakes import (
    FakeScoreAnswer,
    FakeSystemOneResponse,
    FakeUsage,
    RecordingBackend,
    RecordingClientFactory,
    RecordingSystemOneClient,
)


def _score_question() -> Score:
    return Score(instructions="How urgent is the state?", criteria=["low", "medium", "high"])


def _score_response(
    score: float,
    probabilities: dict[int, float],
    *,
    model: str = "jev-score-test",
) -> FakeSystemOneResponse:
    return FakeSystemOneResponse(
        model=model,
        answers={
            "target": FakeScoreAnswer(
                score=score,
                confidence=max(probabilities.values()),
                legend={0: "low", 1: "medium", 2: "high"},
                probabilities=probabilities,
            )
        },
        usage=FakeUsage(input_tokens=14, output_tokens=3),
    )


def _backend_with_batches(
    *batches: list[FakeSystemOneResponse],
) -> tuple[RecordingBackend, list[RecordingSystemOneClient]]:
    clients = [RecordingSystemOneClient(batch) for batch in batches]
    return RecordingBackend(RecordingClientFactory(clients)), clients


def test_regressor_constructor_is_clone_safe_and_identified() -> None:
    regressor = JevScoreRegressor(
        _score_question(),
        question_id="rating",
        model="jev-pinned",
        state_builder="auto",
    )

    cloned = clone(regressor)

    assert is_regressor(regressor)
    assert is_regressor(cloned)
    assert cloned.question_id == "rating"
    assert cloned.model == "jev-pinned"
    assert not hasattr(cloned, "is_fitted_")


@pytest.mark.parametrize("method", ["evaluate", "predict"])
def test_regressor_inference_requires_fit(method: str) -> None:
    with pytest.raises(NotFittedError):
        getattr(JevScoreRegressor(_score_question()), method)(["sample"])


def test_score_fit_is_network_free() -> None:
    backend = RecordingBackend(RecordingClientFactory([]))
    regressor = JevScoreRegressor(_score_question(), backend=backend)

    returned = regressor.fit(["first", "second"], y=[0.25, 1.75])

    assert returned is regressor
    assert backend.open_count == 0
    assert regressor.question_specs_[0].kind == "score"
    assert regressor.question_specs_[0].levels == (0, 1, 2)
    assert tuple(regressor.questions_) == ("target",)


@pytest.mark.parametrize(
    "question",
    [Noul(instructions="Is it urgent?"), Choice(criteria={"low": None, "high": None})],
)
def test_score_regressor_rejects_wrong_question_type_without_network(question: Any) -> None:
    backend = RecordingBackend(RecordingClientFactory([]))
    regressor = JevScoreRegressor(question, backend=backend)

    with pytest.raises(JevConfigurationError, match="requires a Score question"):
        regressor.fit(["sample"])

    assert backend.open_count == 0
    assert not hasattr(regressor, "is_fitted_")


@pytest.mark.parametrize(
    ("target", "message"),
    [
        ([[0.0], [1.0]], "one-dimensional"),
        ([float("nan")], "finite"),
        (["low"], "numerical"),
        ([-0.1], "declared Score range"),
        ([2.1], "declared Score range"),
    ],
)
def test_score_regressor_validates_optional_target(target: Any, message: str) -> None:
    with pytest.raises(JevConfigurationError, match=message):
        JevScoreRegressor(_score_question()).fit(["sample"] * len(target), y=target)


def test_score_prediction_and_metadata_use_validated_expected_values() -> None:
    batch = [
        _score_response(0.3, {2: 0.1, 0: 0.8, 1: 0.1}, model="jev-resolved"),
        _score_response(1.7, {2: 0.8, 0: 0.1, 1: 0.1}, model="jev-resolved"),
    ]
    backend, clients = _backend_with_batches(batch, batch)
    regressor = JevScoreRegressor(_score_question(), backend=backend).fit(["schema"])
    fitted_attributes = tuple(vars(regressor))

    result = regressor.evaluate(["first", "second"])
    predictions = regressor.predict(["first", "second"])

    np.testing.assert_allclose(result.values, [0.3, 1.7])
    np.testing.assert_allclose(predictions, [0.3, 1.7])
    assert result.values.dtype == np.float64
    assert result.values.shape == (2,)
    assert result.resolved_models == ("jev-resolved", "jev-resolved")
    assert result.usage.request_count == 2
    assert result.usage.input_tokens == 28
    assert result.usage.output_tokens == 6
    assert tuple(vars(regressor)) == fitted_attributes
    assert all(len(client.calls) == 2 for client in clients)
    assert all(client.close_count == 1 for client in clients)


def test_score_empty_inference_has_stable_float_shape_without_network() -> None:
    backend = RecordingBackend(RecordingClientFactory([]))
    regressor = JevScoreRegressor(_score_question(), backend=backend).fit(["schema"])

    result = regressor.evaluate([])
    predictions = regressor.predict([])

    assert result.values.shape == (0,)
    assert predictions.shape == (0,)
    assert predictions.dtype == np.float64
    assert result.usage.request_count == 0
    assert backend.open_count == 0


def test_regressor_tags_describe_zero_shot_contract() -> None:
    tags = get_tags(JevScoreRegressor(_score_question()))

    assert tags.estimator_type == "regressor"
    assert tags.non_deterministic is True
    assert tags.target_tags.required is False
    assert tags.regressor_tags.poor_score is True
    assert tags.input_tags.one_d_array is True
    assert tags.input_tags.two_d_array is True
    assert tags.input_tags.string is True
    assert tags.input_tags.dict is True
    assert tags.input_tags.sparse is False
