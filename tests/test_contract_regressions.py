"""Offline regressions for evaluation, stopping, and response-boundary contracts."""

from __future__ import annotations

import math
import pickle
import traceback
from collections.abc import Callable
from types import SimpleNamespace
from typing import Any

import httpx2
import numpy as np
import pytest
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.metrics import log_loss
from typesafe_sdk import Choice, Noul, Score, SystemOneResponse

from scikit_jev import JevChoiceClassifier, JevNoulClassifier, JevScoreRegressor, JevTransformer, TypeSafeBackend
from scikit_jev.exceptions import JevResponseError
from tests.fakes import (
    FakeChoiceAnswer,
    FakeScoreAnswer,
    FakeSystemOneResponse,
    RecordingClientFactory,
    RecordingSystemOneClient,
    noul_response,
)


def test_choice_sorted_classes_work_with_log_loss_and_preserve_feature_order() -> None:
    question = Choice(criteria={"z": None, "a": None, "m": None})
    states = ["z", "a", "m"] * 2
    responses = [
        FakeSystemOneResponse(
            model="synthetic-contract-model",
            answers={
                "target": FakeChoiceAnswer(
                    choice=state,
                    confidence=0.98,
                    probabilities={label: 0.98 if label == state else 0.01 for label in question.criteria},
                )
            },
        )
        for state in states
    ]
    clients = [RecordingSystemOneClient(responses) for _ in range(4)]
    backend = TypeSafeBackend(client_factory=RecordingClientFactory(clients))
    classifier = JevChoiceClassifier(question, backend=backend).fit(states, states)

    probabilities = classifier.predict_proba(states)

    assert classifier.classes_.tolist() == ["a", "m", "z"]
    np.testing.assert_array_equal(classifier.predict(states), states)
    assert log_loss(states, probabilities, labels=classifier.classes_) == pytest.approx(-math.log(0.98))
    result = classifier.evaluate(states)
    assert result.responses[0].answers[0].labels == ("z", "a", "m")
    np.testing.assert_allclose(result.values, probabilities)
    transformer = JevTransformer({"target": question}, backend=backend).fit(states)
    assert transformer.get_feature_names_out().tolist() == ["target__p[z]", "target__p[a]", "target__p[m]"]
    np.testing.assert_allclose(transformer.transform(states)[0], [0.98, 0.01, 0.01])

    class StateClient:
        def system_one(self, state: Any, questions: Any, *, model: str | None = None) -> FakeSystemOneResponse:
            return responses[states.index(state)]

        def close(self) -> None:
            pass

    calibration_classifier = JevChoiceClassifier(question, backend=TypeSafeBackend(client_factory=StateClient)).fit(
        states, states
    )
    calibrated = CalibratedClassifierCV(FrozenEstimator(calibration_classifier), cv=2).fit(states, states)
    assert calibrated.classes_.tolist() == ["a", "m", "z"]
    np.testing.assert_array_equal(calibrated.predict(states), states)
    assert all(client.close_count == 1 for client in clients)


@pytest.mark.parametrize(("negative", "positive"), [("z", "a"), (5, 1), (True, False)])
@pytest.mark.parametrize(("threshold", "middle_is_positive"), [(0.5, True), (0.8, False)])
def test_noul_sorted_columns_keep_positive_threshold_orientation(
    negative: Any, positive: Any, threshold: float, middle_is_positive: bool
) -> None:
    responses = [noul_response({"target": p}) for p in [0.2, 0.5, 0.8]]
    clients = [RecordingSystemOneClient(responses) for _ in range(2)]
    classifier = JevNoulClassifier(
        Noul(instructions="Positive?"),
        negative_label=negative,
        positive_label=positive,
        threshold=threshold,
        backend=TypeSafeBackend(client_factory=RecordingClientFactory(clients)),
    ).fit(["schema"])

    assert classifier.classes_.tolist() == [positive, negative]
    probabilities = classifier.predict_proba(["a", "b", "c"])
    np.testing.assert_allclose(probabilities, [[0.2, 0.8], [0.5, 0.5], [0.8, 0.2]])
    assert log_loss([negative, positive, positive], probabilities, labels=classifier.classes_) == pytest.approx(
        (-2 * math.log(0.8) - math.log(0.5)) / 3
    )
    expected_middle = positive if middle_is_positive else negative
    np.testing.assert_array_equal(classifier.predict(["a", "b", "c"]), [negative, expected_middle, positive])


def test_choice_ties_follow_sorted_classes_without_reordering_the_request() -> None:
    question = Choice(criteria={"z": None, "a": None})
    response = FakeSystemOneResponse(
        model="synthetic-contract-model",
        answers={"target": FakeChoiceAnswer("z", 0.5, {"z": 0.5, "a": 0.5})},
    )
    client = RecordingSystemOneClient([response])
    classifier = JevChoiceClassifier(question, backend=TypeSafeBackend(client_factory=lambda: client)).fit(["schema"])

    np.testing.assert_array_equal(classifier.predict(["sample"]), ["a"])
    assert tuple(client.calls[0].questions["target"].criteria) == ("z", "a")


@pytest.mark.parametrize(
    ("factory", "invalid"),
    [
        (lambda b: JevTransformer({"target": Noul()}, backend=b), noul_response({"target": 1.2})),
        (lambda b: JevNoulClassifier(Noul(), backend=b), noul_response({"target": 1.2})),
        (
            lambda b: JevChoiceClassifier(Choice(criteria={"a": None, "b": None}), backend=b),
            FakeSystemOneResponse(
                model="synthetic-contract-model",
                answers={"target": FakeChoiceAnswer("a", 0.5, {"a": 0.5})},
            ),
        ),
        (
            lambda b: JevScoreRegressor(Score(criteria=["low", "high"]), backend=b),
            FakeSystemOneResponse(
                model="synthetic-contract-model",
                answers={"target": FakeScoreAnswer(0.9, 0.8, {0: "low", 1: "high"}, {0: 0.8, 1: 0.2})},
            ),
        ),
    ],
    ids=["transformer", "noul", "choice", "score"],
)
@pytest.mark.parametrize("close_fails", [False, True])
def test_invalid_first_response_stops_before_later_rows_and_closes_once(
    factory: Callable[[Any], Any], invalid: FakeSystemOneResponse, close_fails: bool
) -> None:
    class Client(RecordingSystemOneClient):
        def close(self) -> None:
            super().close()
            if close_fails:
                raise RuntimeError("synthetic cleanup diagnostic")

    client = Client([invalid])
    estimator = factory(TypeSafeBackend(client_factory=lambda: client)).fit(["schema"])

    with pytest.raises(JevResponseError, match="sample 0") as caught:
        estimator.evaluate(["first", "must not be sent", "also not sent"])

    assert len(client.calls) == 1
    assert client.close_count == 1
    assert caught.value.__cause__ is None
    assert caught.value.__context__ is None


@pytest.mark.parametrize("field", ["extra_id", "answer_type"])
def test_response_values_do_not_enter_ordinary_errors(field: str) -> None:
    sentinel = "SYNTHETIC_UNTRUSTED_RESPONSE_SENTINEL"
    answers: dict[str, Any] = {"target": {"type": "noul", "noul": 0.8}}
    if field == "extra_id":
        answers[sentinel] = {"type": "noul", "noul": 0.5}
    else:
        answers["target"]["type"] = sentinel
    response = SimpleNamespace(model="synthetic-contract-model", usage={}, answers=answers)
    client = RecordingSystemOneClient([lambda _: response])
    estimator = JevTransformer({"target": Noul()}, backend=TypeSafeBackend(client_factory=lambda: client)).fit(
        ["schema"]
    )

    with pytest.raises(JevResponseError) as caught:
        estimator.transform(["sample"])

    error = caught.value
    assert sentinel not in str(error)
    assert sentinel not in "".join(traceback.format_exception(error))
    assert sentinel.encode() not in pickle.dumps(error)
    assert error.__cause__ is None
    assert error.__context__ is None


@pytest.mark.parametrize("extra_type", ["noul", "future_answer"])
def test_sdk_raw_answer_ids_are_checked_even_if_decoding_drops_an_answer(extra_type: str) -> None:
    payload = {
        "model": "synthetic-contract-model",
        "usage": {},
        "answers": {
            "target": {"type": "noul", "noul": 0.8},
            "extra": {"type": extra_type, "noul": 0.2},
        },
    }
    response = SystemOneResponse.from_http_response(httpx2.Response(200, json=payload))
    client = RecordingSystemOneClient([lambda _: response])
    estimator = JevTransformer({"target": Noul()}, backend=TypeSafeBackend(client_factory=lambda: client)).fit(
        ["schema"]
    )

    with pytest.raises(JevResponseError, match="ID mismatch"):
        estimator.transform(["sample", "must not be sent"])

    assert len(client.calls) == 1
    assert client.close_count == 1


def test_valid_sdk_raw_response_returns_only_canonical_data() -> None:
    payload = {
        "model": "synthetic-contract-model",
        "usage": {},
        "answers": {"target": {"type": "noul", "noul": 0.8}},
    }
    response = SystemOneResponse.from_http_response(httpx2.Response(200, json=payload))
    client = RecordingSystemOneClient([lambda _: response])
    estimator = JevTransformer({"target": Noul()}, backend=TypeSafeBackend(client_factory=lambda: client)).fit(
        ["schema"]
    )

    result = estimator.evaluate(["sample"])

    np.testing.assert_allclose(result.values, [[0.8]])
    assert result.usage.input_tokens is None
    assert not hasattr(result.responses[0], "raw_http_response")


def test_raw_decoder_failure_drops_private_diagnostics_and_stops() -> None:
    sentinel = "SYNTHETIC_RAW_DECODER_SENTINEL"

    def fail_to_decode() -> Any:
        raise ValueError(sentinel)

    response = SimpleNamespace(
        model="synthetic-contract-model",
        usage={},
        answers={"target": {"type": "noul", "noul": 0.8}},
        raw_http_response=SimpleNamespace(json=fail_to_decode),
    )
    client = RecordingSystemOneClient([lambda _: response])
    estimator = JevTransformer({"target": Noul()}, backend=TypeSafeBackend(client_factory=lambda: client)).fit(
        ["schema"]
    )

    with pytest.raises(JevResponseError, match="could not be inspected") as caught:
        estimator.evaluate(["sample", "must not be sent"])

    assert len(client.calls) == 1
    assert client.close_count == 1
    assert sentinel not in "".join(traceback.format_exception(caught.value))
    assert sentinel.encode() not in pickle.dumps(caught.value)
    assert caught.value.__cause__ is None
    assert caught.value.__context__ is None
