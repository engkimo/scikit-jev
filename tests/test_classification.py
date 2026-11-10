"""Contracts for Noul and Choice classifiers."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from sklearn.base import clone, is_classifier
from sklearn.exceptions import NotFittedError
from sklearn.utils import get_tags
from typesafe_sdk import Choice, Noul, Score

from scikit_jev.classification import JevChoiceClassifier, JevNoulClassifier
from scikit_jev.exceptions import JevConfigurationError
from tests.fakes import (
    FakeChoiceAnswer,
    FakeSystemOneResponse,
    FakeUsage,
    RecordingBackend,
    RecordingClientFactory,
    RecordingSystemOneClient,
    noul_response,
)


def _noul_question() -> Noul:
    return Noul(instructions="Does the state satisfy the target?")


def _choice_question() -> Choice:
    return Choice(
        instructions="What is the tone?",
        criteria={"calm": None, "angry": None, "happy": None},
    )


def _choice_response(
    choice: str,
    probabilities: dict[str, float],
    *,
    model: str = "jev-choice-test",
) -> FakeSystemOneResponse:
    return FakeSystemOneResponse(
        model=model,
        answers={
            "target": FakeChoiceAnswer(
                choice=choice,
                confidence=max(probabilities.values()),
                probabilities=probabilities,
            )
        },
        usage=FakeUsage(input_tokens=12, output_tokens=2),
    )


def _backend_with_batches(
    *batches: list[FakeSystemOneResponse],
) -> tuple[RecordingBackend, list[RecordingSystemOneClient]]:
    clients = [RecordingSystemOneClient(batch) for batch in batches]
    return RecordingBackend(RecordingClientFactory(clients)), clients


@pytest.mark.parametrize(
    "estimator",
    [
        JevNoulClassifier(
            _noul_question(),
            question_id="decision",
            model="jev-pinned",
            threshold=0.7,
            negative_label="no",
            positive_label="yes",
        ),
        JevChoiceClassifier(_choice_question(), question_id="decision", model="jev-pinned"),
    ],
)
def test_classifier_constructors_are_clone_safe_and_identified(estimator: Any) -> None:
    cloned = clone(estimator)

    assert is_classifier(estimator)
    assert is_classifier(cloned)
    assert cloned.question_id == "decision"
    assert cloned.model == "jev-pinned"
    assert not hasattr(cloned, "is_fitted_")


@pytest.mark.parametrize(
    "estimator",
    [JevNoulClassifier(_noul_question()), JevChoiceClassifier(_choice_question())],
)
@pytest.mark.parametrize("method", ["evaluate", "predict", "predict_proba"])
def test_classifier_inference_requires_fit(estimator: Any, method: str) -> None:
    with pytest.raises(NotFittedError):
        getattr(estimator, method)(["sample"])


def test_noul_fit_is_network_free_and_binds_declared_class_order() -> None:
    backend = RecordingBackend(RecordingClientFactory([]))
    classifier = JevNoulClassifier(
        _noul_question(),
        backend=backend,
        negative_label="negative",
        positive_label="positive",
    )

    returned = classifier.fit(["first", "second"], y=["negative", "positive"])

    assert returned is classifier
    assert backend.open_count == 0
    assert classifier.classes_.tolist() == ["negative", "positive"]
    assert classifier.question_specs_[0].kind == "noul"
    assert tuple(classifier.questions_) == ("target",)


def test_choice_fit_is_network_free_and_sorts_classes() -> None:
    backend = RecordingBackend(RecordingClientFactory([]))
    classifier = JevChoiceClassifier(_choice_question(), backend=backend)

    classifier.fit(["first", "second"], y=["happy", "calm"])

    assert backend.open_count == 0
    assert classifier.classes_.tolist() == ["angry", "calm", "happy"]
    assert classifier.question_specs_[0].labels == ("calm", "angry", "happy")
    assert classifier.question_specs_[0].kind == "choice"


@pytest.mark.parametrize(
    ("estimator", "expected_name"),
    [
        (JevNoulClassifier(_choice_question()), "JevNoulClassifier"),
        (JevChoiceClassifier(_noul_question()), "JevChoiceClassifier"),
        (JevChoiceClassifier(Score(criteria=["low", "high"])), "JevChoiceClassifier"),
    ],
)
def test_classifier_rejects_wrong_question_type_without_network(estimator: Any, expected_name: str) -> None:
    backend = RecordingBackend(RecordingClientFactory([]))
    estimator.backend = backend

    with pytest.raises(JevConfigurationError, match=expected_name):
        estimator.fit(["sample"])

    assert backend.open_count == 0
    assert not hasattr(estimator, "is_fitted_")


@pytest.mark.parametrize("threshold", [-0.1, 1.1, float("nan"), float("inf"), True, "0.5"])
def test_noul_rejects_invalid_threshold_at_fit(threshold: Any) -> None:
    classifier = JevNoulClassifier(_noul_question(), threshold=threshold)

    with pytest.raises(JevConfigurationError, match="threshold"):
        classifier.fit(["sample"])

    assert not hasattr(classifier, "is_fitted_")


@pytest.mark.parametrize(
    ("negative_label", "positive_label", "message"),
    [
        ("same", "same", "distinct"),
        (["no"], ["yes"], "hashable scalar"),
        (("no",), ("yes",), "hashable scalar"),
        (float("nan"), 1.0, "reflexive"),
        ("no", 1, "mutually sortable"),
    ],
)
def test_noul_rejects_invalid_class_labels(negative_label: Any, positive_label: Any, message: str) -> None:
    classifier = JevNoulClassifier(
        _noul_question(),
        negative_label=negative_label,
        positive_label=positive_label,
    )

    with pytest.raises(JevConfigurationError, match=message):
        classifier.fit(["sample"])


@pytest.mark.parametrize(
    "classifier",
    [
        JevNoulClassifier(_noul_question(), negative_label="no", positive_label="yes"),
        JevChoiceClassifier(_choice_question()),
    ],
)
def test_classifier_rejects_target_outside_declared_domain(classifier: Any) -> None:
    with pytest.raises(JevConfigurationError, match="declared class domain"):
        classifier.fit(["sample"], y=["unknown"])


def test_noul_probability_prediction_threshold_and_metadata() -> None:
    batch = [
        noul_response({"target": 0.2}, model="jev-resolved"),
        noul_response({"target": 0.5}, model="jev-resolved"),
        noul_response({"target": 0.9}, model="jev-resolved"),
    ]
    backend, clients = _backend_with_batches(batch, batch, batch)
    classifier = JevNoulClassifier(
        _noul_question(),
        backend=backend,
        threshold=0.5,
        negative_label="no",
        positive_label="yes",
    ).fit(["schema"])
    fitted_attributes = tuple(vars(classifier))

    result = classifier.evaluate(["a", "b", "c"])
    probabilities = classifier.predict_proba(["a", "b", "c"])
    predictions = classifier.predict(["a", "b", "c"])

    expected = np.asarray([[0.8, 0.2], [0.5, 0.5], [0.1, 0.9]])
    np.testing.assert_allclose(result.values, expected)
    np.testing.assert_allclose(probabilities, expected)
    np.testing.assert_array_equal(predictions, ["no", "yes", "yes"])
    assert result.resolved_models == ("jev-resolved",) * 3
    assert result.usage.request_count == 3
    assert result.usage.input_tokens == 30
    assert result.usage.output_tokens == 3
    assert tuple(vars(classifier)) == fitted_attributes
    assert all(len(client.calls) == 3 for client in clients)
    assert all(client.close_count == 1 for client in clients)


def test_choice_probabilities_and_prediction_follow_sorted_classes() -> None:
    batch = [
        _choice_response("angry", {"happy": 0.1, "angry": 0.7, "calm": 0.2}),
        _choice_response("happy", {"happy": 0.6, "angry": 0.1, "calm": 0.3}),
    ]
    backend, clients = _backend_with_batches(batch, batch, batch)
    classifier = JevChoiceClassifier(_choice_question(), backend=backend).fit(["schema"])
    fitted_attributes = tuple(vars(classifier))

    result = classifier.evaluate(["first", "second"])
    probabilities = classifier.predict_proba(["first", "second"])
    predictions = classifier.predict(["first", "second"])

    expected = np.asarray([[0.7, 0.2, 0.1], [0.1, 0.3, 0.6]])
    np.testing.assert_allclose(result.values, expected)
    np.testing.assert_allclose(probabilities, expected)
    np.testing.assert_array_equal(predictions, ["angry", "happy"])
    assert classifier.classes_.tolist() == ["angry", "calm", "happy"]
    assert tuple(vars(classifier)) == fitted_attributes
    assert all(len(client.calls) == 2 for client in clients)


def test_choice_prediction_uses_sorted_first_argmax_for_tie() -> None:
    question = Choice(criteria={"first": None, "second": None})
    response = FakeSystemOneResponse(
        model="jev-test",
        answers={
            "target": FakeChoiceAnswer(
                choice="second",
                confidence=0.5,
                probabilities={"second": 0.5, "first": 0.5},
            )
        },
        usage=FakeUsage(),
    )
    backend, _ = _backend_with_batches([response])
    classifier = JevChoiceClassifier(question, backend=backend).fit(["schema"])

    np.testing.assert_array_equal(classifier.predict(["sample"]), ["first"])


@pytest.mark.parametrize(
    ("factory", "n_classes"),
    [
        (lambda backend: JevNoulClassifier(_noul_question(), backend=backend), 2),
        (lambda backend: JevChoiceClassifier(_choice_question(), backend=backend), 3),
    ],
)
def test_classifier_empty_inference_has_stable_shapes_without_network(factory: Any, n_classes: int) -> None:
    backend = RecordingBackend(RecordingClientFactory([]))
    classifier = factory(backend).fit(["schema"])

    result = classifier.evaluate([])
    probabilities = classifier.predict_proba([])
    predictions = classifier.predict([])

    assert result.values.shape == (0, n_classes)
    assert probabilities.shape == (0, n_classes)
    assert probabilities.dtype == np.float64
    assert predictions.shape == (0,)
    assert result.usage.request_count == 0
    assert backend.open_count == 0


@pytest.mark.parametrize(
    "classifier",
    [JevNoulClassifier(_noul_question()), JevChoiceClassifier(_choice_question())],
)
def test_classifier_tags_describe_zero_shot_contract(classifier: Any) -> None:
    tags = get_tags(classifier)

    assert tags.estimator_type == "classifier"
    assert tags.non_deterministic is True
    assert tags.target_tags.required is False
    assert tags.classifier_tags.poor_score is True
    assert tags.input_tags.one_d_array is True
    assert tags.input_tags.two_d_array is True
    assert tags.input_tags.string is True
    assert tags.input_tags.dict is True
    assert tags.input_tags.sparse is False
