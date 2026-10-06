"""Cross-component sklearn integration contracts using local fake clients."""

from __future__ import annotations

import pickle
from contextlib import contextmanager
from typing import Any

import numpy as np
from packaging.version import Version
from sklearn import __version__ as sklearn_version
from sklearn.base import BaseEstimator, clone
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.utils.estimator_checks import check_estimator
from typesafe_sdk import Choice, Noul, Score

from scikit_jev import (
    JevChoiceClassifier,
    JevNoulClassifier,
    JevScoreRegressor,
    JevTransformer,
    TypeSafeBackend,
)
from tests.fakes import (
    FakeChoiceAnswer,
    FakeNoulAnswer,
    FakeScoreAnswer,
    FakeSystemOneResponse,
    FakeUsage,
    RecordingBackend,
    RecordingClientFactory,
    RecordingSystemOneClient,
    noul_response,
)


def _numeric_state_builder(sample: Any) -> Any:
    """Give generic sklearn checks a valid JSON-array state per sample."""

    converted = sample.tolist() if hasattr(sample, "tolist") else sample
    if isinstance(converted, (str, dict, list)):
        return converted
    return [converted]


class _ConstantSystemOneClient:
    """Stateless response generator for the variable call counts in checks."""

    def system_one(self, state: Any, questions: Any, *, model: str | None = None) -> FakeSystemOneResponse:
        del state
        answers = {}
        for question_id, question in questions.items():
            if question.type == "noul":
                answers[question_id] = FakeNoulAnswer(0.5)
            elif question.type == "choice":
                labels = tuple(question.criteria)
                probability = 1.0 / len(labels)
                answers[question_id] = FakeChoiceAnswer(
                    choice=labels[0],
                    confidence=probability,
                    probabilities={label: probability for label in labels},
                )
            else:
                levels = tuple(range(len(question.criteria)))
                probability = 1.0 / len(levels)
                answers[question_id] = FakeScoreAnswer(
                    score=sum(level * probability for level in levels),
                    confidence=probability,
                    legend={level: criterion for level, criterion in enumerate(question.criteria)},
                    probabilities={level: probability for level in levels},
                )
        return FakeSystemOneResponse(
            model=model or "jev-check",
            answers=answers,
            usage=FakeUsage(),
        )

    def close(self) -> None:
        return None


class _ConstantSystemOneBackend(BaseEstimator):
    """Clone-safe, no-network backend used only by estimator checks."""

    def __init__(self, max_concurrency: int = 1) -> None:
        self.max_concurrency = max_concurrency

    @contextmanager
    def open_client(self):
        client = _ConstantSystemOneClient()
        try:
            yield client
        finally:
            client.close()


def _sklearn_16_check_failures() -> dict[str, str]:
    """Return failures caused by sklearn 1.6 checks, never later versions."""

    if Version(sklearn_version).release[:2] != (1, 6):
        return {}
    return {
        "check_positive_only_tag_during_fit": (
            "sklearn 1.6 converts categorical X to int32, then subtracts a floating-point mean in place before calling "
            "the estimator"
        )
    }


def _sklearn_16_noul_target_failures() -> dict[str, str]:
    """Name sklearn 1.6 checks whose generated y is outside False/True."""

    if Version(sklearn_version).release[:2] != (1, 6):
        return {}
    reason = "sklearn 1.6 supplies target labels outside the Noul classifier's declared False/True domain"
    return {
        "check_estimators_overwrite_params": reason,
        "check_estimators_fit_returns_self": reason,
        "check_readonly_memmap_input": reason,
    }


def test_all_public_estimators_are_clone_safe() -> None:
    backend = TypeSafeBackend(timeout=2.0)
    estimators = (
        JevTransformer({"target": Noul(instructions="Relevant?")}, backend=backend),
        JevNoulClassifier(Noul(instructions="Relevant?"), backend=backend),
        JevChoiceClassifier(
            Choice(instructions="Tone?", criteria={"calm": None, "angry": None}),
            backend=backend,
        ),
        JevScoreRegressor(Score(instructions="Quality?", criteria=["low", "high"]), backend=backend),
    )

    for estimator in estimators:
        cloned = clone(estimator)
        assert type(cloned) is type(estimator)
        assert cloned.get_params(deep=False).keys() == estimator.get_params(deep=False).keys()
        assert not hasattr(cloned, "is_fitted_")


def test_deep_parameters_include_backend_configuration() -> None:
    estimator = JevTransformer(
        {"target": Noul(instructions="Relevant?")},
        backend=TypeSafeBackend(timeout=3.0, max_concurrency=1),
    )

    parameters = estimator.get_params(deep=True)

    assert parameters["backend__timeout"] == 3.0
    assert parameters["backend__max_concurrency"] == 1
    assert parameters["backend__client_factory"] is None
    assert all("api_key" not in name for name in parameters)


def test_fit_is_network_free_and_transform_calls_once_per_sample() -> None:
    client = RecordingSystemOneClient(
        [
            noul_response({"target": 0.2}),
            noul_response({"target": 0.8}),
        ]
    )
    backend = RecordingBackend(RecordingClientFactory([client]))
    transformer = JevTransformer({"target": Noul(instructions="Relevant?")}, backend=backend)

    transformer.fit(["schema one", "schema two"])
    assert backend.open_count == 0

    values = transformer.transform(["evaluation one", "evaluation two"])

    np.testing.assert_allclose(values, [[0.2], [0.8]])
    assert backend.open_count == 1
    assert len(client.calls) == 2
    assert all(tuple(call.questions) == ("target",) for call in client.calls)


def test_transformer_runs_in_a_normal_logistic_regression_pipeline() -> None:
    training_probabilities = (0.05, 0.15, 0.85, 0.95)
    prediction_probabilities = (0.1, 0.9)
    training_client = RecordingSystemOneClient(
        [noul_response({"signal": probability}) for probability in training_probabilities]
    )
    prediction_client = RecordingSystemOneClient(
        [noul_response({"signal": probability}) for probability in prediction_probabilities]
    )
    backend = RecordingBackend(RecordingClientFactory([training_client, prediction_client]))
    pipeline = Pipeline(
        [
            ("jev", JevTransformer({"signal": Noul(instructions="Positive?")}, backend=backend)),
            ("model", LogisticRegression(random_state=0)),
        ]
    )

    pipeline.fit(["a", "b", "c", "d"], [0, 0, 1, 1])
    predictions = pipeline.predict(["e", "f"])

    np.testing.assert_array_equal(predictions, [0, 1])
    assert len(training_client.calls) == 4
    assert len(prediction_client.calls) == 2
    assert backend.open_count == 2


def test_repr_parameters_and_pickle_do_not_capture_environment_secret_or_fit_states(
    monkeypatch,
) -> None:
    api_secret = "typesafe-api-secret-that-must-not-appear"
    private_state = "private-fit-state-that-must-not-appear"
    monkeypatch.setenv("TYPESAFE_API_KEY", api_secret)
    estimator = JevTransformer({"target": Noul(instructions="Relevant?")}).fit([private_state])

    representation = repr(estimator)
    serialized = pickle.dumps(estimator)

    assert api_secret not in representation
    assert private_state not in representation
    assert api_secret.encode() not in serialized
    assert private_state.encode() not in serialized
    assert all("api_key" not in name for name in estimator.get_params(deep=True))


def test_transformer_modern_estimator_checks_have_only_named_semantic_input_exclusions() -> None:
    estimator = JevTransformer(
        {"target": Noul(instructions="Relevant?")},
        backend=_ConstantSystemOneBackend(),
        state_builder=_numeric_state_builder,
    )
    expected_failures = {
        "check_dont_overwrite_parameters": (
            "sklearn 1.8 converts X to one dimension when one_d_array is supported, then its check indexes X as 2D"
        ),
        "check_n_features_in_after_fitting": (
            "a one-dimensional semantic-state container has no conventional input feature axis or n_features_in_"
        ),
        **_sklearn_16_check_failures(),
    }

    check_estimator(estimator, legacy=False, expected_failed_checks=expected_failures)


def test_noul_classifier_modern_estimator_checks_have_only_named_fixed_schema_exclusions() -> None:
    estimator = JevNoulClassifier(
        Noul(instructions="Relevant?"),
        backend=_ConstantSystemOneBackend(),
        state_builder=_numeric_state_builder,
    )
    expected_failures = {
        "check_dont_overwrite_parameters": (
            "sklearn converts X to one dimension when one_d_array is supported, then its check indexes X as 2D"
        ),
        "check_n_features_in_after_fitting": (
            "a one-dimensional semantic-state container has no conventional input feature axis or n_features_in_"
        ),
        **_sklearn_16_noul_target_failures(),
        **_sklearn_16_check_failures(),
    }

    check_estimator(estimator, legacy=False, expected_failed_checks=expected_failures)


def test_choice_classifier_modern_estimator_checks_have_only_named_fixed_schema_exclusions() -> None:
    estimator = JevChoiceClassifier(
        Choice(instructions="Class?", criteria={"first": None, "second": None, "third": None}),
        backend=_ConstantSystemOneBackend(),
        state_builder=_numeric_state_builder,
    )
    target_reason = (
        "generic sklearn checks supply numeric y, but official Choice labels are strings and y is intentionally "
        "validated against the declared class domain"
    )
    expected_failures = {
        "check_fit_score_takes_y": target_reason,
        "check_estimators_overwrite_params": target_reason,
        "check_dont_overwrite_parameters": (
            "sklearn converts X to one dimension when one_d_array is supported, then its check indexes X as 2D"
        ),
        "check_estimators_fit_returns_self": target_reason,
        "check_readonly_memmap_input": target_reason,
        "check_n_features_in_after_fitting": target_reason,
        "check_positive_only_tag_during_fit": target_reason,
        **_sklearn_16_check_failures(),
    }

    check_estimator(estimator, legacy=False, expected_failed_checks=expected_failures)


def test_score_regressor_modern_estimator_checks_have_only_named_fixed_schema_exclusions() -> None:
    estimator = JevScoreRegressor(
        Score(instructions="Score?", criteria=[str(level) for level in range(10)]),
        backend=_ConstantSystemOneBackend(),
        state_builder=_numeric_state_builder,
    )
    expected_failures = {
        "check_dont_overwrite_parameters": (
            "sklearn converts X to one dimension when one_d_array is supported, then its check indexes X as 2D"
        ),
        "check_n_features_in_after_fitting": (
            "the generic check supplies y outside the declared Score rubric and one-dimensional semantic states have "
            "no conventional n_features_in_"
        ),
        **_sklearn_16_check_failures(),
    }

    check_estimator(estimator, legacy=False, expected_failed_checks=expected_failures)
