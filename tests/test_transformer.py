"""End-to-end contracts for the System One semantic feature transformer."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.exceptions import NotFittedError
from sklearn.utils import get_tags
from typesafe_sdk import Choice, Noul, Score

from scikit_jev.backends import TypeSafeBackend
from scikit_jev.exceptions import JevConfigurationError, JevResponseError
from scikit_jev.transformer import JevTransformer
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


def _noul_questions() -> dict[str, Noul]:
    return {
        "relevant": Noul(instructions="Is this relevant?"),
        "urgent": Noul(instructions="Is this urgent?"),
    }


def _mixed_questions() -> dict[str, Any]:
    return {
        "risk": Noul(instructions="Is there risk?"),
        "tone": Choice(
            instructions="What is the tone?",
            criteria={"calm": None, "angry": None},
        ),
        "severity": Score(
            instructions="How severe is it?",
            criteria=["low", "medium", "high"],
        ),
    }


def _mixed_response(*, model: str = "jev-test") -> FakeSystemOneResponse:
    return FakeSystemOneResponse(
        model=model,
        answers={
            "risk": FakeNoulAnswer(noul=0.75),
            "tone": FakeChoiceAnswer(
                choice="angry",
                confidence=0.8,
                probabilities={"angry": 0.7, "calm": 0.3},
            ),
            "severity": FakeScoreAnswer(
                score=1.3,
                confidence=0.6,
                legend={0: "low", 1: "medium", 2: "high"},
                probabilities={2: 0.5, 0: 0.2, 1: 0.3},
            ),
        },
        usage=FakeUsage(input_tokens=21, output_tokens=4),
    )


def test_constructor_is_clone_safe_and_assignment_only() -> None:
    questions = _noul_questions()
    backend = TypeSafeBackend(timeout=2.0)
    transformer = JevTransformer(
        questions,
        model="jev-pinned",
        backend=backend,
        state_builder="auto",
        score_encoding="expected_and_std",
        include_confidence=True,
    )

    cloned = clone(transformer)

    assert cloned.get_params(deep=False).keys() == transformer.get_params(deep=False).keys()
    assert tuple(cloned.questions) == tuple(questions)
    assert cloned.model == "jev-pinned"
    assert cloned.backend.timeout == 2.0
    assert cloned.score_encoding == "expected_and_std"
    assert cloned.include_confidence is True
    assert not hasattr(cloned, "is_fitted_")


def test_fit_is_network_free_and_binds_default_backend() -> None:
    transformer = JevTransformer(_noul_questions())

    returned = transformer.fit(["first", "second"], y=[0, 1])

    assert returned is transformer
    assert transformer.is_fitted_ is True
    assert tuple(transformer.questions_) == ("relevant", "urgent")
    assert tuple(spec.question_id for spec in transformer.question_specs_) == ("relevant", "urgent")
    assert isinstance(transformer.backend_, TypeSafeBackend)
    assert not hasattr(transformer, "n_features_in_")
    np.testing.assert_array_equal(
        transformer.get_feature_names_out(),
        np.asarray(["relevant__p_true", "urgent__p_true"], dtype=object),
    )


def test_fit_does_not_open_injected_backend() -> None:
    factory = RecordingClientFactory([])
    backend = RecordingBackend(factory)

    JevTransformer(_noul_questions(), backend=backend).fit(["first", "second"])

    assert backend.open_count == 0
    assert factory.calls == []


@pytest.mark.parametrize("y", [[1], [1, 2, 3], 1])
def test_fit_rejects_inconsistent_or_scalar_y_without_network(y: Any) -> None:
    backend = RecordingBackend(RecordingClientFactory([]))
    transformer = JevTransformer(_noul_questions(), backend=backend)

    with pytest.raises(JevConfigurationError, match="same number"):
        transformer.fit(["first", "second"], y=y)

    assert backend.open_count == 0
    assert not hasattr(transformer, "is_fitted_")


def test_fit_rejects_empty_input_and_failed_refit_clears_fitted_state() -> None:
    transformer = JevTransformer(_noul_questions()).fit(["first"])

    with pytest.raises(JevConfigurationError, match="at least one"):
        transformer.fit([])

    assert not hasattr(transformer, "is_fitted_")
    assert not hasattr(transformer, "feature_schema_")


def test_fit_transform_calls_once_per_sample_and_packs_all_questions() -> None:
    first = noul_response({"relevant": 0.1, "urgent": 0.2})
    second = noul_response({"relevant": 0.8, "urgent": 0.9})
    client = RecordingSystemOneClient([first, second])
    backend = RecordingBackend(RecordingClientFactory([client]))
    transformer = JevTransformer(_noul_questions(), backend=backend, model="jev-pinned")

    values = transformer.fit_transform(["first", {"row": 2}])

    np.testing.assert_allclose(values, [[0.1, 0.2], [0.8, 0.9]])
    assert [call.state for call in client.calls] == ["first", {"row": 2}]
    assert all(tuple(call.questions) == ("relevant", "urgent") for call in client.calls)
    assert all(call.model == "jev-pinned" for call in client.calls)
    assert client.close_count == 1


def test_fit_transform_consumes_generator_and_state_builder_once_per_sample() -> None:
    first = noul_response({"relevant": 0.1, "urgent": 0.2})
    second = noul_response({"relevant": 0.8, "urgent": 0.9})
    client = RecordingSystemOneClient([first, second])
    backend = RecordingBackend(RecordingClientFactory([client]))
    built: list[int] = []

    def state_builder(value: int) -> dict[str, int]:
        built.append(value)
        return {"value": value}

    transformer = JevTransformer(
        _noul_questions(),
        backend=backend,
        state_builder=state_builder,
    )

    values = transformer.fit_transform(iter([1, 2]))

    np.testing.assert_allclose(values, [[0.1, 0.2], [0.8, 0.9]])
    assert built == [1, 2]
    assert [call.state for call in client.calls] == [{"value": 1}, {"value": 2}]
    assert not hasattr(transformer, "states_")


def test_mixed_questions_preserve_declared_feature_order() -> None:
    client = RecordingSystemOneClient([_mixed_response()])
    backend = RecordingBackend(RecordingClientFactory([client]))
    transformer = JevTransformer(
        _mixed_questions(),
        backend=backend,
        include_confidence=True,
    ).fit(["schema sample"])

    values = transformer.transform(["evaluation sample"])

    np.testing.assert_allclose(values, [[0.75, 0.3, 0.7, 0.8, 0.2, 0.3, 0.5, 0.6]])
    np.testing.assert_array_equal(
        transformer.get_feature_names_out(),
        np.asarray(
            [
                "risk__p_true",
                "tone__p[calm]",
                "tone__p[angry]",
                "tone__confidence",
                "severity__p[0]",
                "severity__p[1]",
                "severity__p[2]",
                "severity__confidence",
            ],
            dtype=object,
        ),
    )


def test_evaluate_returns_values_and_metadata_from_one_inference() -> None:
    client = RecordingSystemOneClient([_mixed_response(model="jev-resolved")])
    backend = RecordingBackend(RecordingClientFactory([client]))
    transformer = JevTransformer(_mixed_questions(), backend=backend).fit(["schema sample"])

    result = transformer.evaluate(["evaluation sample"])

    assert len(client.calls) == 1
    assert result.values.shape == (1, 6)
    assert result.resolved_models == ("jev-resolved",)
    assert result.usage.request_count == 1
    assert result.usage.input_tokens == 21
    assert result.usage.output_tokens == 4
    assert result.warnings == ()
    with pytest.raises(ValueError, match="read-only"):
        result.values[0, 0] = 0.0


def test_invalid_response_reports_safe_sample_index() -> None:
    valid = noul_response({"relevant": 0.1, "urgent": 0.2})
    missing_answer = noul_response({"relevant": 0.9})
    client = RecordingSystemOneClient([valid, missing_answer])
    backend = RecordingBackend(RecordingClientFactory([client]))
    transformer = JevTransformer(_noul_questions(), backend=backend).fit(["schema sample"])

    with pytest.raises(JevResponseError, match="sample 1") as caught:
        transformer.transform(["private first state", "private second state"])

    assert "private first state" not in str(caught.value)
    assert "private second state" not in str(caught.value)


def test_transform_empty_input_has_stable_shape_and_no_backend_call() -> None:
    backend = RecordingBackend(RecordingClientFactory([]))
    transformer = JevTransformer(_noul_questions(), backend=backend).fit(["schema sample"])
    attributes_before = tuple(vars(transformer))

    values = transformer.transform([])
    result = transformer.evaluate([])

    assert values.shape == (0, 2)
    assert values.dtype == np.float64
    assert result.values.shape == (0, 2)
    assert result.usage.request_count == 0
    assert backend.open_count == 0
    assert tuple(vars(transformer)) == attributes_before


def test_get_feature_names_out_requires_fit() -> None:
    with pytest.raises(NotFittedError):
        JevTransformer(_noul_questions()).get_feature_names_out()


def test_two_dimensional_feature_count_drift_is_rejected_before_inference() -> None:
    backend = RecordingBackend(RecordingClientFactory([]))
    transformer = JevTransformer(_noul_questions(), backend=backend).fit(np.asarray([[1, 2], [3, 4]]))

    assert transformer.n_features_in_ == 2
    with pytest.raises(JevConfigurationError, match="number of features"):
        transformer.transform(np.asarray([[1, 2, 3]]))
    with pytest.raises(JevConfigurationError, match="container schema"):
        transformer.transform(["not a two-dimensional row container"])

    assert backend.open_count == 0


def test_dataframe_column_order_drift_is_rejected_before_inference() -> None:
    backend = RecordingBackend(RecordingClientFactory([]))
    frame = pd.DataFrame({"title": ["a", "b"], "body": ["c", "d"]})
    transformer = JevTransformer(_noul_questions(), backend=backend).fit(frame)

    assert transformer.n_features_in_ == 2
    np.testing.assert_array_equal(transformer.feature_names_in_, ["title", "body"])
    with pytest.raises(JevConfigurationError, match="feature names or column order"):
        transformer.transform(frame[["body", "title"]])
    with pytest.raises(JevConfigurationError, match="container schema"):
        transformer.transform(frame.to_numpy())

    assert backend.open_count == 0


def test_refit_from_dataframe_to_state_sequence_removes_feature_metadata() -> None:
    transformer = JevTransformer(_noul_questions())
    transformer.fit(pd.DataFrame({"text": ["first"]}))
    assert hasattr(transformer, "feature_names_in_")

    transformer.fit(["second"])

    assert not hasattr(transformer, "n_features_in_")
    assert not hasattr(transformer, "feature_names_in_")


def test_fit_snapshots_questions_against_caller_mutation() -> None:
    questions = {"relevant": Noul(instructions="Is this relevant?")}
    response = noul_response({"relevant": 0.9})
    client = RecordingSystemOneClient([response])
    backend = RecordingBackend(RecordingClientFactory([client]))
    transformer = JevTransformer(questions, backend=backend).fit(["schema sample"])
    questions["later"] = Noul(instructions="Added after fit")

    values = transformer.transform(["evaluation sample"])

    np.testing.assert_allclose(values, [[0.9]])
    assert tuple(client.calls[0].questions) == ("relevant",)


def test_tags_describe_remote_json_transformer_contract() -> None:
    tags = get_tags(JevTransformer(_noul_questions()))

    assert tags.non_deterministic is True
    assert tags.requires_fit is True
    assert tags.target_tags.required is False
    assert tags.transformer_tags is not None
    assert tags.input_tags.one_d_array is True
    assert tags.input_tags.two_d_array is True
    assert tags.input_tags.three_d_array is False
    assert tags.input_tags.string is True
    assert tags.input_tags.dict is True
    assert tags.input_tags.sparse is False
    assert tags.input_tags.allow_nan is False
