"""Contract tests for lazy TypeSafe clients and ordered batch execution."""

from __future__ import annotations

import inspect
import pickle
import traceback
from collections.abc import Mapping
from typing import Any

import pytest
from sklearn.base import clone
from typesafe_sdk import Noul, RetryPolicy

from scikit_jev._engine import execute_system_one
from scikit_jev.backends import TypeSafeBackend
from scikit_jev.exceptions import JevConfigurationError, JevInferenceError
from tests.fakes import RecordingClientFactory, RecordingSystemOneClient, noul_response


def _questions() -> Mapping[str, Any]:
    return {
        "relevant": Noul(instructions="Is the state relevant?"),
        "urgent": Noul(instructions="Is the state urgent?"),
    }


def test_backend_construction_and_clone_do_not_create_a_client() -> None:
    client = RecordingSystemOneClient([])
    factory = RecordingClientFactory([client])
    retry = RetryPolicy(max_retries=0)

    backend = TypeSafeBackend(timeout=2.5, retry=retry, client_factory=factory)
    assert factory.calls == []

    cloned = clone(backend)

    assert cloned.timeout == 2.5
    assert cloned.retry == retry
    assert cloned.max_concurrency == 1
    assert factory.calls == []
    assert "api_key" not in backend.get_params(deep=True)
    assert "api_key" not in inspect.signature(TypeSafeBackend).parameters
    assert "api_key" not in repr(backend)


def test_execution_is_ordered_and_packs_all_questions_per_state() -> None:
    first = noul_response({"relevant": 0.2, "urgent": 0.1}, model="jev-test-a")
    second = noul_response({"relevant": 0.8, "urgent": 0.9}, model="jev-test-b")
    client = RecordingSystemOneClient([first, second])
    factory = RecordingClientFactory([client])
    backend = TypeSafeBackend(client_factory=factory)
    questions = _questions()
    states = ({"row": 1}, {"row": 2})

    responses = execute_system_one(states, questions, backend=backend, model="jev-pinned")

    assert responses == (first, second)
    assert len(factory.calls) == 1
    assert factory.calls[0].args == ()
    assert factory.calls[0].kwargs == {}
    assert [call.state for call in client.calls] == list(states)
    assert all(call.questions is questions for call in client.calls)
    assert all(tuple(call.questions) == ("relevant", "urgent") for call in client.calls)
    assert all(call.model == "jev-pinned" for call in client.calls)
    assert client.close_count == 1
    client.assert_consumed()


def test_empty_state_batch_does_not_create_a_client() -> None:
    factory = RecordingClientFactory([])

    responses = execute_system_one((), _questions(), backend=TypeSafeBackend(client_factory=factory))

    assert responses == ()
    assert factory.calls == []


def test_provider_failure_has_safe_indexed_error_and_closes_client() -> None:
    first = noul_response({"relevant": 0.2, "urgent": 0.1})
    cause = RuntimeError("provider diagnostics that must not enter the wrapper message")
    client = RecordingSystemOneClient([first, cause])
    factory = RecordingClientFactory([client])
    private_state = "private state must not enter the wrapper message"

    with pytest.raises(JevInferenceError) as caught:
        execute_system_one(
            ({"safe": 1}, private_state),
            _questions(),
            backend=TypeSafeBackend(client_factory=factory),
        )

    error = caught.value
    assert error.sample_index == 1
    assert error.question_ids == ("relevant", "urgent")
    assert error.__cause__ is None
    assert error.__context__ is None
    assert private_state not in str(error)
    assert str(cause) not in str(error)
    rendered_traceback = "".join(traceback.format_exception(error))
    serialized = pickle.dumps(error)
    assert private_state not in rendered_traceback
    assert str(cause) not in rendered_traceback
    assert private_state.encode() not in serialized
    assert str(cause).encode() not in serialized
    assert pickle.loads(serialized).sample_index == 1
    assert client.close_count == 1


def test_client_creation_failure_is_wrapped_without_state_data() -> None:
    private_state = {"secret": "never show this"}
    cause = RuntimeError("could not configure provider")

    def fail_to_create() -> RecordingSystemOneClient:
        raise cause

    with pytest.raises(JevInferenceError) as caught:
        execute_system_one(
            (private_state,),
            _questions(),
            backend=TypeSafeBackend(client_factory=fail_to_create),
        )

    assert caught.value.sample_index == 0
    assert caught.value.__cause__ is None
    assert caught.value.__context__ is None
    assert "never show this" not in str(caught.value)
    assert str(cause) not in str(caught.value)


def test_client_is_closed_when_system_one_and_close_both_fail() -> None:
    inference_error = RuntimeError("inference failed")

    class FailingCloseClient(RecordingSystemOneClient):
        def close(self) -> None:
            super().close()
            raise RuntimeError("cleanup failed")

    client = FailingCloseClient([inference_error])

    with pytest.raises(JevInferenceError) as caught:
        execute_system_one(
            ("state",),
            _questions(),
            backend=TypeSafeBackend(client_factory=lambda: client),
        )

    assert caught.value.__cause__ is None
    assert caught.value.__context__ is None
    assert client.close_count == 1


def test_cleanup_failure_after_success_is_wrapped() -> None:
    cleanup_error = RuntimeError("cleanup failed")

    class FailingCloseClient(RecordingSystemOneClient):
        def close(self) -> None:
            super().close()
            raise cleanup_error

    client = FailingCloseClient([noul_response({"relevant": 0.2, "urgent": 0.1})])

    with pytest.raises(JevInferenceError) as caught:
        execute_system_one(
            ("state",),
            _questions(),
            backend=TypeSafeBackend(client_factory=lambda: client),
        )

    assert caught.value.sample_index == 0
    assert caught.value.__cause__ is None
    assert caught.value.__context__ is None
    assert client.close_count == 1


@pytest.mark.parametrize("max_concurrency", [True, 0, 2, 1.0, "1"])
def test_non_sequential_concurrency_is_rejected_at_execution(max_concurrency: Any) -> None:
    factory = RecordingClientFactory([])
    backend = TypeSafeBackend(max_concurrency=max_concurrency, client_factory=factory)

    with pytest.raises(JevConfigurationError, match="max_concurrency"):
        execute_system_one(("state",), _questions(), backend=backend)

    assert factory.calls == []


def test_default_official_client_is_created_lazily_with_backend_settings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = RecordingSystemOneClient([noul_response({"relevant": 0.2, "urgent": 0.1})])
    creations: list[dict[str, Any]] = []

    def fake_typesafe_client(**kwargs: Any) -> RecordingSystemOneClient:
        creations.append(kwargs)
        return client

    monkeypatch.setattr("typesafe_sdk.TypeSafeClient", fake_typesafe_client)
    retry = RetryPolicy(max_retries=0)
    backend = TypeSafeBackend(timeout=3.0, retry=retry)
    assert creations == []

    responses = execute_system_one(("state",), _questions(), backend=backend)

    assert len(responses) == 1
    assert creations == [{"timeout": 3.0, "retry": retry}]
    assert client.close_count == 1
