from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import pytest

from scikit_jev._states import build_states, materialize_states
from scikit_jev.exceptions import JevConfigurationError


@pytest.mark.parametrize("X", ["one state", {"message": "one state"}, b"one state", np.array("one state")])
def test_scalar_outer_containers_are_rejected(X: object) -> None:
    with pytest.raises(JevConfigurationError, match="container of samples|scalar array"):
        build_states(X)


def test_one_dimensional_container_treats_each_element_as_a_state() -> None:
    assert build_states(["first", {"message": "second"}, ["third", 3]]) == (
        "first",
        {"message": "second"},
        ["third", 3],
    )


def test_two_dimensional_array_converts_each_row_to_a_json_array() -> None:
    X = np.array([[1, 2], [3, 4]], dtype=np.int64)

    assert materialize_states(X) == ([1, 2], [3, 4])


def test_dataframe_converts_rows_to_records_in_column_order() -> None:
    X = pd.DataFrame({"message": ["a", "b"], "count": np.array([1, 2], dtype=np.int64)})

    assert build_states(X) == (
        {"message": "a", "count": 1},
        {"message": "b", "count": 2},
    )


@dataclass
class Ticket:
    body: str
    priority: int


def test_callable_builder_receives_each_sample() -> None:
    X = [Ticket("a", 1), Ticket("b", 2)]

    states = build_states(X, lambda ticket: {"body": ticket.body, "priority": ticket.priority})

    assert states == ({"body": "a", "priority": 1}, {"body": "b", "priority": 2})


def test_numpy_values_are_losslessly_converted_inside_states() -> None:
    states = build_states([{"count": np.int64(3), "scores": np.array([0.25, 0.75])}])

    assert states == ({"count": 3, "scores": [0.25, 0.75]},)


@pytest.mark.parametrize(
    "state",
    [
        42,
        {"value": float("nan")},
        {"value": float("inf")},
        {1: "non-string key"},
        {"value": object()},
        b"binary",
    ],
)
def test_auto_builder_rejects_non_json_states_without_stringifying(state: object) -> None:
    with pytest.raises(JevConfigurationError):
        build_states([state])


def test_state_errors_do_not_echo_private_values_or_keys() -> None:
    secret = "do-not-echo-this-state"

    with pytest.raises(JevConfigurationError) as exc_info:
        build_states([{"private-key": object(), "secret": secret}])

    assert secret not in str(exc_info.value)
    assert "private-key" not in str(exc_info.value)


def test_builder_failures_do_not_echo_builder_exception_text() -> None:
    def fail(_: object) -> object:
        raise RuntimeError("private input appeared here")

    with pytest.raises(JevConfigurationError, match="sample 0") as exc_info:
        build_states(["state"], fail)

    assert "private input" not in str(exc_info.value)


def test_reference_cycles_are_rejected() -> None:
    state: list[object] = []
    state.append(state)

    with pytest.raises(JevConfigurationError, match="reference cycle"):
        build_states([state])


def test_empty_input_has_no_states() -> None:
    assert build_states([]) == ()


def test_state_builder_must_be_auto_or_callable() -> None:
    with pytest.raises(JevConfigurationError, match="state_builder"):
        build_states(["state"], state_builder="stringify")
