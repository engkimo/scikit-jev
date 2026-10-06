"""Safety and hierarchy contracts for public exceptions."""

from __future__ import annotations

import pickle

import pytest

from scikit_jev.exceptions import (
    JevBatchError,
    JevConfigurationError,
    JevError,
    JevInferenceError,
    JevResponseError,
)


@pytest.mark.parametrize(
    ("exception_type", "standard_base"),
    [
        (JevConfigurationError, ValueError),
        (JevResponseError, ValueError),
    ],
)
def test_validation_errors_have_library_and_standard_bases(
    exception_type: type[JevError], standard_base: type[Exception]
) -> None:
    assert issubclass(exception_type, JevError)
    assert issubclass(exception_type, standard_base)


def test_all_public_errors_share_the_library_base() -> None:
    assert issubclass(JevInferenceError, JevError)
    assert issubclass(JevBatchError, JevError)


def test_inference_error_reports_only_safe_request_coordinates() -> None:
    private_state = "private-customer-message-71e4"
    secret = "typesafe-secret-token-91a2"

    error = JevInferenceError(sample_index=7, question_ids=("spam", "tone"))

    assert error.sample_index == 7
    assert error.question_ids == ("spam", "tone")
    assert "sample 7" in str(error)
    assert "'spam'" in str(error)
    assert "'tone'" in str(error)
    for rendered in (str(error), repr(error), repr(error.args)):
        assert private_state not in rendered
        assert secret not in rendered
        assert "provider failed" not in rendered


def test_inference_error_pickle_round_trip_contains_only_safe_coordinates() -> None:
    error = JevInferenceError(sample_index=0, question_ids=("target",))

    serialized = pickle.dumps(error)
    restored = pickle.loads(serialized)

    assert error.args == (0, ("target",))
    assert restored.sample_index == 0
    assert restored.question_ids == ("target",)
    assert str(restored) == str(error)
    assert b"private provider response body" not in serialized
