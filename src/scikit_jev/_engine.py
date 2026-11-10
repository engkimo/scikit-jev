"""Ordered synchronous execution of normalized System One requests."""

from __future__ import annotations

from collections.abc import Sequence

from typesafe_sdk import JSONContent

from scikit_jev._results import CanonicalResponse
from scikit_jev._validation import validate_questions, validate_response
from scikit_jev.backends import QuestionMapping, SystemOneBackend, TypeSafeBackend
from scikit_jev.exceptions import JevConfigurationError, JevInferenceError, JevResponseError


def execute_system_one(
    states: Sequence[JSONContent],
    questions: QuestionMapping,
    *,
    backend: SystemOneBackend | None = None,
    model: str | None = None,
) -> tuple[CanonicalResponse, ...]:
    """Evaluate and validate each state before sending the next state.

    At most one ``system_one`` call is made for each state. Every call receives
    the complete question mapping, so adding questions does not multiply the
    number of remote requests.  The client is created once for the operation and
    always closed by its backend. A provider or response-validation failure
    stops later requests, and a cleanup failure never masks that first failure.
    This counts logical client calls; SDK retries may cause more HTTP attempts.

    Provider exceptions are replaced by a safe ``JevInferenceError`` containing
    only the sample index and declared question IDs.  The original exception is
    deliberately neither retained nor chained because provider diagnostics can
    contain response bodies, credentials, or private states.
    """

    state_batch = tuple(states)
    if not state_batch:
        return ()

    selected_backend = TypeSafeBackend() if backend is None else backend
    specs = validate_questions(questions)
    question_ids = tuple(questions)
    responses: list[CanonicalResponse] = []
    current_index = 0
    failure: JevInferenceError | JevResponseError | None = None

    try:
        with selected_backend.open_client() as client:
            for current_index, state in enumerate(state_batch):
                try:
                    response = client.system_one(state, questions, model=model)
                except Exception:
                    # Record a data-free error and leave the exception handler
                    # before raising it.  ``raise ... from None`` would hide the
                    # provider traceback when formatted but still retain it in
                    # ``__context__``.
                    failure = JevInferenceError(current_index, question_ids)
                    break
                try:
                    canonical = validate_response(response, specs)
                except JevResponseError as error:
                    failure = JevResponseError(f"Invalid System One response for sample {current_index}: {error}")
                    break
                responses.append(canonical)
    except JevConfigurationError:
        if failure is None:
            raise
    except Exception:
        # Client creation and cleanup are operation-level failures.  Index zero
        # identifies the first unevaluated sample during setup; after execution
        # it identifies the last sample whose operation could not be completed.
        if failure is None:
            failure = JevInferenceError(current_index, question_ids)

    # Stay outside provider and validation handlers: even a suppressed context
    # could retain untrusted response objects or cleanup diagnostics.
    if failure is not None:
        raise failure

    return tuple(responses)


__all__ = ["execute_system_one"]
