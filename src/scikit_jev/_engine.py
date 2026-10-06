"""Ordered synchronous execution of normalized System One requests."""

from __future__ import annotations

from collections.abc import Sequence

from typesafe_sdk import JSONContent, SystemOneResponse

from scikit_jev.backends import QuestionMapping, SystemOneBackend, TypeSafeBackend
from scikit_jev.exceptions import JevConfigurationError, JevInferenceError


def execute_system_one(
    states: Sequence[JSONContent],
    questions: QuestionMapping,
    *,
    backend: SystemOneBackend | None = None,
    model: str | None = None,
) -> tuple[SystemOneResponse, ...]:
    """Evaluate normalized states sequentially and preserve their input order.

    Exactly one ``system_one`` call is made for each state.  Every call receives
    the complete question mapping, so adding questions does not multiply the
    number of remote requests.  The client is created once for the operation and
    always closed by its backend.

    Provider exceptions are replaced by a safe ``JevInferenceError`` containing
    only the sample index and declared question IDs.  The original exception is
    deliberately neither retained nor chained because provider diagnostics can
    contain response bodies, credentials, or private states.
    """

    state_batch = tuple(states)
    if not state_batch:
        return ()

    selected_backend = TypeSafeBackend() if backend is None else backend
    question_ids = tuple(questions)
    responses: list[SystemOneResponse] = []
    current_index = 0
    failure: JevInferenceError | None = None

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
                responses.append(response)
    except JevConfigurationError:
        raise
    except Exception:
        # Client creation and cleanup are operation-level failures.  Index zero
        # identifies the first unevaluated sample during setup; after execution
        # it identifies the last sample whose operation could not be completed.
        failure = JevInferenceError(current_index, question_ids)

    # This raise must stay outside every provider-exception handler so Python
    # does not attach the original exception as an implicit ``__context__``.
    if failure is not None:
        raise failure

    return tuple(responses)


__all__ = ["execute_system_one"]
