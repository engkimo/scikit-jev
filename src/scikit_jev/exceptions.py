"""Public exceptions raised by scikit-jev."""

from __future__ import annotations

from collections.abc import Sequence


class JevError(Exception):
    """Base exception for scikit-jev failures."""


class JevConfigurationError(JevError, ValueError):
    """An estimator, question, state, or feature configuration is invalid."""


class JevResponseError(JevError, ValueError):
    """A backend response violates the declared question or probability schema."""


class JevInferenceError(JevError):
    """A backend failed while evaluating one input sample.

    The exception intentionally stores only the sample index and question IDs.
    Provider exceptions are not retained or chained because their messages,
    response bodies, and tracebacks may contain credentials or private states.
    """

    def __init__(self, sample_index: int, question_ids: Sequence[str]) -> None:
        self.sample_index = sample_index
        self.question_ids = tuple(question_ids)
        # Keep reconstructible coordinates in Exception.args so ordinary
        # exception pickling round-trips without serializing provider data.
        super().__init__(sample_index, self.question_ids)

    def __str__(self) -> str:
        joined = ", ".join(repr(question_id) for question_id in self.question_ids)
        return f"System One inference failed for sample {self.sample_index}; questions: {joined}"


class JevBatchError(JevError):
    """Reserved for a future batch result containing multiple safe failures."""
