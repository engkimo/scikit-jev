"""Structural interfaces shared by System One backends."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from contextlib import AbstractContextManager
from typing import Protocol, TypeAlias

from typesafe_sdk import JSONContent, Question, SystemOneResponse

QuestionMapping: TypeAlias = Mapping[str, Question]
"""Named System One questions in declaration order."""


class SystemOneClient(Protocol):
    """Smallest synchronous client contract required by the execution engine.

    This deliberately matches the common subset implemented by the official
    ``TypeSafeClient`` and the separate ``SystemOneAdapterClient``.  Provider-
    specific retry and timeout configuration belongs at client construction
    time, rather than leaking into the estimator execution path.
    """

    def system_one(
        self,
        state: JSONContent,
        questions: QuestionMapping,
        *,
        model: str | None = None,
    ) -> SystemOneResponse:
        """Answer every named question for one state."""

    def close(self) -> None:
        """Release resources owned by the client."""


SystemOneClientFactory: TypeAlias = Callable[[], SystemOneClient]
"""Clone-safe, no-argument factory for an owned synchronous client."""


class SystemOneBackend(Protocol):
    """Configuration capable of opening one owned client per operation."""

    max_concurrency: int

    def open_client(self) -> AbstractContextManager[SystemOneClient]:
        """Open an owned client that is closed when the context exits."""
