"""Deterministic System One test doubles.

The response objects intentionally expose the same public attributes as the
``typesafe-sdk`` 0.7 response models.  They are small structural fakes rather
than subclasses of SDK models, so unit tests can exercise the backend boundary
without constructing HTTP responses or credentials.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Callable, Iterable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Literal, TypeAlias


@dataclass(frozen=True, slots=True)
class FakeNoulAnswer:
    """A structural fake for ``typesafe_sdk.NoulAnswer``."""

    noul: float
    type: Literal["noul"] = "noul"


@dataclass(frozen=True, slots=True)
class FakeChoiceAnswer:
    """A structural fake for ``typesafe_sdk.ChoiceAnswer``."""

    choice: str
    confidence: float
    probabilities: dict[str, float]
    type: Literal["choice"] = "choice"


@dataclass(frozen=True, slots=True)
class FakeScoreAnswer:
    """A structural fake for ``typesafe_sdk.ScoreAnswer``."""

    score: float
    confidence: float
    legend: dict[int, Any]
    probabilities: dict[int, float]
    type: Literal["score"] = "score"


FakeAnswer: TypeAlias = FakeNoulAnswer | FakeChoiceAnswer | FakeScoreAnswer


@dataclass(frozen=True, slots=True)
class FakeUsage:
    """A structural fake for ``typesafe_sdk.Usage``."""

    input_tokens: int | None = None
    output_tokens: int | None = None


@dataclass(frozen=True, slots=True)
class FakeSystemOneResponse:
    """A structural fake for ``typesafe_sdk.SystemOneResponse``."""

    model: str
    answers: dict[str, FakeAnswer]
    usage: FakeUsage = field(default_factory=FakeUsage)

    @property
    def nouls(self) -> dict[str, FakeNoulAnswer]:
        return {name: answer for name, answer in self.answers.items() if isinstance(answer, FakeNoulAnswer)}

    @property
    def choices(self) -> dict[str, FakeChoiceAnswer]:
        return {name: answer for name, answer in self.answers.items() if isinstance(answer, FakeChoiceAnswer)}

    @property
    def scores(self) -> dict[str, FakeScoreAnswer]:
        return {name: answer for name, answer in self.answers.items() if isinstance(answer, FakeScoreAnswer)}


@dataclass(frozen=True, slots=True)
class SystemOneCall:
    """The arguments observed by one fake ``system_one`` call."""

    state: Any
    questions: Mapping[str, Any]
    model: str | None
    retry: Any
    timeout: Any
    extra_headers: Mapping[str, str] | None
    extra_body: Mapping[str, Any] | None
    response_model: type[Any] | None


ResponseFactory: TypeAlias = Callable[[SystemOneCall], FakeSystemOneResponse]
SystemOneOutcome: TypeAlias = FakeSystemOneResponse | BaseException | ResponseFactory


class RecordingSystemOneClient:
    """Return scripted responses while recording every System One request.

    An outcome may be a response, an exception to raise, or a callable receiving
    the recorded call.  Outcomes are consumed in order, making request-count and
    ordering failures deterministic.
    """

    def __init__(self, outcomes: Iterable[SystemOneOutcome]) -> None:
        self._outcomes = deque(outcomes)
        self.calls: list[SystemOneCall] = []
        self.close_count = 0

    def system_one(
        self,
        state: Any,
        questions: Mapping[str, Any],
        *,
        model: str | None = None,
        retry: Any = None,
        timeout: Any = None,
        extra_headers: Mapping[str, str] | None = None,
        extra_body: Mapping[str, Any] | None = None,
        response_model: type[Any] | None = None,
    ) -> FakeSystemOneResponse:
        call = SystemOneCall(
            state=state,
            questions=questions,
            model=model,
            retry=retry,
            timeout=timeout,
            extra_headers=extra_headers,
            extra_body=extra_body,
            response_model=response_model,
        )
        self.calls.append(call)
        if not self._outcomes:
            raise AssertionError("Unexpected System One call: no scripted outcome remains")

        outcome = self._outcomes.popleft()
        if isinstance(outcome, BaseException):
            raise outcome
        if callable(outcome):
            return outcome(call)
        return outcome

    @property
    def remaining_outcomes(self) -> int:
        """Number of responses or failures that have not yet been consumed."""

        return len(self._outcomes)

    def assert_consumed(self) -> None:
        """Fail when a test did not make every expected request."""

        assert not self._outcomes, f"{len(self._outcomes)} scripted System One outcome(s) were not consumed"

    def close(self) -> None:
        self.close_count += 1

    def __enter__(self) -> RecordingSystemOneClient:
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()


@dataclass(frozen=True, slots=True)
class ClientFactoryCall:
    """Arguments observed by a recording client factory."""

    args: tuple[Any, ...]
    kwargs: Mapping[str, Any]


class RecordingClientFactory:
    """Supply scripted clients and record lazy client construction."""

    def __init__(self, clients: Iterable[RecordingSystemOneClient]) -> None:
        self._clients = deque(clients)
        self.calls: list[ClientFactoryCall] = []
        self.created: list[RecordingSystemOneClient] = []

    def __call__(self, *args: Any, **kwargs: Any) -> RecordingSystemOneClient:
        self.calls.append(ClientFactoryCall(args=args, kwargs=dict(kwargs)))
        if not self._clients:
            raise AssertionError("Unexpected client construction: no scripted client remains")
        client = self._clients.popleft()
        self.created.append(client)
        return client

    @property
    def remaining_clients(self) -> int:
        return len(self._clients)


class RecordingBackend:
    """A backend double with the production ``open_client`` contract.

    The backend owns every client returned by its factory and closes it exactly
    once when the batch context exits, including exceptional exits.
    """

    def __init__(self, client_factory: RecordingClientFactory, *, max_concurrency: int = 1) -> None:
        self.client_factory = client_factory
        self.max_concurrency = max_concurrency
        self.open_count = 0

    @contextmanager
    def open_client(self) -> Iterator[RecordingSystemOneClient]:
        self.open_count += 1
        client = self.client_factory()
        try:
            yield client
        finally:
            client.close()


def noul_response(
    answers: Mapping[str, float],
    *,
    model: str = "jev-test-2026-09-15",
    input_tokens: int | None = 10,
    output_tokens: int | None = 1,
) -> FakeSystemOneResponse:
    """Build a deterministic response containing only Noul answers."""

    return FakeSystemOneResponse(
        model=model,
        answers={name: FakeNoulAnswer(noul=value) for name, value in answers.items()},
        usage=FakeUsage(input_tokens=input_tokens, output_tokens=output_tokens),
    )
