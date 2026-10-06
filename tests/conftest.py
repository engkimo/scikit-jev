"""Global pytest safety fixtures."""

from __future__ import annotations

import os
import socket
from collections.abc import Iterator

import pytest

_LIVE_ENV = "SCIKIT_JEV_RUN_LIVE"


def _live_enabled() -> bool:
    return os.environ.get(_LIVE_ENV) == "1"


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """Keep credentialed, paid tests opt-in even when collected explicitly."""

    if _live_enabled():
        return
    skip = pytest.mark.skip(reason=f"live test requires {_LIVE_ENV}=1")
    for item in items:
        if item.get_closest_marker("live") is not None:
            item.add_marker(skip)


@pytest.fixture(autouse=True)
def _forbid_network(monkeypatch: pytest.MonkeyPatch, request: pytest.FixtureRequest) -> Iterator[None]:
    """Fail ordinary tests immediately if they attempt a network connection.

    Local Unix-domain sockets remain available for process-based sklearn and
    joblib tests; they cannot reach a remote provider.
    """

    is_enabled_live_test = request.node.get_closest_marker("live") is not None and _live_enabled()
    if is_enabled_live_test:
        yield
        return

    # Do not let a developer's real credential affect a unit test or appear in
    # a failure produced by accidentally constructing the official client.
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)

    def blocked(*_: object, **__: object) -> None:
        raise AssertionError("Network access is forbidden in non-live scikit-jev tests")

    original_connect = socket.socket.connect
    original_connect_ex = socket.socket.connect_ex
    unix_family = getattr(socket, "AF_UNIX", object())

    def guarded_connect(sock: socket.socket, address: object) -> object:
        if sock.family == unix_family:
            return original_connect(sock, address)  # type: ignore[arg-type]
        return blocked()

    def guarded_connect_ex(sock: socket.socket, address: object) -> object:
        if sock.family == unix_family:
            return original_connect_ex(sock, address)  # type: ignore[arg-type]
        return blocked()

    monkeypatch.setattr(socket, "create_connection", blocked)
    monkeypatch.setattr(socket.socket, "connect", guarded_connect)
    monkeypatch.setattr(socket.socket, "connect_ex", guarded_connect_ex)
    yield
