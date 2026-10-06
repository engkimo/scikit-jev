"""Official TypeSafe System One backend configuration."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from sklearn.base import BaseEstimator
from typesafe_sdk import RetryPolicy

from scikit_jev.backends._protocol import SystemOneClient, SystemOneClientFactory
from scikit_jev.exceptions import JevConfigurationError


class TypeSafeBackend(BaseEstimator):
    """Clone-safe configuration for the official synchronous TypeSafe client.

    Construction is deliberately side-effect free.  The official client is
    imported and instantiated only when an inference operation opens a client.
    Authentication therefore remains in the SDK's environment-based
    configuration and never becomes an estimator parameter.

    ``client_factory`` is an advanced injection seam for tests and compatible
    clients.  It is called without arguments and the returned client is owned
    and closed by this backend.  A custom factory is responsible for applying
    any desired timeout or retry configuration itself.
    """

    def __init__(
        self,
        *,
        timeout: float | None = None,
        retry: RetryPolicy | None = None,
        max_concurrency: int = 1,
        client_factory: SystemOneClientFactory | None = None,
    ) -> None:
        self.timeout = timeout
        self.retry = retry
        self.max_concurrency = max_concurrency
        self.client_factory = client_factory

    def _validate_configuration(self) -> None:
        if isinstance(self.max_concurrency, bool) or not isinstance(self.max_concurrency, int):
            raise JevConfigurationError("max_concurrency must be the integer 1 in the sequential implementation")
        if self.max_concurrency != 1:
            raise JevConfigurationError("max_concurrency must be 1; concurrent execution is not implemented yet")
        if self.client_factory is not None and not callable(self.client_factory):
            raise JevConfigurationError("client_factory must be callable or None")

    def _create_client(self) -> SystemOneClient:
        self._validate_configuration()
        if self.client_factory is not None:
            return self.client_factory()

        # Keep SDK configuration, environment lookup, and HTTP object creation
        # out of __init__, sklearn cloning, repr, and parameter inspection.
        from typesafe_sdk import TypeSafeClient

        return TypeSafeClient(timeout=self.timeout, retry=self.retry)

    @contextmanager
    def open_client(self) -> Iterator[SystemOneClient]:
        """Yield one lazily created, operation-owned client and close it once."""

        client = self._create_client()
        primary_error: BaseException | None = None
        try:
            yield client
        except BaseException as error:
            primary_error = error
            raise
        finally:
            try:
                client.close()
            except Exception:
                # Preserve the inference failure when cleanup also fails.  A
                # cleanup error is propagated normally when it is the only
                # failure from the operation.
                if primary_error is None:
                    raise

    def __sklearn_tags__(self) -> Any:
        """Use BaseEstimator's standard tags without claiming estimator roles."""

        return super().__sklearn_tags__()
