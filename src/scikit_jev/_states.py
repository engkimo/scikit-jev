"""Convert sklearn-style input containers into System One JSON states.

The outer value is always a sample container.  In particular, a plain string
or mapping is rejected so that accidentally passing one sample cannot silently
turn into a sequence of characters or mapping keys.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import TYPE_CHECKING, Any, TypeAlias

import numpy as np

from .exceptions import JevConfigurationError

if TYPE_CHECKING:
    from typesafe_sdk import JSONContent
else:
    JSONContent = Any

StateBuilder: TypeAlias = str | Callable[[Any], JSONContent]


def build_states(X: Any, state_builder: StateBuilder = "auto") -> tuple[JSONContent, ...]:
    """Materialize *X* and return one JSON-compatible state per sample.

    ``state_builder="auto"`` applies only lossless, documented conversions:
    NumPy scalars become Python scalars, arrays and sequences become lists, and
    mappings become dictionaries with string keys.  It never stringifies an
    unsupported object.  A callable is invoked once per raw sample and its
    result is subjected to the same JSON validation.

    Parameters
    ----------
    X:
        A one-dimensional sample container, a two-dimensional NumPy array, or a
        pandas DataFrame.  A scalar string or mapping is deliberately invalid.
    state_builder:
        ``"auto"`` or a callable accepting one sample and returning a string,
        JSON object, or JSON array.
    """

    builder = _validate_state_builder(state_builder)
    samples = _materialize_samples(X)
    states: list[JSONContent] = []
    for sample_index, sample in enumerate(samples):
        if builder is not None:
            try:
                sample = builder(sample)
            except Exception:
                # A builder exception may contain a repr of private input data.
                raise JevConfigurationError(f"state_builder failed for sample {sample_index}") from None
        try:
            state = _normalize_json_content(sample, context=f"state for sample {sample_index}")
        except JevConfigurationError:
            # _normalize_json_content reports only the safe context supplied
            # above, never a value, mapping key, or object repr.
            raise
        states.append(state)
    return tuple(states)


# The longer name is useful at call sites that emphasize eager realization and
# keeps the boundary unsurprising for backend/engine code.
materialize_states = build_states


def _validate_state_builder(state_builder: StateBuilder) -> Callable[[Any], JSONContent] | None:
    if isinstance(state_builder, str):
        if state_builder == "auto":
            return None
        raise JevConfigurationError("state_builder must be 'auto' or a callable")
    if callable(state_builder):
        return state_builder
    raise JevConfigurationError("state_builder must be 'auto' or a callable")


def _materialize_samples(X: Any) -> list[Any]:
    if isinstance(X, (str, bytes, bytearray)) or isinstance(X, Mapping):
        raise JevConfigurationError("X must be a container of samples; wrap a single state in a list")
    if _is_sparse(X):
        raise JevConfigurationError("sparse input is not supported")
    if _is_dataframe(X):
        # orient="records" preserves DataFrame column order and makes the
        # conversion independent of pandas' Series scalar indexing behavior.
        return list(X.to_dict(orient="records"))
    if isinstance(X, np.ndarray):
        if X.ndim == 0:
            raise JevConfigurationError("X must be a container of samples, not a scalar array")
        if X.ndim == 1:
            return list(X)
        if X.ndim == 2:
            return [row for row in X]
        raise JevConfigurationError("X must be one- or two-dimensional")
    if not isinstance(X, Iterable):
        raise JevConfigurationError("X must be an iterable container of samples")
    try:
        return list(X)
    except Exception:
        raise JevConfigurationError("X could not be materialized as a sample container") from None


def _is_dataframe(value: Any) -> bool:
    cls = type(value)
    return cls.__name__ == "DataFrame" and cls.__module__.split(".", 1)[0] == "pandas"


def _is_sparse(value: Any) -> bool:
    cls = type(value)
    return cls.__module__.startswith("scipy.sparse") or (
        hasattr(value, "format") and hasattr(value, "tocsr") and hasattr(value, "nnz")
    )


def _normalize_json_content(value: Any, *, context: str) -> JSONContent:
    """Return a JSON-content value without coercing arbitrary objects to text.

    ``context`` must be a safe description chosen by the caller.  The error
    deliberately excludes the rejected value and nested mapping keys.
    """

    normalized = _normalize_json_value(value, context=context, active=set())
    if not isinstance(normalized, (str, dict, list)):
        raise JevConfigurationError(f"{context} must be a string, JSON object, or JSON array")
    return normalized


def _normalize_json_value(value: Any, *, context: str, active: set[int]) -> Any:
    if isinstance(value, np.generic):
        value = value.item()
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise JevConfigurationError(f"{context} contains a non-finite number")
        return value
    if isinstance(value, (bytes, bytearray, memoryview)):
        raise JevConfigurationError(f"{context} contains a binary value, which is not JSON-compatible")

    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            return _normalize_json_value(value.item(), context=context, active=active)
        value = value.tolist()

    if isinstance(value, Mapping):
        object_id = id(value)
        if object_id in active:
            raise JevConfigurationError(f"{context} contains a reference cycle")
        active.add(object_id)
        try:
            result: dict[str, Any] = {}
            for key, item in value.items():
                if not isinstance(key, str):
                    raise JevConfigurationError(f"{context} contains a JSON object with a non-string key")
                result[key] = _normalize_json_value(item, context=context, active=active)
            return result
        finally:
            active.remove(object_id)

    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        object_id = id(value)
        if object_id in active:
            raise JevConfigurationError(f"{context} contains a reference cycle")
        active.add(object_id)
        try:
            return [_normalize_json_value(item, context=context, active=active) for item in value]
        finally:
            active.remove(object_id)

    raise JevConfigurationError(f"{context} contains a value that is not JSON-compatible")
