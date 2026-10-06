"""No-network contracts for sklearn model-selection workflows."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, clone
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.pipeline import Pipeline
from typesafe_sdk import Noul

from scikit_jev import JevTransformer
from tests.fakes import FakeNoulAnswer, FakeSystemOneResponse, FakeUsage, SystemOneCall


@dataclass
class _SharedCallLedger:
    """Collect calls made by every backend clone in one model-selection run."""

    calls: list[SystemOneCall] = field(default_factory=list)
    open_count: int = 0
    close_count: int = 0

    def __deepcopy__(self, memo: dict[int, Any]) -> _SharedCallLedger:
        """Let sklearn clones report to the same test-only observation point."""

        del memo
        return self


class _StateDrivenClient:
    """Generate deterministic Noul probabilities from a semantic state."""

    def __init__(self, ledger: _SharedCallLedger) -> None:
        self._ledger = ledger

    def system_one(
        self,
        state: Any,
        questions: Any,
        *,
        model: str | None = None,
    ) -> FakeSystemOneResponse:
        assert isinstance(state, str)
        probability = 0.95 if state.startswith("positive") else 0.05
        self._ledger.calls.append(
            SystemOneCall(
                state=state,
                questions=questions,
                model=model,
                retry=None,
                timeout=None,
                extra_headers=None,
                extra_body=None,
                response_model=None,
            )
        )
        return FakeSystemOneResponse(
            model=model or "jev-model-selection-test",
            answers={question_id: FakeNoulAnswer(probability) for question_id in questions},
            usage=FakeUsage(),
        )

    def close(self) -> None:
        self._ledger.close_count += 1


class _StateDrivenBackend(BaseEstimator):
    """Clone-safe backend with no credentials, state, or network dependency."""

    def __init__(self, ledger: _SharedCallLedger, max_concurrency: int = 1) -> None:
        self.ledger = ledger
        self.max_concurrency = max_concurrency

    @contextmanager
    def open_client(self):
        self.ledger.open_count += 1
        client = _StateDrivenClient(self.ledger)
        try:
            yield client
        finally:
            client.close()


def _pipeline(ledger: _SharedCallLedger) -> Pipeline:
    return Pipeline(
        [
            (
                "jev",
                JevTransformer(
                    {"sentiment": Noul(instructions="Is this state positive?")},
                    backend=_StateDrivenBackend(ledger),
                    model="jev-pinned-test",
                ),
            ),
            ("model", LogisticRegression(C=1_000_000.0, random_state=0)),
        ]
    )


def _dataset() -> tuple[list[str], np.ndarray]:
    states = [
        "negative-0",
        "positive-0",
        "negative-1",
        "positive-1",
        "negative-2",
        "positive-2",
        "negative-3",
        "positive-3",
    ]
    return states, np.asarray([0, 1, 0, 1, 0, 1, 0, 1])


def test_clone_fit_is_network_free_but_fit_transform_calls_once_per_sample() -> None:
    ledger = _SharedCallLedger()
    transformer = clone(_pipeline(ledger).named_steps["jev"])
    states, _ = _dataset()

    returned = transformer.fit(states)

    assert returned is transformer
    assert ledger.calls == []
    assert ledger.open_count == 0

    values = transformer.fit_transform(states)

    np.testing.assert_allclose(values[:, 0], [0.05, 0.95] * 4)
    assert len(ledger.calls) == len(states)
    assert ledger.open_count == 1
    assert ledger.close_count == 1
    assert all(tuple(call.questions) == ("sentiment",) for call in ledger.calls)
    assert all(call.model == "jev-pinned-test" for call in ledger.calls)


def test_cross_val_score_clones_pipeline_and_counts_every_inference_call() -> None:
    ledger = _SharedCallLedger()
    states, targets = _dataset()
    folds = 4

    scores = cross_val_score(
        _pipeline(ledger),
        states,
        targets,
        cv=folds,
        n_jobs=1,
        error_score="raise",
    )

    np.testing.assert_array_equal(scores, np.ones(folds))
    # In every fold, each row is inferred exactly once: in either the training
    # fit_transform or the validation transform.
    assert len(ledger.calls) == len(states) * folds
    assert ledger.open_count == folds * 2
    assert ledger.close_count == ledger.open_count


def test_grid_search_clones_candidates_and_refits_with_exact_call_count() -> None:
    ledger = _SharedCallLedger()
    states, targets = _dataset()
    folds = 4
    candidates = (1_000.0, 1_000_000.0)
    search = GridSearchCV(
        _pipeline(ledger),
        {"model__C": candidates},
        cv=folds,
        n_jobs=1,
        error_score="raise",
        refit=True,
    )

    search.fit(states, targets)

    assert search.best_score_ == 1.0
    assert search.best_estimator_.named_steps["jev"].is_fitted_
    # Each candidate sees every row once per fold across training and scoring.
    # GridSearchCV then fit_transforms every row once for the final refit.
    expected_calls = len(states) * len(candidates) * folds + len(states)
    assert len(ledger.calls) == expected_calls
    expected_operations = len(candidates) * folds * 2 + 1
    assert ledger.open_count == expected_operations
    assert ledger.close_count == expected_operations
