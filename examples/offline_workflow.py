"""Run a credential-free, synthetic sklearn workflow: python examples/offline_workflow.py.

The client returns hand-authored probabilities for fixed synthetic examples.
No semantic model runs, and predictions are not evidence of task accuracy.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, ClassVar

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from typesafe_sdk import Choice, ChoiceAnswer, Noul, NoulAnswer, Question, Score, ScoreAnswer, SystemOneResponse, Usage

from scikit_jev import JevTransformer, TypeSafeBackend


@dataclass(frozen=True)
class SyntheticExample:
    text: str
    priority: str
    refund: float
    billing: float
    urgency: tuple[float, float, float]

    def state(self) -> dict[str, str]:
        return {"text": self.text, "channel": "synthetic-demo"}


TRAIN = (
    SyntheticExample("Please send a copy of my invoice.", "standard", 0.05, 0.95, (0.8, 0.15, 0.05)),
    SyntheticExample("How do I change my billing address?", "standard", 0.05, 0.90, (0.75, 0.2, 0.05)),
    SyntheticExample("Where can I find the API documentation?", "standard", 0.02, 0.10, (0.8, 0.15, 0.05)),
    SyntheticExample("Can I change the dashboard theme?", "standard", 0.02, 0.10, (0.85, 0.1, 0.05)),
    SyntheticExample("The service is down for our entire team.", "urgent", 0.05, 0.05, (0.05, 0.1, 0.85)),
    SyntheticExample("Payment failed and our account is locked.", "urgent", 0.15, 0.85, (0.05, 0.15, 0.8)),
    SyntheticExample("Refund the duplicate charge today, please.", "urgent", 0.95, 0.95, (0.05, 0.15, 0.8)),
    SyntheticExample("Production API calls fail on every request.", "urgent", 0.05, 0.05, (0.05, 0.1, 0.85)),
)
EVALUATION = (
    SyntheticExample("I need last month's invoice when convenient.", "standard", 0.05, 0.95, (0.8, 0.15, 0.05)),
    SyntheticExample("We cannot access the service right now.", "urgent", 0.05, 0.05, (0.05, 0.1, 0.85)),
)
EXAMPLES = {example.text: example for example in (*TRAIN, *EVALUATION)}
RUBRIC = ["can wait", "needs attention soon", "needs immediate attention"]
QUESTIONS: dict[str, Question] = {
    "refund": Noul(instructions="Does the message request a refund?"),
    "topic": Choice(instructions="Which topic fits?", criteria={"technical": None, "billing": None}),
    "urgency": Score(instructions="How urgent is the request?", criteria=RUBRIC),
}


class SyntheticClient:
    """A local test double, not an alternative model or a prediction fallback.

    The class-level counters make logical calls and cleanup visible in this
    sequential demo. Estimators hold only the factory class, never a live client.
    Unknown examples fail instead of generating invented probabilities.
    """

    calls: ClassVar[int] = 0
    opens: ClassVar[int] = 0
    closes: ClassVar[int] = 0

    def __init__(self) -> None:
        type(self).opens += 1

    def system_one(
        self, state: Any, questions: Mapping[str, Question], *, model: str | None = None
    ) -> SystemOneResponse:
        if tuple(questions) != tuple(QUESTIONS):
            raise ValueError("This demo supports only its declared question bank")
        example = EXAMPLES[state["text"]]
        type(self).calls += 1
        return SystemOneResponse(
            model="synthetic-demo-v1",
            usage=Usage(),  # Unknown tokens remain unknown; no provider is called.
            answers={
                "refund": NoulAnswer(noul=example.refund),
                "topic": ChoiceAnswer(
                    choice="billing" if example.billing > 0.5 else "technical",
                    confidence=max(example.billing, 1 - example.billing),
                    probabilities={"billing": example.billing, "technical": 1 - example.billing},
                ),
                "urgency": ScoreAnswer(
                    score=sum(level * probability for level, probability in enumerate(example.urgency)),
                    confidence=max(example.urgency),
                    probabilities=dict(enumerate(example.urgency)),
                    legend=dict(enumerate(RUBRIC)),
                ),
            },
        )

    def close(self) -> None:
        type(self).closes += 1


def run_demo() -> dict[str, Any]:
    SyntheticClient.calls = SyntheticClient.opens = SyntheticClient.closes = 0
    train_states = [example.state() for example in TRAIN]
    evaluation_states = [example.state() for example in EVALUATION]
    features = JevTransformer(QUESTIONS, backend=TypeSafeBackend(client_factory=SyntheticClient))

    features.fit(train_states)
    fit_calls = SyntheticClient.calls
    pipeline = Pipeline([("features", features), ("classifier", LogisticRegression())])
    pipeline.fit(train_states, [example.priority for example in TRAIN])
    training_calls = SyntheticClient.calls - fit_calls

    # Get features and metadata in one evaluation. The already-fitted classical
    # head predicts locally; no second semantic evaluation is needed.
    result = pipeline.named_steps["features"].evaluate(evaluation_states)
    predictions = pipeline.named_steps["classifier"].predict(result.values)
    return {
        "mode": "Synthetic offline contract demo; no semantic model or accuracy benchmark",
        "feature_names": features.get_feature_names_out().tolist(),
        "feature_shape": list(result.values.shape),
        "features": result.values.tolist(),
        "illustrative_predictions": predictions.tolist(),
        "resolved_models": list(result.resolved_models),
        "usage": {
            "evaluation_logical_calls": result.usage.request_count,
            "input_tokens": result.usage.input_tokens,
            "output_tokens": result.usage.output_tokens,
        },
        "execution": {
            "fit_logical_calls": fit_calls,
            "pipeline_training_logical_calls": training_calls,
            "total_logical_calls": SyntheticClient.calls,
            "client_opens": SyntheticClient.opens,
            "client_closes": SyntheticClient.closes,
        },
    }


if __name__ == "__main__":
    print(json.dumps(run_demo(), indent=2))
