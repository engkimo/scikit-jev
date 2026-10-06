# scikit-jev

> Status: functional MVP, 2026-10-06

`scikit-jev` is a scikit-learn integration for TypeSafe System One and Jev. It
turns typed, probabilistic answers about text or structured JSON into stable
sklearn predictions and feature matrices.

The project is not a generic LLM prompt wrapper. Its core is a semantic feature
engine:

```text
states -> System One questions -> validated probability tensors -> sklearn
```

The primary estimator is `JevTransformer`, which sends all questions for one
sample in a single System One request. Thin task-specific estimators share the
same execution and validation layer:

- `JevNoulClassifier`
- `JevChoiceClassifier`
- `JevScoreRegressor`
- `JevTransformer`

## Quickstart

Install the current source checkout and its test dependencies with:

```console
uv sync --extra test
```

Set `TYPESAFE_API_KEY` in the process environment when using the default
`TypeSafeBackend`. Credentials are resolved lazily by the official SDK; they are
not estimator parameters.

```python
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.pipeline import make_pipeline
from typesafe_sdk import Noul, Score

from scikit_jev import JevTransformer

features = JevTransformer(
    questions={
        "contains_request": Noul(instructions="Does the message contain an actionable request?"),
        "urgency": Score(
            instructions="How urgent is this request?",
            criteria=["not urgent", "soon", "immediate"],
        ),
    },
    score_encoding="probabilities",
    model="jev-1.13.0",
)

pipeline = make_pipeline(features, HistGradientBoostingClassifier())
pipeline.fit(train_states, train_labels)
predictions = pipeline.predict(test_states)
```

`fit()` is intentionally network-free. `fit_transform()`, `transform()`,
`predict()`, and `predict_proba()` perform remote inference.

One input sample produces one System One request containing every declared
question. Consequently, `Pipeline.fit()` normally performs remote inference:
scikit-learn calls the transformer's `fit_transform()` to obtain training
features. Cross-validation and grid search repeat those paid calls.

Call `evaluate(X)` when both numerical values and immutable metadata are needed
from the same inference:

```python
result = features.fit(train_states).evaluate(test_states)
result.values
result.resolved_models
result.usage
result.warnings
```

The implemented public estimators are:

- `JevTransformer` for mixed Noul, Choice, and Score feature schemas
- `JevNoulClassifier` for binary probabilities and thresholded predictions
- `JevChoiceClassifier` for declaration-ordered multiclass probabilities
- `JevScoreRegressor` for probability-weighted scores
- `TypeSafeBackend` for lazy, clone-safe official SDK access

The first implementation is deliberately synchronous and sequential. It has no
implicit persistent cache and never invents a fallback prediction for an invalid
or failed response. Input states are sent to the configured remote backend during
inference; review that backend's data-handling terms before using sensitive data.
Provider exceptions are replaced with coordinate-only errors and are not chained,
because provider bodies and tracebacks may contain private state or credentials.

## Development

```console
.venv/bin/pytest -q
.venv/bin/mypy
ruff check .
ruff format --check .
uv build
```

The deterministic test suite blocks ordinary network access and requires no API
credential. Live tests, when added, must be explicitly opted into.

## Design documents

- [Architecture](docs/architecture.md)
- [Implementation plan](docs/implementation-plan.md)

## Evidence behind the design

The design follows the public contract of
[`typesafe-sdk` 0.7.2](https://github.com/typesafe-ai/typesafe-sdk-python/tree/f078f1e208a0d885154dc758344ae4fce77ac168).
TypeSafe's official
[AutoResearch feature-discovery cookbook](https://docs.typesafe.ai/cookbooks/autoresearch_feature_discovery)
already demonstrates the underlying pattern: one request per state, all questions
packed into that request, probability-preserving feature encodings, and a
classical model downstream. `scikit-jev` turns that pattern into reusable,
testable sklearn estimators.
