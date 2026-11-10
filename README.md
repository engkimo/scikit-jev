# scikit-jev

> Status: pre-alpha functional MVP, 2026-11-10

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

Install from GitHub (the package is not yet published to PyPI):

```console
pip install "scikit-jev @ git+https://github.com/engkimo/scikit-jev.git"
```

Install the current source checkout and its test dependencies with:

```console
uv sync --extra test
```

Try the credential-free example from a source checkout before configuring a
remote provider:

```console
python -m pip install .
python examples/offline_workflow.py
```

It shows a mixed Noul/Choice/Score transformer, a logistic-regression pipeline,
named features, immutable result metadata, and logical call counts. The example
uses hand-authored responses for synthetic support messages: no API key, network
inference, or billing, and no claim of semantic-model accuracy. It prints a
`[2, 6]` evaluation feature shape, zero calls for `fit`, eight calls for pipeline
training, and two calls for evaluation; unknown token counts stay `null`.

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
    model="jev-latest",  # Moving alias; inspect result.resolved_models after inference.
)

pipeline = make_pipeline(features, HistGradientBoostingClassifier())
pipeline.fit(train_states, train_labels)
predictions = pipeline.predict(test_states)
```

`fit()` is intentionally network-free. `fit_transform()`, `transform()`,
`predict()`, and `predict_proba()` perform remote inference.

Each evaluated sample makes one logical `system_one` call containing every
declared question; SDK retries may create multiple HTTP attempts. Consequently,
`Pipeline.fit()` normally performs remote inference:
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
- `JevChoiceClassifier` for multiclass probabilities in sorted `classes_` order
- `JevScoreRegressor` for probability-weighted scores
- `TypeSafeBackend` for lazy, clone-safe official SDK access

The first implementation is deliberately synchronous and sequential. It has no
implicit persistent cache and never invents a fallback prediction for an invalid
or failed response. Input states are sent to the configured remote backend during
inference; review that backend's data-handling terms before using sensitive data.
Provider exceptions are replaced with coordinate-only errors and are not chained,
because provider bodies and tracebacks may contain private state or credentials.

## Contracts and current limits

Transformer features and canonical answer distributions follow declared question,
label, and rubric order. Classifiers instead sort `classes_` and align every
`predict_proba` and `evaluate().values` column to that order, as sklearn expects.
Choice ties select the first class in sorted order. Noul thresholds always apply
to the configured positive label, even when that label sorts first.

Each response is validated before sending the next sample. Invalid distributions,
missing or extra answers, and inconsistent answer types or scalars raise an
indexed error. When an SDK response exposes its raw HTTP response, the original
answer IDs are also checked, including unknown answer types dropped by SDK
decoding. Structural clients without raw metadata can only be checked against
the answers they expose. Raw HTTP payloads are not retained in canonical results.

Ordinary library response errors report declared IDs, fixed field descriptions,
and counts instead of arbitrary response keys or values. These guarantees cover
library errors and canonical results, not third-party logging or arbitrary custom
client behavior; the SDK can log details during decoding before validation runs.

Model aliases may change. Recording a resolved model version does not establish
that the service accepts it as a request model or guarantees replay. The current
probability-sum and scalar tolerances are `1e-6`; broader handling of quantized
probabilities remains an open compatibility question, not a validated live-service
guarantee. The [architecture](docs/architecture.md) documents the exact contracts
and links to deterministic regression tests.

The sorted classifier columns change the earlier pre-alpha behavior. Refit
classifiers created with older source revisions; align any code consuming their
probability columns to `classes_`. Transformer feature order is unchanged.

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

## Related projects

[`sklearn-decision`](https://github.com/jaketimothy/sklearn-decision) also provides
decision-model estimators and question-derived features for sklearn, with local
models, calibration, and caching. [`tab-jev`](https://github.com/edamame-labs/tab-jev)
combines Jev judgments with tabular predictors, while
[`jevframe`](https://github.com/ktaletsk/jevframe) adds typed row evaluation to
pandas and Polars. See the [dated landscape review](docs/landscape.md) for scope
and evidence. No performance comparison against these projects has been run.

## License

MIT. See [LICENSE](LICENSE). This is an independent community project;
the software license does not grant access to the hosted TypeSafe service or
rights to third-party models or trademarks.
