# scikit-jev architecture

Status: functional MVP and contract corrections implemented; operational hardening deferred
Updated: 2026-11-10
SDK baseline: `typesafe-sdk>=0.7.2,<0.8`

## 1. Product thesis

The essential product is not "an LLM behind `predict()`". It is a compiler from
a declared semantic schema—System One questions—to a stable numerical schema
understood by scikit-learn.

```text
                         one request per sample
X (states) ──> state conversion ──> System One execution ──> response validation
                                                               │
                                                               v
                                                     canonical answer batch
                                                       /        |        \
                                                      /         |         \
                                          feature matrix   predictions   metadata
                                           Transformer     Clf / Regr     + usage
```

The workflow itself is shared with `sklearn-decision` and adjacent libraries.
The selected direction is to test the value of explicit feature-schema,
probability, sklearn-contract, request-accounting, and failure semantics. These
contracts are a product hypothesis, not evidence of adoption or superior accuracy.

## 2. Ground truth from the official API

The design is based on the public Python SDK at commit
[`f078f1e`](https://github.com/typesafe-ai/typesafe-sdk-python/tree/f078f1e208a0d885154dc758344ae4fce77ac168)
and the official primitive documentation for
[`Noul`](https://docs.typesafe.ai/primitives/noul),
[`Choice`](https://docs.typesafe.ai/primitives/choice), and
[`Score`](https://docs.typesafe.ai/primitives/score).

`TypeSafeClient.system_one()` accepts one JSON-compatible `state`, a non-empty
mapping of named questions, and an optional model. It returns one answer per
question plus the resolved model and usage. There is no public multi-state batch
endpoint.

| Question | Answer | sklearn projection |
| --- | --- | --- |
| `Noul` | probability of true | binary `predict_proba`; one non-redundant feature |
| `Choice` | selected label, confidence, full label distribution | multiclass `predict_proba`; one feature per label |
| `Score` | expected level, confidence, full level distribution | regression prediction; configurable numerical features |

The official
[AutoResearch cookbook](https://docs.typesafe.ai/cookbooks/autoresearch_feature_discovery)
confirms the intended feature workflow: one call per row, all row questions in
that call, bounded thread concurrency, Noul probability features, Score
mean/spread or full-distribution features, and a classical model downstream.

## 3. Goals and non-goals

### Goals

1. Preserve Jev probabilities and declared feature/rubric order; align classifier
   probability columns with sorted sklearn class labels.
2. Make one System One call per sample and pack every question for that sample.
3. Behave predictably under `clone`, `Pipeline`, CV, `GridSearchCV`, joblib, and
   serialization.
4. Make invalid or partial responses explicit; never invent an answer.
5. Keep credentials and live HTTP objects out of estimator parameters, reprs,
   exceptions, logs, and pickles.
6. Support the official client and System One-compatible clients through a narrow
   structural interface.
7. Expose immutable inference metadata without mutating estimators during
   prediction.

### Non-goals for the first release

- Generic chat completion or prompt orchestration.
- Fine-tuning, few-shot example selection, or automatic question generation.
- Hiding network cost or pretending remote inference is deterministic.
- An async sklearn estimator.
- A distributed request scheduler.
- Silent fallback to random, majority, zero, or 0.5 outputs.
- Heterogeneous answers forced into a classifier or regressor contract; mixed
  question sets belong in the transformer.

## 4. Public API

The distribution name is `scikit-jev`; the import package is `scikit_jev`.

```python
JevNoulClassifier(
    question,
    *,
    question_id="target",
    model=None,
    backend=None,
    threshold=0.5,
    negative_label=False,
    positive_label=True,
    state_builder="auto",
)

JevChoiceClassifier(
    question,
    *,
    question_id="target",
    model=None,
    backend=None,
    state_builder="auto",
)

JevScoreRegressor(
    question,
    *,
    question_id="target",
    model=None,
    backend=None,
    state_builder="auto",
)

JevTransformer(
    questions,
    *,
    model=None,
    backend=None,
    state_builder="auto",
    score_encoding="probabilities",
    include_confidence=False,
)

TypeSafeBackend(
    *,
    timeout=None,
    retry=None,
    max_concurrency=1,
    client_factory=None,
)
```

`model=None` deliberately inherits the official SDK configuration. Use a concrete
request model only when the selected service documents that it accepts that ID.
Aliases such as `jev-latest` can move. Every rich result records the resolved
`response.model`, but recording that value neither proves it is requestable nor
guarantees replay. Live fixed-version support remains unverified.

All constructor parameters are assigned verbatim. Validation, copying, client
creation, and network access happen later so sklearn cloning remains valid.

### Rich result API

Standard sklearn methods return only their conventional arrays. Each estimator
also exposes an `evaluate(X)` method returning an immutable `JevBatchResult`:

```python
result = transformer.evaluate(X)
result.values
result.responses
result.resolved_models
result.usage
result.warnings
```

`transform`, `predict`, and `predict_proba` project from the same internal result
model, but do not store `last_result_`. A caller that needs values and metadata
together calls `evaluate()` once, avoiding a second paid request.

The result does not retain the input state. Raw HTTP bodies/debug traces are not
included by default.

## 5. Feature schema

Question insertion order is authoritative. Within a question, criterion/rubric
order is authoritative. Response dictionary order is never used.

Default `JevTransformer` encoding:

| Type | Default columns |
| --- | --- |
| Noul | `{question}__p_true` |
| Choice | `{question}__p[{label}]` for every declared label |
| Score | `{question}__p[{level}]` for every declared level |

Noul needs one column because `p_false = 1 - p_true`; no information is lost.

`score_encoding` supports:

- `"probabilities"`: every level probability; default and lossless.
- `"expected"`: the probability-weighted level.
- `"expected_and_std"`: expected level plus distribution standard deviation,
  matching the useful compact representation in the official cookbook.

`include_confidence=True` adds the reported Choice/Score confidence as an
additional column. Confidence is still retained in rich results when not included
as a feature.

`get_feature_names_out()` returns the exact, stable column order. Labels are
escaped by a documented reversible encoder when necessary; simple labels remain
readable.

Task estimators use these contracts:

- Both classifiers sort `classes_` numerically or lexicographically and align
  `predict_proba` and `evaluate().values` with it. Noul labels must be distinct,
  reflexive, mutually sortable scalar values; incompatible mixed label types are
  rejected instead of silently converted to strings.
- Noul maps `p_true` to the configured positive label's column and its complement
  to the negative label's column. `p_true >= threshold` is positive regardless of
  where that label sorts. Boolean defaults still produce `[1-p_true, p_true]`.
- Choice requests, canonical answer distributions, and Transformer features keep
  declared criteria order; only classifier output is remapped to sorted classes.
  The returned `choice` must agree with an argmax, allowing a reported tie;
  otherwise validation fails. Classifier prediction breaks ties using the first
  sorted class.
- `JevScoreRegressor.predict` returns the reported expected score as `float64`.
  It must agree with the weighted distribution within tolerance.

## 6. Input and fit semantics

The outer object passed as `X` always represents samples.

- A sequence/one-dimensional array: each element is one state.
- A two-dimensional NumPy array: each row becomes a JSON array.
- A DataFrame: each row becomes a JSON object keyed by column name.
- A scalar string or mapping is rejected; one sample must be written `[state]`.
- A JSON-array state must also be nested in the outer sample container.

`state_builder="auto"` performs only documented conversions to JSON-compatible
Python values. It never falls back to `str(value)`. A callable state builder is
available for domain objects, custom missing-value policy, and explicit column
selection.

`fit(X, y=None)` is a schema-binding operation:

1. Validate question configuration and output schema.
2. Validate the input container and, when supplied, `X`/`y` length.
3. Validate labels against the declared domain.
4. Set fitted attributes and input feature metadata where meaningful.
5. Make zero network calls.

`fit_transform`, `transform`, `predict`, `predict_proba`, and `evaluate` perform
inference. This distinction must be prominent because CV and pipeline evaluation
can cause repeated paid calls.

## 7. Backend boundary and ownership

The estimator layer depends on a small structural `SystemOneClient` protocol, not
on `isinstance(TypeSafeClient)`:

```python
class SystemOneClient(Protocol):
    def system_one(
        self,
        state,
        questions,
        *,
        model=None,
    ) -> SystemOneResponse: ...
```

The backend owns conversion from this client call into canonical responses and
controls row concurrency. Estimators own feature/prediction semantics.

Rules:

- Default clients are created lazily for a single inference operation and closed.
- API keys come from `TYPESAFE_API_KEY`; there is no estimator `api_key`
  parameter.
- A live client is not a constructor parameter.
- Advanced injection uses a clone-safe top-level `client_factory`.
- A client created by the package is owned and closed by the package.
- If a later advanced API accepts a borrowed client, it is never closed by the
  package.
- Retries are delegated to the SDK/backend. Estimators do not wrap them in a
  second retry loop.

The official `SystemOneAdapterClient` is not a `TypeSafeClient` subclass but
returns a `SystemOneResponse` subclass and implements the common call shape. A
separate optional backend can adapt its extra `provider` configuration without
polluting the core protocol.

## 8. Execution semantics

The first vertical slice is synchronous and sequential. Bounded row concurrency
is added only after the sequential contract is proven.

For every inference operation:

1. Materialize and validate the sample sequence.
2. Convert each sample to one JSON-compatible state.
3. Send all declared questions for that state in one request.
4. Validate and canonicalize that response before sending the next state. Stop on
   failure; cleanup runs once and does not replace the primary inference or
   response failure.
5. Preserve input ordering, even when requests later run concurrently.
6. Project the canonical batch to sklearn arrays or a rich result. Raw SDK/HTTP
   responses never become batch-result entries.

One call per sample means one logical `system_one` invocation for each evaluated
row, not one HTTP attempt or guaranteed billable event. SDK retries may multiply
HTTP attempts. A fail-fast operation can evaluate fewer rows than were supplied.

The API uses `max_concurrency`, not `batch_size`, because the service has no
multi-state batch request. The initial default is one. A later thread-based
implementation may allow explicit concurrency; the official cookbook uses eight
workers with a shared synchronous client.

The documentation must show that effective in-flight requests can be
`CV n_jobs × backend max_concurrency`. The library will not call `asyncio.run()`
inside sklearn methods.

## 9. Response validation and errors

The SDK validates Pydantic structure but does not fully enforce the estimator
contract. The common normalizer additionally verifies:

1. Expected and returned question IDs are an exact match. When a response exposes
   raw HTTP metadata, inspect raw answer IDs as well as decoded IDs: SDK forward
   compatibility can otherwise hide unknown answer types. Directly constructed
   SDK models and structural clients without raw metadata use decoded IDs only.
2. Every answer type matches its question type.
3. Choice probability keys exactly match criteria keys.
4. Score probability and legend keys exactly match rubric levels.
5. Every probability is finite and in `[0, 1]`.
6. Probability sums are within tolerance of one.
7. Small floating-point error may be normalized; material error is rejected.
8. Noul is finite and in `[0, 1]`.
9. Choice contains 2–255 alternatives and Score contains 2–10 levels.
10. Reported Choice/Score scalar answers agree with their distributions.

Exceptions:

- `JevConfigurationError`: invalid question, label, state-builder, or feature
  configuration.
- `JevResponseError`: structurally or mathematically inconsistent response.
- `JevInferenceError`: replaces an SDK/backend failure with only the sample index
  and question IDs. Provider exceptions are deliberately not retained or chained
  because their bodies and tracebacks may contain credentials or private states.
- `JevBatchError`: future collect-mode error containing safe per-sample failure
  records.

Ordinary library error messages never include raw input states, credentials, or
full provider bodies. Response errors include safe declared IDs, fixed field
descriptions, and unexpected-ID counts, not returned arbitrary keys or type
values. Engine response failures are raised outside validation/provider/cleanup
handlers so their `__cause__` and `__context__` do not retain those exceptions.
The first release is fail-fast. No random choice and no numeric imputation are
allowed.

The current sum and scalar-agreement tolerances are `1e-6`. This rejects a
synthetic `[0.33, 0.33, 0.33]` distribution. Whether such quantized outputs should
be accepted from a real backend needs separate compatibility evidence and an
explicit normalization policy; broader live compatibility is not assumed.

Offline evidence for the 2026-11-10 corrections:

| Contract | Evidence in `tests/test_contract_regressions.py` |
| --- | --- |
| Sorted classifier columns with unchanged feature/canonical order | `test_choice_sorted_classes_work_with_log_loss_and_preserve_feature_order` also exercises standard frozen-classifier calibration |
| Noul positive-label orientation survives sorting | `test_noul_sorted_columns_keep_positive_threshold_orientation` covers strings, numbers, and booleans |
| Choice ties use sorted classes without changing the request | `test_choice_ties_follow_sorted_classes_without_reordering_the_request` |
| Invalid responses stop later calls and clients close once | `test_invalid_first_response_stops_before_later_rows_and_closes_once` covers all four estimators, including cleanup failure |
| Arbitrary response values do not enter ordinary errors | `test_response_values_do_not_enter_ordinary_errors` checks messages, tracebacks, chains, and pickles |
| SDK decoding cannot hide extra raw answer IDs | `test_sdk_raw_answer_ids_are_checked_even_if_decoding_drops_an_answer` |
| Raw decoding errors do not retain private diagnostics | `test_raw_decoder_failure_drops_private_diagnostics_and_stops` |

These are synthetic contract tests, not task-accuracy benchmarks or measurements
of real provider failure rates. Existing validation tests cover missing/extra
probability keys, finite ranges, sums, scalar agreement, and unknown usage.

## 10. Usage, caching, and privacy

Usage totals distinguish known totals from missing reports. Missing token counts
are not treated as zero. Resolved model IDs are retained per response.

Prediction methods do not mutate estimator state, which preserves thread safety
and avoids surprising clone/CV behavior.

Caching is opt-in because both sides are hazardous:

- no cache can repeat paid calls during CV;
- a disk cache can retain sensitive states/responses and become stale when a model
  alias moves.

Before the first public release, an ADR must define a cache key containing the
normalized state, normalized questions, explicit model, encoding version, and
backend identity. Persistent caching must require an explicit path, warn that it
contains sensitive material, and recommend requestable pinned model versions
where supported. Alias invalidation remains necessary otherwise. There is no
implicit global cache.

The README and API docs must explicitly state that states are sent to the selected
remote backend. Debug HTTP logging remains off by default.
The library does not sanitize third-party logging, custom-client side effects, or
traceback tools that explicitly dump frame locals. SDK decoding may log unknown
answer names and types before the wrapper can validate the response. The tested
privacy boundary covers ordinary library exception rendering/serialization and
retained canonical results, not process-wide information-flow guarantees.

## 11. sklearn contract

- Mixin order is `ClassifierMixin` / `RegressorMixin` / `TransformerMixin` before
  `BaseEstimator`.
- `__init__` contains explicit parameters and assignments only.
- Runtime clients are never fitted attributes.
- `check_is_fitted` guards inference.
- Refit replaces all learned trailing-underscore state.
- Sparse matrices are rejected in the first release.
- `n_features_in_` and `feature_names_in_` are set only where meaningful.
- Estimator tags declare non-determinism, supported input forms, optional target,
  and poor-score expectations accurately.
- `check_estimator(..., legacy=False)` must have no unexpected failures.
- Any legacy check excluded for a remote, fixed-schema zero-shot estimator is
  named individually with a written reason; blanket skip is forbidden.

Tests use a deterministic fake backend. PR tests never require credentials or
network access. Live contract smoke tests are manual or scheduled and have an
explicit request/cost ceiling.

## 12. Package layout

```text
src/scikit_jev/
├── __init__.py
├── _engine.py             # ordered per-state execution
├── _features.py           # fixed feature-schema compilation
├── _results.py            # immutable canonical and public results
├── _states.py             # X -> JSON-compatible states
├── _validation.py         # questions and response invariants
├── classification.py
├── regression.py
├── transformer.py
├── exceptions.py
└── backends/
    ├── _protocol.py
    ├── typesafe.py
    └── system_one_adapter.py   # optional, after MVP
```

Internal modules may be consolidated during the first vertical slice; the
dependency direction must remain:

```text
estimators -> engine/features -> backend protocol
                              -> canonical result types
TypeSafe SDK ----------------> TypeSafe backend only
```

## 13. Fixed invariants

These are release-blocking invariants, not preferences:

- each evaluated state plus all its questions equals one logical backend call;
- Transformer feature and canonical probability columns are declaration-ordered;
- classifier probability columns follow sorted `classes_` with explicit remapping;
- sequential validation finishes before sending the next state;
- no network call in `__init__`, `fit`, `clone`, repr, or parameter inspection;
- no credential in estimator state;
- no invented fallback output;
- standard inference does not mutate the estimator;
- fake-backend tests cover every mathematical mapping;
- raw states are absent from normal exceptions and result metadata.

## 14. Decisions still requiring explicit sign-off

1. Project license: MIT as of 2026-11-10, replacing the BSD-3-Clause choice used
   for the initial 2026-11-09 source publication.
2. Final import name (`scikit_jev` is proposed).
3. Exact cache API and persistence warning.
4. Whether a public v0.1 includes concurrency or keeps it experimental.
5. Whether arbitrary non-string sklearn class labels are supported through a
   reversible key codec; the official Choice API itself uses string keys.
6. Trademark/name review before publishing `scikit-jev`.
