# scikit-jev implementation plan

Status: functional MVP implemented; Phase 4 and release hardening remain
Date: 2026-10-06

## Outcome

The first public release must prove one vertical claim:

> A declared set of Noul, Choice, and Score questions can be cloned as an sklearn
> estimator, evaluated with one request per sample, validated without information
> loss, and consumed safely by a normal sklearn pipeline.

The transformer is the core deliverable. Task-specific estimators are projections
of the same engine, not separate integrations.

## Progress snapshot

Completed on 2026-10-06:

- Phase 0 packaging, CI, deterministic no-network harness, wheel, and sdist
- Phase 1 end-to-end Noul transformer slice
- Phase 2 mixed Noul/Choice/Score feature compiler and rich results
- Phase 3 task-specific classifier and regressor projections
- Phase 5 core clone, pipeline, pickle, tags, and modern estimator checks

The current suite contains 155 deterministic tests. Phase 4 operational features
and Phase 6 live/release gates remain intentionally open. The repository itself
has not been initialized as Git because the license and desired default branch
still require explicit sign-off.

## Phase 0 — Repository and decision baseline

Deliverables:

- Initialize the repository only after the desired default branch and license are
  confirmed.
- Add `pyproject.toml` with a `src/` layout.
- Set Python support to 3.10–3.14.
- Initially constrain `typesafe-sdk>=0.7.2,<0.8` and
  `scikit-learn>=1.6,<2`.
- Add Ruff, a static type checker, pytest, coverage, build validation, and a
  no-network test guard.
- Record ADRs for backend ownership, pure `fit`, probability validation, and
  caching.
- Freeze the four public estimator signatures described in the architecture.

Exit gate:

- Source and test packages import from a clean environment.
- Wheel and sdist build.
- `python -m build` artifacts pass metadata inspection.
- No credential-bearing configuration exists in repository files.

## Phase 1 — Core vertical slice: Noul transformer

Implement the smallest end-to-end path before generalizing:

1. `SystemOneClient`/backend protocol.
2. Clone-safe `TypeSafeBackend` configuration with lazy client creation.
3. Deterministic fake and recording backends.
4. State-container validation and `state_builder="auto"`.
5. Canonical request/response result types.
6. Strict Noul response validation.
7. `JevTransformer` with Noul questions, `fit`, `transform`, `evaluate`, and
   `get_feature_names_out`.

The first implementation is synchronous and sequential. Do not implement threads,
async bridging, persistent cache, or partial-failure collection here.

Exit gate:

- `clone(transformer)` succeeds.
- `fit` makes zero backend calls.
- `fit_transform` makes exactly one call per sample.
- Multiple Noul questions for one sample produce one call, not one call per
  question.
- `Pipeline(JevTransformer, LogisticRegression)` runs entirely against the fake
  backend.
- A malformed, missing, extra, NaN, infinite, or out-of-range answer fails with a
  safe indexed exception.
- Estimator repr, exceptions, and pickle contain neither API key nor input state.

## Phase 2 — Complete feature compiler

Add:

- Choice question validation and declaration-ordered probability features.
- Score validation and three encodings: `probabilities`, `expected`, and
  `expected_and_std`.
- Optional Choice/Score confidence columns.
- Stable escaping and `get_feature_names_out` for arbitrary valid question IDs
  and labels.
- DataFrame row and two-dimensional NumPy input conversion.
- Exact input-order preservation.
- Immutable `JevBatchResult` with resolved models, answer metadata, warnings, and
  usage accounting.

Exit gate:

- Mixed Noul/Choice/Score questions generate a fixed numerical matrix.
- Feature shape and names are determined at fit time and remain stable.
- Shuffled response map order cannot change output columns.
- Small probability rounding error is normalized and reported; material error is
  rejected.
- Usage with missing token counts is represented as unknown, never as zero.
- Calling `evaluate` once provides both values and metadata without another
  backend call.

## Phase 3 — Task-specific estimators

Build thin projections over the shared engine:

- `JevNoulClassifier`
- `JevChoiceClassifier`
- `JevScoreRegressor`

Required semantics:

- Noul returns `[1-p_true, p_true]` and has a tested inclusive threshold boundary.
- Choice probability columns follow `classes_`, independent of response order.
- Choice prediction and distribution are mutually consistent, including ties.
- Score returns `float64` expected levels and validates them against the full
  distribution.
- `y`, when supplied, is used for shape/domain validation; it does not cause a
  remote training call.

Exit gate:

- `is_classifier`/`is_regressor` identify every estimator correctly.
- `predict`, `predict_proba`, and `predict` shape/dtype contracts hold for empty,
  single-sample, and multi-sample cases as applicable.
- `Pipeline`, `cross_val_score`, and `GridSearchCV` run with deterministic fake
  responses.
- Refit resets fitted schema state.
- Inference does not add or mutate fitted attributes.

## Phase 4 — Operational reliability

Add only after sequential semantics are stable:

- Bounded row concurrency using `ThreadPoolExecutor`.
- Stable reorder by original sample index.
- Explicit `max_concurrency`; default remains one until live evidence supports a
  larger default.
- Client ownership and close behavior for success, validation failure, API
  failure, and cancellation.
- Optional `SystemOneAdapterBackend` extra for OpenAI/Anthropic/Gemini baselines.
- A reasoned cache implementation or a documented decision to defer it.
- Safe per-sample failure collection in rich results; standard sklearn methods
  remain fail-fast.

Exit gate:

- Sequential and concurrent results are identical for the same recording backend.
- Maximum observed in-flight calls never exceeds `max_concurrency`.
- A result at index `i` always corresponds to input sample `i`.
- SDK retries are not multiplied by an outer retry loop.
- Owned clients close exactly once; borrowed clients are never closed.
- Documentation demonstrates the multiplication of CV `n_jobs` and backend
  concurrency.
- The adapter backend passes the same canonical contract tests as TypeSafeBackend.

## Phase 5 — sklearn compliance and quality

Run the full contract matrix:

### Estimator mechanics

- `get_params(deep=True)` / `set_params`
- `clone`
- repr
- joblib/cloudpickle where supported
- fitted-state detection
- refit reset
- tags
- feature-name propagation

### Numerical mapping

- Noul complement and threshold boundary
- Choice criteria order and ties
- Score expected value and standard deviation
- probability normalization tolerance
- all result shapes and dtypes

### Invalid responses

- missing/extra answer IDs
- wrong answer type
- missing/extra probability keys
- NaN, infinity, negative value, value above one
- materially invalid probability sum
- invalid Choice cardinality or Score rubric
- reported scalar/distribution disagreement

### Input containers

- strings and JSON mappings
- nested JSON-array states
- object arrays and two-dimensional arrays
- DataFrame rows
- scalar rejection
- empty input
- ragged input
- sparse rejection
- `X`/`y` length mismatch
- feature-count/name drift

### Integration

- estimator checks with `legacy=False`: zero unexpected failures
- Pipeline with classifier/regressor downstream
- CV and grid search
- nested joblib process/thread behavior
- no-network guarantee in all PR tests

Any expected failure from legacy sklearn checks must be listed individually with a
specific reason. Blanket `_skip_test`, broad `xfail`, or disabling estimator checks
is not accepted.

## Phase 6 — Documentation, live contract tests, and v0.1 release

Documentation:

- Quickstart for each estimator.
- Transformer-to-classical-model example mirroring the official cookbook pattern.
- Input conversion rules and one-sample syntax.
- Cost model: one request per sample, all questions packed.
- Reproducibility: pin concrete Jev model IDs.
- Privacy: states leave the process; cache/debug logging implications.
- CV/GridSearch cost and concurrency warning.
- Error-handling and metadata examples.
- TypeSafe vs System One Adapter baseline comparison.

Live tests:

- Manual or scheduled only; never required for ordinary PRs.
- Explicit credentials supplied ephemerally by CI secrets.
- Fixed maximum samples/questions and request-count assertion.
- One Noul, one Choice, one Score, and one mixed-question smoke path.
- Validate public shape, not exact model probabilities.

Release gate:

- All fake-backend contract tests pass on the compatibility matrix: minimum
  sklearn on its oldest supported Python, latest sklearn on Python 3.10–3.14,
  and any additional valid boundary combinations. An impossible pairing such as
  an old sklearn release with a newer unsupported Python is not required.
- Linux is required; macOS and Windows receive at least smoke coverage.
- Wheel/sdist install and import in clean environments.
- Public API docs contain no unresolved behavioral ambiguity.
- Package/import name, license, and trademark review are resolved.
- Changelog clearly labels remote, probabilistic, paid side effects.
- No secret or captured live state exists in the repository or distributions.

## Ordered issue backlog

The implementation should be tracked in this dependency order:

1. Packaging, CI, and no-network test harness.
2. Backend protocol and fake/recording backend.
3. State conversion contract.
4. Canonical response models and safe exceptions.
5. Probability and answer-schema validator.
6. Feature-schema compiler.
7. Noul-only transformer vertical slice.
8. Choice and Score transformer support.
9. Rich immutable result and usage aggregation.
10. Noul classifier.
11. Choice classifier.
12. Score regressor.
13. sklearn estimator checks and integration fixtures.
14. TypeSafeBackend contract and manual live smoke.
15. Bounded concurrency.
16. System One Adapter backend.
17. Cache ADR and implementation decision.
18. Documentation and v0.1 packaging.

Items 1–7 form the first demonstrable slice. Items 1–14 form the functional MVP.
Items 15–18 form the public-release hardening set.

## Deferred beyond v0.1

- Multi-output classifier/regressor convenience estimators.
- Noul multi-label estimator.
- `predict_distribution` convenience API for Score.
- Probability calibration meta-estimator and abstention policy.
- pandas/polars `set_output` polish beyond basic feature names.
- Async non-estimator helper.
- Distributed execution and rate-limit coordination.
- Automatic question discovery/selection.

These extensions must reuse the canonical batch and feature-schema layers; they
must not introduce alternate response parsing paths.

## Principal risks and mitigations

| Risk | Mitigation |
| --- | --- |
| SDK changes rapidly | Pin `<0.8`, contract-test supported versions, isolate SDK in one backend |
| CV repeats paid calls | Pure fit, cost documentation, explicit cache ADR, recording backend tests |
| Moving model alias changes behavior | Recommend pinned model IDs and record resolved IDs |
| Probability shape is subtly invalid | Central strict validator before every projection |
| Secrets break clone/pickle safety | Environment-based auth and lazy factory; no live client param |
| Parallelism exceeds rate limits | Default concurrency one, bounded executor, document nested parallelism |
| Disk cache leaks private states | No implicit cache; explicit sensitive-data warning and path |
| Generic sklearn checks assume a local learner | Accurate tags plus named, justified legacy exclusions only |
| Project becomes a thin vendor wrapper | Keep feature schema, validation, metadata, and sklearn contract as core IP |

## Definition of done for v0.1

`scikit-jev` v0.1 is done when a user can put `JevTransformer` in a real sklearn
pipeline, know exactly how many remote calls it will make, obtain stable named
features and metadata from valid Jev answers, receive an explicit safe error for
invalid answers, clone/CV the estimator without copying credentials or clients,
and reproduce all behavior locally with a fake backend.
