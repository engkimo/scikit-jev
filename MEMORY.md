# Project memory: scikit-jev

Last updated: 2026-11-09

## Objective

Build `scikit-jev`: a scikit-learn-native semantic feature engine for TypeSafe
System One and Jev.

The project is not a generic LLM prompt wrapper. Its durable value is the strict,
probability-preserving conversion of typed System One answers into stable sklearn
predictions and feature matrices.

## Core design decisions

- `JevTransformer` is the center of the design.
- One input sample means one `system_one` request containing all questions for
  that sample.
- `JevNoulClassifier`, `JevChoiceClassifier`, and `JevScoreRegressor` are thin
  projections of the shared execution, validation, and result layers.
- `fit()` is network-free. Remote inference occurs in `fit_transform()`,
  `transform()`, `predict()`, `predict_proba()`, and `evaluate()`.
- Probabilities are never flattened to unvalidated text and invalid responses are
  never replaced with random, majority, zero, or 0.5 fallback values.
- Provider failures are replaced with coordinate-only `JevInferenceError`
  objects. Original provider exceptions are neither stored nor chained because
  their messages, response bodies, and tracebacks may contain private data.
- API keys and live clients must not appear in estimator parameters, reprs,
  exceptions, logs, or pickles. Authentication is resolved lazily from the
  official SDK environment configuration.
- The estimator layer talks through a narrow System One backend/client protocol,
  allowing fake clients and the official System One Adapter without coupling to
  one concrete client class.
- Standard inference methods do not mutate estimator state. Rich immutable
  results carry usage, resolved model, answers, and warnings.
- The implementation is synchronous and sequential. Bounded opt-in concurrency
  remains deferred to Phase 4.
- Persistent caching is never implicit because cached states and responses may be
  sensitive and model aliases may move.

## API baseline

- Python: 3.10+
- `typesafe-sdk>=0.7.2,<0.8`
- `scikit-learn>=1.6,<2`
- Import package: `scikit_jev`
- Public estimators:
  - `JevTransformer`
  - `JevNoulClassifier`
  - `JevChoiceClassifier`
  - `JevScoreRegressor`
  - `TypeSafeBackend`

## Required invariants

- Exact question-ID and answer-type matching.
- Declaration-ordered Choice and Score probability columns.
- Finite probabilities in `[0, 1]` with a valid sum.
- Noul: one probability feature; classifier probability order is
  `[negative, positive]`.
- Choice: full declared label distribution.
- Score: full distribution by default, with expected and expected-plus-standard-
  deviation encodings available.
- No network in `__init__`, `fit`, `clone`, repr, or parameter inspection.
- No raw input state in ordinary errors or result metadata.
- PR tests use deterministic fake backends and do not require credentials or
  network access.

## Evidence

- Official SDK baseline:
  https://github.com/typesafe-ai/typesafe-sdk-python/tree/f078f1e208a0d885154dc758344ae4fce77ac168
- Official AutoResearch feature workflow:
  https://docs.typesafe.ai/cookbooks/autoresearch_feature_discovery
- Detailed architecture: `docs/architecture.md`
- Ordered implementation plan: `docs/implementation-plan.md`

## Current implementation state

The functional MVP is implemented through Phase 3 plus the core sklearn quality
gate:

- package metadata, CI, a no-network test harness, and build artifacts;
- lazy TypeSafe backend, structural backend/client protocol, and ordered
  sequential execution;
- JSON state conversion, immutable canonical results, strict response
  validation, and a probability-preserving feature compiler;
- `JevTransformer`, `JevNoulClassifier`, `JevChoiceClassifier`, and
  `JevScoreRegressor`;
- public exports, type marker, fake-backend pipeline integration, cloning,
  pickling, and modern sklearn estimator checks.

As of 2026-10-06, 155 deterministic tests pass without network access. Wheel and
sdist builds succeed. Operational Phase 4 work remains deliberately deferred:
bounded concurrency, a System One Adapter backend, an explicit cache decision,
and partial-failure collection. Public-release work also still requires live
contract smoke tests, final documentation, import/package-name and
trademark sign-off, and full clean-environment compatibility-matrix execution.

## Public-source publication and landscape

- The user authorized adding a license and making the GitHub repository public
  on 2026-11-09. BSD-3-Clause was selected, matching the sklearn ecosystem.
- GitHub source publication is separate from a stable v0.1 or PyPI release;
  the package remains pre-alpha and has not been uploaded to PyPI.
- The 2026-11-09 review found `jaketimothy/sklearn-decision`, a direct overlap:
  question featurizer plus Noul/Choice classifiers and Score regressor, Jev and
  local backends, calibration, and caching. It was created on 2026-09-26 and
  missed by the original review. Do not claim the category is unoccupied.
- scikit-llm main still has no Jev/TypeSafe/System One integration in the
  inspected source; latest PyPI release is 1.4.3. Adjacent projects include
  tab-jev and jevframe. See `docs/landscape.md` for primary-source links.

Update this file when a durable design decision changes. Do not use it as a
changelog or store secrets, temporary debugging notes, or credentials here.
