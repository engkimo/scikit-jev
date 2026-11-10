# Project memory: scikit-jev

Last updated: 2026-11-10

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
- Every sequential response is canonicalized and validated before the next row
  is sent. Validation failures stop later calls and survive cleanup failures.
  Raw answer IDs are inspected when SDK HTTP metadata exists, without retaining
  raw HTTP responses in results. These are logical-call contracts; SDK retries
  can create multiple HTTP attempts.
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
- Declaration-ordered Transformer and canonical Choice/Score probability columns;
  classifier columns are explicitly remapped to sorted `classes_`.
- Finite probabilities in `[0, 1]` with a valid sum.
- Noul: one probability feature; classifier columns follow sorted labels and the
  configured positive label retains its threshold orientation even if it sorts
  first. Default Boolean labels still yield `[negative, positive]`.
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

The contract-correction and offline-demo milestone on 2026-11-10 has 181 tests passing
without network access on Python 3.12.8 / sklearn 1.9.1 / SDK 0.7.2, Python
3.10.14 / sklearn 1.6.0 / SDK 0.7.5, and Python 3.14.3 / sklearn 1.9.1 / SDK 0.7.5.
Classifier sorting, sequential validation, raw SDK ID checks, and
response-error sanitization are implemented locally with regression tests.
Wheel and sdist builds are part of the publication gate. Operational Phase 4 work
remains deferred:
bounded concurrency, a System One Adapter backend, an explicit cache decision,
and partial-failure collection. Public-release work also still requires live
contract smoke tests, final documentation, import/package-name and
trademark sign-off, and full clean-environment compatibility-matrix execution.

## Public-source publication and landscape

- The user authorized adding a license and making the GitHub repository public
  on 2026-11-09. BSD-3-Clause was selected, matching the sklearn ecosystem.
- On 2026-11-10, the user requested a simpler permissive license (MIT/Apache were
  suggested) and authorized external publication. MIT was selected for its short
  terms; LICENSE, package metadata, README, and architecture are aligned. Earlier
  published revisions retain their historical BSD-3-Clause license.
- GitHub source publication is separate from a stable v0.1 or PyPI release;
  the package remains pre-alpha and has not been uploaded to PyPI.
- The 2026-11-09 review found `jaketimothy/sklearn-decision`, a direct overlap:
  question featurizer plus Noul/Choice classifiers and Score regressor, Jev and
  local backends, calibration, and caching. It was created on 2026-09-26 and
  missed by the original review. Do not claim the category is unoccupied.
- scikit-llm main still has no Jev/TypeSafe/System One integration in the
  inspected source; latest PyPI release is 1.4.3. Adjacent projects include
  tab-jev and jevframe. See `docs/landscape.md` for primary-source links.

## Selected next investment

- On 2026-11-09, the user selected the response-contract, evaluation-correctness,
  and reproducible-execution direction after reviewing sklearn-decision, then
  requested a concrete plan followed by wider promotion. The visibility ambition
  is 3,000 GitHub stars; this is a goal, not an adoption metric or forecast.
- The next-cycle plan in `docs/implementation-plan.md` takes priority over the
  historical Phase 4 feature backlog: correct known contracts, consolidate offline
  evidence, make a useful runnable demo, prepare a preview, then distribute and
  evaluate actual use. The initial contract corrections are implemented;
  a reusable demo and wider release/distribution work remain next.
- Classifier sorting intentionally changes the earlier pre-alpha contract.
  Transformer/canonical declaration order is preserved, while classifier outputs
  follow sorted labels. Choice ties now use the first sorted class. Refit older
  classifiers and align consumers to `classes_`.
- Keep hosted-model rounding and requestable fixed-version IDs as unresolved
  compatibility questions. A recorded resolved version alone does not guarantee
  replay, and fake tests alone do not establish semantic accuracy.
- No broad local-model/encoder feature race is planned. Continue independently
  only if users value the focused contracts; shared test assets and upstream
  contributions remain alternatives. Live inference remains opt-in and outside
  the current scope.
- On 2026-11-10, the user selected X, English articles, and overseas communities
  in addition to the GitHub/Japanese launch. Prepare concrete publication copy
  for user review before publishing. Specific account handles and destinations
  are not yet established; this is not authorization to publish unreviewed copy.
- The tested information boundary covers ordinary package error messages,
  formatted exception chains/pickles, and canonical results. Third-party SDK
  logging and frame-local traceback dumps are outside that guarantee.
- A runnable credential-free example now exists at `examples/offline_workflow.py`
  with a no-network integration test. It uses hand-authored synthetic responses,
  not a semantic model or an accuracy benchmark, and exposes named mixed features,
  a classical head, logical calls, and unknown token counts.
- Initial English/Japanese X drafts and an English article brief are recorded in
  `docs/implementation-plan.md` for review. None is approved or posted; full
  articles, demo video, concrete publication accounts, and public preview remain
  outstanding.

Update this file when a durable design decision changes. Do not use it as a
changelog or store secrets, temporary debugging notes, or credentials here.
