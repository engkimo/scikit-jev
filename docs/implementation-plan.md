# scikit-jev implementation plan

Status: functional MVP implemented; contract corrections and launch preparation next
Updated: 2026-11-10

## Outcome

The first public release must prove one vertical claim:

> A declared set of Noul, Choice, and Score questions can be cloned as an sklearn
> estimator, evaluated with one logical call per evaluated sample, validated
> without information loss, and consumed safely by a normal sklearn pipeline.

The transformer is the core deliverable. Task-specific estimators are projections
of the same engine, not separate integrations.

## Progress snapshot

Completed on 2026-10-06:

- Phase 0 packaging, CI, deterministic no-network harness, wheel, and sdist
- Phase 1 end-to-end Noul transformer slice
- Phase 2 mixed Noul/Choice/Score feature compiler and rich results
- Phase 3 task-specific classifier and regressor projections
- Phase 5 core clone, pipeline, pickle, tags, and modern estimator checks

The contract-correction and offline-demo milestone on 2026-11-10 has 181 deterministic tests
passing with Python 3.12.8 / sklearn 1.9.1 / SDK 0.7.2, Python 3.10.14 /
sklearn 1.6.0 / SDK 0.7.5, and Python 3.14.3 / sklearn 1.9.1 / SDK 0.7.5.
Mypy and Ruff pass in the development environment. Phase 4 operational features
and Phase 6 live/release gates remain open. The repository was initially
published on GitHub under BSD-3-Clause on 2026-11-09. On 2026-11-10, the user
requested simpler permissive licensing and authorized external publication;
the current source uses MIT. Source publication is not a PyPI release or
evidence of production readiness.

The MIT source, contract corrections, tests, and offline example were published
to GitHub main on 2026-11-10 at
`4d7248bc5da6c8e2406bde4d31ec85197b44c650`. The wheel/sdist license metadata and
the demo from a clean wheel install were checked. This is not a PyPI upload or a
social announcement; X account selection remains pending. GitHub Actions run
`42742517598` passed all ten jobs for the code commit: Python 3.10-3.14,
minimum sklearn, macOS/Windows, lint/type checks, and packaging.

## Next cycle: contract corrections, useful demo, and adoption

The user selected the contract/evaluation/reproducibility direction on
2026-11-09 and requested a plan before wider promotion. The growth ambition is
3,000 GitHub stars. Treat that as an outcome to work toward, not a forecast or a
substitute for actual use. The GitHub API reported 0 stars and 0 forks at this
review; no launch campaign has been executed.

The priority order below supersedes the historical Phase 0-6 backlog for the
next cycle. Time estimates are working ranges, not scheduled commitments.

### Positioning to test

Help Python ML practitioners turn questions about text and JSON into named,
probabilistic features that work in ordinary sklearn evaluation and pipelines.
Make the column mapping, response validation, failure behavior, and logical
request accounting explicit and reproducible.

The visible story should begin with a useful workflow, supported by evidence for
the contracts. A validator by itself is not evidence of product demand.
`sklearn-decision` already implements the broad workflow, and `jevframe` already
has substantial response validation. Do not claim a new category, superior
accuracy, or universal reliability. Link related projects and explain the
specific tradeoffs.

### Work packages and exit gates

| Order | Work package | Deliverables | Exit gate |
| --- | --- | --- | --- |
| 1, approximately week 1 | Correct the known contracts | Sorted classifier `classes_` with explicit remapping; declaration-ordered Transformer/canonical data; per-row validation before sending the next row; raw and parsed answer-ID checks when raw HTTP metadata exists; response-derived values removed from ordinary error messages | Unsorted Choice labels and reversed Noul label orientation give correct predictions and sklearn log-loss; a bad first response prevents later sequential calls; unknown SDK-filtered answer types are detected; synthetic sentinels do not appear in library error messages/chains/pickles; cleanup happens once |
| 2, approximately weeks 1-2 | Make evidence reusable | A named offline contract matrix using deterministic synthetic responses; SDK decoding-boundary tests; compatibility results; updated architecture and behavior-change notes | Existing checks plus the new matrix pass with sockets blocked; minimum and latest supported sklearn combinations are verified; no competitor code or captured live responses are shipped as fixtures |
| 3, approximately weeks 2-3 | Make the workflow easy to try | A support-triage example using synthetic text/JSON, mixed questions, features, a classical head, and result metadata; a credential-free contract demo; a short walkthrough and an accurate README | A new user can install in a clean environment and run the offline demo with one documented command; fake outputs are clearly labeled and never presented as measured semantic accuracy; a small trial checks a provisional target of under five minutes to first result |
| 4, approximately weeks 3-4 | Prepare a credible preview release | Versioned package/install path; API and input/output examples; cost/privacy/model-alias limitations; release notes; runnable examples and supported-version evidence | Packaging and clean install pass; every advertised implemented feature has a reproducible example or test; unresolved hosted-model rounding/version questions are explicit; publication identity and destination are verified before publishing |
| 5, after the preview gate | Publish and learn | Japanese and English explanations, a short demo clip, channel-specific announcements, a feedback template, and an aggregate distribution log | Announcements use verified claims and correct public links; publication account and community fit are established; users can report problems; initial questions and installation failures are triaged |
| 6, approximately weeks 4-6 | Decide the next investment | Adoption review and a continue/narrow/contribute-upstream decision | At least three independent prospective users attempt a real workflow, two identify the contract/execution behavior as useful, and one reports repeat use; otherwise revise the thesis or narrow to shared contracts/upstream contributions |

Work package 1 is implemented as of 2026-11-10, including regression
tests, architecture, and README behavior-change notes. It changes existing public
semantics: classifier probability columns follow sorted `classes_`; Transformer
features and canonical answer data retain declaration order. Noul thresholds
still apply to the configured positive label even when that label sorts first.
Choice ties select the first sorted class. The offline contract tests cover all
four estimators stopping before later rows, including simultaneous cleanup
failure, and raw SDK decoding-boundary checks. Work package 3 now has an initial
credential-free example at `examples/offline_workflow.py` and a regression test.
It produces a 2-by-6 evaluation feature matrix from hand-authored responses for
synthetic messages, trains a classical head, reports zero logical calls for fit,
eight for training, and two for evaluation, and leaves token counts unknown.
It is not an empirical semantic-accuracy result or an onboarding user study.
Work packages 2-4 remain open: the new tests are not yet a separately packaged
conformance toolkit, and demo video, external first-run feedback, API guide,
and preview publication remain to do.

Keep the initial evidence scope precise: one `system_one` invocation per sample
is a logical-call contract, not a promise of one HTTP attempt or one billable
event. SDK retries may create multiple attempts. Missing usage remains unknown.
The package cannot sanitize third-party debug logging or arbitrary custom-client
side effects; ordinary library exceptions and retained results are the bounded
privacy contract to test.

### Questions to resolve before stronger claims

- Probability rounding: current sum/scalar tolerances are `1e-6`. Rejection of a
  synthetic `[0.33, 0.33, 0.33]` is verified, but whether such distributions are
  legitimate hosted-service output is not established. Do not relax tolerances
  or advertise live compatibility without evidence and a documented raw versus
  normalized-value policy.
- Model versions: a resolved response version does not prove it is accepted as a
  request model. Remove unverified pinned-ID examples; document supported request
  names separately from recorded versions and model-drift detection.
- Empirical utility: compare a direct Choice question, calibrated direct answers,
  a question bank plus a classical head, and a text baseline. Use held-out data,
  record coverage/rejections as well as accuracy and log-loss, and distinguish
  model quality from wrapper behavior.
- Live inference remains outside the current authorized scope. Any later live
  study needs an explicit dataset, request ceiling, budget, and opt-in before
  execution. Ordinary tests, PRs, demos in CI, and this planning work stay offline.

### Distribution sequence

Use one concrete example across channels, adapted to the audience. Explain the
workflow first, show the working result, state the current limitations, and link
to the repo and a runnable example. Publish in waves so feedback can improve the
next announcement rather than sending identical messages everywhere at once.

| Wave | Candidate channel | Material | Readiness/dependency |
| --- | --- | --- | --- |
| Foundation | GitHub repository and a versioned preview release | Clear README, examples, install instructions, changelog, related work, focused contribution tasks | Work packages 1-4; preserve pre-alpha/preview labeling until the actual release gates pass |
| First audience | Japanese technical article on Zenn or Qiita, plus the user's own social account if available | A task-driven article and a short before/after workflow clip | Choose one primary article destination; confirm the actual account; article numbers must come from reproducible evidence |
| Broader audience | An English technical article and the user's X account; overseas communities | An English walkthrough, concise demo, and transparent tradeoff comparison | Selected by the user on 2026-11-10; exact handles/destinations pending; user reviews publication copy before posting |
| Specialist audience | Relevant Python/sklearn/TypeSafe communities and newsletters | A contribution useful to that community: an integration example, contract lesson, or reproducible experiment | Identify the specific venue and its rules before posting or contacting editors; no assumed access or endorsement |
| Optional | Show HN | A project the reader can actually try, with a maker available for discussion | HN requires human-written posts and disallows automated posting; the user must write/post/respond personally; do not prepare generated submission copy for HN |
| Optional, later | A permitted Reddit project/self-promotion thread | A relevant technical explanation and working example | The exact current subreddit/thread and rules must be verified; public rules were not readable in this review, so this channel is not cleared |
| Later credibility | sklearn related-project listings or contrib consideration | Documentation, compatibility evidence, and a substantive maintained integration | Review current inclusion requirements first; inclusion and repository transfer are separate decisions, not launch prerequisites |

The project owner requested wider promotion after planning and selected X,
English articles, and overseas communities on 2026-11-10, with publication copy
reviewed before publishing. Destination-specific accounts, tools, final content,
and timing are still to be established. Prepare concrete artifacts for that
review; do not treat the current plan as evidence that an account is connected.
Do not publish drafts until the user has reviewed them, or claim that local
unpublished fixes are already available in the public release.

Read on 2026-11-09:

- [Show HN guidelines](https://news.ycombinator.com/showhn.html): a runnable thing
  people can try, with the maker participating.
- [HN guidelines](https://news.ycombinator.com/newsguidelines.html): human-written
  posts and comments; no automated posting. These rules make HN a user-owned
  action, unlike ordinary publishing automation.
- [sklearn related projects](https://scikit-learn.org/stable/related_projects.html)
  and [contrib workflow](https://github.com/scikit-learn-contrib/scikit-learn-contrib/blob/master/workflow.md)
  describe related-project and quality expectations; neither is an endorsement
  of scikit-jev.

### Initial publication copy for user review

Drafted on 2026-11-10; not approved, scheduled, or posted. The user explicitly
requires review before publication. Verify the actual posting account and public
artifact availability first. These short posts avoid claims about task accuracy,
production readiness, first-in-category status, or superiority over competitors.
They link to the existing public repository; announcement of the new example
must wait until its source is public.

English X post:

```text
Turn typed questions about text and JSON into probability features for scikit-learn.

scikit-jev is a pre-alpha Jev integration with a question transformer and zero-shot estimators. Looking for feedback on ML workflows.
https://github.com/engkimo/scikit-jev
```

Japanese X post:

```text
文章・JSONへの質問を、確率付きの特徴量に。scikit-jevはTypeSafe Jevをscikit-learnのPipelineで使うためのpre-alphaライブラリです。特徴量化・評価まわりのフィードバックを募集中です。
https://github.com/engkimo/scikit-jev
```

English article brief, to expand into full copy after the example is reviewed:

- Working title: "From questions about text to probability features in scikit-learn".
- Reader: a Python ML practitioner combining support text/JSON and classical models.
- Lead with the concrete question bank: refund, topic, urgency; show named columns
  and their mapping to the 2-by-6 example result.
- Walk through the offline example, clearly identifying every model response as
  synthetic. Explain which parts are sklearn behavior and which are the backend.
- Demonstrate metadata from one `evaluate` call, zero-call fit, and repeated-call
  implications of CV. Distinguish logical calls, HTTP retries, and unknown tokens.
- Explain what requires hosted-service credentials and what remains unverified:
  semantic utility on a real dataset, probability quantization, and model replay.
- Credit sklearn-decision, tab-jev, and jevframe; describe the specific contract
  choices rather than claiming the workflow is new.
- Ask for feedback on integration and evaluation, with links to public examples
  and issue reporting only once those artifacts exist at the destination.

This is an article brief, not a finished or approved article. A demo clip and
the long-form English/Japanese copy are still pending. HN is excluded from these
generated drafts because its rules require the user's own writing.

### Growth measurements and decision rules

Use 100, 500, 1,000, and 3,000 stars as visibility checkpoints without promising
dates. Check these alongside adoption evidence:

| Signal | Measurement | Action |
| --- | --- | --- |
| Visibility | Aggregate repository visitors/referrers where available, article views, demo views, and star growth | Establish actual channel conversion before buying effort with more cross-posting |
| Activation | Reported successful installs and example completions; opt-in feedback on time to first result | Fix installation and onboarding before broadening the campaign |
| Use | Independent users trying their own workflows, repeat-use reports, actionable issues and external contributions | Prioritize features that remove observed blockers rather than matching competitor checklists |
| Quality | Offline contract results, version/OS compatibility, unresolved correctness defects | A correctness regression blocks release promotion until addressed |
| Positioning | Whether users mention feature generation, evaluation correctness, or execution control as their reason to use it | If the specific value is not a reason to adopt, narrow or collaborate instead of adding more wrappers |

Keep the distribution log to public post URLs, dates, aggregate counts, and
lessons. Do not add telemetry by default or store account credentials, private
messages, or user/customer records. A first review should follow the initial
launch wave; the four-to-six-week adoption review decides whether to continue
independently, specialize further, or contribute the reusable contract work
upstream. The 3,000-star ambition can span multiple releases and is not the exit
gate for this development cycle.

### Work deferred until evidence requires it

Bounded concurrency, a cache, budgets, and an adapter remain candidates. Prioritize
them only after a concrete measured blocker appears, with call/attempt semantics,
cache isolation and deletion, and unknown usage specified first. Local
Transformers inference, exemplar encoders, automatic question discovery, and a
generic multi-provider platform are not part of this cycle. Do not start a new
website, paid promotion, or a broad feature race before the runnable workflow and
adoption evidence exist.

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
- Reproducibility: document requestable model names separately from resolved
  versions, and use pinned request IDs only where the service supports them.
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
| Moving model alias changes behavior | Record resolved IDs; use requestable pinned versions only where supported; do not promise replay from a response version alone |
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
