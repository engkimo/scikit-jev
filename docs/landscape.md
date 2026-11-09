# Decision-model sklearn landscape

Reviewed: 2026-11-09. This review records public interfaces and repository facts,
not independent accuracy, reliability, or performance benchmarks.

## Direct overlap: sklearn-decision

[`jaketimothy/sklearn-decision`](https://github.com/jaketimothy/sklearn-decision)
was created on 2026-09-26; its latest pushed timestamp at review was 2026-09-29.
It predates scikit-jev and was missed by the original review.

The public README and source expose `QuestionFeaturizer`, `NoulClassifier`,
`ChoiceClassifier`, `ScoreRegressor`, and `ChoiceEncoder`. They support hosted
Jev and local Transformers backends, calibration, and memory/SQLite caching.
The package is installed from GitHub; its PyPI JSON endpoint returned 404.

The overlap with scikit-jev's core thesis is substantial. A question featurizer
plus zero-shot sklearn estimators alone is not a unique product position.
Potential work to evaluate next is strict response-contract coverage,
predictable request accounting, credential-safe errors, and compatibility across
sklearn versions. These are priorities for scikit-jev, not verified claims that
other implementations lack them. No strategic rewrite was made in this review.

## scikit-llm

The [official repository](https://github.com/fnnx-ai/scikit-llm) main HEAD remains
`cf668038e311f6b73d13bed2b08c5455221de554`, dated 2026-01-21. A scan of its
72 Python, TOML, and Markdown files found no Jev, TypeSafe, `system_one`, or
System One references. Searches of public issues/PRs for Jev, TypeSafe, and
"system one" returned no results. This does not cover private plans or every
historical branch. The [PyPI release](https://pypi.org/project/scikit-llm/)
remains 1.4.3, uploaded on 2026-01-21.

## Adjacent projects

| Project | Public scope | Relation |
| --- | --- | --- |
| [tab-jev](https://github.com/edamame-labs/tab-jev) | Typed Jev judgments combined with tabular predictors, rubric features, pluggable backends | Substantial overlap in semantic features plus a downstream model |
| [jevframe](https://github.com/ktaletsk/jevframe) | pandas/Polars row evaluation, complete distributions, async concurrency, optional caching | DataFrame-oriented adjacent integration |

Both projects predate the original October review. PyPI currently lists
[`tab-jev` 0.0.1](https://pypi.org/project/tab-jev/) and
[`jevframe` 0.1.0](https://pypi.org/project/jevframe/).

## Search limits

Sources included public GitHub repository/README search, project source,
scikit-llm issue/PR search, PyPI JSON metadata, and web search. Search coverage is
not proof that no other equivalent project exists. The prior claim that no
comprehensive sklearn decision-model package existed is superseded by the
sklearn-decision finding above.
