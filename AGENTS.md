# Local Codex instructions

- Read `MEMORY.md`, `docs/architecture.md`, and the relevant portion of
  `docs/implementation-plan.md` before changing public behavior.
- Treat `MEMORY.md` as the durable project-memory summary and update it only when
  a lasting design decision or implementation milestone changes.
- Preserve the release-blocking invariants in `docs/architecture.md`.
- Never add API keys, tokens, captured private states, or live provider responses
  to source, tests, fixtures, logs, configuration, or documentation.
- Unit and PR tests must use fake or recording backends and must not make network
  requests. Live API tests must be explicitly marked and opt-in.
- Keep sklearn estimator constructors assignment-only and clone-safe. Do not put
  live clients or runtime sessions in estimator parameters or fitted state.
- Do not add silent fallback predictions for failed or invalid responses.
- Use `apply_patch` for repository file edits.
