# lib/router/worker-selection

`dynamo-worker-selection` is the worker selection stage and the public plugin API for custom policies (`WorkerFilter`, `WorkerScorer`, `WorkerPicker`). Hosts outside `lib/router` import through `host` and plugins through `dynamo_kv_router::plugins::prelude`.

## Working rules: the public plugin API

`WorkerFilter`, `WorkerScorer`, `WorkerPicker`, the contexts, the candidate views, and `WorkerInputs` are the public Rust contract for custom worker selection (reached by plugins as `dynamo_kv_router::plugins::worker_selection` and the prelude). Treat each public item as a versioned external API.

- Do not add a public field, accessor, input group, type, or re-export unless the task explicitly requires a new external policy capability. An internal need in the default scorer, logging, tests, or the selection service does not justify a public API addition.
- Trace a proposed value to its source before you expose it; record whether it is raw state, a derived estimate, or an intermediate in Dynamo's default formula. Expose raw facts or complete user-facing abstractions, never partial credit, weighted overlap, legacy arithmetic, or another intermediate whose meaning depends on the default policy.
- Keep struct fields private; add the narrowest accessor that supports the approved use case. Return `Option` for absent data; do not replace absence with a sentinel.
- When a protocol context is projected into a worker-selection type, destructure the source without `..` and handle every field explicitly; map only approved policy fields, bind each field that stays internal by name and say why, so a new source field is a compile error until its policy meaning, cost, documentation, and contract test are reviewed.
- Document the source, units, lifetime, staleness, missing-data behavior, weighting, and clamping for each public value.
- Require callers to name each `WorkerInputs` group they use; do not add a public `ALL` shortcut. Before adding a value to an existing group, account for its calculation and retained-column cost for every policy that requests that group.
- Do not pass the full `SchedulingRequest`, worker maps, router configuration internals, default-score weights, or host-owned eligibility and reservation state to custom policies. Eligibility, picker-row validation, accounting, and reservation stay in the host path.
- The builtin default policy lives in `lib/router-plugins/builtin` and uses only this public API; the kernel's reference selector is test- and benchmark-only. Plugins own their score calculation and worker choice.
- Before each public API addition: search all accessors, re-exports, documentation, examples, and external-looking call sites; add one focused contract test that uses the new value through `WorkerFilter`, `WorkerScorer`, or `WorkerPicker`; update `docs/fern/pages/developer-guide/knowledge-base/modular-components/router/custom-worker-selection.mdx` and one canonical example; if the signal adds work, storage, allocation, or another scan to the selection path, run the worker-selection benchmark.
