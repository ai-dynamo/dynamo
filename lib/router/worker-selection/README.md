<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# dynamo-worker-selection

Candidate rows in, a pick out. The router's worker selection stage: the plugin API (`WorkerFilter`, `WorkerScorer`, `WorkerPicker`) and the composed `WorkerSelectionPolicy` that runs them over the rows a `CandidateRowSource` yields and returns a `RowPick` or a `SelectionFailure`; the `WorkerSelectionContext` view of the request; the candidate rows (`CandidateData`, `WorkerInputRows`) and their gated views; `CandidatePreparer` and `PreparedRows` for rows built before the request reaches the actor; runtime parameter updates (`RuntimePolicy*`); the selection slice of the router config (`SelectionConfig`); the bounded selection trace; the conditional-disaggregation policies; and the LoRA narrowing filter. The kv-transfer source summary (`WorkerKvTransferKeys`, `KvTransferSourceSummary`) lives in `dynamo-kv-hints`; the kernel builds the `WorkerTransferInput` column from it. `lib/kv-router` re-exports the crate at `plugins::worker_selection` and keeps the kernel side: `WorkerSelector`, the request-to-row source, the config- and projection-derived columns, the policy factory and registry, and the reference selector.

Dependencies: `dynamo-router-protocol`, `dynamo-slot-tracker` (for the fullness rule, `RankLoad` and `BusyThresholds`, which the slot tracker owns), and `dynamo-session-affinity` (for `WorkerAffinityTarget`, the target a session binds to) and nothing router-side beyond them; no axum, reqwest, zmq, Prometheus, or `dynamo-runtime` under any feature. The `bench` and `testing` features add the reference picker (`parking_lot`) for parity tests and benchmarks.

Part of the router split (see `lib/router/README.md`).

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
