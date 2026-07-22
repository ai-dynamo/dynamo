# lib/worker-selection-api

This crate defines the stable native contract used by trusted Dynamo worker-selection plugins. Keep it zero-dependency and independent of the router runtime and loader.

## Structure

- Keep one flat public API at the crate root; internal source modules stay private and their public types are re-exported from `lib.rs`.
- `lib.rs` owns the raw `repr(C)` ABI types, constants, root exports, and contract tests.
- `plugin.rs` owns the safe authoring API, input validation, panic containment, export shim, and export macro.
- Runtime loading, pipeline orchestration, eligibility, pinning, validation, reservation, and accounting belong in the KV router.
- Future filter, scorer, and picker contracts should share this crate and its worker types, but do not add modules or ABI entries before those contracts exist.

## Builder Contract

Implement `WorkerSelectorPlugin` and invoke `export_worker_selector_plugin!` from a trusted Rust `cdylib`. The safe trait is the supported authoring surface; raw `*V1` types exist for the host ABI and non-Rust implementations.

The host creates one plugin instance per decode or prefill router role, calls `required_worker_inputs()` once, then calls `select()` for each unpinned request. Calls for one instance are serialized. `Selection::Worker(index)` selects an index shared by every present worker column; `Selection::UseDefault` delegates the request to Dynamo's configured selector. Dynamo bounds-checks and revalidates a selected worker before reservation and dispatch.

### Request Data

`SelectionInput` always exposes this request data without a `WorkerInputs` flag:

| Accessor | Meaning |
|---|---|
| `block_size()` | KV block size in tokens |
| `isl_tokens()` | Input sequence length in tokens |
| `expected_output_tokens()` | Optional expected output length in tokens |
| `request_id()` | Optional request ID |
| `session_id()` | Optional session ID |
| `selection_mode()` | Query-only, tracked, or tracked-with-admission mode |
| `tracks_prefill_tokens()` | Whether Dynamo tracks active prefill tokens |
| `has_shared_cache_hits()` | Whether the request has shared-cache overlap data |

### Worker Data

Worker columns contain only currently eligible workers. Every present column has `worker_count()` entries and uses the same unspecified order. Combine required flags with `|`; accessors for unrequested columns return `None`.

| `WorkerInputs` flag | Accessor | Per-worker value |
|---|---|---|
| `NONE` | None | No workers are enumerated; request data remains available and the plugin can return `UseDefault` |
| `IDENTITY` | `worker_ids()`, `dp_ranks()` | Runtime worker ID and data-parallel rank; automatically included for every nonempty declaration |
| `CACHED_TOKENS` | `cached_tokens()` | Effective cached prompt tokens |
| `CACHE_TIERS` | `cache_tiers()` | Effective, device, host-pinned, disk, and shared-cache overlap in KV blocks |
| `LOAD` | `loads()` | Active prefill tokens, active decode blocks, and additional blocks introduced by this request |
| `CAPACITY` | `capacities()` | Published total KV-block and maximum batched-token capacities; either may be unavailable |
| `ROUTING` | `routing()`, `worker_stable_routing_id()` | Optional restart-stable routing ID and preferred-taint cost multiplier |
| `DEFAULT_COST` | `default_costs()` | Dynamo's complete configured cost before temperature sampling; lower is preferred |
| `DEFAULT_KV_OVERLAP` | `default_kv_overlaps()` | Weighted and decay-adjusted KV-overlap credit used by the default cost, in blocks |
| `DEFAULT_DECODE_LOAD` | `default_decode_loads()` | Projected decode load used by the default cost, in blocks |

### Execution Constraints

- `select()` runs synchronously on the scheduler actor's request hot path. Keep it bounded and non-blocking; do not perform network or filesystem I/O.
- Input slices and strings are borrowed only for the callback and must not be retained.
- Mutable strategy state is allowed because calls for one instance are serialized. Separate decode and prefill instances may execute on different threads.
- Returning an error fails that selection call; return `UseDefault` when delegation is intended.
- Pinning, eligibility, final validation, accounting, reservation, and dispatch remain host-owned and cannot be overridden by the plugin.

## ABI Rules

- Do not reorder, remove, or change v1 ABI fields or constants. Add a new ABI version for incompatible changes and update the layout test for every ABI change.
- No Rust-owned allocation, trait object, reference, string, or collection crosses the dynamic-library boundary. The ABI uses C-layout values, raw pointers, lengths, and caller-owned error storage.
- Treat every pointer, length, struct size, enum value, bitset, and returned worker index as untrusted at the boundary and validate it before use.
- Worker columns are borrowed, demand-declared, and index-aligned. Materialization and allocation reuse remain host responsibilities.
- Never unwind across the ABI. Generated callbacks must contain plugin panics and return an ABI status.

## Validation

- `cargo test -p dynamo-worker-selection-api`
- `cargo clippy -p dynamo-worker-selection-api --all-targets -- -D warnings`
- `cargo doc -p dynamo-worker-selection-api --no-deps`
- `cargo fmt --all --check`
- `cargo metadata --locked --no-deps --format-version 1`
