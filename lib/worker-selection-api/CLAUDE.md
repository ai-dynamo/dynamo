# lib/worker-selection-api

This crate defines the stable native contract used by trusted Dynamo worker-selection plugins. Keep it zero-dependency and independent of the router runtime and loader.

## Structure

- Keep one flat public API at the crate root; internal source modules stay private and their public types are re-exported from `lib.rs`.
- `lib.rs` owns the raw `repr(C)` ABI types, constants, root exports, and contract tests.
- `plugin.rs` owns the safe authoring API, input validation, panic containment, export shim, and export macro.
- Runtime loading, pipeline orchestration, eligibility, pinning, validation, reservation, and accounting belong in the KV router.
- Future filter, scorer, and picker contracts should share this crate and its worker types, but do not add modules or ABI entries before those contracts exist.

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
