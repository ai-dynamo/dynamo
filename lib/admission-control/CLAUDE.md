# lib/admission-control

This crate contains built-in implementations of `PolicyClassAdmissionStrategy`. Keep the generic strategy contract and request ownership in `dynamo-kv-router`; keep algorithm-specific state and decisions in a submodule here.

## Structure

- `src/lib.rs` exposes strategy modules and preserves the crate's public convenience re-exports.
- `src/session_aware/` contains the complete Session-Aware Admission Control implementation and its local instructions.

## Boundaries

- Strategies receive read-only request facts and return `AdmissionDecision` or `AdmissionAction` values.
- The KV-router admission controller owns queued requests, validates actions, applies placement, and delivers lifecycle events.
- A strategy must not own `SchedulingRequest`, call the selector, reserve worker slots, or reproduce queue accounting.
- Keep each concrete algorithm together. Do not split accounting, pressure, victim selection, packing, or placement into separate plug-ins without a second implementation that requires the seam.
- Preserve root re-exports when moving implementation details so downstream users do not need import changes.

## Validation

Run `cargo test -p dynamo-admission-control` and `cargo clippy -p dynamo-admission-control --all-targets -- -D warnings` after changes.
