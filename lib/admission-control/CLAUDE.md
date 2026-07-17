# lib/admission-control

This crate contains built-in implementations of `PolicyClassAdmissionPolicy`. Keep the generic policy contract and request ownership in `dynamo-kv-router`; keep algorithm-specific state and decisions in a submodule here.

## Structure

- `src/lib.rs` exposes policy modules and preserves the crate's public convenience re-exports.
- `src/session_aware/` contains the complete Session-Aware Admission Control implementation and its local instructions.

## Boundaries

- Policies receive read-only request facts and return `AdmissionDecision` or `AdmissionAction` values.
- The KV-router policy queue owns queued requests, validates actions, applies placement, and delivers lifecycle events.
- A policy must not own `SchedulingRequest`, call the selector, reserve worker slots, or reproduce queue accounting.
- Policy-family and cache-bucket classification, including exact-placement reclassification, remain KV-router behavior outside every concrete admission policy.
- Session-Aware Admission Control binds either to an explicit class or to the sole bucket of a policy family. Do not attach it to one class in a multi-bucket family; a session can be reclassified into another physical bucket on a later turn and escape that class-local program table.
- Keep each concrete algorithm together. Do not split accounting, pressure, victim selection, packing, or placement into separate plug-ins without a second implementation that requires the seam.
- Preserve root re-exports when moving implementation details so downstream users do not need import changes.

## Validation

Run `cargo test -p dynamo-admission-control` and `cargo clippy -p dynamo-admission-control --all-targets -- -D warnings` after changes.
