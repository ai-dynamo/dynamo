@AGENTS.md

## Policy-Class Admission Strategy API

The public contract in `lib/kv-router/src/scheduling/queue_admission/` is the narrow boundary between the router and a policy-class admission algorithm. This layer defines vocabulary only; request storage, queue integration, lifecycle delivery, strategy registration, and concrete algorithms belong to the runtime host and composition layers.

### Contract

- `AdmissionId` is the only universal request fact. It is host-issued and must be the strategy's lifecycle key.
- `AdmissionRequest` borrows optional session identity, full logical `context_tokens`, and a live `WorkerEligibility` capability. A strategy may ignore optional facts or return `Bypass` when they do not apply.
- `WorkerEligibility` is read-only and may be retained by deferred work. Its snapshot distinguishes structurally legal workers from workers currently available after transient router conditions.
- `Bypass` opts out of strategy lifecycle handling. `Ready(Any)` preserves ordinary routing, `Ready(Exact(worker))` adds a placement constraint, and `Defer` asks the runtime host to retain the request.
- `AdmissionEvent` describes dispatch, successful completion, abort, and reconciliation. `Completed` carries the authoritative final logical context; `Aborted` commits no new context.
- `MakeReady` releases a deferred admission ID with optional exact placement. Duplicate or unknown actions are harmless by contract.
- `reconcile_interval` requests a maximum interval between reconsideration opportunities while work is deferred; it is not a dedicated strategy thread or timer.

### Ownership and extension rules

- A strategy owns algorithm state only. It must not own `SchedulingRequest`, request payloads, queue ordering, worker slots, or final selection.
- Exact placement constrains the existing selector; it does not bypass structural validation, scoring, or booking.
- Add request facts as typed accessors only after a concrete strategy proves they are necessary. Expose changing system state through a narrow read-only capability instead of copying actor state or adding an untyped metadata bag.
- Keep algorithm-specific configuration in flattened `QueueAdmissionConfig::options`; the generic router validates only the strategy name and container shape.
- This is a Rust dependency-inversion boundary, not a stable dynamic-plugin ABI. Built-in implementations should live outside `dynamo-kv-router` and use explicit registration at a composition root.
