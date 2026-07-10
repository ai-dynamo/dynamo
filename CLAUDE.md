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
- Land algorithm-specific configuration and its concrete registration atomically; configuration must not accept strategy names or options that no composition root can validate.
- This is a Rust dependency-inversion boundary, not a stable dynamic-plugin ABI. Built-in implementations should live outside `dynamo-kv-router` and use explicit registration at a composition root.

## Policy-Class Admission Runtime

The KV router hosts admission strategies behind the API above. The scheduler queue actor owns orchestration and lifecycle correlation; strategies own only algorithm state.

```text
request
  -> scheduler queue actor
  -> policy queue
  -> configured policy-class admission strategy
  -> ready/deferred storage owned by PolicyQueue
  -> worker selector
  -> engine
```

- `PolicyQueue` owns all request payloads. A strategy returns `Bypass`, `Ready`, or `Defer`; it never owns a `SchedulingRequest`.
- The actor correlates admitted work by `AdmissionId`, delivers dispatch/completion/abort/reconcile events, and applies `MakeReady` actions.
- Admission receives a live, read-only `RequestProgress` handle. The paired updater records monotonic logical context growth with one relaxed atomic operation on the response path; strategies read it during their own admission or reconciliation work.
- `RequestGuard` owns terminal lifecycle delivery. Successful completion reports the authoritative final logical context; dropping a tracked request before completion reports an abort.
- `free()` remains worker-slot cleanup. It does not terminate admission state and retains its existing public behavior.
- `Ready(Exact(worker))` is a constraint on the normal selector, not a routing bypass. The selector still validates, scores, and books the worker.
- Built-in strategies are registered explicitly in `dynamo-llm`. Caller-provided strategies use the same map and are not replaced by built-ins.
- Do not add per-output-token actor messages. Changing request state should use a narrow capability such as `RequestProgress` only after an admission algorithm proves it needs the data.
