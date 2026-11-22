# lib/router/session-affinity

`dynamo-session-affinity` is the session affinity stage: it binds a session to a worker and turns that binding into a routing constraint. Hosts outside `lib/router` import through `host`.

## Rules

- **Adding a public item a host needs.** Re-export it by name from `src/host.rs`; that list is the host contract.
- **Changing how a session binds.** Keep the order acquire, commit, release. `try_acquire` returns a `Hold`; the host commits it with the worker it dispatched to; the `AffinityLease` keeps the binding alive until it drops. In `Hard` mode a dispatch to another worker or rank is an error (`validate_dispatch_target`); in `Soft` mode the session rebinds to it (`rebound_target`).
- **Changing `apply_replica_update`.** A replica update never replaces an `Initializing` entry. It replaces the target of an unexpired binding only when its `AffinityVersion` is strictly higher (sequence first, then writer id). Every result is a `ReplicaApplyOutcome` variant; add a variant for a new case.
- **Changing `AffinityBindingEvent` (`src/wire.rs`).** Router replicas send it to each other. Give a new field `#[serde(default)]` so an event without it still decodes, and don't rename or remove a field.
- **Adding a transport.** Not here. The table publishes through `AffinityReplicaSink`; the mesh that carries the events is in `lib/router/kv-router/src/services/common/replica_sync.rs`.
- **Adding a test hook.** Put it behind `#[cfg(any(test, feature = "testing"))]`, beside the existing hooks. No normal dependency may enable `testing`.

## Test

```bash
cargo test -p dynamo-session-affinity --lib
cargo test -p dynamo-llm --lib session_affinity::   # replication between tables
```
