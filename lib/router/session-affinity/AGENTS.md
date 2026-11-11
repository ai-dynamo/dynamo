# lib/router/session-affinity

`dynamo-session-affinity` is the session affinity stage: session bindings in, constraints out. The binding table, leases, versions, the replica sink port, config, the worker target a session binds to (`WorkerAffinityTarget`), and the replication wire; the mesh that carries the wire and the frontend coordinator stay outside (kv-router adapter, lib/llm host).

- Depends on (Dynamo crates): `dynamo-router-protocol`.
- Must not depend on: other stages (no stage depends on it except admission and worker-selection, for the affinity target), the kernel, kv-router, the runtime, any socket transport (tokio for `Notify` and timers only).
- Hosts import this crate through its `host` module.
- Modules are private except `host`. The root list in `src/lib.rs` is the in-router contract (kernel, composition, other stages); `host` is the subset promised to hosts outside `lib/router`. A new cross-crate name goes on the root list, and on `host` only when a host needs it.
- Check: `bash scripts/router_layering_check.sh` (and `--host`).
- Design: `lib/router/README.md`.
