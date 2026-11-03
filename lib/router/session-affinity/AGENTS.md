# lib/router/session-affinity

`dynamo-session-affinity` is the session affinity stage: session bindings in, constraints out. The binding table, leases, versions, the replica sink port, config, and the replication wire; the mesh that carries the wire and the frontend coordinator stay outside (kv-router adapter, lib/llm host).

- Depends on (Dynamo crates): `dynamo-router-protocol`.
- Must not depend on: other stages, the kernel, kv-router, the runtime, any socket transport (tokio for `Notify` and timers only).
- Hosts import this crate through its `host` module.
- Check: `bash scripts/router_layering_check.sh` (and `--host`).
- Design: `lib/router/README.md`.
