# lib/router/slot-tracker

`dynamo-slot-tracker` is the slot tracker stage: bookings per rank in, load and fullness out. Active sequences, the prompt registry, block and prefill trackers, load projection, the fullness rule (`RankLoad`, `BusyThresholds`), the booking descriptor, and the replication wire; the replica mesh, the registry, and the HTTP server stay in `dynamo-kv-router` as adapters.

- Depends on (Dynamo crates): `dynamo-router-protocol`, `dynamo-tokens`.
- Must not depend on: other stages, the kernel, kv-router, the runtime, any socket transport (tokio for channels and timers only).
- Hosts import this crate through its `host` module.
- Modules are private except `host`. The root list in `src/lib.rs` is the in-router contract (kernel, composition, other stages); `host` is the subset promised to hosts outside `lib/router`. A new cross-crate name goes on the root list, and on `host` only when a host needs it.
- Check: `bash scripts/router_layering_check.sh` (and `--host`).
- Design: `lib/router/README.md`.
