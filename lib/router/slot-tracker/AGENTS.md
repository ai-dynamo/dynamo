# lib/router/slot-tracker

`dynamo-slot-tracker` is the slot tracker stage: bookings per rank in, load and fullness out. Active sequences, the prompt registry, block and prefill trackers, load projection, the fullness rule (`RankLoad`, `BusyThresholds`), the booking descriptor, and the replication wire; the replica mesh, the registry, and the HTTP server stay in `dynamo-kv-router` as adapters.

- Depends on (Dynamo crates): `dynamo-router-protocol`, `dynamo-tokens`.
- Must not depend on: other stages, the kernel, kv-router, the runtime, any socket transport (tokio for channels and timers only).
- Hosts import this crate through its `host` module.
- Check: `bash scripts/router_layering_check.sh` (and `--host`).
- Design: `lib/router/README.md`.

`record_full_worker_ranks_at` fills a caller-owned `FullRankCounts` from the derived
prompt registry at one instant. It holds one read guard and creates no projection
map or per-request ownership state. The scheduler uses it when exclusive-affinity
costing or queued readiness needs current host-reject fullness.
