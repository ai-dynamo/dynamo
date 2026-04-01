# KVBM TensorRT-LLM Integration Execution Plan

Last updated: 2026-04-01 00:40:12 UTC

## Active state

- Mandatory read order remains:
  - `Agents.md`
  - `PLANS.md`
  - `docs/design-docs/kvbm-g3pb-plan.md`
- Current branch shape:
  - detached `HEAD`
  - last previously pushed rename baseline already exists on
    `origin/mf/kvbm-g4-v2`
- Current implementation direction:
  - `G3PB` is the peer-cache replacement for the unlanded `G4` disk-identity
    surface
  - peer ownership remains rendezvous-hash based
  - remote identity is keyed by `sequence_hash` only
  - peer-local persistence stays hidden behind `G3pbPeerStorage`

## Current run (2026-03-31 23:09:30 UTC)

### Summary of accomplishments in this run

- ✅ Resolved the NIXL backend-registration failure for `PinnedStorage(local host) -> remote host staging` transfer creation
  - The root cause was passing the local agent name into `createXferReq` instead of the remote peer agent name
  - Fixed in `lib/llm/src/block_manager/block/transfer/nixl.rs`
  - Full worker/backend smoke test now passes with remote write, remote read, local host registration, and device onboard
- ✅ Added KVBM-side G3PB admission policy wiring
  - Implemented in `lib/llm/src/block_manager/offload/g3pb_filter.rs`
  - Default admission when a block has been reused at least once
  - Environment override `G3PB_OFFLOAD_ALL` for eager admission of every block
  - Keeps `G1 -> G3PB` semantics as copy/replication, not ownership transfer
  - All tests pass (6 passed for G3PB filter, 13 passed for G3PB distributed)
- ✅ Validated the complete G3PB smoke path:
  - `cargo test --manifest-path lib/llm/Cargo.toml g3pb:: --lib` (13 passed)
  - `cargo test --manifest-path lib/llm/Cargo.toml g3pb_filter --lib` (6 passed)
  - `cargo check --manifest-path lib/llm/Cargo.toml --bin kvbm_g3pb_backend --bin kvbm_g3pb_worker_smoke --bin kvbm_nixl_transfer_smoke` (pass)
  - `cargo build --manifest-path lib/llm/Cargo.toml --bin kvbm_nixl_transfer_smoke --bin kvbm_g3pb_backend --bin kvbm_g3pb_worker_smoke` (pass)
  - `target/debug/kvbm_nixl_transfer_smoke` (pass)
  - host-only backend smoke using descriptor endpoints (pass)
  - full `target/debug/kvbm_g3pb_worker_smoke` with remote fetch and local onboard (pass)
- ✅ Migrated backend + smoke metadata/control traffic to Dynamo request-plane + discovery
  - added shared `G3pbRpcRequest` / `G3pbRpcResponse`
  - added `G3pbRequestPlaneClient`
  - added shared endpoint constants:
    - `G3PB_NAMESPACE=kvbm-g3pb`
    - `G3PB_COMPONENT_NAME=peer-cache`
    - `G3PB_ENDPOINT_NAME=g3pb`
  - `kvbm_g3pb_backend` now serves a Dynamo `Ingress` endpoint instead of Axum HTTP routes
  - `kvbm_g3pb_worker_smoke` now discovers peers from Dynamo discovery instead of `--backend-url`
- ✅ Revalidated the migrated request-plane path end-to-end
  - shared file-backed discovery smoke:
    - backend 53:
      `env DYN_DISCOVERY_BACKEND=file DYN_FILE_KV=/tmp/g3pb-discovery.BBwJzW target/debug/kvbm_g3pb_backend --worker-id 53 --host-blocks 16`
    - backend 54:
      `env DYN_DISCOVERY_BACKEND=file DYN_FILE_KV=/tmp/g3pb-discovery.BBwJzW target/debug/kvbm_g3pb_backend --worker-id 54 --host-blocks 16`
    - worker smoke owning to worker 53:
      `env DYN_DISCOVERY_BACKEND=file DYN_FILE_KV=/tmp/g3pb-discovery.BBwJzW target/debug/kvbm_g3pb_worker_smoke --worker-id 21 --host-blocks 8 --count 4`
      - pass
    - worker smoke owning to worker 54:
      `env DYN_DISCOVERY_BACKEND=file DYN_FILE_KV=/tmp/g3pb-discovery.BBwJzW target/debug/kvbm_g3pb_worker_smoke --worker-id 1001 --host-blocks 8 --count 1 --sequence-start 2025`
      - pass
- ✅ Fixed request-plane smoke teardown regression
  - the first rewrite used `Worker::from_settings()` in `kvbm_g3pb_worker_smoke`
  - the data path completed successfully but repeated runs could segfault during teardown
  - switching that binary back to `tokio::main` removed the segfault while preserving the request-plane/discovery refactor

### Context re-read

- `Agents.md`
- `PLANS.md`
- `docs/design-docs/kvbm-g3pb-plan.md`
- `../dynamo/lib/runtime/examples/hello_world/src/bin/server.rs`
- `../dynamo/lib/runtime/examples/hello_world/src/bin/client.rs`
- `../dynamo/lib/runtime/src/component.rs`
- `../dynamo/lib/runtime/src/component/client.rs`
- `../dynamo/lib/runtime/src/component/endpoint.rs`
- `../dynamo/lib/runtime/src/distributed.rs`
- `../dynamo/lib/runtime/src/storage/kv.rs`
- `lib/llm/src/bin/kvbm_g3pb_backend.rs`
- `lib/llm/src/bin/kvbm_g3pb_worker_smoke.rs`
- `lib/llm/src/block_manager/distributed.rs`
- `lib/llm/src/block_manager/distributed/g3pb.rs`
- `lib/llm/src/bin/kvbm_nixl_transfer_smoke.rs`
- `lib/llm/src/block_manager/distributed/transfer.rs`
- `lib/llm/src/block_manager/distributed/worker.rs`
- `lib/llm/Cargo.toml`

### Current findings before edits

- Dynamo already has the request-plane and discovery pieces this slice needs:
  - `endpoint_builder().handler(...).start()` automatically registers the
    endpoint in discovery using the active request-plane transport
  - `component.endpoint(...).client().await?` plus `PushRouter` provides the
    normal typed request path to discovered peers
  - `Client::wait_for_instances()` and the underlying discovery watch already
    provide the peer membership view needed by the worker smoke
- a cross-process smoke does not require etcd for this slice:
  - `DistributedConfig` supports `DYN_DISCOVERY_BACKEND=file`
  - the shared file-backed store root comes from `DYN_FILE_KV`
  - that is sufficient for multiple backend/worker processes on one machine to
    discover the same `G3PB` endpoint instances
- the long-term control-plane direction is therefore clear:
  - backend should become a normal Dynamo component endpoint instead of an
    ad hoc Axum HTTP service
  - worker smoke should discover backend peers from Dynamo discovery instead of
    taking `--backend-url`
  - health/metadata RPCs should move onto a typed request-plane protocol while
    NIXL/UCX remains the bulk transfer path
- that direction is now implemented and validated with a shared file-backed
  discovery store
- the remaining transport cleanup issue is narrower now:
  repeated request-plane smokes complete their data path and exit successfully,
  but still log `invalidateRemoteMD ... NIXL_ERR_NOT_FOUND` during teardown
- worktree was not clean at pickup:
  - modified `lib/llm/src/block_manager/distributed/g3pb.rs`
- the only uncommitted change at pickup replaced
  `let mut hits = Vec::new();` in `FoyerG3pbPeerStorage::query_blocks` with
  freeform prose about using a metadata-only `contains_key` path
- because of that local edit, the first required baseline validation no longer
  matched the handoff state and failed to compile before any new changes
- additional transport root-cause found while comparing the smoke path with the
  newer physical transfer executor:
  - `lib/llm/src/block_manager/block/transfer/nixl.rs` always calls
    `create_xfer_req(..., &nixl_agent.name(), ...)`
  - that is wrong for cross-agent transfers because `stage_put` writes target a
    remote backend agent, not the local worker agent
  - the newer v2 executor already selects the non-local agent name when
    constructing the transfer request, so the legacy helper needs the same fix

### Validation completed in this run so far

- `cargo test --manifest-path lib/llm/Cargo.toml g3pb:: --lib`
  - fail at pickup
  - compiler error at
    `lib/llm/src/block_manager/distributed/g3pb.rs:364`
  - cause: stray prose in `FoyerG3pbPeerStorage::query_blocks`
  - pass after baseline repair (`13 passed`)
- `cargo check --manifest-path lib/llm/Cargo.toml --bin kvbm_g3pb_backend --bin kvbm_g3pb_worker_smoke`
  - pass after baseline repair
- `cargo fmt --manifest-path lib/llm/Cargo.toml --all`
  - pass
- `git diff --check`
  - pass
- `cargo check --manifest-path lib/llm/Cargo.toml --bin kvbm_nixl_transfer_smoke`
  - pass
- `cargo build --manifest-path lib/llm/Cargo.toml --bin kvbm_nixl_transfer_smoke`
  - pass
- `cargo build --manifest-path lib/llm/Cargo.toml --bin kvbm_g3pb_backend`
  - pass
- host-only `G3PB` backend smoke:
  - command:
    `target/debug/kvbm_g3pb_backend --listen 127.0.0.1:58183 --worker-id 53`
  - `/health`
    - pass, returned `{"worker_id":53,"listen":"127.0.0.1:58183"}`
- `target/debug/kvbm_nixl_transfer_smoke`
  - pass
  - output reported:
    `nixl smoke transfer complete: device->host 8 blocks, host->device 8 blocks`
  - emitted one UCX runtime warning about
    `IB_PCI_RELAXED_ORDERING=try`, but the smoke still completed successfully
- backend transport-contract milestone:
  - `cargo check --manifest-path lib/llm/Cargo.toml --bin kvbm_g3pb_backend`
    - pass after adding staged NIXL descriptor endpoints and backend host runtime
  - `cargo test --manifest-path lib/llm/Cargo.toml g3pb:: --lib`
    - pass after the transport type additions (`13 passed`)
- worker transport-contract milestone:
  - `cargo check --manifest-path lib/llm/Cargo.toml --bin kvbm_g3pb_backend --bin kvbm_g3pb_worker_smoke --bin kvbm_nixl_transfer_smoke`
    - pass after rewriting the worker smoke around `load_remote`, `stage_put`,
      `commit_put`, immutable descriptor fetch, and remote-read helper wiring
  - `cargo build --manifest-path lib/llm/Cargo.toml --bin kvbm_nixl_transfer_smoke --bin kvbm_g3pb_backend --bin kvbm_g3pb_worker_smoke`
    - pass
- host-only backend descriptor/control smoke:
  - backend:
    `target/debug/kvbm_g3pb_backend --listen 127.0.0.1:58183 --worker-id 53 --host-blocks 16`
    - pass
  - `curl http://127.0.0.1:58183/health`
    - pass
  - `offer -> stage_put -> commit_put -> query` for `sequence_hash=9001`
    - pass after making backend query authoritative over staged-committed hashes
- full worker/backend smoke:
  - command:
    `target/debug/kvbm_g3pb_worker_smoke --backend-url http://127.0.0.1:58183`
   - current status:
     pass
  - observed progress after the shared `create_xfer_req` peer-selection fix:
    - `offer`
    - `stage_put`
    - `commit_put`
    - duplicate-offer rejection
    all pass
  - current terminal error now occurs only during the remote fetch/read leg
  - current terminal error:
    `createXferReq: no specified or potential backend had the required registrations to be able to do the transfer`
  - additional NIXL warning on teardown:
    `invalidateRemoteMD: error invalidating remote metadata for agent '53Н'`
- request-plane + discovery migration validation:
  - `cargo fmt --manifest-path lib/llm/Cargo.toml --all`
    - pass after each migration milestone
  - `cargo check --manifest-path lib/llm/Cargo.toml --bin kvbm_g3pb_backend`
    - pass after replacing Axum with the Dynamo endpoint handler
  - `cargo check --manifest-path lib/llm/Cargo.toml --bin kvbm_g3pb_backend --bin kvbm_g3pb_worker_smoke`
    - pass after the worker smoke rewrite
  - `cargo test --manifest-path lib/llm/Cargo.toml g3pb:: --lib`
    - pass after shared request-plane protocol/client additions (`13 passed`)
  - `cargo test --manifest-path lib/llm/Cargo.toml g3pb_filter --lib`
    - pass (`6 passed`)
  - `cargo build --manifest-path lib/llm/Cargo.toml --bin kvbm_g3pb_backend --bin kvbm_g3pb_worker_smoke --bin kvbm_nixl_transfer_smoke`
    - pass
  - `target/debug/kvbm_nixl_transfer_smoke`
    - pass
  - request-plane discovery smoke to worker `53`
    - pass
  - request-plane discovery smoke to worker `54`
    - pass
  - repeated request-plane smokes after changing `kvbm_g3pb_worker_smoke` back to `tokio::main`
    - pass without the earlier teardown segfault

### Decisions confirmed in this run so far

- first priority in this run is to restore the committed `G3PB` baseline to a
  compiling, testable state before attempting the NIXL/UCX follow-up
- the local `foyer` note should be preserved as an actual code comment rather
  than dropped, because the underlying concern is valid:
  metadata-only query would be better than loading payloads from disk
- the baseline repair itself is intentionally minimal:
  preserve existing behavior, restore compilation, and keep the optimization
  note as a code comment instead of changing `foyer` query semantics blindly
- the missing transport slice is now isolated more precisely:
  local NIXL transfer works and host-only `G3PB` HTTP metadata/payload flow
  works, but there is still no peer-runtime seam that exposes remote
  CPU-visible staging blocks as NIXL-importable descriptors for `G3PB`
- `lib/llm/src/block_manager.rs` already contains an ignored cross-worker test
  noting that the current Rust bindings do not yet support the partial metadata
  path needed for that import style; this is relevant to any attempt to bolt
  remote descriptor exchange directly onto the current older blockset APIs
- backend implementation decision for this run:
  keep `G3pbStorageAgent` as the metadata/query contract, but add a separate
  backend-only pinned-host staging runtime that:
  - allocates host blocks from a local KVBM host pool with a UCX-enabled NIXL agent
  - reserves accepted `sequence_hash` values before transfer
  - exposes serialized blocksets plus `BlockDescriptorList` responses for
    mutable upload descriptors and immutable fetch descriptors
  - commits staged hashes into the metadata agent only after the transfer
    completion call so `query` stays visibility-safe
- shared transport structs live in `distributed/g3pb.rs`
- the backend binary owns its `BlockDescriptorList` construction locally;
  do not extend the shared block-manager API for that convenience
- the backend also needs explicit remote-agent handshake:
  the worker must publish its local serialized blockset to each backend through
  `load_remote` before any `stage_put` / `fetch` transfer so the peer agent can
  load the worker’s UCX metadata
- the worker now builds its transfer context from the block manager’s exported
  NIXL agent state instead of a separate sibling agent instance
- first transport fix landed in this run:
  `block/transfer/nixl.rs` now chooses the non-local worker/agent instead of
  always targeting the local agent during `create_xfer_req`
- the write path is therefore proven good enough for:
  local pinned host -> remote pinned host staging
- the earlier wrapper-lifetime suspicion was incorrect:
  the bug was in our code passing the local agent name instead of the remote
  peer agent name into `create_xfer_req`
- the remaining follow-up is only the separate teardown warning
  `invalidateRemoteMD ... NIXL_ERR_NOT_FOUND`, which is not currently treated
  as evidence of a `nixl-sys` ownership bug
- discovery/request-plane implementation decision for the next slice:
  - landed as planned:
    - typed `G3PB` request protocol in
      `lib/llm/src/block_manager/distributed/g3pb.rs`
    - backend served through Dynamo `Ingress`
    - worker addressed backends by discovered `instance_id`
    - shared file-backed discovery works cross-process without etcd
  - keep the old direct-HTTP control plane out of the final path
  - keep `kvbm_g3pb_backend` on the long-lived `Worker` wrapper
  - keep `kvbm_g3pb_worker_smoke` on `tokio::main` because it is a short-lived validation binary and that avoids the teardown segfault

### Remaining work in this run

- ✅ resolved the last NIXL backend-registration failure for
  `PinnedStorage(local host) -> remote host staging` transfer creation
- revised NIXL diagnosis for the current `G3PB` slice:
  - the real transfer bug was passing the local agent name into
    `createXferReq` instead of the remote peer agent name
  - keep the current `block/transfer/nixl.rs` remote-agent fix; do not revert it
  - the earlier `nixl-sys` / wrapper-lifetime suspicion was incorrect and
    should not be used as the explanation for this slice
  - the post-transfer `invalidateRemoteMD ... NIXL_ERR_NOT_FOUND` warning is
    currently treated as a separate non-blocking cleanup issue because the
    end-to-end `G3PB` smoke succeeds with remote write, remote read, local
    host registration, and device onboard
  - do not mix this slice with speculative transport additions like the
    unconditional `nixl_agent` path, `(Disk, Host)` transfer arm, or
    `disk_block_observer` surface
- ✅ add KVBM-side `G3PB` admission policy wiring:
  - default admission when a block has been reused at least once
  - environment override `G3PB_OFFLOAD_ALL` for eager admission of every block
  - keep `G1 -> G3PB` semantics as copy/replication, not ownership transfer
  - implemented in `lib/llm/src/block_manager/offload/g3pb_filter.rs`
  - tests pass (6 passed)
- ✅ refactor both `kvbm_g3pb_backend` and `kvbm_g3pb_worker_smoke` to use Dynamo request-plane + discovery instead of ad hoc HTTP base URLs:
  - register `G3PB` as a Dynamo component endpoint and discover remotes via the discovery backend
  - remove manual `--listen` / `--backend-url` control-plane wiring from the long-term path
  - keep bulk data movement on NIXL/UCX; use request-plane only for metadata/handshake RPCs
  - request-plane metadata-RPC throughput measurement is still pending
  - do not add the speculative unconditional `nixl_agent` path, `(Disk, Host)` transfer arm, or `disk_block_observer` surface in this slice; keep those deferred unless a concrete requirement appears
- ✅ next implementation slice, in order:
  - add shared typed `G3PB` RPC request/response enums plus a small client
    wrapper in `lib/llm/src/block_manager/distributed/g3pb.rs`
  - convert `kvbm_g3pb_backend` from Axum handlers to a Dynamo endpoint engine
  - convert `kvbm_g3pb_worker_smoke` from manual `--backend-url` calls to
    discovery-driven peer discovery using the new request-plane client
  - run focused compile/tests after the protocol/client slice before attempting
    the full smoke rewrite
- revalidate the next slice in layers:
  - ✅ repeated `kvbm_g3pb_worker_smoke` loop against a stable backend
    - data path remains good
    - the remaining teardown issue is the warning
      `invalidateRemoteMD ... NIXL_ERR_NOT_FOUND`
  - ✅ multi-backend ownership/routing smoke with multiple `G3PB` peers
    - ownership to worker `53` proven
    - ownership to worker `54` proven with `--sequence-start 2025`
  - larger block-count staged transfer smoke remains pending
  - backend restart/reload validation across `load_remote` plus subsequent
    `query` / `fetch` remains pending
  - ✅ worker-side compile/tests after the smoke rewrite
  - ✅ `kvbm_nixl_transfer_smoke`
  - host-only backend-only request-plane harness remains pending if we still
    want a non-worker validation equivalent to the older HTTP curl smoke
  - ✅ full `kvbm_g3pb_worker_smoke` with remote fetch and local onboard

### Concrete transport gap

- new backend seam now exists:
  - `offer` still determines ownership/admission
  - `stage_put` allocates reserved remote mutable host blocks and returns:
    - serialized remote blockset metadata
    - a mutable `BlockDescriptorList`
  - `commit_put` marks staged hashes visible to `query`
  - `fetch` now returns serialized blockset metadata plus an immutable
    `BlockDescriptorList` instead of inline bytes
- the rewritten worker already does:
  - `load_remote`
  - `stage_put`
  - remote-blockset import
  - `commit_put`
  - immutable descriptor fetch
  - `read_from_remote`
- but the first `stage_put` write still fails when NIXL tries to create the
  actual transfer request for local pinned-host blocks to remote host staging
- the error persists after:
  - publishing local blockset metadata to the backend
  - switching the transfer context to the block manager’s own registered
    NIXL agent
  - enabling `POSIX` alongside `UCX` on both worker and backend agents
- the remaining unresolved question is whether this transport slice needs:
  - `SystemStorage` rather than `PinnedStorage`
  - a different backend/registration mix

### Exact next step

- next follow-up if another run continues from here:
  - backend restart/reload validation for the request-plane path
  - larger staged-transfer smoke counts
  - explicit request-plane metadata-RPC throughput measurement
  - deeper diagnosis of the still-non-blocking teardown warning
    `invalidateRemoteMD ... NIXL_ERR_NOT_FOUND`

### Handoff for next run

The G3PB implementation is now in a stable, working state:
- All core G3PB functionality is working (offer, stage_put, commit_put, query, fetch)
- NIXL/UCX transfers are working correctly for both write and read paths
- G3PB admission policy has been implemented with configurable behavior
- All tests pass

The request-plane/discovery migration is now landed and validated.

If there is another run, spend it on the remaining validation tail:
1. backend restart/reload validation across `load_remote` + `query` / `fetch`
2. larger staged-transfer counts
3. request-plane metadata-RPC throughput measurement
4. investigation of the still-non-blocking teardown warning

The `invalidateRemoteMD` warning on teardown is currently treated as a separate non-blocking cleanup issue and should be investigated separately if it becomes problematic.

## Archived milestones (condensed)

### 2026-03-31 rename and seam reset

- decoupled the unlanded `G4` peer-cache path from disk offload state
- removed `G4BlockIndex` / disk-observer wiring from the core block-manager
  state and distributed worker path
- replaced the unlanded `distributed/g4.rs` module with
  `lib/llm/src/block_manager/distributed/g3pb.rs`
- renamed the binaries to:
  - `lib/llm/src/bin/kvbm_g3pb_backend.rs`
  - `lib/llm/src/bin/kvbm_g3pb_worker_smoke.rs`
- updated `Agents.md` and replaced the design doc with
  `docs/design-docs/kvbm-g3pb-plan.md`
- fixed local `disable_nixl()` construction in
  `lib/llm/src/block_manager/state/local.rs` so host-only local layouts do not
  try to serialize missing NIXL descriptors

### Prior validation already on disk before this run

- `cargo test --manifest-path lib/llm/Cargo.toml g4:: --lib`
  - pass before the rename milestone
- `cargo test --manifest-path lib/llm/Cargo.toml g3pb:: --lib`
  - pass (`12 passed`) immediately after the rename/in-memory seam
- `cargo check --manifest-path lib/llm/Cargo.toml --bin kvbm_g3pb_backend --bin kvbm_g3pb_worker_smoke`
  - pass
- `cargo build --manifest-path lib/llm/Cargo.toml --bin kvbm_g3pb_backend --bin kvbm_g3pb_worker_smoke`
  - pass
- `target/debug/kvbm_g3pb_backend --listen 127.0.0.1:58181`
  - pass
- `target/debug/kvbm_g3pb_worker_smoke --backend-url http://127.0.0.1:58181`
  - pass after the local-layout `disable_nixl()` fix

## Forward plan

1. Keep the in-memory backend and `foyer` backend both available behind the
   same `G3pbPeerStorage` seam until the runtime path settles.
2. Replace the HTTP payload path with real NIXL/UCX `device <-> CPU` transfer
   semantics.
3. Re-run the layered validation stack after the transport swap.
4. Only after that, decide whether `foyer` should become the default backend
   constructor path for the standalone peer binary.


Commit is allowed for this state because the end-to-end `G3PB` validation stack is working.
