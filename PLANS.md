# KVBM TensorRT-LLM Integration Execution Plan

Last updated: 2026-04-01 05:16:06 UTC

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

## Current run (2026-04-01 00:46:42 UTC)

### Summary of accomplishments in this run

- ✅ Re-read the required handoff/design context before touching code:
  - `Agents.md`
  - `PLANS.md`
  - `docs/design-docs/kvbm-g3pb-plan.md`
- ✅ Reconfirmed the repo baseline from the current detached `HEAD`
- ✅ Revalidated the focused `G3PB` unit baseline:
  - `cargo test --manifest-path lib/llm/Cargo.toml g3pb:: --lib`
    - pass (`13 passed`)
- ✅ Revalidated the `G3PB` admission policy suite:
  - `cargo test --manifest-path lib/llm/Cargo.toml g3pb_filter --lib`
    - pass (`6 passed`)
- ✅ Revalidated the same focused baseline again from the current clean worktree
  before starting the remaining validation tail:
  - `cargo test --manifest-path lib/llm/Cargo.toml g3pb:: --lib`
    - pass (`13 passed`)
  - `cargo test --manifest-path lib/llm/Cargo.toml g3pb_filter --lib`
    - pass (`6 passed`)
- ✅ Rebuilt the active request-plane smoke binaries:
  - `cargo build --manifest-path lib/llm/Cargo.toml --bin kvbm_g3pb_backend --bin kvbm_g3pb_worker_smoke`
    - pass
- ✅ Completed backend restart/reload validation across `load_remote` + `query` / `fetch`
  - first attempt exposed a discovery-root setup mistake (`DYN_FILE_KV` must point
    at a directory root, not a pre-created file)
  - corrected shared discovery root worked
  - first smoke run passed against worker `53`
  - backend was terminated and restarted against the same discovery + foyer dirs
  - second smoke run also passed against worker `53`
- ✅ Completed larger staged-transfer validation on the request-plane/discovery path
  - a `24`-block smoke passed against a backend with `48` host blocks
  - a follow-up `40`-block smoke on the same backend failed only because the
    previous committed `24` blocks still occupied host staging capacity, leaving
    `24` blocks available
  - restarting onto a fresh backend with `96` host blocks allowed the same
    `40`-block smoke to pass cleanly
- ✅ Added explicit metadata-RPC timing output to `kvbm_g3pb_worker_smoke`
  - the smoke now reports per-RPC and total request-plane timing for:
    `health`, `load_remote`, `offer`, `stage_put`, `commit_put`, `query`, and `fetch`
  - this keeps the measurement on the real G3PB validation path rather than
    creating a separate synthetic harness
- ✅ Diagnosed the teardown warning root cause
  - the corrupted remote-agent names in
    `invalidateRemoteMD ... NIXL_ERR_NOT_FOUND` come from `nixl-sys`, not the
    repo-local G3PB code
  - `nixl-sys` uses `remote_agent.as_ptr()` from a Rust `&str` when calling
    `nixl_capi_invalidate_remote_md`, instead of passing a null-terminated
    `CString`
  - relevant dependency paths:
    - `/root/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/nixl-sys-0.10.1/src/agent.rs`
    - `/root/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/nixl-sys-0.10.1/wrapper.cpp`
- ✅ Started the structural cleanup follow-up from the prior handoff
  - renamed the transitional storage/config symbols so new code paths stop
    exporting the migration-era `G2G3G3pb*` names
  - landed stable names:
    - `G3pbStorageConfig`
    - `G3pbCacheStorage`
    - internal metadata/location helpers now use `G3pbCache*` terminology
  - behavior intentionally unchanged in this slice
- ✅ Added a reusable request-plane discovery/client seam for G3PB peers
  - lifted peer discovery, worker-id to instance-id resolution, and
    broadcast `load_remote` wiring out of `kvbm_g3pb_worker_smoke`
  - new shared helpers/types live in `lib/llm/src/block_manager/distributed/g3pb.rs`:
    - `discover_g3pb_peers`
    - `G3pbDiscoveredPeers`
    - `G3pbPeerInstance`
  - `kvbm_g3pb_worker_smoke` now reuses that shared seam instead of carrying
    its own ad hoc discovery bookkeeping
- ✅ Refactored the standalone backend around a cleaner service boundary
  - `kvbm_g3pb_backend` now keeps runtime/storage/RPC behavior on
    `G3pbBackendService`
  - endpoint transport wiring is a thin wrapper over that service instead of
    being interleaved with request handling state
  - no protocol or storage behavior changes in this slice
- ✅ Landed the first native KVBM config surface for G3PB admission policy
  - added shared `G3pbAdmissionPolicy` / `G3pbAdmissionConfig` to
    `lib/llm/src/block_manager/config.rs`
  - `KvBlockManagerConfig` now carries optional `g3pb_admission`
  - local/logical KVBM state construction now installs the `G3PB` host
    offload filter automatically when `g3pb_admission` is set and the host
    layout did not already provide an explicit offload filter
  - `G3pbAdmissionFilter` now builds from the shared config type and treats the
    legacy `G3PB_OFFLOAD_ALL` env var only as a compatibility fallback
  - added unit coverage for the config-to-host-filter wiring in
    `lib/llm/src/block_manager/state.rs`

### Current findings before edits

- the remaining work from the prior handoff is still the validation tail, not a
  known missing implementation slice
- `kvbm_g3pb_worker_smoke` already exercises the core request-plane flow needed
  for the pending milestones:
  - discovery-driven peer enumeration
  - `load_remote`
  - staged remote writes
  - `commit_put`
  - ownership-aware `query`
  - descriptor-based `fetch`
  - local host registration plus device onboard
- explicit timing/throughput output is now available directly from
  `kvbm_g3pb_worker_smoke`
- `DYN_FILE_KV` for the file discovery backend must be a directory root; using a
  `mktemp` file path fails backend registration with:
  `Unable to register service for discovery ... Internal filesystem error: Not a directory (os error 20)`
- restart/reload behavior for the current `G3PB` smoke path is now validated:
  a restarted backend can accept a fresh `load_remote`, complete staged upload,
  answer `query`, and serve descriptor-based `fetch` in the same shared
  discovery namespace
- the teardown warning still reproduces after each successful worker smoke:
  `invalidateRemoteMD ... NIXL_ERR_NOT_FOUND`
- `KvBlockManagerState::import_remote_blockset()` currently rejects duplicate
  worker ids and does not expose any explicit unload/update path for remote NIXL
  metadata, which is relevant context for the teardown-warning investigation
- the warning diagnosis is now specific:
  repo-local remote-blockset lifetime is not the immediate cause of the garbled
  remote-agent names; the names are already corrupted inside `nixl-sys`
  invalidation calls because the dependency passes non-null-terminated Rust
  string pointers to the C API during remote metadata teardown
- the first structural cleanup slice is low-risk and independently landable:
  the naming cleanup is isolated from request-plane behavior, NIXL transfer
  behavior, and storage semantics
- the next structural cleanup slice is also now proven low-risk:
  peer discovery and instance resolution can live in the shared G3PB client
  surface without changing request-plane timings or transfer behavior
- backend cleanup can also stay structural:
  the standalone backend’s request handling, runtime policy, and endpoint
  startup can be separated without changing the request-plane contract
- concrete config-surface findings from this run:
  - KVBM/core policy candidates:
    - offload admission policy
    - CPU-buffer retention/eviction thresholds
    - promotion/demotion thresholds between CPU buffer and `foyer`
  - backend-runtime candidates:
    - worker identity
    - pinned host staging capacity
    - backend device id for pinned registration
    - `foyer` directory and capacity settings
  - smoke-only validation controls:
    - demo block counts
    - sequence seed/range
    - synthetic local device/host block counts used only by validation binaries
  - the first concrete shared KVBM knob is now implemented:
    `KvBlockManagerConfig::g3pb_admission`
  - ownership split after this slice is clearer:
    - core KVBM config:
      `g3pb_admission`
    - backend-runtime config:
      worker identity, pinned host staging block count, device id, `foyer`
      directories, G2 bytes, `foyer` memory bytes, `foyer` disk bytes
    - smoke-only validation CLI:
      synthetic device block counts, local host block counts, sequence seed,
      demo block count, per-run workload shaping
  - compatibility note:
    `G3PB_OFFLOAD_ALL` still works, but now only as a legacy fallback for the
    new native config surface rather than being the only policy entry point

### Validation completed in this run so far

- `cargo test --manifest-path lib/llm/Cargo.toml g3pb:: --lib`
  - pass (`13 passed`)
- `cargo test --manifest-path lib/llm/Cargo.toml g3pb_filter --lib`
  - pass (`6 passed`)
- repeated focused baseline from the clean worktree before the next milestone:
  - `cargo test --manifest-path lib/llm/Cargo.toml g3pb:: --lib`
    - pass (`13 passed`)
  - `cargo test --manifest-path lib/llm/Cargo.toml g3pb_filter --lib`
    - pass (`6 passed`)
- `cargo build --manifest-path lib/llm/Cargo.toml --bin kvbm_g3pb_backend --bin kvbm_g3pb_worker_smoke`
  - pass
- backend restart/reload validation with shared file discovery + persistent foyer dir:
  - incorrect discovery-root attempt:
    `env DYN_DISCOVERY_BACKEND=file DYN_FILE_KV=/tmp/g3pb-discovery.restart.XBheCV target/debug/kvbm_g3pb_backend --worker-id 53 --host-blocks 16 --foyer-dir /tmp/g3pb-foyer.restart.PpnPrN`
    - fail at startup because `DYN_FILE_KV` pointed to a file instead of a directory
  - corrected backend start:
    `env DYN_DISCOVERY_BACKEND=file DYN_FILE_KV=/tmp/g3pb-discovery.restartdir.rx0f7E target/debug/kvbm_g3pb_backend --worker-id 53 --host-blocks 16 --foyer-dir /tmp/g3pb-foyer.restart.PpnPrN`
    - pass
  - first worker smoke:
    `env DYN_DISCOVERY_BACKEND=file DYN_FILE_KV=/tmp/g3pb-discovery.restartdir.rx0f7E target/debug/kvbm_g3pb_worker_smoke --worker-id 21 --host-blocks 8 --count 4`
    - pass
  - backend graceful shutdown via `Ctrl+C`
    - pass
  - restarted backend with same discovery + foyer dirs:
    `env DYN_DISCOVERY_BACKEND=file DYN_FILE_KV=/tmp/g3pb-discovery.restartdir.rx0f7E target/debug/kvbm_g3pb_backend --worker-id 53 --host-blocks 16 --foyer-dir /tmp/g3pb-foyer.restart.PpnPrN`
    - pass
  - second worker smoke after restart:
    `env DYN_DISCOVERY_BACKEND=file DYN_FILE_KV=/tmp/g3pb-discovery.restartdir.rx0f7E target/debug/kvbm_g3pb_worker_smoke --worker-id 21 --host-blocks 8 --count 4`
    - pass
  - observations:
    - both worker smokes transferred `4` blocks / `32768` bytes
    - both runs validated `load_remote`, staged upload, duplicate-offer rejection,
      `query`, remote `fetch`, local host registration, and device onboard
    - both runs still emitted the non-blocking teardown warning
      `invalidateRemoteMD ... NIXL_ERR_NOT_FOUND`
- larger staged-transfer validation:
  - rebuilt binaries before the milestone:
    `cargo build --manifest-path lib/llm/Cargo.toml --bin kvbm_g3pb_backend --bin kvbm_g3pb_worker_smoke`
    - pass
  - backend with `48` host blocks:
    `env DYN_DISCOVERY_BACKEND=file DYN_FILE_KV=/tmp/g3pb-discovery.large.EKfFBR target/debug/kvbm_g3pb_backend --worker-id 53 --host-blocks 48 --foyer-dir /tmp/g3pb-foyer.large.2HfqDv`
    - pass
  - `24`-block worker smoke:
    `env DYN_DISCOVERY_BACKEND=file DYN_FILE_KV=/tmp/g3pb-discovery.large.EKfFBR target/debug/kvbm_g3pb_worker_smoke --worker-id 21 --num-device-blocks 64 --host-blocks 40 --count 24`
    - pass
    - transferred `24` blocks / `196608` bytes
  - same backend, `40`-block worker smoke:
    `env DYN_DISCOVERY_BACKEND=file DYN_FILE_KV=/tmp/g3pb-discovery.large.EKfFBR target/debug/kvbm_g3pb_worker_smoke --worker-id 22 --num-device-blocks 96 --host-blocks 56 --count 40 --sequence-start 5000`
    - fail with:
      `Not enough blocks available, requested: 40, available: 24`
    - this was a retained-capacity limit, not a transport failure
  - fresh backend with `96` host blocks:
    `env DYN_DISCOVERY_BACKEND=file DYN_FILE_KV=/tmp/g3pb-discovery.xl.uu9wfv target/debug/kvbm_g3pb_backend --worker-id 53 --host-blocks 96 --foyer-dir /tmp/g3pb-foyer.xl.em2H7Z`
    - pass
  - `40`-block worker smoke on the fresh backend:
    `env DYN_DISCOVERY_BACKEND=file DYN_FILE_KV=/tmp/g3pb-discovery.xl.uu9wfv target/debug/kvbm_g3pb_worker_smoke --worker-id 22 --num-device-blocks 96 --host-blocks 56 --count 40 --sequence-start 5000`
    - pass
    - transferred `40` blocks / `327680` bytes
- post-instrumentation validation:
  - `cargo fmt --manifest-path lib/llm/Cargo.toml --all`
    - pass
  - `git diff --check`
    - pass
  - `cargo test --manifest-path lib/llm/Cargo.toml g3pb:: --lib`
    - pass (`13 passed`)
  - `cargo test --manifest-path lib/llm/Cargo.toml g3pb_filter --lib`
    - pass (`6 passed`)
  - `cargo build --manifest-path lib/llm/Cargo.toml --bin kvbm_g3pb_backend --bin kvbm_g3pb_worker_smoke`
    - pass
  - timed `40`-block worker smoke with instrumentation:
    `env DYN_DISCOVERY_BACKEND=file DYN_FILE_KV=/tmp/g3pb-discovery.xl.uu9wfv target/debug/kvbm_g3pb_worker_smoke --worker-id 23 --num-device-blocks 96 --host-blocks 56 --count 40 --sequence-start 9000`
    - pass
    - transferred `40` blocks / `327680` bytes
    - request-plane timings:
      - `health`: `1` op in `0.002878s` (`347.41 ops/s`)
      - `load_remote`: `1` op in `0.003572s` (`279.98 ops/s`)
      - `offer`: `1` op in `0.001408s` (`710.46 ops/s`)
      - `stage_put`: `1` op in `0.002331s` (`429.05 ops/s`)
      - `commit_put`: `1` op in `0.001698s` (`589.01 ops/s`)
      - `query`: `1` op in `0.001343s` (`744.58 ops/s`)
      - `fetch`: `1` op in `0.001933s` (`517.42 ops/s`)
      - total metadata RPCs: `7` ops in `0.015162s` (`461.68 ops/s`)
- post-naming-cleanup validation:
  - `cargo fmt --manifest-path lib/llm/Cargo.toml --all`
    - pass
  - `cargo test --manifest-path lib/llm/Cargo.toml g3pb:: --lib`
    - pass (`13 passed`)
  - `cargo test --manifest-path lib/llm/Cargo.toml g3pb_filter --lib`
    - pass (`6 passed`)
  - `cargo build --manifest-path lib/llm/Cargo.toml --bin kvbm_g3pb_backend --bin kvbm_g3pb_worker_smoke`
    - pass
- post-client-seam validation:
  - `cargo fmt --manifest-path lib/llm/Cargo.toml --all`
    - pass
  - `cargo test --manifest-path lib/llm/Cargo.toml g3pb:: --lib`
    - pass (`15 passed`)
    - added unit coverage for duplicate-worker discovery rejection and
      deterministic worker-id ordering in discovered peer registries
  - `cargo test --manifest-path lib/llm/Cargo.toml g3pb_filter --lib`
    - pass (`6 passed`)
  - `cargo build --manifest-path lib/llm/Cargo.toml --bin kvbm_g3pb_backend --bin kvbm_g3pb_worker_smoke`
    - pass
- post-backend-boundary validation:
  - `cargo fmt --manifest-path lib/llm/Cargo.toml --all`
    - pass
  - `cargo test --manifest-path lib/llm/Cargo.toml g3pb:: --lib`
    - pass (`15 passed`)
  - `cargo test --manifest-path lib/llm/Cargo.toml g3pb_filter --lib`
    - pass (`6 passed`)
  - `cargo build --manifest-path lib/llm/Cargo.toml --bin kvbm_g3pb_backend --bin kvbm_g3pb_worker_smoke`
    - pass
- post-native-config validation:
  - `cargo fmt --manifest-path lib/llm/Cargo.toml --all`
    - pass
  - `cargo test --manifest-path lib/llm/Cargo.toml g3pb_filter --lib`
    - first pass failed in `test_default_trait` because an earlier env-mutating
      test left `G3PB_OFFLOAD_ALL` set; fixed by clearing the env var inside the
      test
    - final pass (`6 passed`)
  - `cargo test --manifest-path lib/llm/Cargo.toml g3pb:: --lib`
    - pass (`15 passed`)
  - `cargo build --manifest-path lib/llm/Cargo.toml --bin kvbm_g3pb_backend --bin kvbm_g3pb_worker_smoke`
    - pass
  - `git diff --check`
    - pass

### Decisions confirmed in this run so far

- keep `PLANS.md` as the live execution log and update it before each major
  milestone
- execute the remaining work in the exact order already handed off:
  1. backend restart/reload validation
  2. larger staged-transfer smoke counts
  3. request-plane metadata-RPC throughput measurement
  4. teardown-warning investigation
- after the validation tail, pick up the structural cleanup work before adding
  more G3PB surface area:
  1. refactor the G3PB client seam
  2. refactor the G3PB backend seam
  3. clean up externally visible naming so new APIs and configs do not inherit
     awkward transitional names such as `g2g2g3*`
  4. design the long-term KVBM-native config exposure path for G3PB behavior
- only make commits after the corresponding end-to-end `G3PB` validation stack
  is green for that milestone, per `Agents.md`
- treat the file-discovery-root mistake as an operator/setup note only; no code
  change is needed for it
- the restart milestone is complete, so the next effort should stay focused on
  scaling the same smoke path rather than broadening functionality
- first teardown-warning readback should stay narrow:
  inspect local remote-metadata ownership/unload behavior before assuming the
  warning requires transport-path changes
- keep the request-plane timing output in the worker smoke:
  it is cheap, uses the real G3PB path, and gives enough throughput visibility
  for future validation without adding a separate harness
- treat the teardown warning as an upstream `nixl-sys` cleanup bug rather than a
  blocker for the landed G3PB slice
- structural cleanup should land in small validated slices:
  - first naming cleanup
  - then reusable client seam cleanup
  - then backend/service boundary cleanup
- keep the worker smoke timing buckets stable while refactoring:
  move discovery bookkeeping into shared helpers, but leave timing ownership in
  the smoke binary so measurements stay comparable across runs
- keep the backend cleanup structural:
  factor request handling behind a service object first, and defer any protocol
  or storage-policy changes until a later run with separate validation
- config-surface design outcome from this run:
  - do not hide long-lived G3PB behavior only in binary-local CLI/env wiring
  - keep synthetic workload knobs local to smoke binaries
  - treat backend transport/storage knobs separately from core KVBM tier-policy
    knobs so the long-term surface is easier to reason about
- native-config decision landed in code:
  - keep the first shared KVBM surface narrow and policy-focused
  - model the current admission behavior explicitly as
    `G3pbAdmissionPolicy::{AfterFirstReuse,Eager}`
  - inject the filter during state construction only when
    `KvBlockManagerConfig.g3pb_admission` is set
  - preserve explicit host `offload_filter` wiring as the stronger override
  - retain `G3PB_OFFLOAD_ALL` only as backward-compatibility fallback for the
    new shared config path

### Remaining work in this run

- no additional code slice is required to satisfy the current handoff
- the remaining follow-up is later integration work, not missing work in this
  run:
  - adopt `KvBlockManagerConfig.g3pb_admission` from real non-smoke KVBM
    construction paths once those callers are ready to opt into `G3PB`
    admission policy
  - design the next shared config layer for CPU-buffer retention / `foyer`
    retention separately from this admission-policy slice
  - remove the legacy `G3PB_OFFLOAD_ALL` env path only after relevant callers
    have moved onto the native config surface

### Exact next step

- if another run continues from here, the next concrete slice should be config
  adoption rather than config invention:
  - wire `KvBlockManagerConfig.g3pb_admission` into the first real KVBM caller
    that should opt into `G3PB` admission policy
  - keep backend-runtime-only and smoke-only knobs out of that adoption change
  - keep CPU-buffer retention / `foyer` retention as a separate follow-up after
    an actual caller for those knobs is identified
  - rerun the focused `g3pb` / `g3pb_filter` tests plus backend/worker build
    after that adoption slice

### Handoff for next run

- the validation tail, structural cleanup pass, and first native-config slice
  are complete
- landed cleanup/config work in this run:
  - stable storage/config naming
  - shared peer discovery/client seam
  - cleaner backend service boundary
  - native `KvBlockManagerConfig.g3pb_admission`
- next run should focus on config adoption rather than more transport or
  endpoint restructuring
- concrete next slice:
  1. wire `KvBlockManagerConfig.g3pb_admission` into the first real KVBM caller
     that should opt into `G3PB`
  2. keep backend-runtime-only knobs and smoke-only knobs out of that adoption
     change
  3. rerun:
     - `cargo test --manifest-path lib/llm/Cargo.toml g3pb:: --lib`
     - `cargo test --manifest-path lib/llm/Cargo.toml g3pb_filter --lib`
     - `cargo build --manifest-path lib/llm/Cargo.toml --bin kvbm_g3pb_backend --bin kvbm_g3pb_worker_smoke`
- config ownership guidance already established in this run:
  - KVBM/core policy:
    - offload admission policy
    - CPU-buffer retention/eviction thresholds
    - promotion/demotion thresholds between CPU buffer and `foyer`
  - backend runtime:
    - worker identity
    - pinned host staging capacity
    - backend device id
    - `foyer` directory and capacity settings
  - smoke-only validation:
    - demo block counts
    - sequence seed/range
    - synthetic local device/host block counts
- optional follow-on work outside the main slice remains:
  1. upstream or locally patch `nixl-sys` remote metadata invalidation to use
     null-terminated `CString` values
  2. decide whether backend-side eviction/reclamation is needed for long-lived
     retained committed blocks beyond the current smoke coverage

## Current run (2026-04-01 01:51:53 UTC)

### Summary of accomplishments in this run

- ✅ Re-read the required handoff/design context before editing:
  - `Agents.md`
  - `PLANS.md`
  - `docs/design-docs/kvbm-g3pb-plan.md`
- ✅ Confirmed the remaining concrete slice from the prior handoff:
  `KvBlockManagerConfig.g3pb_admission` existed in core KVBM state wiring, but
  no real non-smoke caller was setting it yet
- ✅ Adopted the native `g3pb_admission` config surface in the first real KVBM
  caller path:
  - `lib/bindings/kvbm/src/block_manager.rs`
  - both Python `BlockManager::new` and `BlockManagerBuilder::build()` now read
    `DYN_KVBM_G3PB_ADMISSION_POLICY` and, when set, pass
    `KvBlockManagerConfig.g3pb_admission`
  - supported values:
    - `after_first_reuse`
    - `eager`
    - `disabled`
- ✅ Added focused parser coverage for the bindings adoption path
  - default/unset behavior
  - `after_first_reuse`
  - `eager`
  - invalid-value rejection
- ✅ Fixed the bindings parser test race
  - the first test run exposed process-wide env mutation leakage across
    parallel tests
  - resolved by serializing the env-mutating parser tests with a local mutex
- ✅ Updated repo-local docs to match the landed request-plane architecture and
  the new real-caller config adoption:
  - `docs/design-docs/kvbm-g3pb-plan.md`
  - `lib/bindings/kvbm/README.md`
  - `lib/runtime/src/config/environment_names.rs`

### Current findings before final handoff

- the prior “next concrete slice” is now implemented:
  real KVBM callers can opt into native `G3PB` admission policy without using
  only the legacy `G3PB_OFFLOAD_ALL` fallback
- the smallest production-facing adoption point is the bindings layer, because:
  - the connector leaders already flow through `BlockManagerBuilder`
  - that path already consumes other KVBM env-driven runtime policy
  - the underlying core config surface remained unchanged in this slice
- the adoption stays backward-compatible:
  - unset `DYN_KVBM_G3PB_ADMISSION_POLICY` preserves existing behavior
  - explicit disk offload filters remain stronger than `g3pb_admission`
    because the core state wiring only installs the `G3PB` host filter when
    no explicit host offload filter is already present
- the design doc was stale before this run:
  it still described the old HTTP backend/request path even though the landed
  implementation already uses Dynamo request-plane + discovery
- the only bug uncovered during this slice was test-only:
  the new bindings parser tests needed serialization because they mutate a
  shared process environment variable

### Validation completed in this run so far

- `cargo fmt --manifest-path lib/llm/Cargo.toml --all`
  - pass
- `cargo fmt --manifest-path lib/bindings/kvbm/Cargo.toml --all`
  - pass
- `cargo test --manifest-path lib/llm/Cargo.toml g3pb:: --lib`
  - pass (`15 passed`)
- `cargo test --manifest-path lib/llm/Cargo.toml g3pb_filter --lib`
  - pass (`6 passed`)
- `cargo build --manifest-path lib/llm/Cargo.toml --bin kvbm_g3pb_backend --bin kvbm_g3pb_worker_smoke`
  - pass
- `cargo test --manifest-path lib/bindings/kvbm/Cargo.toml read_g3pb_admission_config`
  - first run failed because the new env-mutating tests raced each other under
    parallel execution
  - pass after serializing env mutation (`4 passed`)
- `git diff --check`
  - pass

### Decisions confirmed in this run so far

- treat the bindings layer as the first real KVBM caller for native
  `g3pb_admission` adoption
- keep the adoption path explicit and backward-compatible:
  do not force-enable `G3PB` policy for all callers, and do not broaden the
  core config surface beyond admission policy in this slice
- repair stale docs in the same slice because the design doc and bindings README
  are part of the active handoff surface for future runs

### Remaining work in this run

- make a signed small commit for the config-adoption/docs slice

### Exact next step

- create the signed commit for:
  - bindings-side native `g3pb_admission` adoption
  - env-name/docs updates
  - `PLANS.md` handoff refresh

### Handoff for next run

- this run already landed the concrete “config adoption” slice from the prior
  handoff in the real KVBM bindings caller path
- after the commit is cut, the remaining work should return to follow-on items
  rather than more admission config plumbing:
  1. upstream or locally patch the `nixl-sys` teardown warning if needed
  2. decide whether retained committed blocks need backend-side reclamation
  3. design any future CPU-buffer / `foyer` retention knobs as a separate slice

## Current run (2026-04-01 02:00:50 UTC)

### Summary of accomplishments in this run

- ✅ Re-read `PLANS.md` after the config-adoption commit to verify whether any
  concrete work remained unrecorded
- ✅ Revalidated the focused `G3PB` / bindings stack from the current clean
  `HEAD` instead of assuming the prior run log was still sufficient
- ✅ Found and fixed one remaining test-only reliability gap:
  - `lib/llm/src/block_manager/offload/g3pb_filter.rs`
  - the `g3pb_filter` env-legacy tests were still mutating
    `G3PB_OFFLOAD_ALL` without serialization
  - a fresh validation run exposed the race in
    `test_legacy_env_false_string`
  - fixed by serializing the env-mutating tests with a local mutex, matching
    the bindings-side parser test strategy
- ✅ Refreshed the handoff state in `PLANS.md` after the post-commit audit

### Current findings before final handoff

- the config-adoption/docs slice from the prior run was already committed and
  signed before this run started:
  - `126c3590d bindings: adopt native g3pb admission config`
- the only remaining issue discovered by re-running the planned validation was
  the `g3pb_filter` env-test race
- no additional product/runtime behavior changes are required for the current
  `G3PB` slice once that test reliability fix is in
- the audit-run reliability fix is now committed and signed:
  - `701d74a95 tests: serialize g3pb filter env cases`

### Validation completed in this run so far

- `cargo test --manifest-path lib/llm/Cargo.toml g3pb:: --lib`
  - initial audit pass: pass (`15 passed`)
  - post-fix rerun: pass (`15 passed`)
- `cargo test --manifest-path lib/llm/Cargo.toml g3pb_filter --lib`
  - initial audit pass: fail in
    `block_manager::offload::g3pb_filter::tests::test_legacy_env_false_string`
    because `G3PB_OFFLOAD_ALL` was racing another env-mutating test
  - post-fix rerun: pass (`6 passed`)
- `cargo build --manifest-path lib/llm/Cargo.toml --bin kvbm_g3pb_backend --bin kvbm_g3pb_worker_smoke`
  - pass
- `cargo test --manifest-path lib/bindings/kvbm/Cargo.toml read_g3pb_admission_config`
  - audit pass before the llm-side test fix: pass (`4 passed`)
  - post-fix hygiene rerun: pass (`4 passed`)
- `cargo fmt --manifest-path lib/llm/Cargo.toml --all`
  - pass
- `cargo fmt --manifest-path lib/bindings/kvbm/Cargo.toml --all`
  - pass
- `git diff --check`
  - pass

### Decisions confirmed in this run so far

- keep re-reading `PLANS.md` after each landed slice; the post-commit audit
  found a real test gap that the prior handoff had not captured
- treat env-mutating test serialization as required in both bindings and llm
  test modules whenever policy/env compatibility paths are covered
- keep the current follow-on queue focused on the already-known non-blocking
  items rather than inventing new `G3PB` surface area

### Remaining work in this run

- none

### Exact next step

- if another run continues from here, resume with the follow-on backlog rather
  than more validation cleanup:
  1. upstream or locally patch the `nixl-sys` teardown warning if needed
  2. decide whether retained committed blocks need backend-side reclamation
  3. design any future CPU-buffer / `foyer` retention knobs as a separate slice

### Handoff for next run

- the config-adoption/docs slice is already committed
- this audit run found and fixed the last currently known focused-validation
  gap: the `g3pb_filter` env-test race
- this audit run is also committed:
  - `701d74a95 tests: serialize g3pb filter env cases`
- repo state at handoff:
  - clean worktree
  - detached `HEAD`
- the remaining follow-on work returns to the same non-blocking backlog already
  captured in the prior run:
  1. upstream or locally patch the `nixl-sys` teardown warning if needed
  2. decide whether retained committed blocks need backend-side reclamation
  3. design any future CPU-buffer / `foyer` retention knobs as a separate slice

## Current run (2026-04-01 05:02:41 UTC)

### Summary of accomplishments in this run

- ✅ Re-read the required handoff/design context before doing any work:
  - `Agents.md`
  - `PLANS.md`
  - `docs/design-docs/kvbm-g3pb-plan.md`
- ✅ Re-read `PLANS.md` completely to verify whether any concrete work remained
  before making changes
- ✅ Audited the current repo state against the latest handoff:
  - clean worktree at pickup
  - detached `HEAD`
  - latest commits already match the prior handoff:
    - `21c45d28f docs: finalize g3pb handoff state`
    - `701d74a95 tests: serialize g3pb filter env cases`
    - `126c3590d bindings: adopt native g3pb admission config`
- ✅ Confirmed that no additional implementation slice is currently required to
  satisfy the active `G3PB` handoff
- ✅ Refreshed `PLANS.md` with this completion audit so the next run does not
  have to rediscover that the remaining items are follow-on backlog, not
  missing work in the current slice
- ✅ Recorded the audit/handoff refresh as signed small docs commits in this
  run

### Current findings before final handoff

- the latest committed repo state matches the previous handoff without drift:
  the bindings-side `g3pb_admission` adoption, docs refresh, and llm-side env
  test serialization are already present on `HEAD`
- there is no unimplemented code slice left in the active `PLANS.md` body;
  the only remaining items are the same non-blocking follow-on backlog already
  captured in prior runs
- this run is therefore an execution audit, not a new product/runtime change
- `PLANS.md` itself was the only file that needed updating in this run so the
  handoff remains precise for the next execution
- the completion-audit refresh is now committed; no uncommitted implementation
  or handoff work remains after this run

### Validation completed in this run so far

- `cargo test --manifest-path lib/llm/Cargo.toml g3pb:: --lib`
  - pass (`15 passed`)
- `cargo test --manifest-path lib/llm/Cargo.toml g3pb_filter --lib`
  - pass (`6 passed`)
- `cargo build --manifest-path lib/llm/Cargo.toml --bin kvbm_g3pb_backend --bin kvbm_g3pb_worker_smoke`
  - pass
- `cargo test --manifest-path lib/bindings/kvbm/Cargo.toml read_g3pb_admission_config`
  - pass (`4 passed`)
- `git diff --check`
  - pass

### Decisions confirmed in this run so far

- do not invent additional `G3PB` code work when the active plan already says
  the implementation slice is complete
- keep the completion audit on disk in `PLANS.md` so the next run can start
  from a verified state instead of repeating the same audit work
- still rerun the focused validation stack after this documentation-only audit,
  because the user explicitly asked for a completion check rather than trusting
  prior run logs

### Remaining work in this run

- none

### Exact next step

- if another run continues from here, resume only from the existing non-blocking
  follow-on backlog:
  1. upstream or locally patch the `nixl-sys` teardown warning if needed
  2. decide whether retained committed blocks need backend-side reclamation
  3. design any future CPU-buffer / `foyer` retention knobs as a separate slice

### Handoff for next run

- no additional implementation work remains in the active `G3PB` plan
- this completion-audit run is already committed in signed docs-only commits
- the next run should resume only from the existing non-blocking follow-on
  backlog:
  1. upstream or locally patch the `nixl-sys` teardown warning if needed
  2. decide whether retained committed blocks need backend-side reclamation
  3. design any future CPU-buffer / `foyer` retention knobs as a separate slice

## Current run (2026-04-01 05:09:21 UTC)

### Summary of accomplishments in this run

- ✅ Re-read the required handoff/design context before doing any work:
  - `Agents.md`
  - `PLANS.md`
  - `docs/design-docs/kvbm-g3pb-plan.md`
- ✅ Re-read `PLANS.md` completely before making changes to verify whether
  anything in the active plan had become stale or incomplete
- ✅ Audited the repo state against the latest handoff:
  - detached `HEAD`
  - clean worktree at pickup
  - latest handoff commits on `HEAD`:
    - `f2ac59dec docs: stabilize g3pb audit handoff`
    - `f1219b704 docs: finalize g3pb audit handoff`
    - `d8cd551c5 docs: refresh g3pb completion audit`
    - `21c45d28f docs: finalize g3pb handoff state`
    - `701d74a95 tests: serialize g3pb filter env cases`
- ✅ Revalidated the focused `G3PB` stack from the current tip instead of
  relying on prior run logs
- ✅ Refreshed `PLANS.md` again so the latest completion audit, repo state, and
  exact follow-on backlog are all recorded on disk for the next run

### Current findings before final handoff

- the repo still matches the active handoff: there is no unimplemented code
  slice remaining in `PLANS.md`
- the only remaining items are the same non-blocking follow-on backlog already
  captured in prior runs:
  1. upstream or locally patch the `nixl-sys` teardown warning if needed
  2. decide whether retained committed blocks need backend-side reclamation
  3. design any future CPU-buffer / `foyer` retention knobs as a separate slice
- broad `TODO` markers still exist elsewhere in the repo, but the audit found
  no new `G3PB`-specific gap that belongs in the active execution slice
- this run remains an execution audit and handoff refresh, not a new
  product/runtime change
- this run is now committed in a signed docs-only audit refresh commit

### Validation completed in this run so far

- `cargo test --manifest-path lib/llm/Cargo.toml g3pb:: --lib`
  - pass (`15 passed`)
- `cargo test --manifest-path lib/llm/Cargo.toml g3pb_filter --lib`
  - pass (`6 passed`)
- `cargo test --manifest-path lib/bindings/kvbm/Cargo.toml read_g3pb_admission_config`
  - pass (`4 passed`)
- `cargo build --manifest-path lib/llm/Cargo.toml --bin kvbm_g3pb_backend --bin kvbm_g3pb_worker_smoke`
  - pass
- `git diff --check`
  - pass

### Decisions confirmed in this run so far

- continue treating the current `G3PB` plan as complete unless a fresh audit
  exposes a concrete regression or missing validation
- keep rerunning the focused validation stack during audit-only runs so
  `PLANS.md` reflects verified current state, not stale assumptions
- keep the next-run handoff narrow: resume only from the known follow-on
  backlog rather than inventing new scope inside the completed slice

### Remaining work in this run

- none

### Exact next step

- if another run continues from here, resume only from the existing
  non-blocking follow-on backlog:
  1. upstream or locally patch the `nixl-sys` teardown warning if needed
  2. decide whether retained committed blocks need backend-side reclamation
  3. design any future CPU-buffer / `foyer` retention knobs as a separate slice

### Handoff for next run

- the active `G3PB` implementation slice remains complete after a fresh audit
- this run revalidated the focused test/build stack from the current `HEAD`
- this run is committed as a signed docs-only handoff refresh
- if another run continues from here, resume only from the existing
  non-blocking follow-on backlog:
  1. upstream or locally patch the `nixl-sys` teardown warning if needed
  2. decide whether retained committed blocks need backend-side reclamation
  3. design any future CPU-buffer / `foyer` retention knobs as a separate slice

## Current run (2026-04-01 05:15:11 UTC)

### Summary of accomplishments in this run

- ✅ Re-read the required handoff/design context before doing any work:
  - `Agents.md`
  - `PLANS.md`
  - `docs/design-docs/kvbm-g3pb-plan.md`
- ✅ Re-read `PLANS.md` completely before changing anything to confirm whether
  the active `G3PB` plan still contained unfinished implementation work
- ✅ Audited the current repo state against the latest handoff:
  - detached `HEAD`
  - clean worktree at pickup
  - latest audit/handoff commits on `HEAD`:
    - `745fbef1c docs: finalize g3pb audit handoff`
    - `3cf603369 docs: refresh g3pb completion audit`
    - `f2ac59dec docs: stabilize g3pb audit handoff`
    - `f1219b704 docs: finalize g3pb audit handoff`
    - `d8cd551c5 docs: refresh g3pb completion audit`
- ✅ Revalidated the focused `G3PB` and bindings stack from the current tip
- ✅ Refreshed `PLANS.md` again so this run records the same conclusion on
  disk: the active `G3PB` implementation slice is complete and the remaining
  work is follow-on backlog only

### Current findings before final handoff

- the repo still matches the active handoff: there is no unimplemented code
  slice remaining in `PLANS.md`
- the bindings-side native `g3pb_admission` adoption remains present in the
  tree, so the previously identified “first real caller” follow-up is still
  complete
- the focused validation stack is green from the current `HEAD`, not only from
  prior run logs
- no new `G3PB`-specific TODO, cleanup item, docs gap, or validation gap was
  uncovered in this audit
- the only remaining work continues to be the existing non-blocking backlog:
  1. upstream or locally patch the `nixl-sys` teardown warning if needed
  2. decide whether retained committed blocks need backend-side reclamation
  3. design any future CPU-buffer / `foyer` retention knobs as a separate slice

### Validation completed in this run so far

- `cargo test --manifest-path lib/llm/Cargo.toml g3pb:: --lib`
  - pass (`15 passed`)
- `cargo test --manifest-path lib/llm/Cargo.toml g3pb_filter --lib`
  - pass (`6 passed`)
- `cargo test --manifest-path lib/bindings/kvbm/Cargo.toml read_g3pb_admission_config`
  - pass (`4 passed`)
- `cargo build --manifest-path lib/llm/Cargo.toml --bin kvbm_g3pb_backend --bin kvbm_g3pb_worker_smoke`
  - pass
- `git diff --check`
  - pass

### Decisions confirmed in this run so far

- continue treating the active `G3PB` implementation slice as complete unless a
  fresh audit exposes a concrete regression
- keep writing audit results back into `PLANS.md` so the next run can start
  from verified state instead of repeating discovery work
- do not invent more `G3PB` implementation scope when the active plan body and
  current tree already match

### Remaining work in this run

- none

### Exact next step

- if another run continues from here, resume only from the existing
  non-blocking follow-on backlog

### Handoff for next run

- the active `G3PB` implementation slice remains complete after another fresh
  audit
- this run revalidated the focused llm/bindings test and build stack from the
  current `HEAD`
- this run is recorded on disk and committed in the latest signed docs-only
  handoff refresh commits on `HEAD`
- no uncommitted work should remain after this run
- if another run continues from here, resume only from the existing
  non-blocking follow-on backlog:
  1. upstream or locally patch the `nixl-sys` teardown warning if needed
  2. decide whether retained committed blocks need backend-side reclamation
  3. design any future CPU-buffer / `foyer` retention knobs as a separate slice
