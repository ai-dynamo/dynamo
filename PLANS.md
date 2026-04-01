# KVBM TensorRT-LLM Integration Execution Plan

Last updated: 2026-04-01 02:02:11 UTC

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

### Remaining work in this run

- the validation tail is complete; this run is now executing the structural
  cleanup TODOs directly:
  - refactor `kvbm_g3pb_worker_smoke` / shared client-side request-plane logic
    into a cleaner G3PB client seam that is easier to reuse outside the smoke
    binary
  - refactor `kvbm_g3pb_backend` around a cleaner backend/service boundary so
    transport wiring, storage policy, and endpoint handling are less entangled
  - the client refactor should be broader than the smoke binary wrapper:
    capture the Dynamo-facing component/client layer and client-side placement
    behavior such as ring-hash logic in a reusable G3PB client seam
  - treat the backend refactor as a true server cleanup:
    separate endpoint handling, transport wiring, and storage/lifecycle policy
  - remove or rename ugly transitional identifiers before they spread further;
    specifically avoid future names in the style of `g2g2g3*` and prefer
    stable, intent-revealing KVBM / peer-cache / G3PB terminology
  - simplify the `foyer` to-disk policy: prefer write-all, last-touched/LRU
    eviction, and no complex recovery path when entries drop out
  - if a block drops out of `foyer`, nothing special needs to happen:
    it was either promoted back into the G2 CPU buffer on the G3PB worker or it
    can simply be dropped; losing an event here is acceptable
  - metadata layout for `foyer` can be sharded or serialized with the payload;
    choose the simpler implementation unless a clear scaling reason appears
  - think about offload policy primarily as a block transition problem from GPU
    memory into offload tiers
  - keep the existing default admission when a block has been reused at least
    once, and keep the env-driven eager path for mostly-prefill workers
  - treat G2 in the G3PB worker as the CPU buffer tier, with LRU-style eviction
  - produce a concrete long-term plan for exposing this configuration natively
    in KVBM rather than only through binary-local flags/env wiring
    - expected output from that design pass:
      - identify which knobs are truly KVBM policy/configuration versus smoke-
        only validation options
      - decide the owning config layer for each knob:
        block-manager config, backend runtime config, or standalone smoke-only
        CLI
      - define names that are backend-agnostic enough to survive beyond the
        current G3PB slice
      - prefer config that is visible from core KVBM types/state so the feature
        is discoverable and not hidden in one-off binaries
      - make the long-term surface describe tier transitions, CPU-buffer
        retention, and simple `foyer` retention directly

### Exact next step

- continue this run with the reusable G3PB client seam:
  - lift peer discovery, health resolution, and worker-id to instance-id mapping
    out of `kvbm_g3pb_worker_smoke`
  - keep placement/routing in shared `distributed/g3pb.rs`
  - rerun the focused `g3pb` / `g3pb_filter` tests plus backend/worker build
    after that milestone before touching the backend boundary

### Handoff for next run

- the landed request-plane/discovery slice is complete, but the next agent
  should continue with two follow-up tracks:
  1. finish the outstanding validation tail in the existing order
  2. start the structural cleanup work for the long-lived G3PB shape
- explicit structural cleanup TODOs for that next run:
  - refactor the G3PB client seam
  - refactor the G3PB backend seam
  - scrub ugly transitional naming from any new/public-facing path; do not
    propagate names like `g2g2g3*`
  - turn the current ad hoc config discussion into a KVBM-native config plan
- explicit architecture direction for that follow-up:
  - the G3PB client refactor should absorb the Dynamo component/client logic and
    client-side placement behavior such as ring-hash handling
  - the G3PB backend refactor should produce a cleaner server boundary between
    endpoint handling, transport, and storage policy
  - simplify `foyer`: write all, evict by last-touch/LRU, and accept drop-on-evict
  - treat G2 on the G3PB worker as the CPU buffer tier with LRU-style eviction
  - preserve the default offload trigger on first reuse, plus env-driven eager
    offload for mostly-prefill workers
- long-term config direction to evaluate during that follow-up:
  - expose durable G3PB/KVBM behavior through native KVBM config/state where it
    will be visible to maintainers and future integrations
  - keep smoke-only validation controls local to the smoke binary
  - avoid baking backend-specific or migration-era names into the long-term
    config surface
  - model the long-term knobs around tier transition policy, CPU-buffer
    retention, and simple `foyer` retention rather than one-off env wiring
  - `foyer` metadata can be sharded or serialized with payload; choose the
    simpler path unless scaling forces something else
- if a future run picks this up, only pursue follow-on work outside this slice:
  1. upstream or locally patch `nixl-sys` remote metadata invalidation to use
     null-terminated `CString` values
  2. decide whether backend-side eviction/reclamation is needed for long-lived
     retained committed blocks beyond the current smoke coverage
