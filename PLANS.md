# KVBM TensorRT-LLM Integration Execution Plan

Last updated: 2026-03-31 22:26:30 UTC

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

## Current run (2026-03-31 21:14:41 UTC)

### Context re-read

- `Agents.md`
- `PLANS.md`
- `docs/design-docs/kvbm-g3pb-plan.md`
- `lib/llm/src/block_manager/distributed.rs`
- `lib/llm/src/block_manager/distributed/g3pb.rs`
- `lib/llm/src/bin/kvbm_g3pb_backend.rs`
- `lib/llm/src/bin/kvbm_g3pb_worker_smoke.rs`
- `lib/llm/src/bin/kvbm_nixl_transfer_smoke.rs`
- `lib/llm/src/block_manager/distributed/transfer.rs`
- `lib/llm/src/block_manager/distributed/worker.rs`
- `lib/llm/Cargo.toml`

### Current findings before edits

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
    fail
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

### Remaining work in this run

- resolve the last NIXL backend-registration failure for
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
- add KVBM-side `G3PB` admission policy wiring:
  - default admission when a block has been reused at least once
  - environment override `G3PB_OFFLOAD_ALL` for eager admission of every block
  - keep `G1 -> G3PB` semantics as copy/replication, not ownership transfer
- refactor both `kvbm_g3pb_backend` and `kvbm_g3pb_worker_smoke` to use Dynamo request-plane + discovery instead of ad hoc HTTP base URLs:
  - register `G3PB` as a Dynamo component endpoint and discover remotes via the discovery backend
  - remove manual `--listen` / `--backend-url` control-plane wiring from the long-term path
  - keep bulk data movement on NIXL/UCX; use request-plane only for metadata/handshake RPCs
  - assume request-plane control traffic should comfortably handle at least ~100 RPS for `offer` / `query` / `stage_put` / `commit_put` / `fetch` metadata calls, then validate with measurement after the refactor
  - do not add the speculative unconditional `nixl_agent` path, `(Disk, Host)` transfer arm, or `disk_block_observer` surface in this slice; keep those deferred unless a concrete requirement appears
- revalidate the next slice in layers:
  - repeated `kvbm_g3pb_worker_smoke` loop against a stable backend to see
    whether the `invalidateRemoteMD` warning remains harmless under repetition
  - multi-backend ownership/routing smoke with multiple `G3PB` peers
  - larger block-count staged transfer smoke
  - backend restart/reload validation across `load_remote` plus subsequent
    `query` / `fetch`
  - worker-side compile/tests after the smoke rewrite
  - `kvbm_nixl_transfer_smoke`
  - host-only `G3PB` backend smoke with `stage_put` / `commit_put` / descriptor
    fetch
  - full `kvbm_g3pb_worker_smoke` with remote fetch and local onboard

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
  - or reuse of an existing KVBM transfer-handler path instead of the ad hoc
    smoke `TransferContext`

### Exact next step

- next file:
  `lib/llm/src/bin/kvbm_g3pb_worker_smoke.rs`
- next commands:
  - trim remaining branch-only drift in the worker smoke binary before the
    request-plane/discovery refactor
  - keep the current remote-agent fix in
    `lib/llm/src/block_manager/block/transfer/nixl.rs`
  - treat the remaining `invalidateRemoteMD` warning as a separate follow-up,
    not as evidence of a NIXL wrapper bug
  - rerun:
    - `cargo test --manifest-path lib/llm/Cargo.toml g3pb:: --lib`
    - `cargo check --manifest-path lib/llm/Cargo.toml --bin kvbm_g3pb_backend --bin kvbm_g3pb_worker_smoke --bin kvbm_nixl_transfer_smoke`
    - `cargo build --manifest-path lib/llm/Cargo.toml --bin kvbm_nixl_transfer_smoke --bin kvbm_g3pb_backend --bin kvbm_g3pb_worker_smoke`
    - `target/debug/kvbm_nixl_transfer_smoke`
    - host-only backend smoke using descriptor endpoints
    - full `target/debug/kvbm_g3pb_worker_smoke ...`

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
