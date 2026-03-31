# KVBM TensorRT-LLM Integration Execution Plan

Last updated: 2026-03-31 21:24:38 UTC

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

### Remaining work in this run

- reconnect the remote data path to real NIXL/UCX `device <-> CPU` transfers
  instead of HTTP payload exchange
- revalidate the next slice in layers:
  - host-only `G3PB` backend smoke
  - `kvbm_nixl_transfer_smoke`
  - full `kvbm_g3pb_worker_smoke` with remote fetch and local onboard

### Concrete transport gap

- current `G3PB` backend endpoints:
  - `offer`
  - `put`
  - `put_payload`
  - `query`
  - `fetch`
- current worker smoke uses those endpoints for both metadata and bytes:
  - upload path: `offer -> put_payload`
  - fetch path: `query -> fetch`
- this means the peer only stores opaque bytes (`Vec<u8>`), not remote host
  staging memory that another worker can import through NIXL
- replacing the HTTP byte path requires a new seam, not just a small swap in
  `kvbm_g3pb_worker_smoke.rs`:
  - peer side must own registered pinned host staging blocks
  - peer side must expose/import NIXL metadata or serialized remote layouts for
    those staging blocks
  - worker side must import the remote metadata and issue NIXL read/write
    against those remote host blocks
  - sequence-hash routing still stays in `G3PB`, but payload transport must no
    longer be `Vec<u8>` based once the remote descriptors exist

### Exact next step

- next file:
  `lib/llm/src/bin/kvbm_g3pb_backend.rs`
- next commands:
  - extend the backend to allocate and retain pinned host staging blocks for
    accepted `sequence_hash` entries rather than storing only `Vec<u8>`
  - add a metadata endpoint that returns the remote NIXL-importable description
    for those staging blocks
  - then update `lib/llm/src/bin/kvbm_g3pb_worker_smoke.rs` to:
    - import the remote metadata
    - write accepted local host blocks to the peer via NIXL
    - read fetched remote host blocks back via NIXL before local onboard
  - rerun:
    - `cargo test --manifest-path lib/llm/Cargo.toml g3pb:: --lib`
    - `cargo check --manifest-path lib/llm/Cargo.toml --bin kvbm_g3pb_backend --bin kvbm_g3pb_worker_smoke --bin kvbm_nixl_transfer_smoke`
    - `target/debug/kvbm_nixl_transfer_smoke`
    - host-only backend smoke
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
