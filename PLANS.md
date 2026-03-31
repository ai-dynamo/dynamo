# KVBM TensorRT-LLM Integration Execution Plan

Last updated: 2026-03-31 21:10:20 UTC

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

## Current run (2026-03-31 20:53:29 UTC)

### Context re-read

- `Agents.md`
- `PLANS.md`
- `docs/design-docs/kvbm-g3pb-plan.md`
- `lib/llm/src/block_manager/distributed.rs`
- `lib/llm/src/block_manager/distributed/g3pb.rs`
- `lib/llm/src/bin/kvbm_g3pb_backend.rs`
- `lib/llm/Cargo.toml`

### Milestone completed in working tree

- implementation commit recorded in this run:
  - `6b6326172` `Add foyer-backed G3PB peer storage`
- follow-up signed `PLANS.md` handoff refresh commits were recorded after the
  implementation commit
- worktree was clean immediately before each push in this run
- added a concrete `FoyerG3pbPeerStorage` behind the existing
  `G3pbPeerStorage` trait in
  `lib/llm/src/block_manager/distributed/g3pb.rs`
- added `G3pbFoyerStorageConfig` with bounded default memory/disk capacities so
  the peer cache backend can stay internal to `G3PB`
- kept `InMemoryG3pbPeerStorage` as the contract-reference backend and default
  agent constructor path
- re-exported the new `foyer` storage types from
  `lib/llm/src/block_manager/distributed.rs`
- wired `lib/llm/src/bin/kvbm_g3pb_backend.rs` to support:
  - `--foyer-dir`
  - `--foyer-memory-bytes`
  - `--foyer-disk-bytes`
- added targeted library coverage for the new backend through the same
  storage-agent semantics already used by the in-memory path
- added the `foyer` crate dependency with its `serde` feature in
  `lib/llm/Cargo.toml`

### Validation completed in this run

- `cargo fmt --manifest-path lib/llm/Cargo.toml --all`
  - pass
- `cargo test --manifest-path lib/llm/Cargo.toml g3pb:: --lib`
  - pass (`13 passed`)
  - includes the new `foyer_storage_supports_agent_query_and_fetch` test
- `cargo check --manifest-path lib/llm/Cargo.toml --bin kvbm_g3pb_backend`
  - pass
- `cargo check --manifest-path lib/llm/Cargo.toml --bin kvbm_g3pb_backend --bin kvbm_g3pb_worker_smoke`
  - pass
- `cargo build --manifest-path lib/llm/Cargo.toml --bin kvbm_g3pb_backend`
  - pass
- `git diff --check`
  - pass
- live host-only backend validation with `foyer`:
  - command:
    `target/debug/kvbm_g3pb_backend --listen 127.0.0.1:58182 --worker-id 52 --foyer-dir /tmp/tmp.l1l75f5kuN`
  - `/health`
    - pass, returned `{"worker_id":52,"listen":"127.0.0.1:58182"}`
  - direct HTTP `offer -> put_payload -> query -> fetch`
    for `sequence_hash=123`
    - pass

### Decisions confirmed in this run

- the `foyer` backend is now a drop-in peer-local implementation, not a new
  user-visible KVBM tier
- the in-memory backend remains the default constructor path until the fuller
  worker/peer runtime has more operational evidence with `foyer`
- `foyer` persistence/recovery semantics were intentionally not promoted as a
  contract in this run; the landed validation target is the peer-cache API
  contract (`offer/query/fetch/put`) behind the existing trait seam

### Remaining work after this run

- reconnect the remote data path to real NIXL/UCX `device <-> CPU` transfers
  instead of HTTP payload exchange
- revalidate the next slice in layers:
  - host-only `G3PB` backend smoke
  - `kvbm_nixl_transfer_smoke`
  - full `kvbm_g3pb_worker_smoke` with remote fetch and local onboard

### Branch update completed in this run

- pushed detached `HEAD` to `origin/mf/kvbm-g4-v2`
- remote branch now contains the `foyer` storage milestone plus the condensed
  handoff updates from this run

### Exact next step for the next run

- next file:
  `lib/llm/src/block_manager/distributed/g3pb.rs`
- next commands:
  - `cargo test --manifest-path lib/llm/Cargo.toml g3pb:: --lib`
  - `cargo check --manifest-path lib/llm/Cargo.toml --bin kvbm_g3pb_backend --bin kvbm_g3pb_worker_smoke`
  - then start the NIXL/UCX transfer follow-up from
    `lib/llm/src/block_manager/distributed/g3pb.rs`
    and `lib/llm/src/bin/kvbm_g3pb_worker_smoke.rs`

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
