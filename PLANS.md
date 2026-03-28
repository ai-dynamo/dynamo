# KVBM TensorRT-LLM Integration Execution Plan

Last updated: 2026-03-28 UTC

## Objective

Implement the KVBM-backed TensorRT-LLM KV manager path described in
`docs/design-docs/kvbm-trtllm-integration.md`, keep this file as the compact
working log, and iterate until the supported path is complete or blocked by an
external dependency that cannot be resolved from this machine.

## Constraints And Environment Findings

- Local repo instructions were read from `Agents.md`.
- The design source of truth is
  `docs/design-docs/kvbm-trtllm-integration.md`.
- Local TensorRT-LLM checkout exists at `/tmp/trtllm-latest`.
- Pinned upstream TensorRT-LLM commit for this work:
  `3318aca3f4cabf71a323c6e2868f6586817d03cb`.
- Current shell environment has `python3`, `cargo`, and `uv`.
- Current shell environment does not have importable `torch`,
  `tensorrt_llm`, or `pytest` on the default interpreter path.
- Tests therefore need to prefer:
  - Rust unit tests runnable with `cargo test`
  - Python stdlib tests with stubbed modules
  - compile/import checks that do not require the full TRT-LLM runtime

## Phase 0 Decisions

Status: completed

- Supported upstream seam is the PyTorch v2-shaped manager surface in
  `tensorrt_llm/_torch/pyexecutor/resource_manager.py`.
- Initial support matrix is intentionally narrow:
  - TensorRT-LLM pinned to `3318aca3f4cabf71a323c6e2868f6586817d03cb`
  - single GPU
  - beam width 1
  - aggregated execution only
  - no pipeline parallelism
  - no tensor parallelism
  - no speculative decoding in the first runnable path
- Initial attention/backend target is the TRTLLM attention path that consumes:
  - `copy_batch_block_offsets`
  - `get_block_ids_per_seq`
  - `get_buffers`
  - `get_unique_primary_pool`
- Initial exported KV tensor layout target is `NHD`.
- `kv_cache_manager.impl` consumers have to be handled explicitly; they are not
  an optional follow-up.

## Supported-Path Inventory

Status: completed

Primary v2-shaped surface found in local TRT-LLM checkout:

- `prepare_context`
- `resize_context`
- `try_allocate_generation`
- `is_request_active`
- `suspend_request`
- `prepare_resources`
- `update_resources`
- `free_resources`
- `copy_batch_block_offsets`
- `get_cache_indices`
- `get_batch_cache_indices`
- `get_block_ids_per_seq`
- `get_buffers`
- `get_num_free_blocks`
- `get_num_available_tokens`
- `get_num_kv_blocks`
- `add_dummy_requests`
- `get_kv_cache_stats`

Observed `impl` and compatibility touch points:

- `/tmp/trtllm-latest/tensorrt_llm/_torch/pyexecutor/_util.py`
- `/tmp/trtllm-latest/tensorrt_llm/_torch/pyexecutor/kv_cache_transceiver.py`
- `/tmp/trtllm-latest/tensorrt_llm/_torch/disaggregation/resource/kv_extractor.py`
- `/tmp/trtllm-latest/tensorrt_llm/_torch/auto_deploy/shim/ad_executor.py`
- `/tmp/trtllm-latest/tensorrt_llm/_torch/pyexecutor/py_executor.py`

Observed existing repo state:

- `kvbm.trtllm_integration` currently exposes connector classes only.
- `kvbm.trtllm_integration.rust` still imports `_vllm_integration`.
- No dedicated TensorRT-LLM KV cache manager module exists yet.
- Existing Rust DLPack export code is block-oriented in:
  - `lib/bindings/kvbm/src/block_manager/block.rs`
  - `lib/bindings/kvbm/src/block_manager/layer.rs`

## Implementation Phases

### Phase 1: Dedicated TRTLLM integration module

Status: completed

Goal:

- Stop reusing `_vllm_integration` as the public TensorRT-LLM module name.
- Add a dedicated Python manager entry point and dedicated Rust module surface.

Planned changes:

- Add `kvbm.trtllm_integration.kv_cache_manager`.
- Add a dedicated `_trtllm_integration` Rust pymodule that re-exports the
  shared request/block primitives plus TRTLLM-specific connector classes.
- Update the Python Rust loader to import `_trtllm_integration`.
- Add unit tests that validate the loader and manager surface using stubbed
  TensorRT-LLM modules.

### Phase 2: Manager shell and request-state contract

Status: completed

Goal:

- Land a thin Python `KvbmKVCacheManager` with the v2-shaped contract.
- Back it with minimal internal state and explicit `NotImplementedError`
  boundaries where KVBM-owned tensor export is not wired yet.

Implemented so far:

- constructor and attribute validation
- request lifecycle bookkeeping aligned more closely with pinned TRTLLM v2
- first-chunk versus later-chunk context preparation semantics
- host block-offset bookkeeping with per-slot clearing/restoration
- generation allocation and rewind bookkeeping
- cache-index and padded block-id export helpers
- dummy-request seeding
- small aggregated-path `impl` compatibility shim
- explicit `NotImplementedError` boundaries for tensor export

Deferred to later phases:

- integration with a real Rust-backed allocator/export path
- richer draft-path semantics beyond mirrored bookkeeping
- optional environment bootstrapping for richer TRTLLM/PyTorch validation later
- any redesign of `BAD_PAGE_INDEX` beyond the current supported path
### Phase 3: Buffer and primary-pool export

Status: completed

Goal:

- Replace placeholder tensor ownership with KVBM-backed exported views.

Planned changes:

- Add non-block-oriented Rust tensor export helpers
- export primary pool tensor
- export per-layer tensors compatible with TRTLLM expectations

Implemented so far:

- activated the previously dormant Rust export modules by wiring them into
  `lib/bindings/kvbm/src/block_manager.rs`
- generalized the local DLPack wrapper to preserve explicit strides instead of
  forcing contiguous shapes
- added pooled block-list primary-pool export support
- added pooled per-layer export support using explicit non-contiguous strides
- added Rust unit tests for the exported pool/layer shape-stride contracts
- added a dedicated Rust TRTLLM primary-pool constructor that builds a local
  KVBM-owned `FullyContiguous` slab and exposes it as a `BlockList`
- extended the generic block wrappers so those exported views can own raw local
  KVBM blocks instead of requiring pool-managed mutable blocks only
- wired `KvbmKVCacheManager` to auto-create and use that Rust-backed pool when
  uniform TRTLLM layer head counts are available
- reshaped the Rust DLPack exports in Python to the pinned TRTLLM v2 layouts:
  - primary pool: `[blocks, layers, kv_factor, page_size, num_heads, head_dim]`
  - layer buffer NHD: `[blocks, kv_factor, page_size, num_heads, head_dim]`
  - layer buffer HND: `[blocks, kv_factor, num_heads, page_size, head_dim]`
- added stdlib-only manager tests that verify native pool autowiring without
  requiring a full TRTLLM runtime or torch at test time

### Phase 4: Full request lifecycle ownership

Status: completed

Goal:

- Move context/generation growth, teardown, and reuse fully under the KVBM
  manager for the supported path.

Implemented so far:

- added a Rust `TrtllmStateManager` that owns:
  - request-to-block allocation state
  - request slot assignment
  - generation growth / rewind bookkeeping
  - free-block accounting
- updated the Python TRTLLM manager to auto-create and delegate to that Rust
  lifecycle helper when available, while keeping the old Python logic as a
  fallback for stub-only tests and environments without the extension
- added stdlib-only tests that verify the Python manager can delegate request
  lifecycle state to the native helper surface
- moved supported-path dummy request allocation onto the native helper too,
  instead of seeding dummy state through Python `prepare_context()` plus
  direct resize calls
- fixed `get_kv_cache_stats()` to report native-backed allocations instead of
  incorrectly reading only the dormant Python fallback free-list
- kept host block-offset materialization as a thin Python-side formatting step;
  the native helper already owns slot assignment plus padded block rows, and
  the extra Python copy remains sufficient for the pinned aggregated path
- added repeated allocate/free, slot reuse, public shutdown, and shimmed
  `impl.shutdown()` coverage for both fallback and native-backed paths

Decision:

- host block-offset formatting remains in Python for now
- request/block ownership remains native on the supported path
- no additional native host-row formatter is justified until a real runtime path
  demonstrates a correctness or performance need

### Phase 5: `impl` compatibility surface

Status: completed

Goal:

- Implement or patch the minimal `impl` surface still needed by the pinned
  supported path.

Implemented so far:

- retained the minimal aggregated-path shim for:
  - `get_primary_pool_data()`
  - `get_unique_primary_pool()`
  - `clear_reusable_blocks()`
  - `shutdown()`
- routed shim shutdown through the real manager teardown path so both direct
  `manager.shutdown()` and wrapper-style `impl.shutdown()` teardown behave
  correctly against the pinned TRT-LLM call sites
- re-checked the pinned upstream consumers and found no additional aggregated
  supported-path `impl` methods that need to move into Rust today

### Phase 6: Rank-local multi-GPU expansion

Status: completed

Goal:

- Support non-disaggregated TensorRT-LLM bring-up for:
  - `tp=4, pp=1`
  - `tp=2, pp=2`
- Keep TensorRT-LLM responsible for TP/PP scheduling and collectives.
- Make KVBM own rank-local KV memory and rank-local request/block state on each
  TRT-LLM worker process.
- Include baseline TRT-LLM MLA support in this worker model:
  - single latent-cache pool
  - `kv_factor=1`
  - compatible with `load_paged_kv_cache_for_mla`,
    `load_chunked_kv_cache_for_mla`, and
    `mla_rope_append_paged_kv_assign_q`

Required API and implementation changes:

- Extend `KvbmKVCacheManager` construction with explicit topology/device
  metadata:
  - `device_id`
  - `world_size`
  - `tp_size`
  - `tp_rank`
  - `pp_size`
  - `pp_rank`
- Make local KV geometry rank-aware:
  - local exported heads must use the TP shard, not global head count
  - local exported layers must respect the PP stage's `pp_layers`
- Split MHA/MQA assumptions from MLA assumptions:
  - standard attention uses head-sharded K/V exports
  - MLA uses SELFKONLY-style latent cache with `kv_factor=1`
  - MLA cache sizing under TP is not the same as `num_heads / tp_size`
- Pass `device_id` through to the Rust primary-pool constructor so each rank
  allocates on its own GPU instead of implicitly assuming device 0/default
- Extend the native lifecycle helper so its state is explicitly scoped to one
  TRT-LLM rank/stage and can expose that identity for debugging and validation
- Re-check host block-offset and cache-index helpers against rank-local block
  numbering; baseline TP/PP should keep physical block IDs local to a rank
  unless the TRT-LLM call site proves a wider namespace is required

Design decision for this phase:

- The baseline supported architecture is one KVBM-backed TRT-LLM KV manager per
  process/rank, not a shared distributed allocator
- The first MLA target is baseline TRT-LLM MLA only:
  - support the single-pool latent-cache contract used by the TRT-LLM backend
  - do not require DSA sparse MLA in the first milestone
  - do not require FlashInfer MLA dual-cache layout in the first milestone
- TensorRT-LLM continues to own:
  - request scheduling
  - TP collectives
  - PP send/recv and stage coordination
- KVBM owns:
  - rank-local KV tensors
  - rank-local block allocation
  - rank-local request/block lifecycle invariants

Exit criteria:

- Constructor/API can represent `tp>1` and `pp>1`
- Exported pool/layer views are correct for:
  - standard local heads / local layers
  - baseline MLA latent-cache layout
- The manager can be instantiated independently on each TRT-LLM rank for
  `tp=4,pp=1` and `tp=2,pp=2` shapes without pretending KV is globally shared

Implemented so far:

- extended `KvbmKVCacheManager` construction with explicit worker topology:
  - `device_id`
  - `world_size`
  - `tp_size`
  - `tp_rank`
  - `pp_size`
  - `pp_rank`
- added explicit cache-mode selection:
  - `standard` keeps K/V layout with `kv_factor=2`
  - `mla` keeps baseline TRT-LLM latent-cache layout with `kv_factor=1`
- made local exported KV geometry rank-aware:
  - standard attention now derives rank-local heads with the same ceil-TP shard
    rule used by pinned TRT-LLM
  - baseline MLA keeps latent-cache head geometry local without pretending it
    follows standard `num_heads / tp_size` sharding
  - default `layer_offsets` now map global PP layers to local offsets, so the
    manager surface is PP-rank aware even without manual overrides
- threaded worker identity plus `kv_factor` / `cache_mode` into the native Rust
  `TrtllmStateManager`
- threaded `device_id` into Rust primary-pool creation so each manager can
  allocate its own rank-local export slab
- exposed a stable `get_worker_identity()` surface in Python and native helper
  identity getters in Rust for later validation/debugging work
- added stdlib-only tests that validate:
  - `tp=4,pp=1` style standard construction/export semantics
  - `tp=2,pp=2` style rank-local standard construction/export semantics
  - baseline MLA latent-cache construction/export semantics with `kv_factor=1`
  - native state helper construction receives the full worker topology scope

### Phase 7: Disaggregated serving convergence

Status: in progress

Goal:

- Extend the TRT-LLM manager path so a TRT-LLM worker can still own its local
  KV memory via KVBM while also exposing KVBM storage tiers and transfer hooks
  to TRT-LLM's disaggregation runtime.

Current blockers identified from the pinned TRT-LLM disaggregation surface:

- The current aggregated-path `_ImplCompat` shim is too small for
  `KVRegionExtractorV1` / `KvCacheTransceiverV2`
- The TRT-LLM disaggregation path expects manager/impl storage metadata that
  the current manager does not expose, including:
  - `impl.layer_grouping`
  - `impl._storage`
  - `impl._init_config`
  - `impl._life_cycles`
  - `impl.get_indexer_k_cache_pool()`
- The current manager does not model multi-life-cycle/layer-group storage
  ownership, so TRT-LLM cannot build a V2 page table from it
- The current TRT-LLM manager path has no local representation of KVBM storage
  tiers (GPU/host/disk) even though the older connector path and KVBM storage
  layer do
- Rank exchange for disaggregation currently exists in TRT-LLM and in the older
  KVBM connector path, but not in the new TRT-LLM KV manager path
- MLA-specific transfer/storage shapes are not modeled yet:
  - standard TRT-LLM MLA uses a single latent-cache pool (`kv_factor=1`)
  - FlashInfer MLA uses two separate caches (`ckv_cache`, `kpe_cache`)
  - DSA sparse MLA adds an indexer K-cache side pool and extra block-offset
    metadata

Required design work:

- Decide whether the TRT-LLM manager should:
  - grow into a V2-style storage-backed manager surface directly, or
  - expose a narrower adapter that wraps KVBM storage/lifecycle primitives in
    the exact TRT-LLM V2 contracts
- Define how KVBM storage tiers map onto TRT-LLM life cycles, pool groups, and
  layer groups for:
  - single-stage TP
  - multi-stage PP
  - disaggregated load/store transfer
- Keep Rust as the source of truth for ownership/state transitions:
  - request lifecycle
  - block lifecycle
  - slot ownership
  - storage-tier residency
  - transfer eligibility
- Decide whether MLA support is in-scope for the first disaggregated worker:
  - baseline MLA on the TRT-LLM backend only
  - DSA sparse MLA with indexer cache later
  - FlashInfer MLA dual-cache layout later

Exit criteria:

- TRT-LLM can build a valid disaggregation page table from the KVBM-backed
  manager
- KVBM-backed TRT-LLM workers can participate in disaggregated transfer without
  abandoning Rust-owned lifecycle/storage invariants

Implemented so far:

- extended the local TRTLLM `impl` compatibility surface with a narrow fake-V2
  metadata adapter:
  - `impl.layer_grouping`
  - `impl._storage`
  - `impl._init_config`
  - `impl._life_cycles`
- added manager-owned disaggregation metadata synthesis from the exported
  primary-pool tensor layout instead of trying to emulate TRT-LLM ownership:
  - computes slot bytes and per-layer role offsets from the
    `[blocks, layers, kv_factor, page_size, num_heads, head_dim]` export
  - exposes one life cycle / one pool group for the currently supported
    non-sliding-window path
  - emits key-only metadata automatically for baseline MLA (`kv_factor=1`)
- added a lightweight `mapping` shim plus derived `max_batch_size`,
  `is_vswa=False`, and `max_draft_len=0` so TRT-LLM's Python
  disaggregation/rank-info helpers have the manager fields they read first
- added stdlib-only tests that validate the fake-V2 metadata contract for:
  - standard K/V layout
  - baseline MLA key-only layout

## Progress Log

- Completed design inventory against local TRT-LLM checkout.
- Confirmed the current repo has TRT-LLM connector integration but not a KV
  manager integration.
- Confirmed the first practical coding milestone is a dedicated TRTLLM module
  boundary plus a manager shell and tests that can run without the full runtime.
- Added a dedicated Rust `_trtllm_integration` pymodule in
  `lib/bindings/kvbm/src/block_manager/trtllm.rs`.
- Updated the Python TRTLLM Rust loader to import `_trtllm_integration` instead
  of `_vllm_integration`.
- Added `kvbm.trtllm_integration.kv_cache_manager.KvbmKVCacheManager` as a thin
  dependency-light manager shell.
- Added stdlib-only tests in `lib/bindings/kvbm/tests/test_trtllm_integration.py`.
- Added host slot and block-offset bookkeeping to the manager shell so
  `copy_batch_block_offsets()` now mirrors TRTLLM-style cached block rows
  instead of directly returning raw block IDs.
- Tightened the Python manager shell against pinned TRTLLM v2 semantics:
  - non-first context chunks now require existing request state
  - context resize preserves existing capacity like upstream v2
  - generation allocation on unknown requests now fails cleanly instead of
    raising
  - suspended requests are skipped during late updates
  - generation updates now apply rewind before committing the new history
- Added a minimal aggregated-path `impl` compatibility shim that delegates
  `get_primary_pool_data()` and `get_unique_primary_pool()` back to the Python
  manager surface instead of leaving `impl=None`.
- Extended stdlib-only tests to cover:
  - non-first context chunk semantics
  - missing-generation behavior
  - multi-pool host block-offset copying
  - layer export resolution through global/local layer ids
  - the new `impl` compatibility shim
- Activated the Rust export module tree (`block`, `block_list`, `layer`,
  `dlpack`) so the previously written export helpers are compiled and testable.
- Added stride-aware DLPack support in the local binding wrapper instead of
  assuming all exported tensors are contiguous.
- Added pooled block-list exports:
  - `BlockList.__dlpack__()` now exposes a primary-pool-style block-first view
  - `BlockList.layer_view(layer_idx)` returns a per-layer pooled export with
    explicit non-contiguous strides
- Added a dedicated TRTLLM primary-pool constructor in
  `lib/bindings/kvbm/src/block_manager/trtllm.rs` that builds a local
  KVBM-owned `FullyContiguous` slab rather than reusing the logical/distributed
  Python `BlockManager`.
- Wired the Python TRTLLM manager to use that Rust-backed pool automatically
  when the local layer head counts are uniform, and to reshape the exported
  DLPack views into TRTLLM v2-compatible NHD/HND tensors lazily via torch only
  when a real DLPack consumer is present.
- Added a native Rust `TrtllmStateManager` and started delegating request
  lifecycle bookkeeping from the Python manager into that object for the
  supported path.
- Added native helper support for dummy request allocation so the supported
  path no longer needs Python-side dummy request seeding before allocation.
- Fixed native-backed KV cache stats reporting so allocated bytes now reflect
  native free-block state plus full TRTLLM block geometry.
- Added public/shimmed shutdown coverage plus repeated slot-reuse teardown
  coverage, and completed the supported-path decision to keep host block-offset
  row materialization in Python.
- Completed the rank-local multi-GPU expansion milestone:
  - topology-aware TRT-LLM manager construction now models one KVBM-backed
    manager per TRT-LLM worker/rank
  - standard attention derives local KV head geometry from TP size
  - baseline MLA keeps a single latent-cache pool with `kv_factor=1`
  - native pool/state helpers now carry device/rank/stage identity
- Landed the first disaggregation-facing adapter step:
  - the manager now synthesizes a fake-V2 storage/page-table metadata surface
    directly from the KVBM-owned primary pool
  - the manager now exposes the lightweight `mapping` and batch-size fields
    that TRT-LLM's Python transfer/rank-info helpers read before transfer
    starts
- Inspected KVBM storage/layout internals and found the key phase-3 tension:
  `FullyContiguous` can provide a clean primary-pool slab, but the current
  DLPack helper only exports contiguous shapes, while TRTLLM also needs
  per-layer views that are not naturally contiguous in that layout.

## New Findings

- Current DLPack export in `lib/bindings/kvbm/src/block_manager/dlpack.rs`
  assumes contiguous tensors only.
- The pinned TRTLLM v2 control path is close enough to model in Python without
  native TRTLLM bindings; the biggest local semantic gaps were chunked-context
  handling and late generation rewind, not raw allocator shape.
- Aggregated execution can avoid most `impl` coupling, but leaving `impl=None`
  was too weak even for compatibility work; a tiny shim is enough for the
  currently supported non-disaggregated path.
- `dlpark` already supports explicit strides, so the export blocker was in the
  local wrapper rather than the upstream dependency.
- The old Rust export files were present on disk but not wired into the active
  module tree; once activated, they also needed API updates from the current
  `block_data_mut()` signature before they would compile.
- The first concrete export-model decision is now made:
  use `FullyContiguous`-compatible pooled exports with explicit strides for
  per-layer views instead of adding a second tensor-only layout immediately.
- `FullyContiguous` layout is a good fit for `get_unique_primary_pool()`.
- The logical/distributed `BlockManager` binding cannot be the source of TRTLLM
  DLPack exports because logical blocks do not expose local views.
- The supported-path answer is therefore:
  build a dedicated local KVBM-owned `FullyContiguous` export slab for TRTLLM,
  then reshape the existing block-list DLPack views in Python into TRTLLM's
  exact NHD/HND tensor layouts.
- That keeps ownership inside KVBM without inventing a second storage format,
  and it avoids forcing the Python manager to synthesize fake tensor metadata.
- The same logical/local split applies to lifecycle ownership: the Python shell
  can delegate request accounting to Rust without reusing the logical
  distributed `BlockManager`, because the supported path only needs local
  request/block bookkeeping plus exported tensor views today.
- `get_kv_cache_stats()` was still a real correctness gap after the native
  lifecycle work: it read only the Python fallback free-list, so native-backed
  allocations always reported zero bytes until fixed.
- After re-reading the current native lifecycle boundary, the only remaining
  supported-path Python-owned state that materially affects correctness is host
  block-offset row formatting; request/block ownership can stay native.
- Re-reading the pinned upstream shutdown paths showed both direct
  `manager.shutdown()` and wrapper-style `impl.shutdown()` matter; routing the
  shimmed shutdown through the manager is sufficient for the pinned aggregated
  path.
- No additional aggregated-path `impl` methods were found beyond the current
  shim once the public manager teardown behavior was added.
- The next multi-GPU step should not start with KVBM's older distributed
  leader/worker transfer path. For baseline TRT-LLM TP/PP, the correct model is
  one KVBM-backed manager per TRT-LLM rank with TensorRT-LLM still owning
  schedule/collective coordination.
- The main disaggregated-serving blocker is no longer generic transfer support;
  it is the missing V2-style storage/page-table API expected by TRT-LLM's
  disaggregation extractor/transceiver code.
- The current TRT-LLM manager constructor is missing the topology surface needed
  to represent a real worker instance:
  - `device_id`
  - `world_size`
  - `tp_size`
  - `tp_rank`
  - `pp_size`
  - `pp_rank`
- The current native TRT-LLM pool/state helpers are local-only and therefore
  usable as the basis of per-rank ownership, but they do not yet encode rank or
  stage identity for validation, debugging, or disaggregated coordination.
- TRT-LLM disaggregation expects richer manager metadata than the current shim
  exposes, notably:
  - `impl.layer_grouping`
  - `impl._storage`
  - `impl._init_config`
  - `impl._life_cycles`
  - `impl.get_indexer_k_cache_pool()`
- The current TRT-LLM manager and plan are still MHA-shaped:
  - this is now fixed for the baseline worker path:
    - the manager supports both `standard` (`kv_factor=2`) and baseline `mla`
      (`kv_factor=1`) modes
    - local exported head geometry is now TP-aware for standard attention
    - baseline MLA latent-cache geometry no longer pretends to use standard TP
      head sharding
  - the remaining MLA gap is disaggregated/storage-aware support for indexer or
    dual-cache variants, not the baseline worker-local latent-cache shape
- The default `layer_offsets` mapping needed to change from synthetic
  `0..N-1 -> 0..N-1` aliases to `global_pp_layer -> local_offset`; otherwise a
  multi-stage manager without explicit overrides could not resolve its own
  global layer IDs correctly.
- Rank/stage identity is now explicit on both sides of the Python/Rust seam,
  which is enough for local worker validation and future disaggregation
  adapters. The remaining phase-7 blocker is not identity; it is missing
  storage/page-table metadata.
- After adding the fake-V2 metadata surface, the next real runtime blocker
  moved outside this repo:
  - importing the installed TRT-LLM disaggregation builder in
    `/workspace/model-performance/michaelfeil1209/mfdynamo/.venv`
    still aborts inside Open MPI / PMIx initialization on this machine
  - this happens even with `TLLM_DISABLE_MPI=1`, so the current runtime issue is
    not the KVBM manager API anymore; it is the local TRT-LLM / MPI import
    environment
- TRT-LLM's Python disaggregation helpers also read more than page-table data:
  `RankInfo.from_kv_cache_manager()` and `KvCacheTransceiverV2` expect
  `mapping` and `max_batch_size`, so the phase-7 minimum viable adapter had to
  include those fields as well.

## Testing Log

- Passed: `python3 -m unittest discover -s lib/bindings/kvbm/tests -p 'test_*.py'`
- Passed: `cargo test --manifest-path lib/bindings/kvbm/Cargo.toml --no-run`
- Passed: `cargo check --manifest-path lib/bindings/kvbm/Cargo.toml`
- Passed again after host block-offset changes:
  `python3 -m unittest discover -s lib/bindings/kvbm/tests -p 'test_*.py'`
- Passed after aligning pinned v2 Python semantics and adding `impl` compat
  coverage:
  `python3 -m unittest discover -s lib/bindings/kvbm/tests -p 'test_*.py'`
- Passed after the same milestone:
  `cargo check --manifest-path lib/bindings/kvbm/Cargo.toml`
- Passed after starting the Rust lifecycle delegation milestone:
  `python3 -m unittest discover -s lib/bindings/kvbm/tests -p 'test_*.py'`
- Passed after the same milestone:
  `cargo check --manifest-path lib/bindings/kvbm/Cargo.toml`
- Passed after activating and extending the Rust export module tree:
  `cargo test --manifest-path lib/bindings/kvbm/Cargo.toml --lib`
- Passed again after the same milestone:
  `python3 -m unittest discover -s lib/bindings/kvbm/tests -p 'test_*.py'`
- Passed after wiring the Rust TRTLLM primary-pool constructor and Python
  manager autowiring:
  `python3 -m unittest discover -s lib/bindings/kvbm/tests -p 'test_*.py'`
- Passed after the same milestone:
  `cargo check --manifest-path lib/bindings/kvbm/Cargo.toml`
- Passed after moving dummy request allocation to the native helper and fixing
  native-backed stats:
  `python3 -m unittest discover -s lib/bindings/kvbm/tests -p 'test_*.py'`
- Passed after the same milestone:
  `cargo check --manifest-path lib/bindings/kvbm/Cargo.toml`
- Passed after adding public/shimmed shutdown coverage and fallback/native
  teardown regression tests:
  `python3 -m unittest discover -s lib/bindings/kvbm/tests -p 'test_*.py'`
- Passed after the same milestone:
  `cargo check --manifest-path lib/bindings/kvbm/Cargo.toml`
- Passed after the rank-local topology + baseline MLA milestone:
  `python3 -m unittest discover -s lib/bindings/kvbm/tests -p 'test_*.py'`
- Passed after the same milestone:
  `cargo check --manifest-path lib/bindings/kvbm/Cargo.toml`
- Passed after the fake-V2 disaggregation metadata milestone:
  `python3 -m unittest discover -s lib/bindings/kvbm/tests -p 'test_*.py'`
- Passed after the same milestone:
  `cargo check --manifest-path lib/bindings/kvbm/Cargo.toml`
- Failed in the repo `.venv` direct TRT-LLM smoke path before page-table
  validation completed:
  `TLLM_DISABLE_MPI=1 .venv/bin/python ... build_page_table_from_manager(manager)`
  Reason:
  - local import first needed stubbed `nixl` / `kvbm._core` because the repo
    Python package is not installed as a full built wheel in `.venv`
  - after stubbing those, TRT-LLM still aborted during Open MPI / PMIx init
    (`MPI_Init_thread`, `Unable to start a daemon on the local node`)
  - this remains an external runtime-environment blocker on this machine
- Failed once due to wrong package selector:
  `cargo test -p kvbm --manifest-path lib/bindings/kvbm/Cargo.toml --no-run`
  Reason: the crate is named `kvbm-py3`, not `kvbm`.
- Failed in the current environment after the new TRTLLM export milestone:
  `cargo test --manifest-path lib/bindings/kvbm/Cargo.toml --lib`
  Reason: PyO3 test-binary linking could not resolve Python C-API symbols
  (`PyErr_Print`, `PyObject_GetAttr`, etc.). `cargo check` still passes, so the
  code compiles; the blocker is the local Python link environment for the Rust
  test binary.
- Commit hooks could not run to completion in this environment:
  - first due read-only pre-commit cache under `/root/.cache/pre-commit`
  - then due network-restricted hook bootstrap (`git fetch` to GitHub failed)
  Result: signed commits for this work need `--no-verify` unless hook assets are
  already available locally.

## Remaining Work

- Baseline aggregated single-GPU support is complete for the pinned TRTLLM v2
  surface described above.
- Baseline rank-local multi-GPU worker support is now complete for the pinned
  non-disaggregated path:
  - topology-aware manager construction
  - per-rank head/layer export semantics
  - baseline MLA latent-cache export (`kv_factor=1`) on the same worker-local
    path
  - explicit device placement in Rust pool creation
  - rank/stage-aware validation for `tp>1` and `pp>1`
- Disaggregated-serving convergence is still pending:
  - validate the new V2-style storage/page-table surface against a TRT-LLM
    runtime environment that can import disaggregation modules without MPI/PMIx
    aborting
  - mapping KVBM storage tiers onto TRT-LLM life cycles / pool groups
  - preserving Rust-owned lifecycle and residency guarantees through transfer
    flows
- External blocker remains: add a runtime-capable validation path for the Rust
  test binary once the local PyO3/Python link environment is fixed; until then
  rely on `cargo check` plus Python contract tests on this machine.
- Additional external blocker now identified for phase 7 runtime validation:
  the local `.venv` TRT-LLM import path aborts inside Open MPI / PMIx before the
  direct disaggregation page-table smoke check can run.

## Exact Next Step

1. Re-run the direct TRT-LLM smoke check on a host or container where importing
   `.venv` `tensorrt_llm` disaggregation modules does not abort in Open MPI /
   PMIx. Use the same manager construction added in this repo and validate that
   `build_page_table_from_manager(manager)` succeeds end-to-end.
2. Once that runtime import blocker is gone, continue phase 7 by checking
   whether `KvCacheTransceiverV2` can progress past `RankInfo` and transfer
   worker setup using the current fake-V2 metadata, or whether it still needs
   additional `impl` storage-tier/life-cycle detail.
3. If more detail is required, the next file to extend is:
   `/workspace/model-performance/michaelfeil1209/mfdynamo/lib/bindings/kvbm/python/kvbm/trtllm_integration/kv_cache_manager.py`
   specifically the fake-V2 metadata builders:
   - `_build_disagg_metadata()`
   - `get_disagg_storage_metadata()`
   - `get_disagg_init_config()`
   - `get_disagg_life_cycles()`
- /workspace/model-performance/michaelfeil1209/mfdynamo/.venv has a installation of torch and tensorrt_llm active, can be patched.
