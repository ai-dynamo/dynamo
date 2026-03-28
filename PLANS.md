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

Status: in progress

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

Still pending in this phase:

- decide whether host block-offset materialization should also move into Rust or
  remain a thin Python-side formatting step
- add repeated allocate/free and teardown coverage around the now-native dummy
  and lifecycle path

### Phase 5: `impl` compatibility surface

Status: pending

Goal:

- Implement or patch the minimal `impl` surface still needed by the pinned
  supported path.

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

- Phase 4: decide whether host block-offset row materialization should move into
  Rust now that slot assignment and padded block rows are already available
  there.
- Phase 4: add repeated allocate/free and teardown coverage for both fallback
  and native-backed lifecycle paths.
- Phase 5: add the public manager shutdown/teardown behavior still required by
  the pinned executor path, then re-check whether the current tiny aggregated
  `impl` shim is sufficient.
- Add a runtime-capable validation path for the Rust test binary once the local
  PyO3/Python link environment is fixed; until then rely on `cargo check` plus
  Python contract tests for this machine.
- Continue updating this file after every milestone with exact next steps.

## Exact Next Step

1. Add a public `KvbmKVCacheManager.shutdown()` that clears native/fallback
   state, host block-offset rows, and owned export references in the way the
   pinned executor expects.
2. Keep host block-offset formatting in Python unless shutdown/teardown testing
   exposes a real native-only correctness need; padded rows are already native.
3. Extend tests around repeated allocate/free, slot reuse, dummy requests, and
   teardown, then re-check the pinned aggregated-path `impl` consumers.
