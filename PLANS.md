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

Status: in progress

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

Status: pending

Goal:

- Land a thin Python `KvbmKVCacheManager` with the v2-shaped contract.
- Back it with minimal internal state and explicit `NotImplementedError`
  boundaries where KVBM-owned tensor export is not wired yet.

Planned changes:

- Constructor/attribute validation
- request lifecycle bookkeeping
- host block-offset bookkeeping
- cache-index export helpers
- draft path explicitly unsupported in the initial shell - but should be supported asap after, as has high prio.

### Phase 3: Buffer and primary-pool export

Status: pending

Goal:

- Replace placeholder tensor ownership with KVBM-backed exported views.

Planned changes:

- Add non-block-oriented Rust tensor export helpers
- export primary pool tensor
- export per-layer tensors compatible with TRTLLM expectations

### Phase 4: Full request lifecycle ownership

Status: pending

Goal:

- Move context/generation growth, teardown, and reuse fully under the KVBM
  manager for the supported path.

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

## Testing Log

- No meaningful tests run yet.
- Reason: phase 0 was design/inventory only; next milestone will add code and
  then run targeted tests.

## Remaining Work

- Implement Phase 1 module split.
- Add the TensorRT-LLM manager shell.
- Add targeted tests that run in the current dependency-constrained shell.
- Continue updating this file after every milestone with exact next steps.

## Exact Next Step

1. Add `_trtllm_integration` to `lib/bindings/kvbm/src/block_manager`.
2. Update `lib/bindings/kvbm/python/kvbm/trtllm_integration/rust.py`.
3. Add `lib/bindings/kvbm/python/kvbm/trtllm_integration/kv_cache_manager.py`.
4. Add stub-based unit tests for the new module and manager shell.
