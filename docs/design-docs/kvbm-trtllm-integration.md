<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# KVBM TensorRT-LLM Deep Integration Plan

**Status:** Draft design
**Objective:** Replace TensorRT-LLM's KV cache manager with a KVBM-backed implementation that owns the GPU KV buffers directly
**Architecture rule:** KVBM is the sole owner of KV cache placement across GPU, CPU, and disk for the supported path

---

## 1. Objective

TensorRT-LLM currently owns GPU KV allocation in both of its in-tree KV manager implementations. KVBM participates today through connector-style integration, but that does not make KVBM the owner of the GPU cache.

This design targets a stricter end state:

- KVBM allocates and owns the GPU KV pool.
- KVBM exports TensorRT-LLM-compatible buffer views and page indices.
- KVBM remains the source of truth for reuse, eviction, and future offload tiers.

This is a direct manager replacement, not a sidecar extension.

---

## 2. Scope

### In scope

- A TensorRT-LLM-specific KVBM KV manager.
- KVBM-owned GPU buffer allocation.
- TensorRT-LLM-compatible page-index and block-ID export.
- Prefix reuse and request lifecycle under KVBM ownership.
- Later extension to CPU and disk tiers through the same manager.

### Initial supported envelope

The first implementation should support only:

- one pinned TensorRT-LLM version
- single GPU
- beam width 1
- one attention backend
- one KV layout path
- aggregated execution only

The architecture must still leave room for one-model speculative decoding, especially MTP, because that is a required follow-on capability for the PyTorch backend.

The first implementation should not support:

- pipeline parallelism
- tensor parallelism
- speculative decoding
- multimodal
- Mamba or SSM cache managers
- cache transceiver and disaggregated transfer flows

---

## 3. Upstream Findings

The current TensorRT-LLM checkout inspected for this design is under `/tmp/trtllm-latest`.

### 3.1 v1 owns its own pools

In `tensorrt_llm/_torch/pyexecutor/resource_manager.py`, the v1 `KVCacheManager`:

- constructs `KVCacheManagerCpp`
- calls `allocate_pools(False)`
- reads back pool pointers and layer-to-pool mapping
- exports per-layer views via `get_primary_pool_data()`

This makes buffer ownership part of the manager contract.

### 3.2 Attention backends consume manager-owned buffers directly

The manager is called directly by attention code in:

- `tensorrt_llm/_torch/attention_backend/flashinfer.py`
- `tensorrt_llm/_torch/attention_backend/trtllm.py`
- `tensorrt_llm/_torch/attention_backend/trtllm_gen.py`

These paths depend on:

- `get_buffers()`
- `get_batch_cache_indices()`
- `get_block_ids_per_seq()`
- host block-offset or page-index data

The replacement manager must preserve buffer shape, stride, and page-index semantics expected by the selected backend.

### 3.3 Executor setup expects access to the primary GPU pool

In `tensorrt_llm/_torch/pyexecutor/py_executor.py`, the executor calls `get_unique_primary_pool()` and passes the result into connector worker registration.

If KVBM owns the buffers, this method must return a KVBM-owned pool tensor in the same effective format.

### 3.4 Some TensorRT-LLM code depends on `kv_cache_manager.impl`

The public Python manager interface is not the only seam. TensorRT-LLM also passes `kv_cache_manager.impl` into helper code such as:

- `tensorrt_llm/_torch/pyexecutor/_util.py`
- `tensorrt_llm/_torch/pyexecutor/kv_cache_transceiver.py`

The design therefore requires one of two approaches:

- provide a KVBM-backed `impl` compatibility surface
- patch the relevant TensorRT-LLM integration points in this repo

### 3.5 v2 is structurally cleaner but still self-allocates

The v2 manager in `tensorrt_llm/runtime/kv_cache_manager_v2/` is more Python-visible, but it still creates and owns GPU storage internally. It is not an external memory injection interface.

v2 may still be the better reference model for page-index semantics, but it does not solve GPU ownership by itself.

---

## 4. Design Constraints

The replacement manager must satisfy all of the following:

- Own the GPU KV buffers directly.
- Export tensors compatible with TensorRT-LLM attention kernels.
- Export page indices or block IDs in the exact form consumed by the selected attention backend.
- Match TensorRT-LLM request lifecycle behavior.
- Support prefix reuse under KVBM ownership.
- Avoid depending on TensorRT-LLM's native pool allocator.
- Reserve writable KV capacity for speculative tokens without making that capacity immediately reusable.
- Support partial rewind of speculative writes after acceptance.
- Support a coordinated main-cache and draft-cache view if the chosen speculative mode requires separate draft KV layout.

The hardest constraints are not policy-related. They are:

- buffer layout compatibility
- page-index compatibility
- `impl` compatibility or targeted upstream patching
- speculative reservation and rewind semantics

---

## 5. Feasibility Assessment

This design is potentially feasible with the current KVBM codebase. The core reasons are:

### 5.1 Rust already supports true GPU ownership

`DeviceStorage` in `lib/llm/src/block_manager/storage/cuda.rs` already supports:

- owned CUDA allocations via `DeviceStorage::new(...)`
- external tensor wrapping via `DeviceStorage::new_from_torch(...)`

For the TensorRT-LLM integration, the owned-allocation path is the correct foundation. The external-tensor path is still useful during bring-up, testing, and for compatibility experiments.

### 5.2 KVBM already has a real pool and layout abstraction

`KvBlockManagerState` in `lib/llm/src/block_manager/state.rs` already owns:

- a device pool
- a host pool
- a disk pool
- offload and onboard infrastructure

`LayoutType`, `FullyContiguous`, and `LayerSeparate` in `lib/llm/src/block_manager/layout.rs` already provide stable address calculations for:

- block index
- layer index
- outer dimension index

That means KVBM already has the primitives needed to own a TRT-LLM-compatible slab and compute deterministic per-block and per-layer regions inside it.

### 5.3 KVBM already exports GPU memory to Python

The DLPack binding in `lib/bindings/kvbm/src/block_manager/layer.rs` already exposes device-backed block views as Python-consumable tensors.

This is important because the TensorRT-LLM manager replacement needs exactly that kind of export path:

- whole-pool export for `get_unique_primary_pool()`
- per-layer export for `get_buffers()`

The existing binding is too block-oriented for direct use, but the mechanism is already present.

### 5.4 KVBM already has a request-to-block state machine

The Rust vLLM integration in `lib/bindings/kvbm/src/block_manager/vllm.rs` and `lib/bindings/kvbm/src/block_manager/vllm/slot.rs` already implements:

- per-request slot creation
- token progression tracking
- block allocation
- block reuse
- prefix matching
- request free

That logic is framework-agnostic enough to reuse for TensorRT-LLM. What is vLLM-specific is the Python protocol surface, not the underlying state machine.

### 5.5 KVBM already models torch tensors abstractly

`TorchTensor` in `lib/llm/src/block_manager/storage/torch.rs` already gives KVBM a minimal abstraction for:

- device
- data pointer
- size in bytes
- shape
- stride

That is the right kind of adapter boundary for TensorRT-LLM interop.

### 5.6 Existing TRT-LLM connector code proves some of the interop path

The existing TensorRT-LLM connector worker binding in `lib/bindings/kvbm/src/block_manager/vllm/connector/trtllm_worker.rs` already accepts a TRT-LLM KV tensor, wraps it as a `TorchTensor`, and wires it into KVBM worker setup.

That does not solve ownership, but it does prove that:

- TensorRT-LLM-side tensors can already be adapted into KVBM
- the Python to Rust boundary for CUDA tensors is already working

### 5.7 Main feasibility gap

The main missing pieces are not memory primitives. They are:

- a TensorRT-LLM-specific manager facade
- TensorRT-LLM-compatible page-index export
- whole-pool and per-layer tensor export at the manager boundary
- handling direct `kv_cache_manager.impl` consumers

So the project is feasible, but the missing work is at the framework integration boundary, not in the basic GPU ownership layer.

### 5.8 MTP and speculative decoding are a real design constraint

MTP is hard for this integration because TensorRT-LLM does not treat speculative decoding as a pure scheduler feature. It changes the KV cache contract itself.

Local upstream evidence:

- In `tensorrt_llm/_torch/speculative/utils.py`, one-engine speculative modes reserve extra KV capacity through `get_num_extra_kv_tokens(...)`.
- In `tensorrt_llm/_torch/pyexecutor/scheduler/scheduler_v2.py`, scheduling budget includes draft tokens for both context and generation.
- In `tensorrt_llm/_torch/speculative/spec_sampler_base.py`, requests record `py_num_accepted_draft_tokens` and `py_rewind_len` after draft acceptance.
- In `tensorrt_llm/_torch/pyexecutor/resource_manager.py`, the resource manager rewinds KV cache after rejected draft tokens and resizes cache capacity accordingly.
- In `tensorrt_llm/_torch/speculative/interface.py`, one-engine modes may temporarily swap to a separate `draft_kv_cache_manager` and separate block-offset tensors for draft replay.
- In `tensorrt_llm/_torch/speculative/mtp.py`, MTP also allocates a separate hidden-state and past-token pool keyed by per-request slot IDs.

This is difficult for current KVBM because the existing slot model is built around committed sequence progress and reusable block registration. The current KVBM adapter surface still rejects speculative controls:

- `lib/bindings/kvbm/src/block_manager/vllm.rs` rejects `num_lookahead_blocks`
- `lib/bindings/kvbm/src/block_manager/vllm.rs` rejects `delay_cache_blocks`

That means KVBM currently lacks a first-class notion of:

- reserved but not yet committed KV capacity
- tentative writes
- partial rewind of tentative writes
- draft-only KV views with different layout or length semantics

Without those concepts, MTP support will either be incorrect or will devolve into shadow allocation outside KVBM, which violates the ownership goal.

---

## 6. Target Architecture

```text
TensorRT-LLM PyTorch backend
├── Scheduler / executor
├── Attention backend
└── KVBM-backed KV manager
    ├── Python manager surface
    ├── impl compatibility surface or patched call sites
    ├── KVBM-owned GPU pool
    ├── block ID and page-index export
    ├── prefix reuse
    └── request lifecycle
         ↓
      KVBM Rust core
         ├── device pool
         ├── host pool
         ├── disk pool
         └── transfer / reuse / eviction policy
```

The manager boundary must be explicit. TensorRT-LLM should interact with KVBM through a TensorRT-LLM-specific adapter, not through reused vLLM names or temporary re-exports.

---

## 7. Required Compatibility Surface

The KVBM-backed manager must provide the TensorRT-LLM surface actually used by the supported path.

### 7.1 Manager methods and attributes

At minimum, the supported path uses or is expected to use:

- `prepare_resources`
- `update_resources`
- `free_resources`
- `get_cache_indices`
- `get_batch_cache_indices`
- `get_block_ids_per_seq`
- `get_buffers`
- `get_unique_primary_pool`
- `probe_prefix_match_length`
- `get_num_free_blocks`
- `get_num_available_tokens`
- `get_num_kv_blocks`
- `add_dummy_requests`
- `get_kv_cache_stats`
- `flush_iteration_events`
- `get_latest_events`
- `store_blocks_for_reuse`
- `unpin_blocks_by_id`

The supported path may also rely on attributes such as:

- `tokens_per_block`
- `dtype`
- `head_dim`
- `pp_layers`
- `total_num_kv_heads_per_layer`
- `max_seq_len`

### 7.2 Buffer contract

For the selected attention backend, KVBM must match:

- tensor rank
- tensor shape
- tensor stride
- K/V packing convention
- `NHD` or `HND` layout
- page numbering convention
- host page-index or block-offset representation where applicable

### 7.3 `impl` contract

Before implementation starts, the team must inventory every supported-path use of `kv_cache_manager.impl` and decide whether to:

- emulate that interface inside KVBM, or
- patch those call sites

This decision changes the implementation boundary and should not be deferred.

---

## 8. Implementation Proposal

The recommended implementation is to reuse the existing KVBM stack below the framework-adapter boundary and add only a TensorRT-LLM-specific control layer.

### 8.1 Reuse plan by layer

**Reuse without redesign**

- `lib/llm/src/block_manager/state.rs`
  Use `KvBlockManagerState` as the owner of device, host, and disk pools.

- `lib/llm/src/block_manager/storage/cuda.rs`
  Use `DeviceStorage::new(...)` for true Rust-owned GPU allocation.

- `lib/llm/src/block_manager/layout.rs`
  Reuse the layout abstraction to define the TensorRT-LLM KV pool geometry.

- `lib/llm/src/block_manager/block/data/local.rs`
  Reuse `memory_region(...)` and local block views to derive stable addresses for block/layer export and page-index generation.

- `lib/bindings/kvbm/src/block_manager/vllm/slot.rs`
  Reuse slot lifecycle, token accounting, block tracking, and reuse logic.

- `lib/bindings/kvbm/src/block_manager/layer.rs`
  Reuse the DLPack export pattern for device-backed Python tensors.

**Do not reuse as the external interface**

- `lib/bindings/kvbm/python/kvbm/vllm_integration/kv_cache_manager.py`
  Reuse patterns only; do not reuse the protocol class.

- `_vllm_integration` as the long-term TensorRT-LLM binding home
  Keep shared internals if useful, but create a TensorRT-LLM-specific module boundary.

### 8.2 Proposed Rust objects

Add a TensorRT-LLM-focused Rust facade over the existing block manager state.

Suggested shape:

```rust
pub struct TrtllmKvManager {
    block_manager: PyBlockManager,
    slot_manager: Mutex<SlotManager<String>>,
    export: TensorExportMetadata,
}
```

Responsibilities:

- own request-to-block lifecycle
- expose block IDs and page indices
- expose whole-pool and per-layer tensor views
- keep TensorRT-LLM-specific layout metadata close to the manager

Suggested helper objects:

- `TensorExportMetadata`
  Stores dtype, head_dim, num_kv_heads, tokens_per_block, layer-group offsets, and layout type.

- `TrtllmPageIndexConverter`
  Converts KVBM block IDs into the page numbering convention required by the selected TensorRT-LLM attention backend.

- `TrtllmImplCompat`
  Optional compatibility object for the subset of `kv_cache_manager.impl` behavior that cannot be patched away.

### 8.3 Proposed Python surface

Add a dedicated TensorRT-LLM Python module, for example:

- `kvbm.trtllm_integration.kv_cache_manager`
- `KvbmKVCacheManager`

This class should be thin. It should:

- validate constructor arguments from TensorRT-LLM
- hold the Rust manager object
- implement TensorRT-LLM method names
- convert Rust exports into torch tensors or Python lists

It should not:

- own allocation policy
- own block state
- compute page layout itself

### 8.4 Proposed tensor export API

The Rust manager should expose two tensor export paths:

1. base-pool export
2. per-layer export

Suggested methods:

```rust
fn export_primary_pool(&self) -> PyResult<PyObject>;
fn export_layer_tensor(&self, layer_idx: usize, kv_layout: &str) -> PyResult<PyObject>;
```

Semantics:

- `export_primary_pool()` returns one tensor over the full KVBM-owned slab for the connector registration path.
- `export_layer_tensor()` returns a view with TensorRT-LLM-compatible shape and layout for `get_buffers()`.

This should reuse the existing DLPack machinery where possible. If needed, add a non-block-oriented tensor export helper rather than abusing the existing `Layer` wrapper.

### 8.5 Proposed page-index export API

The current KVBM slot manager already yields block IDs. TensorRT-LLM needs backend-specific indices derived from those block IDs.

Suggested methods:

```rust
fn get_cache_indices(&self, request_id: String) -> PyResult<Vec<u32>>;
fn get_batch_cache_indices(&self, request_ids: Vec<String>, layer_idx: Option<usize>) -> PyResult<Vec<Vec<u32>>>;
fn get_block_ids_per_seq(&self, request_ids: Vec<String>) -> PyResult<PyObject>;
```

Implementation rule:

- keep KVBM block IDs as the canonical internal representation
- add a TensorRT-LLM page-index conversion layer only at export time

That preserves reuse of existing KVBM block tracking and limits TensorRT-LLM-specific logic to the adapter boundary.

### 8.6 Layout recommendation

For the first supported path, prefer a single layout strategy and hold it fixed:

- use a fully contiguous device slab if it matches the selected TensorRT-LLM backend cleanly
- only introduce `LayerSeparate` if the chosen backend requires it

The codebase already supports both, but supporting both immediately is unnecessary scope.

### 8.7 `impl` strategy recommendation

Preferred order:

1. patch the minimal supported-path call sites in this repo if they are small and stable
2. only add `TrtllmImplCompat` for the subset that must remain object-compatible

Reason:

- `impl` emulation can easily expand into an accidental shadow implementation of TensorRT-LLM internals
- the existing evidence suggests only a limited number of supported-path consumers need it initially

### 8.8 MTP-ready state model

KVBM should not model speculative decoding as ordinary decode with extra free blocks. It needs explicit tentative state.

Recommended request-local state:

- `committed_tokens`
  Tokens whose KV is durable and may participate in prefix reuse.

- `tentative_tokens`
  Draft tokens whose KV has been written but not yet accepted.

- `reserved_tokens`
  Writable capacity reserved for the current step, including draft slack and any `num_extra_kv_tokens` requirement.

- `draft_view`
  Optional layout metadata for draft layers when TensorRT-LLM requires a separate draft KV cache manager or separate block-offset tensors.

Recommended state transitions:

- `reserve_speculative(request_id, draft_len, extra_kv_tokens)`
  Reserve writable capacity without publishing new reusable blocks.

- `commit_speculative(request_id, accepted_tokens)`
  Advance the committed boundary, register any newly full committed blocks, and retain or trim unused tentative capacity.

- `rewind_speculative(request_id, rejected_tokens)`
  Drop tentative tail state and free pages that are no longer needed.

- `free_request(request_id)`
  Release both committed and tentative capacity.

This keeps KVBM block IDs canonical internally while giving TensorRT-LLM the reservation and rewind semantics it expects.

### 8.9 MTP-specific implementation proposal

The recommended design is:

1. Keep one Rust owner for all KV memory.
2. Add speculative state to the slot manager instead of creating a second ownership path.
3. Export one or two TensorRT-LLM manager facades depending on mode:
   - main KV manager for target layers
   - optional draft KV manager facade for one-engine speculative modes that require different draft layout
4. Back both facades with the same KVBM-owned allocator and coordinated request state.
5. Treat MTP hidden states as a separate resource pool, not as part of the KV allocator, but reuse the same request or slot identity so scheduling and teardown remain aligned.

This matters because TensorRT-LLM sometimes needs different layer-local KV lengths for draft and target execution. If that requirement appears on the selected path, the right answer is not a second allocator. The right answer is two exported views over one KVBM ownership model.

Concrete changes:

- Extend the Rust slot state with tentative-token accounting and rewind support.
- Replace the current `num_lookahead_blocks is not supported` path with real speculative reservation.
- Add non-reusable mutable blocks or pages that can hold draft writes until acceptance.
- Export draft-layer page indices separately when TensorRT-LLM switches to `draft_kv_cache_manager`.
- Keep block reuse tied only to committed tokens, never to tentative tokens.
- Add a small MTP side resource manager for hidden states and past tokens, coordinated by request slot.

### 8.10 Proposed implementation sequence

1. Create `kvbm._core._trtllm_integration`.
2. Add `TrtllmKvManager` in Rust backed by `KvBlockManagerState` and reused `SlotManager`.
3. Add base-pool tensor export.
4. Add per-layer tensor export.
5. Add block-ID and page-index export.
6. Add speculative reservation, commit, and rewind primitives in the slot manager.
7. Add optional draft-manager export over the same KVBM-owned pool.
8. Implement the Python `KvbmKVCacheManager`.
9. Patch or emulate the small supported-path `impl` surface.

This keeps the new code concentrated at the framework boundary and preserves maximum reuse underneath it.

---

## 9. Implementation Plan

### Phase 0: Pin the supported seam

**Goal:** Freeze the exact target before writing replacement code.

**Tasks**

- Pin a TensorRT-LLM commit or release.
- Inventory every supported-path use of `kv_cache_manager` and `kv_cache_manager.impl`.
- Select the initial attention backend and KV layout.
- Record the expected buffer and page-index semantics for that path.
- Decide whether to emulate `impl` or patch call sites.

**Exit criteria**

- One TensorRT-LLM version is explicitly supported.
- One executor path and one attention backend are explicitly supported.
- The required manager and `impl` surface is documented.

### Phase 1: Build the KVBM-backed manager shell

**Goal:** Instantiate a TensorRT-LLM-facing manager that is backed by KVBM.

**Tasks**

- Add a dedicated TensorRT-LLM manager module.
- Add TensorRT-LLM-specific Rust exports instead of relying on `_vllm_integration` reuse.
- Expose required public attributes and lifecycle methods.
- Wire manager creation so KVBM owns the device pool from the beginning.

**Exit criteria**

- The manager can be constructed in the supported TensorRT-LLM path.
- KVBM owns the device pool.

### Phase 2: Export TensorRT-LLM-compatible GPU buffers

**Goal:** Make TensorRT-LLM read KVBM-owned memory correctly.

**Tasks**

- Export per-layer views from the KVBM device pool.
- Export the full primary pool for connector registration.
- Match the selected backend's layout and stride requirements.
- Add assertions and diagnostics for layout mismatch.

**Exit criteria**

- `get_buffers()` returns valid KVBM-backed tensors.
- `get_unique_primary_pool()` returns a valid KVBM-backed pool tensor.

### Phase 3: Match page-index and block-ID semantics

**Goal:** Make TensorRT-LLM address the correct KVBM pages.

**Tasks**

- Implement `get_cache_indices`, `get_batch_cache_indices`, and `get_block_ids_per_seq`.
- Match the selected backend's page numbering convention.
- Add host page-index or block-offset buffers if required by the chosen path.
- Validate prefill and decode page tables against TensorRT-LLM expectations.

**Exit criteria**

- Prefill reads correct cached pages.
- Decode reads and appends correct cached pages.

### Phase 4: Replace request lifecycle ownership

**Goal:** Move request lifecycle fully under KVBM.

**Tasks**

- Implement request creation, growth, rewind, commit, and teardown.
- Match `prepare_resources`, `update_resources`, and `free_resources`.
- Implement warmup and dummy-request handling.
- Implement prefix reuse and block reuse under KVBM ownership.
- Introduce explicit reserve, commit, and rewind semantics for speculative paths even if MTP is enabled in a later phase.

**Exit criteria**

- End-to-end aggregated generation works.
- Prefix reuse works on the supported path.
- Request teardown does not leak blocks.

### Phase 5: Resolve `impl` consumers

**Goal:** Make the executor stack accept the replacement end to end.

**Tasks**

- Implement the required `impl` surface or patch the call sites that depend on it.
- Validate scheduler construction.
- Validate any remaining helper classes that currently expect the native manager implementation.

**Exit criteria**

- The supported path no longer depends on TensorRT-LLM's native KV allocator.
- All supported-path `impl` consumers are handled.

### Phase 6: Expand support

**Goal:** Broaden support only after single-path correctness is proven.

**Tasks**

- Add MTP and other speculative decoding support on top of the explicit speculative state model.
- Add more attention backends or layouts.
- Add connector and disaggregation support.
- Add CPU and disk tiering under the same manager.
- Add distributed support later.

**Exit criteria**

- Each additional supported path is named explicitly and validated independently.

---

## 10. Validation

### Correctness

- single-GPU prefill
- single-GPU decode
- repeated allocate/free cycles
- prefix reuse
- request teardown
- warmup and dummy-request flows

### Buffer correctness

- `get_buffers()` shape and stride
- page-index correctness
- block-ID correctness
- connector pool registration compatibility

### Compatibility

- supported manager surface
- supported `impl` surface or patched call-site coverage
- supported executor path
- supported attention backend path

### Performance

- TTFT against native TensorRT-LLM baseline
- decode throughput against native baseline
- no significant overhead from KVBM page-index translation

---

## 11. Risks

| Risk | Impact | Mitigation |
|------|--------|------------|
| Buffer layout mismatch causes silent corruption | High | Validate shape, stride, and page addressing before broader rollout |
| `impl` coupling is deeper than expected | High | Inventory all supported-path consumers before implementation |
| Different attention backends require different page semantics | High | Support one backend first |
| v1 and v2 are both self-allocating designs | High | Treat this as a full replacement, not a wrapper |
| Speculative decoding requires tentative writes and rewind, not just ordinary block allocation | High | Add explicit speculative state and do not treat lookahead as normal reuse |
| One-engine MTP may require separate draft KV views over the same ownership model | High | Support dual exported manager views without introducing a second allocator |
| Scope expansion increases blast radius too early | Medium | Keep the first supported matrix deliberately narrow |
| Connector and disaggregation code assume native ownership | Medium | Defer until the aggregated path is stable |

---

## 12. Success Criteria

The first milestone is successful when:

- KVBM owns the GPU KV buffers
- TensorRT-LLM attention consumes KVBM-owned buffers correctly
- prefills and decodes succeed on the supported path
- request lifecycle is fully under KVBM ownership
- TensorRT-LLM no longer depends on its native KV allocator in the supported path

After that, CPU and disk tiering can be added on top of the same ownership model.
