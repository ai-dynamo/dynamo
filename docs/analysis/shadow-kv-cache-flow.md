# Shadow Mode KV Cache Flow (Single-Node)

## Overview

Shadow mode allows a standby vLLM engine to initialize without allocating KV cache,
sleep immediately, and later wake up with KV cache allocated on-demand. This enables
sub-second failover because the shadow engine has weights loaded and CUDA graphs
captured — it only needs to allocate KV cache on wake.

## Init Phase (engine startup)

### 1. Environment Setup
- `main.py:437` sets `SHADOW_SKIP_KV_CACHE=1` in the process environment
- `main.py:446-454` forces `cudagraph_mode=PIECEWISE` (required so attention ops
  are stubbed during graph capture — no KV cache exists yet)

### 2. GMSWorker.load_model()
- `worker.py:98-101` checks `SHADOW_SKIP_KV_CACHE=1`
- Sets `model_runner._shadow_init_phase = True` on the model runner
- This flag controls patch behavior during init

### 3. KV Cache Initialization (collective_rpc: initialize_from_config)
- `gpu_worker.py` calls `model_runner.initialize_kv_cache_tensors()`
- **Patch** (`patches.py:216-224`): checks `_shadow_init_phase`
  - If True: stores `kv_cache_config` and `kernel_block_sizes` for later, returns `{}`
  - `self.kv_caches` remains empty (`{}`)
- Result: no GPU memory allocated for KV cache

### 4. CUDA Graph Capture (collective_rpc: compile_or_warm_up_model)
- `gpu_worker.py:471` calls `model_runner.capture_model()`
- Graph capture calls `_dummy_run()` for each batch size
- `_dummy_run()` calls `self._get_slot_mappings()`
- **Patch** (`patches.py:264-268`): checks `not self.kv_caches`
  - If kv_caches empty: returns `(None, None)`
- In `_dummy_run()` at line 4771:
  - PIECEWISE mode: `force_attention=False`, `cudagraph_runtime_mode=PIECEWISE`
  - Condition `force_attention or mode == FULL` → **False**
  - `_build_attention_metadata` is **NOT called** → no assertion hit
  - Attention ops are split out of the graph (PIECEWISE), so no KV cache needed
- CUDA graphs captured successfully (MLP/norm regions only)

### 5. Engine Ready
- Engine is initialized with weights + CUDA graphs but no KV cache
- `self.kv_caches = {}` (empty)

## Sleep Phase

### 6. Shadow Sleep (`main.py:948`)
- `handler.sleep_engine(level=1)` called
- `engine_client.pause_generation()` — drain requests
- `engine_client.sleep(level)` → `collective_rpc("sleep")`
- `GMSWorker.sleep()` (`worker.py:125-165`):
  - Unmaps GMS weight VAs (preserves VA reservations)
  - Disconnects from GMS (releases RO lock)
  - KV cache: nothing to sleep (empty `{}`)
- GPU memory freed (weights unmapped, KV cache was never allocated)

### 7. Lock Wait (`main.py:960-961`)
- Creates `FlockFailoverLock(lock_path)`
- Blocks on `lock.acquire()` — polls every 100ms
- Wakes when active engine dies (kernel releases flock on FD close)

## Wake Phase

### 8. Lock Acquired → Wake (`main.py:964`)
- `handler.wake_engine()` called
- `engine_client.wake_up(tags)` → `collective_rpc("wake_up")`
- `GMSWorker.wake_up()` (`worker.py:167-237`):

#### 8a. Clear shadow flag
- `worker.py:176-179`: `model_runner._shadow_init_phase = False`
- Future calls to `_get_slot_mappings` and `initialize_kv_cache_tensors` will
  proceed normally (patches pass through to originals)

#### 8b. Remap weights
- `manager.connect(RO, timeout=30s)` — reconnect to GMS
- `manager.remap_all_vas()` — remap weights at same virtual addresses
- PyTorch tensors remain valid (same data pointers)

#### 8c. Allocate KV cache on wake
- `worker.py:219`: `model_runner.allocate_kv_cache_on_wake()`
- **Patched method** (`patches.py:296-339`):
  - Uses stored `_shadow_kv_cache_config` and `_shadow_kernel_block_sizes` from init
  - Calls `initialize_kv_cache_tensors()` (now unpatched, `_shadow_init_phase=False`)
  - Allocates full KV cache GPU memory
  - Registers KV caches with NIXL transfer group if available
  - `self.kv_caches` now populated

### 9. Resume
- `engine_client.resume_generation()` — scheduler restarts
- `main.py:967-974`: registers with discovery, starts serving

## Key Invariants

- `_shadow_init_phase=True` during init → patches skip KV cache allocation
- `_shadow_init_phase=False` after wake → patches pass through normally
- PIECEWISE graph mode → attention ops excluded from captured graphs
- CUDA graphs remain valid after KV cache allocation (they don't contain attention)
- VA-stable GMS unmap/remap → tensor pointers survive sleep/wake cycle
- `self.kv_caches` transitions: `{}` (init) → `{}` (sleep) → `{populated}` (wake)

## Multinode Differences (Known Issues)

In multinode MP mode, each rank is in a separate process tree:

1. **SHADOW_SKIP_KV_CACHE** must be passed explicitly to headless worker env (Fix 2)
2. **worker_cls** must be set in `run_dynamo_headless()` (Fix 1)
3. **cudagraph_mode** override happens only on the leader — headless worker must
   receive matching config
4. **CUDAGraphMode escalation**: vLLM's dispatcher may escalate PIECEWISE to
   FULL_AND_PIECEWISE based on hardware. FULL mode calls `_build_attention_metadata`
   which asserts `slot_mappings is not None`. The shadow patch returns None for empty
   kv_caches. This assertion doesn't fire in single-node because the dispatcher
   may behave differently for local vs remote workers, OR because single-node TP uses
   a different rank configuration that doesn't trigger FULL mode escalation.
