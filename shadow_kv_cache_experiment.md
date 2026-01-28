# Shadow Engine KV Cache Experiment

## Goal
Enable shadow engines to start with minimal GPU memory footprint during warmup/CUDA graph capture, then expanding to full size on wake.

---

## Status: ✅ WORKING

**Tested:** 2026-01-28
**Model:** Qwen/Qwen3-0.6B (also tested concepts with Qwen3-14B)

### Test Results

| Metric | Value |
|--------|-------|
| **Failover time** | 382 ms |
| **Wake time** | 357 ms |
| KV cache allocated on wake | 40.04 GiB (28 tensors) |
| Shadow sleeping footprint | ~3.4 GiB (weights only) |
| Full engine footprint | ~44.5 GiB (weights + KV cache) |

### Memory Timeline (Failover Test)

| Stage | GPU Memory | Description |
|-------|-----------|-------------|
| Primary ready | ~44.5 GiB | Primary with full KV cache |
| Primary + Shadow sleeping | 46.9 GiB | Both engines, shadow has no KV |
| After failover | 44.6 GiB | Shadow with KV cache, primary gone |

---

## Current Implementation

**Based on:** dynamo main branch
**vLLM Version:** 0.14.0

### Environment Variables

| Env Var | Purpose |
|---------|---------|
| `SHADOW_SKIP_KV_CACHE=1` | Skip KV cache allocation at startup |
| `SHADOW_SKIP_MEMORY_CHECK=1` | Bypass memory check for shadow engines |

### Architecture

**dynamo changes (`lib/gpu_memory_service/vllm_integration/worker.py`):**
- `GMSWorker.sleep()` - Handles both normal and shadow mode KV cache
- `GMSWorker.wake_up()` - Detects empty `kv_caches` and calls `allocate_kv_cache_on_wake()`

**vLLM changes (`vllm/v1/worker/gpu_model_runner.py`):**
- `initialize_kv_cache()` - Checks `SHADOW_SKIP_KV_CACHE` env var, stores `_shadow_kernel_block_sizes`
- `allocate_kv_cache_on_wake()` - Allocates KV cache on wake using stored config

**vLLM changes (`vllm/v1/worker/utils.py`):**
- `request_memory()` - Checks `SHADOW_SKIP_MEMORY_CHECK` env var

---

## Why PIECEWISE Mode is Required

| CUDA Graph Mode | Attention During Capture | KV Cache Required |
|-----------------|-------------------------|-------------------|
| **FULL** | Actually executes | ✅ Yes |
| **PIECEWISE** | Stubbed/skipped | ❌ No |

In **FULL** mode, during graph capture:
```python
# flash_attn.py line 625
key_cache, value_cache = kv_cache.unbind(0)  # 💥 Fails if no KV cache!
```

In **PIECEWISE** mode:
- Attention ops are "splitting ops" excluded from the graph
- `attn_metadata = None` during capture
- Attention forward is stubbed, KV cache never accessed

---

## Test Scripts

Two automated test scripts are provided:

### Sleep/Wake Test
```bash
# Test single shadow engine sleep/wake cycle with inference
./test_shadow_sleep_wake.sh [MODEL_NAME]

# Example:
./test_shadow_sleep_wake.sh Qwen/Qwen3-0.6B
```

### Failover Test
```bash
# Test full failover: primary dies, shadow takes over
./test_shadow_failover.sh [MODEL_NAME]

# Example:
./test_shadow_failover.sh Qwen/Qwen3-0.6B
```

---

## Manual Test Procedure

### Prerequisites
1. tmux session with multiple panes
2. Environment: dynamo venv activated
3. Branches checked out:
   - dynamo: `gms-shadow-experiment`
   - vLLM: `gms-shadow-experiment` (with shadow patches applied)

### Commands

#### Terminal 1: GPU Memory Service
```bash
cd /home/mabdulwahhab/repos/dynamo-7
source .venv/bin/activate
source .env
python3 -m gpu_memory_service --device 0
```

#### Terminal 2: Shadow Engine
```bash
cd /home/mabdulwahhab/repos/dynamo-7
source .venv/bin/activate
source .env

# Start shadow engine with skip KV cache allocation
SHADOW_SKIP_KV_CACHE=1 SHADOW_SKIP_MEMORY_CHECK=1 \
  DYN_SYSTEM_PORT=8100 python3 -m dynamo.vllm \
  --model Qwen/Qwen3-0.6B -tp 1 \
  --load-format gms \
  --enable-sleep-mode \
  --compilation-config '{"cudagraph_mode": "PIECEWISE"}'
```

#### Terminal 3: GPU Monitor
```bash
watch -n 1 nvidia-smi
```

### Success Criteria

| Checkpoint | What to Look For |
|------------|------------------|
| GMS started | `GPU Memory Service running on device 0` |
| Weights loaded | `[GMS] Read mode: imported X.XX GiB` |
| KV cache skipped | `[Shadow] Skipping KV cache allocation (SHADOW_SKIP_KV_CACHE=1)` |
| CUDA graphs captured | `Capturing CUDA graphs (mixed prefill-decode, PIECEWISE)` |
| **NO** FULL graphs | Should NOT see `Capturing CUDA graphs (decode, FULL)` |
| Engine ready | Engine startup complete |
| Memory usage | GPU memory ~3.4 GiB (vs ~44 GiB with full KV cache) |

### Wake Test

```bash
# Call wake on shadow engine
curl -X POST http://localhost:8100/engine/wake_up

# Expected log output:
# [GMS] KV cache not allocated - allocating on wake
# Allocating KV cache on wake (was skipped during init)
# Allocated KV cache on wake: 40.04 GiB (28 tensors)
# [GMS] Successfully allocated KV cache on wake
```

### Failure Indicators

| Error | Cause | Fix |
|-------|-------|-----|
| `ValueError: not enough values to unpack` | FULL CUDA graph mode | Add `--compilation-config '{"cudagraph_mode": "PIECEWISE"}'` |
| `Free memory on device ... is less than desired` | Memory check failed | Add `SHADOW_SKIP_MEMORY_CHECK=1` |
| `allocate_kv_cache_on_wake not available` | vLLM branch not applied | Checkout `gms-shadow-experiment` and reinstall vLLM |
| `AssertionError: Current vLLM config is not set` | Missing stored config | Ensure `_shadow_kernel_block_sizes` saved in init |

---

## Git References

### dynamo Repository: `/home/mabdulwahhab/repos/dynamo-7`

| Branch | Base | Description |
|--------|------|-------------|
| `gms-shadow-experiment` | `main` | Shadow KV cache with GMS integration |

### vLLM Repository: `/home/mabdulwahhab/repos/vllm`

| Branch | Base | Description |
|--------|------|-------------|
| `gms-shadow-experiment` | `v0.14.0` | Skip KV cache via env vars |

---

## Files Modified

**dynamo:**
- `lib/gpu_memory_service/vllm_integration/worker.py` - `GMSWorker` sleep/wake for shadow mode
- `test_shadow_sleep_wake.sh` - Automated sleep/wake test
- `test_shadow_failover.sh` - Automated failover test
- `shadow_kv_cache_experiment.md` - This documentation

**vLLM:**
- `vllm/v1/worker/gpu_model_runner.py` - `SHADOW_SKIP_KV_CACHE` + `allocate_kv_cache_on_wake()`
- `vllm/v1/worker/utils.py` - `SHADOW_SKIP_MEMORY_CHECK`

---

## Implementation Details

### Key Design Decision: Storing Config at Init

The `allocate_kv_cache_on_wake()` method needs `kernel_block_sizes` and `kv_cache_config` to allocate KV cache. Initially we tried recomputing these on wake, but this required the `vllm_config` context which isn't available during the GMS worker's `wake_up()` call.

**Solution:** Store the necessary config during `initialize_kv_cache()`:
```python
# In initialize_kv_cache():
if skip_kv_cache:
    self._shadow_kernel_block_sizes = kernel_block_sizes  # Save for later
    kv_caches = {}
```

Then retrieve on wake:
```python
# In allocate_kv_cache_on_wake():
assert hasattr(self, '_shadow_kernel_block_sizes'), "Must be set in initialize_kv_cache"
kernel_block_sizes = self._shadow_kernel_block_sizes
```

### Editable vLLM Install
```bash
cd /home/mabdulwahhab/repos/dynamo-7
source .venv/bin/activate
VLLM_USE_PRECOMPILED=1 pip install -e /home/mabdulwahhab/repos/vllm --no-build-isolation
```
