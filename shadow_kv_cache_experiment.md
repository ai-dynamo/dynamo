# Shadow Engine KV Cache Experiment

## Goal
Enable shadow engines to start with minimal GPU memory footprint by skipping KV cache allocation, then expanding to full capacity on wake.

---

## Status: ✅ IMPLEMENTED

**Implementation Date:** 2026-01-28
**Design Document:** `docs/design/gms-shadow-mode.md`

---

## API

```bash
# Shadow engine (skips KV cache, auto-sleeps after init)
python3 -m dynamo.vllm \
    --model Qwen/Qwen3-14B \
    --load-format gms \
    --gms-mode shadow
```

**Why `--gms-mode shadow` instead of separate flags?**
Shadow mode requires GMS for VA-stable weights. Without GMS, wake would be slow (weight reload). Making it a GMS sub-option reflects this dependency.

---

## Behavior

When `--gms-mode shadow`:
1. Skip KV cache allocation at startup (via patches)
2. Force PIECEWISE CUDA graph mode (attention stubbed during capture)
3. Register with discovery, then immediately auto-sleep (unregisters from routing pool)
4. Wait for external `wake_up` call on `DYN_SYSTEM_PORT`
5. On wake: allocate KV cache, re-register with discovery, serve requests

---

## Test Results

| Metric | Value |
|--------|-------|
| **Failover time** | ~380 ms |
| **Wake time** | ~360 ms |
| KV cache allocated on wake | 40.04 GiB (28 tensors for Qwen3-0.6B) |
| Shadow sleeping footprint | ~3.4 GiB (weights only) |
| Full engine footprint | ~44.5 GiB (weights + KV cache) |

### Memory Timeline (Failover Test)

| Stage | GPU Memory | Description |
|-------|-----------|-------------|
| Primary ready | ~44.5 GiB | Primary with full KV cache |
| Primary + Shadow sleeping | ~46.9 GiB | Both engines, shadow has no KV |
| After failover | ~44.6 GiB | Shadow with KV cache, primary gone |

---

## Implementation

### Files Modified

**dynamo:**
- `components/src/dynamo/vllm/args.py` - Add `--gms-mode` argument
- `components/src/dynamo/vllm/main.py` - Shadow mode setup and auto-sleep
- `lib/gpu_memory_service/vllm_integration/patches.py` - Shadow mode patches
- `lib/gpu_memory_service/vllm_integration/worker.py` - Apply patches

### Patches (in `patches.py`)

| Patch | Target | Purpose |
|-------|--------|---------|
| `patch_request_memory()` | `vllm.v1.worker.utils.request_memory` | Bypass memory check in shadow mode |
| `patch_register_kv_caches()` | `NixlConnector.register_kv_caches` | Skip NIXL registration when no KV cache |
| `patch_initialize_kv_cache()` | `GPUModelRunner.initialize_kv_cache` | Skip KV allocation, store config for wake |
| `patch_allocate_kv_cache_on_wake()` | Adds method to `GPUModelRunner` | Allocate KV cache on wake |

### Patch Details

**1. `patch_request_memory()`**
- **Problem:** vLLM checks `free_memory >= requested_memory` at startup, which fails for shadow engines because the primary is using GPU memory.
- **Solution:** When `SHADOW_SKIP_KV_CACHE=1`, bypass the check entirely.

**2. `patch_register_kv_caches()`**
- **Problem:** vLLM registers KV cache with NIXL for RDMA transfers, but calling `register_kv_caches({})` with empty caches causes errors.
- **Solution:** No-op in shadow mode; registration happens in `allocate_kv_cache_on_wake()`.

**3. `patch_initialize_kv_cache()`**
- **Problem:** Need to skip KV cache allocation but preserve config for later.
- **Solution:** Store `kv_cache_config` and `kernel_block_sizes` on self, set `kv_caches = {}`.

**4. `patch_allocate_kv_cache_on_wake()`**
- **Problem:** Need to allocate KV cache dynamically when engine wakes.
- **Solution:** Use stored config to call `initialize_kv_cache_tensors()` and register with KV transfer group.

---

## Why PIECEWISE Mode is Required

| CUDA Graph Mode | Attention During Capture | KV Cache Required |
|-----------------|-------------------------|-------------------|
| **FULL** | Actually executes | ✅ Yes |
| **PIECEWISE** | Stubbed/skipped | ❌ No |

In **FULL** mode, during graph capture:
```python
# flash_attn.py
key_cache, value_cache = kv_cache.unbind(0)  # 💥 Fails if no KV cache!
```

In **PIECEWISE** mode:
- Attention ops are "splitting ops" excluded from the graph
- `attn_metadata = None` during capture
- Attention forward is stubbed, KV cache never accessed

---

## Test Scripts

### Sleep/Wake Test
```bash
# Test shadow engine wake and inference
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
1. Environment: dynamo venv activated
2. GMS, etcd, and NATS running

### Commands

#### Terminal 1: GPU Memory Service
```bash
cd /path/to/dynamo
source .venv/bin/activate
source .env
python3 -m gpu_memory_service --device 0
```

#### Terminal 2: Shadow Engine
```bash
cd /path/to/dynamo
source .venv/bin/activate
source .env

DYN_SYSTEM_PORT=8100 python3 -m dynamo.vllm \
    --model Qwen/Qwen3-0.6B \
    --load-format gms \
    --gms-mode shadow
```

#### Terminal 3: Wake and Test
```bash
# Wake the shadow engine
curl -X POST http://localhost:8100/engine/wake_up

# Test inference (after starting frontend)
curl http://localhost:8000/v1/completions \
    -H "Content-Type: application/json" \
    -d '{"model": "Qwen/Qwen3-0.6B", "prompt": "Hello", "max_tokens": 10}'
```

### Success Criteria

| Checkpoint | What to Look For |
|------------|------------------|
| GMS started | `waiting for connections` |
| KV cache skipped | `[Shadow] Skipping KV cache allocation` |
| Auto-sleep | `[Shadow] Engine is now sleeping` |
| PIECEWISE mode | `Capturing CUDA graphs ... PIECEWISE` |
| Wake successful | `Allocated KV cache on wake: X.XX GiB` |
| Re-registered | `Re-registered endpoint to discovery` |

### Error Indicators

| Error | Cause | Fix |
|-------|-------|-----|
| `--gms-mode shadow requires --load-format gms` | Missing GMS | Add `--load-format gms` |
| `ValueError: not enough values to unpack` | FULL CUDA graph | Shadow mode forces PIECEWISE automatically |
| Memory check failure | Shouldn't happen | Patches bypass memory check in shadow mode |
| `allocate_kv_cache_on_wake` not found | Patches not applied | Ensure GMSWorker is being used |

---

## Git References

### dynamo Repository

| Branch | Base | Description |
|--------|------|-------------|
| `gms-shadow-experiment` | `main` | Shadow mode with `--gms-mode shadow` API |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         SHADOW MODE FLOW                             │
└─────────────────────────────────────────────────────────────────────┘

1. Engine starts with --gms-mode shadow
   ├── SHADOW_SKIP_KV_CACHE=1 set automatically
   ├── PIECEWISE CUDA graph mode forced
   └── Patches check env var at runtime

2. Model initialization
   ├── Weights loaded via GMS (shared with primary)
   ├── patch_initialize_kv_cache() skips KV allocation
   ├── Stores kv_cache_config and kernel_block_sizes
   └── CUDA graphs captured (PIECEWISE, no KV needed)

3. Registration and auto-sleep
   ├── Engine registers with discovery
   ├── handler.sleep() called automatically
   └── Engine unregistered from routing pool

4. Waiting state
   ├── Engine serves endpoints (wake_up accessible)
   ├── Not in routing pool (won't receive requests)
   └── Minimal memory footprint (no KV cache)

5. Wake (triggered externally)
   ├── GMSWorker.wake_up() called
   ├── Weights remapped (VA-stable)
   ├── allocate_kv_cache_on_wake() allocates KV cache
   ├── Register with KV transfer group
   └── Re-register with discovery

6. Active state
   ├── Full memory footprint (weights + KV cache)
   ├── In routing pool (receives requests)
   └── Normal inference operation
```
