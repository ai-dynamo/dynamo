# GMS Shadow Mode Design

## Decision 1: API

**Chosen:** `--gms-mode shadow`

```bash
# Shadow engine (skips KV cache, starts sleeping)
python3 -m dynamo.vllm \
    --model Qwen/Qwen3-14B \
    --load-format gms \
    --gms-mode shadow
```

**Why not independent `--start-sleeping`?**
Shadow mode requires GMS for VA-stable weights. Without GMS, wake is slow (weight reload). Making it a GMS sub-option reflects this dependency.

---

## Behavior

When `--gms-mode shadow`:
1. Skip KV cache allocation at startup
2. Force PIECEWISE CUDA graph mode
3. Go straight to sleep after init
4. Allocate KV cache on wake

---

## Decision 2: Memory Check Handling

**Problem:** vLLM's `request_memory()` checks `free_memory >= requested_memory` at startup. This fails for shadow engines because the primary engine is consuming GPU memory.

**Chosen:** Patch `request_memory` in `patches.py`

- Patch is always applied (consistent with existing patches)
- Check `SHADOW_SKIP_KV_CACHE=1` at runtime inside the patch
- If shadow mode: skip check, log what's happening, return `requested_memory`
- If normal mode: delegate to original `request_memory`

**Why patch instead of env var in vLLM?**
We can't modify vLLM. The existing `patches.py` pattern (used for `empty_cache` and `MemorySnapshot`) lets us inject GMS-specific behavior cleanly.

**Why check inside patch (not conditional patching)?**
Matches existing pattern. More robust—env var can be set anytime before `request_memory()` is called, not just before module import.

---

## Decision 3: KV Transfer Registration

**Problem:** vLLM's `initialize_kv_cache()` calls `register_kv_caches(kv_caches)` to register KV cache memory with NIXL for RDMA transfers. When shadow engines skip KV cache allocation, `kv_caches` is empty, and calling `register_kv_caches({})` causes errors (uninitialized state variables like `num_blocks`).

**Context:**
- Registration happens at startup, actual transfers happen at inference time
- `register_kv_caches()` sets up memory regions for RDMA
- `set_host_xfer_buffer_ops()` sets up D2H/H2D copy operations (also in same block)
- Both are guarded by `if has_kv_transfer_group() and kv_caches:`

**Chosen:** Patch `register_kv_caches` to no-op when in shadow mode

```python
# Conceptual patch in patches.py
_original_register = NixlConnector.register_kv_caches

def patched_register_kv_caches(self, kv_caches):
    if os.environ.get("SHADOW_SKIP_KV_CACHE") == "1":
        logger.info("[Shadow] Skipping KV cache registration (no caches allocated)")
        return
    return _original_register(self, kv_caches)
```

**Why no-op patch vs patching `has_kv_transfer_group()`?**
- More targeted: only affects registration, not all KV transfer checks
- Preserves connector initialization for potential future use on wake
- Cleaner: shadow can later register caches in `allocate_kv_cache_on_wake()`

**Why not `--connector none`?**
- Would completely disable KV transfer infrastructure
- Shadow might need to participate in P/D after waking (future consideration)
- Patch is more surgical—just skips registration at startup

---

## Decision 4: Auto-Sleep Initialization

**Problem:** Shadow engines should initialize into sleep mode automatically, releasing GPU memory immediately after startup while remaining ready to wake.

**Chosen:** Conditional logic in `init()` (Option A)

```python
# In init() after handler creation, before register_vllm_model()
async def init(runtime: DistributedRuntime, config: Config):
    # ... existing setup (setup_vllm_engine, create handler) ...

    # Register sleep/wake routes (always - shadow needs wake_up)
    runtime.register_engine_route("sleep", handler.sleep)
    runtime.register_engine_route("wake_up", handler.wake_up)

    # Shadow mode: auto-sleep and skip discovery registration
    if config.gms_mode == "shadow":
        logger.info("[Shadow] Auto-sleeping engine after initialization")
        await engine_client.sleep(level=1)
        # Skip register_vllm_model() - shadow doesn't advertise until woken
        # Still serve endpoints so wake_up route works
        await _serve_shadow_endpoints(...)
        return

    # Normal mode continues...
    await register_vllm_model(...)
```

**Key behaviors:**
- Sleep happens AFTER engine is fully initialized (weights loaded, model ready)
- Skip discovery registration (shadow doesn't serve until woken)
- Keep serving endpoints so `/engine/wake_up` route remains accessible
- Wake is triggered externally via HTTP call to `/engine/wake_up`

**Why not separate `init_shadow()` function?**
- Would duplicate most of `init()` logic
- Conditional in `init()` is simpler and easier to maintain

**Why not hook in GMSWorker?**
- Worker doesn't control discovery registration
- Sleep needs to happen at Dynamo layer, not vLLM layer

---

## Implementation

- [ ] Add `--gms-mode` to `args.py`
- [ ] Set `SHADOW_SKIP_KV_CACHE=1` env var when `--gms-mode shadow`
- [ ] Force PIECEWISE CUDA graph mode
- [ ] Add shadow mode conditional in `init()` for auto-sleep
- [ ] Add `patch_request_memory()` to `patches.py`
- [ ] Add `patch_register_kv_caches()` to `patches.py`
- [ ] Apply patches in `worker.py`
