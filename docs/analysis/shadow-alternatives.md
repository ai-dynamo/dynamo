# Alternative Approaches to Shadow Engine KV Cache

## The Problem

Two engines share the same GPU(s) via GMS. The active engine needs weights + KV cache.
The shadow engine needs to be a warm standby that can take over in sub-second time.
There isn't enough GPU memory for both engines to have KV cache simultaneously.

## Current Approach: Skip KV Cache at Init (5 patches)

```
Shadow init: load weights → skip KV cache → capture CUDA graphs (PIECEWISE) → sleep
Shadow wake: remap weights → allocate KV cache → serve
```

**Patches required:**
1. `patch_initialize_kv_cache_tensors` — return `{}` during shadow init
2. `patch_get_slot_mappings` — return `(None, None)` for empty kv_caches
3. `patch_request_memory` — bypass memory check
4. `patch_register_kv_caches` — skip NIXL registration when empty
5. `patch_allocate_kv_cache_on_wake` — add method to allocate on wake
6. (multinode) `patch_cudagraph_mode_escalation` — prevent FULL mode captures

**Brittleness concerns:**
- Patches monkey-patch vLLM internals that change across versions
- The slot_mappings patch returned `(None, None)` which hit a new assertion in v0.16
- The cudagraph mode escalation was invisible until multinode testing
- Each new vLLM version may add new codepaths that assume KV cache exists during init

---

## Option A: Full Init → Immediate Sleep

**Concept:** Let vLLM do its normal full initialization (including KV cache allocation
and CUDA graph capture), then immediately call `sleep(level=1)` to free memory.

```
Shadow init: load weights → allocate KV cache → capture CUDA graphs → sleep (free all)
Shadow wake: remap weights → remap KV cache → serve
```

**The memory pressure problem:**
During init, both engines temporarily have KV cache on GPU. For a model that uses
40 GiB KV cache per engine, the GPU needs ~80 GiB KV cache + weights simultaneously.
This is impossible on most hardware.

**Solving memory pressure — staggered init:**
1. Active engine starts first, loads weights, allocates KV cache, serves
2. Active engine calls `sleep(level=1, tags=["kv_cache"])` temporarily — frees KV cache
3. Shadow engine does full init (weights shared via GMS, KV cache allocated)
4. Shadow engine calls `sleep()` — frees its KV cache
5. Active engine calls `wake_up(tags=["kv_cache"])` — reclaims its KV cache

This requires **coordination** between active and shadow but eliminates ALL patches.
vLLM's existing sleep/wake handles everything natively.

**Pros:**
- Zero monkey-patches on vLLM internals
- CUDA graphs captured with real KV cache (all modes work, including FULL)
- Version-resilient — uses only vLLM's public sleep/wake API
- Multinode works automatically (collective_rpc propagates sleep/wake)

**Cons:**
- Requires active engine downtime during shadow init (~2-5 seconds for graph capture)
- Coordination complexity — active engine must temporarily free KV cache
- During the handoff window, no engine has KV cache (no inference possible)
- KV cache addresses change after sleep/wake — FULL mode CUDA graphs become invalid
  (but PIECEWISE graphs survive since attention isn't captured)

**Feasibility:** HIGH — no vLLM patches needed, but needs orchestration logic.

---

## Option B: Profile-Only Init (Deferred Allocation)

**Concept:** Run vLLM's init up to the profiling phase, calculate KV cache size,
but don't allocate. Skip CUDA graph capture. On wake, do the remaining init steps.

```
Shadow init: load weights → profile memory → calculate KV blocks → STOP (no allocation, no graphs)
Shadow wake: allocate KV cache → capture CUDA graphs → serve
```

**What you get at "STOP":**
- Model weights loaded and shared via GMS
- Memory profiled — know exactly how many KV blocks to allocate
- Scheduler configured with block count
- Attention backend NOT initialized (needs KV cache config for some backends)
- CUDA graphs NOT captured

**Wake time would be slower:**
- KV cache allocation: ~100ms
- Attention backend init: ~50ms
- CUDA graph capture: ~3-5 seconds (51 PIECEWISE captures)
- Total: ~3-5 seconds vs current ~170ms

**Pros:**
- Only 1-2 patches needed (skip initialize_from_config and compile_or_warm_up_model)
- Simpler than current approach — fewer vLLM internals to understand
- No slot_mappings or cudagraph mode issues

**Cons:**
- 3-5 second wake time (CUDA graph capture dominates)
- Still requires some patching to stop init midway
- Attention backend may have implicit dependencies on KV cache config
- Scheduler may not work correctly without KV cache metadata

**Feasibility:** MEDIUM — simpler patches but slower wake time.

---

## Option C: Allocate KV Cache, Capture Graphs, Then Free

**Concept:** Like Option A but instead of sleeping the active engine, accept that the
shadow engine temporarily over-commits GPU memory. Rely on the CUDA VMM (virtual
memory management) system to handle the over-commit gracefully — allocate virtual
addresses for KV cache, capture graphs referencing those VAs, then unmap the physical
memory. On wake, remap to the SAME VAs.

```
Shadow init: load weights → allocate KV cache (VMM, may over-commit) → capture graphs → unmap KV physical memory (keep VAs)
Shadow wake: map physical memory to same VAs → serve
```

**Why this might work:**
GMS already uses CUDA VMM (`cuMemCreate` / `cuMemMap` / `cuMemUnmap`). The CuMemAllocator
tracks VA reservations separately from physical mappings. During shadow init:

1. `cuMemAddressReserve` — reserve virtual address range for KV cache
2. `cuMemCreate` + `cuMemMap` — map physical memory temporarily
3. Capture CUDA graphs (they reference VAs, not physical addresses)
4. `cuMemUnmap` + `cuMemRelease` — free physical memory, keep VA reservation
5. On wake: `cuMemCreate` + `cuMemMap` at the SAME VAs — graphs still valid

The over-commit during step 2-3 is temporary (seconds). If the active engine's
KV cache is using different VAs, both can coexist briefly. The physical memory
pressure is real but short-lived.

**This is essentially what `enable_sleep_mode` already does:**
- KV cache allocated within `CuMemAllocator.use_memory_pool(tag="kv_cache")`
- `sleep()` calls `cuMemUnmap` + `cuMemRelease` (keeps VA reservations)
- `wake_up()` calls `cuMemCreate` + `cuMemMap` at original VAs

So `enable_sleep_mode` + normal init + immediate sleep is actually Option A with
VA-stable addresses. The difference from vanilla Option A: if we use CuMemAllocator
for both engines' KV caches, the VAs are reserved per-engine and don't conflict.

**The real question is: can two CuMemAllocators coexist on the same GPU?**

CuMemAllocator is a singleton per process. But with GMS, each engine is a separate
process. Each process has its own CuMemAllocator managing its own VA space.
The CUDA VMM VA space is per-process (each process has its own virtual address space).
Physical GPU memory is shared. So:

- Engine A process: reserves VAs [0x1000-0x5000] for KV cache, maps physical pages
- Engine B process: reserves VAs [0x1000-0x5000] for KV cache (different VA space!), maps different physical pages
- Both have physical pages mapped simultaneously — this is the over-commit

If total physical usage exceeds GPU memory, `cuMemCreate` fails. But if Engine B
immediately unmaps after graph capture, the physical pages are released.

**Pros:**
- CUDA graphs captured with real KV cache at stable VAs
- All graph modes work (PIECEWISE and FULL)
- Wake is instant (~100ms — just `cuMemMap`, no graph recapture)
- Uses existing vLLM sleep/wake infrastructure
- VA stability means graphs survive sleep/wake cycle

**Cons:**
- Temporary physical memory over-commit during shadow init
- `cuMemCreate` may fail if GPU memory is fully consumed by active engine
- Need to ensure active engine leaves enough headroom for shadow's temporary alloc
- Requires `enable_sleep_mode=True` (uses CuMemAllocator, not PyTorch default)

**Feasibility:** HIGH if there's memory headroom; LOW if GPU is fully committed.

---

## Option D: Pre-allocated Shared KV Cache Pool

**Concept:** Instead of each engine managing its own KV cache, create a shared KV cache
pool managed by GMS (like weights). The active engine maps it, the shadow doesn't.
On failover, the shadow maps the same physical KV cache memory.

```
GMS manages: weights (shared RO) + KV cache pool (exclusive, transferred on failover)
Active: maps weights RO + maps KV cache RW
Shadow: maps weights RO + does NOT map KV cache
Failover: shadow maps KV cache (inherits active's state)
```

**The appeal:** The shadow could inherit the active engine's in-flight KV cache state,
enabling truly seamless failover where pending requests continue from where they left off.

**The reality:**
- vLLM's KV cache layout is tightly coupled to its block manager
- Block table metadata (which blocks belong to which sequences) lives in the scheduler
- The scheduler state would also need to be transferred
- KV cache tensor shapes depend on model config and num_blocks — both engines must agree
- CUDA graph addresses would reference the shared KV cache VAs

**Pros:**
- Zero memory overhead for shadow (no KV cache duplication)
- Potentially seamless failover (inherit in-flight state)
- No CUDA graph issues (same VA space for KV cache)

**Cons:**
- Major GMS changes (new resource type beyond weights)
- vLLM scheduler state transfer is complex
- Block manager synchronization across engines
- Fundamentally different architecture from current GMS design

**Feasibility:** LOW for near-term. Interesting long-term direction.

---

## Option E: Leverage vLLM's Native Sleep/Wake Directly (Simplest)

**Concept:** Don't skip KV cache at init at all. Instead, use a completely different
shadow lifecycle:

```
1. Shadow engine starts with --enable-sleep-mode (normal full init)
2. Shadow engine fully initializes (weights, KV cache, CUDA graphs)
3. Shadow engine immediately calls sleep(level=1):
   - Weights: offloaded to CPU (cudaMemcpy, backed up)
   - KV cache: discarded (GPU memory freed)
4. Shadow sits in standby (minimal GPU footprint: ~1-2 GiB)
5. On failover:
   - wake_up(): weights restored from CPU backup, KV cache re-allocated
   - CUDA graphs: PIECEWISE graphs survive (no attention addresses)
   - FULL graphs: need recapture (addresses changed) — or just use PIECEWISE
```

**The key insight:** During step 2, the shadow engine allocates KV cache on GPU.
This temporarily uses GPU memory. But if the active engine hasn't started yet
(sequential startup), there's no conflict. If both are running simultaneously,
the shadow can reduce its gpu_memory_utilization to allocate a smaller KV cache
(just enough for graph capture, not for real inference).

**For the failover scenario:**
- Active engine starts, serves inference
- Shadow engine starts later, does full init including KV cache
  - Active engine is already serving, using most GPU memory
  - Shadow uses `--gpu-memory-utilization 0.1` (just enough for tiny KV cache)
  - Shadow captures PIECEWISE graphs with this tiny KV cache
  - Shadow sleeps: frees both weights (to CPU) and KV cache
- On failover:
  - Shadow wakes: weights from CPU → GPU, KV cache allocated at FULL size
  - The `determine_available_memory()` during wake would need to re-profile
    or use a stored value for the full KV cache size

**Pros:**
- ZERO patches on vLLM internals
- Uses only public API: `--enable-sleep-mode`, `sleep()`, `wake_up()`
- Version-resilient
- Simple to understand

**Cons:**
- Wake time includes weight restore from CPU (~500ms-1s for large models)
- KV cache size at init != KV cache size at wake (need to handle this)
- Tiny KV cache during init means CUDA graphs captured with wrong block count
  - PIECEWISE graphs OK (attention not captured)
  - FULL graphs captured with wrong sizes — need recapture or avoid FULL mode
- Shadow needs GPU memory during init (reduced but nonzero)

**Feasibility:** HIGH — very simple, but wake time is slower.

---

## Comparison Summary

| Approach | Patches | Wake Time | Memory During Init | Graph Modes | Complexity |
|----------|---------|-----------|-------------------|-------------|------------|
| **Current (skip KV)** | 5-6 | ~170ms | Weights only | PIECEWISE only | HIGH (brittle) |
| **A: Full init + sleep** | 0 | ~170ms | Temporary: both KV | All (if VA-stable) | MEDIUM |
| **B: Profile-only** | 1-2 | ~3-5s | Weights only | None at init | LOW-MEDIUM |
| **C: Allocate + unmap** | 0 | ~100ms | Temporary: both KV | All (VA-stable) | MEDIUM |
| **D: Shared KV pool** | 0 | ~0ms | None extra | All | VERY HIGH |
| **E: Native sleep/wake** | 0 | ~500ms-1s | Small KV + weights | PIECEWISE | LOW |

## Recommendation

**For near-term (replacing current patches):** Option E is the safest. Zero vLLM
patches, uses only public API. The slower wake time (~500ms-1s vs 170ms) is
acceptable for most failover SLOs.

**For best performance:** Option C (allocate + unmap via CuMemAllocator) gives the
fastest wake time with no patches, but requires memory headroom during init.

**For production reliability:** Option A (staggered init with active engine
cooperation) is the most robust — no patches, no memory pressure, works with
all graph modes. But requires orchestration.

**Long-term:** Option D (shared KV pool) enables true seamless failover but
requires significant GMS and vLLM architectural changes.
