# GMS warmup memory burst — analysis and theory

*Context: single-node TP8 Kimi-K2.6-NVFP4 on B200 (183 GiB/GPU), eager mode. Comparing a vanilla vLLM engine against a GMS (GPU Memory Service) engine that shares weights out-of-process. All numbers are per-GPU, rank 0, from verified-clean baselines with 1s device sampling + per-phase instrumentation.*

## Symptom

For **identical work**, the GMS engine uses materially more GPU memory during bring-up than vanilla:

| | vanilla | GMS |
|---|---|---|
| pre-warmup baseline (weights + KV) | 143.1 | 141.8 |
| warmup peak | **149.4** | **178.3** |
| steady (serving) | 149.4 | 167.2 |

Normalized to the baseline, vanilla's warmup is a **+6 GiB** blip; GMS's is a **+36 GiB** spike. At peak, vanilla leaves 34 GiB of headroom to the 183 ceiling; GMS leaves 5. That gap is what blocks running a second engine (e.g. a failover shadow) on the same GPUs.

---

## Background 1 — how PyTorch's CUDA caching allocator works

PyTorch never calls the driver (`cudaMalloc`/`cudaFree`) per tensor. It runs a caching allocator:

- It grabs large **segments** from the driver and carves **blocks** for tensors. A freed block returns to the cache (still inside its segment); adjacent free blocks **coalesce**.
- A new request first searches the cache for a fitting free block → reuse, no driver call. If none fits → `cudaMalloc` a new segment.
- A big free block can be **split** to serve a small request; the leftover is a free remainder. The total of those remainders is **`inactive_split_bytes`** — free memory *trapped* as awkward fragments. High value = classic fragmentation.
- If `cudaMalloc` itself **fails** (device near-full), the allocator dumps cached free segments back to the driver and retries — incrementing **`num_alloc_retries`**. A non-zero retry count is a direct signal of real memory pressure near the ceiling.

Two counters: **`allocated`** = bytes in live tensors; **`reserved`** = bytes in all segments (live + cached-free). Their difference is cached-free memory the allocator holds.

## Background 2 — how GMS changes the layout

GMS routes the **weights** through a separate **custom MemPool** (backed by the GPU Memory Service's cross-process `cuMem` memory), distinct from PyTorch's **default pool**. The two pools are **separate arenas** — a free block in one cannot satisfy a request in the other. So under GMS:

- weights (~73 GiB) live in the GMS pool,
- KV (~64) and **all activation / autotune workspaces** live in the default pool.

There is also a **strand**: during weight load, scratch went through the GMS pool, and `prune` freed its *physical* memory but didn't notify PyTorch — so ~43 GiB sits in PyTorch's `reserved` bookkeeping as non-physical GMS-pool cache. (This is a *separate* GMS cost from the warmup burst.)

## Background 3 — what FlashInfer autotune does

Attention has many candidate kernel implementations ("tactics") with different scratch needs. With autotune enabled, FlashInfer runs a full forward at the max batch (8192 tokens) and benchmarks tactic after tactic, each grabbing a **workspace of its own size**, using it, freeing it, moving on — a burst of many varied-size allocate→free cycles in the default pool. This is exactly the pattern that stresses a caching allocator.

---

## Finding — the burst is one sub-step: `flashinfer_autotune`

Splitting warmup into its sub-phases pins the entire GMS burst to that single autotune forward. Everything else (load, profile, KV alloc, DeepGEMM warmup — a no-op here, sampler warmup) behaves identically in both setups.

| during `flashinfer_autotune` (8192) | vanilla | GMS |
|---|---|---|
| device physical Δ | +5.1 | **+25.4** |
| live tensors (`allocated`) Δ | ~0 (peak +3) | ~0 (peak +3) |
| reserved Δ / peak | +5 / 145 | **+25 / 217** |
| **new segments allocated** | **+14** | **+48** |
| **`num_alloc_retries`** | **0** | **15** |
| `inactive_split_bytes` | ~0.3 | ~0.3 |

Reading it: in **both**, the autotune's *live* working set is tiny (~+3 GiB peak — same computation). The difference is what the allocator does underneath. Vanilla serves the tactics by **reusing** cache — 14 segments, +5 GiB, zero `cudaMalloc` failures. GMS **cannot reuse** — it allocates **48 fresh segments / +25 GiB**, and as the pool balloons toward the ceiling, `cudaMalloc` fails **15 times** (free-and-retry). After autotune those segments are cached-free but **not returned**, so steady sits at 167 until an explicit trim pulls it back to ~142.

Crucially, `inactive_split_bytes` is **low in both** — so this is **not** classic intra-segment fragmentation. It's **whole-segment churn**: each tactic's workspace becomes its own segment the next (differently-sized) tactic doesn't reuse, so segments pile up. The clean discriminator is **`num_alloc_retries` (0 vs 15)** and segment count (+14 vs +48).

---

## Theory — why GMS recycles worse than vanilla

Same autotune, same live working set, but GMS's default pool **fails to recycle workspaces across tactics**, so it expands instead. The *why* is strongly indicated but not yet proven — one or more of:

- **Segment placement around the custom pool.** The GMS weights pool occupies a large region of the address space; the default pool's segments are placed around it, which can prevent the coalescing that lets vanilla turn freed workspaces back into one reusable region.
- **Cross-arena scarcity.** Because weights are in a separate pool, the default pool has less contiguous room, so varied workspace sizes don't find fitting free blocks and force fresh segments.
- **Heuristic interaction with the inflated reserved** (~180 GiB incl. the non-physical strand) shaping expand-vs-reuse decisions near the ceiling.

**Confirmed:** the location (autotune), the nature (allocator pool expansion, not compute), that it is *not* inactive-split fragmentation, and the signature (48 segments / 15 retries vs 14 / 0). **Inferred:** which reuse-failure mechanism dominates.

---

## Next test — `expandable_segments`

`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` changes the allocator to reserve one large **growable** virtual-address region and map physical on demand, instead of many fixed segments. That directly attacks the measured failure mode — varied workspace sizes get carved from one contiguous region and reuse cleanly rather than each spawning a fresh segment.

- **If it works** → retries → 0, segments stop ballooning, GMS autotune Δ drops toward vanilla's +5, peak → ~150. Confirms the theory and is very likely the fix.
- **If it doesn't help** → the problem is more specific to the GMS custom-pool interaction; next levers are autotuning in a dedicated scratch pool, or disabling FlashInfer autotune (removes the churn at a performance cost).

## Caveat — two separate GMS costs

The **strand (43 GiB)** and the **burst (25 GiB)** are distinct. The strand is non-physical `reserved` bookkeeping left by load-time prune; the burst is real default-pool expansion during autotune (fresh memory, not the strand being consumed). Both are GMS-specific and both want fixing, but `expandable_segments` targets the burst; the strand is a separate, less urgent matter for the physical ceiling.

---

*One-line summary: same work, but GMS's split-pool layout defeats PyTorch's workspace recycling during FlashInfer autotune, forcing ~48 fresh segments and pushing physical to the ceiling — and `expandable_segments` is the one-knob test that should confirm the diagnosis and fix it at once.*
