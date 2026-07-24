<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Memory Sizing & Minimum TP

## GPU Memory Sizing and Minimum TP Calculation

Before choosing a TP value, **compute** the minimum TP following these steps exactly. Do not skip the arithmetic.

### Weight sizing formulas

- BF16/FP16: `weight_bytes ≈ total_param_count × 2`
- FP8/INT8: `weight_bytes ≈ total_param_count × 1`
- INT4/W4A16/FP4/NVFP4/MXFP4: `weight_bytes ≈ total_param_count × 0.5`
- MoE models: use the **total** param count (all experts), not the active-per-token count

### Step-by-step min_tp calculation

1. **Compute weight size:** `W = total_param_count × bytes_per_param`
2. **Compute raw ratio:** `R = W / per_gpu_memory_bytes`
3. **Ceiling:** `G = ceil(R)`
4. **Round to power of 2:** `min_tp = next_power_of_2(G)`
5. **Verify fit:** confirm `min_tp × per_gpu_memory_bytes > W`. If the total memory across `min_tp` GPUs exceeds the
   weight size, the model **fits** at that TP. Never reject a TP value when total available memory exceeds weight size.


### Worked examples

**Example A — 236B MoE BF16 on B200-192GB:**

1. W = 236 B × 2 = 472 GB
2. R = 472 / 192 = 2.46
3. G = ceil(2.46) = 3
4. min_tp = next_power_of_2(3) = **4**
5. Verify: 4 × 192 = 768 GB > 472 GB — **fits at TP=4**

With 8 GPUs available, run 2 replicas at TP=4, or 1 replica at TP=8.

**Example B — 236B MoE BF16 on H100-80GB:**

1. W = 236 B × 2 = 472 GB
2. R = 472 / 80 = 5.9
3. G = ceil(5.9) = 6
4. min_tp = next_power_of_2(6) = **8**
5. Verify: 8 × 80 = 640 GB > 472 GB — **fits at TP=8**

**The same model requires different min_tp on different GPUs. Always recompute for your hardware.**

### Reference min_tp table

GPU memory varies by hardware — always recompute min_tp for the actual GPU. Do not copy values from this table when
using a different GPU type.

| Model | Precision | Weight size | GPU (mem) | R = W/mem | min_tp |
|---|---|---|---|---|---|
| 8B | BF16 | ~16 GB | H100 (80 GB) | 0.20 | 1 |
| 70B | BF16 | ~140 GB | H100 (80 GB) | 1.75 | 2 |
| 70B | BF16 | ~140 GB | B200 (192 GB) | 0.73 | 1 |
| 70B | FP8 | ~70 GB | H100 (80 GB) | 0.88 | 1 |
| 405B | BF16 | ~810 GB | H100 (80 GB) | 10.1 | 16 (2 nodes) |
| 405B | BF16 | ~810 GB | B200 (192 GB) | 4.22 | 8 |
| Mixtral 8x7B | BF16 | ~87 GB | H100 (80 GB) | 1.09 | 2 |
| DeepSeek V2 (236B) | BF16 | ~472 GB | H100 (80 GB) | 5.90 | 8 |
| DeepSeek V2 (236B) | BF16 | ~472 GB | B200 (192 GB) | 2.46 | 4 |

### Tight Memory: When Weights Leave Little Headroom

After computing min_tp, calculate how much memory **remains per GPU** after weights:

```text
weight_per_gpu = W / min_tp
remaining_per_gpu = per_gpu_memory - weight_per_gpu
headroom_ratio = remaining_per_gpu / per_gpu_memory
```

When `headroom_ratio < 0.50` (weights consume more than half of each GPU's memory), the system is
**memory-constrained**. The remaining memory must cover KV cache, activations, CUDA graphs, and framework overhead. This
is **critical on limited systems** (single GPU, or when only the minimum GPU count is available).

In tight-memory situations, adjust the config to help the runtime fit:

1. **Increase memory utilization** — raise `gpu_memory_utilization` / `free_gpu_memory_fraction` (e.g., 0.90–0.95) to
   maximize the memory available for KV cache. Conservative defaults may be too low when headroom is small.
2. **Decrease maximum sequence length** — lower `max_model_len` / `max_seq_len`, capped to the workload's actual needs
   (e.g., `ISL + OSL` with a small margin). The model default (`max_position_embeddings`) is often 32K–128K; if the
   workload only needs 2K–4K, capping the sequence length frees substantial GPU memory that would otherwise be reserved
   for KV cache metadata.
3. **Combine both strategies** — especially when the model barely fits (headroom_ratio < 0.30), raise memory utilization
   and lower sequence length together. Either alone may be insufficient.

**Worked example — 70B BF16 on 1× B200-192GB:**

1. W = 140 GB, min_tp = 1
2. weight_per_gpu = 140 GB, remaining = 52 GB
3. headroom_ratio = 52 / 192 = 0.27 → **tight**
4. Action: set `gpu_memory_utilization` to ~0.90–0.95 and cap `max_model_len` to `ISL + OSL + margin` so the runtime can
   start and serve at least the target workload.

Without these adjustments, the server may OOM at startup or support only a trivially small number of concurrent
sequences.

---

## Common Sanity Checks

Bad knob combinations to avoid in a single config (not a grid-pruning step).

### Low MNT + High KV fraction → OOM risk

Lower `max_num_tokens` can reduce profiled activation memory and let KV allocation grab too much headroom.

- **Avoid when:** `max_num_tokens <= 2048` and `free_gpu_memory_fraction >= 0.90`
- **Instead:** use `MNT >= 4096` or fraction `<= 0.85`.
