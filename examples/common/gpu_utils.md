# GPU Memory Parameters by Engine

How vLLM, sglang, and TensorRT-LLM interpret memory-related parameters, and how
to estimate total GPU VRAM usage for each.

---

## Quick Reference

| Parameter | vLLM | sglang | TensorRT-LLM |
|---|---|---|---|
| Memory fraction | `--gpu-memory-utilization` | `--mem-fraction-static` | `free_gpu_memory_fraction` (YAML/override) |
| Fraction base | Total VRAM | Total VRAM | Free VRAM (after model load) |
| Default fraction | 0.90 | 0.90 | 0.90 |
| Max sequence length | `--max-model-len` | `--context-length` | `max_seq_len` (YAML/override) |
| KV cache size override | `--kv-cache-memory-bytes` | N/A | `max_gpu_total_bytes` (broken in 1.3.0rc5) |

---

## 1. vLLM

### How `--gpu-memory-utilization` works

This is a fraction of **total** GPU VRAM. The engine budgets everything within
this limit:

```
budget = total_vram * gpu_memory_utilization

KV cache = budget - model_weights - peak_activations - framework_overhead
```

At startup, vLLM profiles actual model weight and activation memory, then
pre-allocates the remaining budget as KV cache blocks. The KV pool size is fixed
for the lifetime of the engine.

### How `--max-model-len` works

Sets the maximum total sequence length (input + output tokens). Longer sequences
require more KV cache per request. If the requested `max-model-len` needs more
KV cache than the budget allows, vLLM errors at startup:

```
ValueError: ... X GiB KV cache is needed, which is larger than the available
KV cache memory (Y GiB). ...
```

Reducing `--max-model-len` is the most effective way to reduce VRAM when the
model fits but the KV cache doesn't.

### How `--kv-cache-memory-bytes` works

When set, this overrides the automatic KV cache sizing from
`gpu-memory-utilization`. The engine allocates exactly this many bytes for KV
cache regardless of the fraction. This means `gpu-memory-utilization` still
controls the *overall* VRAM budget (and thus whether the model fits), but the
KV cache portion is pinned to the explicit byte value.

Consequence for profiling: if a script uses `--kv-cache-memory-bytes`,
changing `_PROFILE_PYTEST_VRAM_FRAC_OVERRIDE` (which maps to
`--gpu-memory-utilization`) won't change the KV cache size, only the leftover
headroom for activations and overhead.

### Estimating total GPU usage

```
total_vram ≈ model_weights + kv_cache + activations + overhead

model_weights ≈ num_params * bytes_per_param
                (e.g. 7B * 2 bytes for BF16 ≈ 14 GiB)

kv_cache_per_token = 2 * num_layers * num_kv_heads * head_dim * bytes_per_element
                     (the factor of 2 is for K and V tensors)

kv_cache_total = kv_cache_per_token * max_model_len * max_concurrent_seqs

overhead ≈ engine-dependent (auto-computed by estimate_worker_vram):
           vllm:   1.2 + 1.0 * sqrt(params_b) GiB  (0.6B≈2.0, 8B≈4.0)
           sglang: 2.5 + 1.5 * sqrt(params_b) GiB  (0.6B≈3.7, 8B≈6.7)
           trtllm: 2.0 + 1.2 * sqrt(params_b) GiB  (0.6B≈2.9, 8B≈5.4)
```

Rule of thumb: set `gpu-memory-utilization` so that
`total_vram * fraction >= model_weights + 2 GiB`. The rest becomes KV cache.

---

## 2. sglang

### How `--mem-fraction-static` works

Like vLLM, this is a fraction of **total** GPU VRAM:

```
budget = total_vram * mem_fraction_static

KV cache pool = budget - model_weights
```

The budget covers model weights and the KV cache pool. Activations and CUDA
graph buffers are allocated *outside* this budget from the remaining VRAM.
This is slightly different from vLLM (which includes activations in the budget).

sglang recommends keeping 5-8 GiB free for activations and overhead. If you
see OOM errors, decrease `--mem-fraction-static` by 0.01-0.05 increments.

### How `--context-length` works

Equivalent to vLLM's `--max-model-len`. Defaults to the model's native context
window. Reducing it shrinks the per-request KV cache requirement and allows more
concurrent sequences.

### Estimating total GPU usage

```
total_vram ≈ model_weights + kv_cache_pool + activations_and_overhead

kv_cache_pool = total_vram * mem_fraction_static - model_weights

activations_and_overhead ≈ 1-8 GiB (depends on model size, batch size, seq len;
                           ~1-2 GiB for small models like 0.6B,
                           ~5-8 GiB for larger models like 8B+ with CUDA graphs)
```

---

## 3. TensorRT-LLM

### How `free_gpu_memory_fraction` works

This is a fraction of **free** VRAM (not total). The engine:

1. Loads model weights and builds the TRT engine (fixed cost).
2. Queries remaining free GPU memory.
3. Allocates `free_memory * free_gpu_memory_fraction` for the KV cache pool.

```
kv_cache = free_vram_after_model_load * free_gpu_memory_fraction
```

This means the same fraction yields different absolute KV cache sizes depending
on how much VRAM the model consumed. A 5 GiB model on a 48 GiB GPU leaves
~43 GiB free; fraction=0.24 gives ~10 GiB KV cache. A 30 GiB model leaves
~18 GiB free; fraction=0.24 gives only ~4 GiB.

Set via YAML config, CLI, or env var:

```bash
--override-engine-args '{"kv_cache_config":{"free_gpu_memory_fraction": 0.24}}'
DYN_TRTLLM_OVERRIDE_ENGINE_ARGS='{"kv_cache_config":{"free_gpu_memory_fraction": 0.24}}'
```

### How `max_seq_len` works

Maximum total sequence length. Defaults to the model's native context.
Sequences exceeding this limit are rejected at runtime.

**VRAM impact: none (PyTorch backend).** Reducing max_seq_len from 40960 to
2048 had zero effect on total VRAM or KV cache size in testing (Qwen3-0.6B,
trtllm 1.3.0rc5). The PyTorch backend does not pre-allocate internal buffers
proportional to max_seq_len; KV cache size is determined solely by
`free_gpu_memory_fraction`. This differs from vLLM/sglang where reducing
context length measurably reduces memory.

Override via:

```bash
--override-engine-args '{"max_seq_len": 4096}'
```

### Override gotcha: sub-dict replacement

Overriding any field inside `kv_cache_config` **replaces the entire sub-dict**.
If your YAML has `enable_block_reuse: true` and you override only
`free_gpu_memory_fraction`, you lose `enable_block_reuse`. Always re-include
all fields you need:

```json
{"kv_cache_config": {"free_gpu_memory_fraction": 0.15, "enable_block_reuse": true}}
```

### How `max_num_tokens` works

Maximum batched input tokens per iteration. Primarily a throughput knob.

**VRAM impact: none.** Reducing from 8192 → 256 had no measurable effect on
total VRAM (41,643 vs 41,465 MiB — within noise; the slight *increase* is
because smaller activation footprint lets the fraction claim marginally more
KV cache).

### `max_gpu_total_bytes` (broken)

Intended as an absolute byte cap for KV cache. As of trtllm 1.3.0rc5, this
field is **ignored**. Setting 5 GiB cap with `free_gpu_memory_fraction=0.95`
still allocated ~42 GiB of KV cache. Setting `free_gpu_memory_fraction=0.0`
with only `max_gpu_total_bytes` causes `"Impossible to fit any sequence in
kvCache"`. Do not rely on this field.

### Override precedence

```
--override-engine-args JSON  >  --extra-engine-args YAML  >  CLI flags
```

The `DYN_TRTLLM_OVERRIDE_ENGINE_ARGS` env var is equivalent to
`--override-engine-args` and avoids shell quoting issues with scripts whose
arg parsers consume unknown flags before passing `"$@"`.

### Estimating total GPU usage

```
total_vram ≈ model_weights + engine_overhead + kv_cache

model_weights ≈ num_params * bytes_per_param / tensor_parallel_size
engine_overhead ≈ 2.0 + 1.2 * sqrt(params_b) GiB  (CUDA context + TRT buffers + activations)
kv_cache = free_vram_after_model_load * free_gpu_memory_fraction
```

Engine overhead is auto-computed by `estimate_worker_vram` when called with the
`trtllm` engine name.  Examples: 0.6B → 2.9 GiB, 8B → 5.4 GiB, 30B → 8.6 GiB.

### Empirical validation (Qwen3-0.6B, RTX 6000 Ada 48 GiB, trtllm 1.3.0rc5)

Controlled test: single worker via agg.sh, one override at a time.

| # | Override | Total VRAM | KV Cache | Tokens |
|---|---------|-----------|----------|--------|
| 1 | Baseline (YAML frac=0.85) | 41,465 MiB | 38.04 GiB | 356,160 |
| 2 | `free_gpu_memory_fraction=0.15` | 9,383 MiB | 6.71 GiB | 62,848 |
| 3 | `max_num_tokens=256` | 41,643 MiB | 38.26 GiB | 358,208 |
| 4 | `max_seq_len=4096` | 41,469 MiB | 38.05 GiB | 356,192 |
| 5 | `max_seq_len=2048` | 41,469 MiB | 38.05 GiB | 356,192 |
| 6 | seq=4096 + frac=0.15 | 9,383 MiB | 6.71 GiB | 62,848 |
| 7 | tokens=256 + seq=4096 + frac=0.15 | 9,377 MiB | 6.75 GiB | 63,200 |

**Conclusion:** `free_gpu_memory_fraction` is the **sole effective knob** for
trtllm VRAM control. Neither `max_seq_len` nor `max_num_tokens` reduce memory.
Combined overrides (test 7) produce no additional benefit over fraction alone
(test 2).

---

## Why vLLM/sglang fractions are NOT interchangeable with TensorRT-LLM

Consider wanting 10 GiB of KV cache on a 48 GiB GPU with a 5 GiB model:

| Engine | Fraction meaning | Calculation | Result |
|---|---|---|---|
| vLLM | 10/48 = 0.21 of total | `48 * 0.21 = 10 GiB` budget (minus model = 5 GiB KV) | Wrong — need higher fraction |
| sglang | Same as vLLM | Same math | Same problem |
| TensorRT-LLM | 10/43 = 0.23 of free | `43 * 0.23 = 10 GiB` KV cache | Correct |

For vLLM/sglang, you actually need `(model + kv) / total = (5 + 10) / 48 = 0.31`
to get 10 GiB of KV cache with a 5 GiB model.

The helper functions in `gpu_utils.sh` handle these differences:
- `gpu_gb_to_total_fraction`: for vLLM/sglang (fraction of total VRAM)
- `gpu_gb_to_free_fraction`: for TensorRT-LLM (fraction of free VRAM)
- `gpu_worker_fraction <engine> <total_gib> <kv_gib>`: converts estimated GiB
  into the engine-appropriate fraction (total for vllm/sglang, free for trtllm).

Launch scripts use `build_gpu_mem_args` which calls these internally:

```bash
GPU_MEM_FRACTION=$(build_gpu_mem_args trtllm --model "$MODEL" --max-model-len "$SEQ_LEN" --max-num-seqs "$CONCURRENCY")
```

---

## KV Cache Memory Per Token

The formula for KV cache memory per token is the same across all engines:

```
kv_bytes_per_token = 2 * num_layers * num_kv_heads * head_dim * bytes_per_element
```

| Model | Layers | KV Heads | Head Dim | Dtype | Per Token |
|---|---|---|---|---|---|
| Qwen3-0.6B | 28 | 8 | 128 | BF16 | 112 KiB |
| Llama-3.1-8B | 32 | 8 | 128 | BF16 | 128 KiB |
| Llama-3.1-70B | 80 | 8 | 128 | BF16 | 320 KiB |
| Qwen2.5-VL-7B | 28 | 4 | 128 | BF16 | 56 KiB |

To estimate KV cache for a given context length:

```
kv_cache_gib = kv_bytes_per_token * max_model_len * max_concurrent_seqs / (1024^3)
```

---

## `_PROFILE_PYTEST_VRAM_FRAC_OVERRIDE`

Environment variable used by Dynamo's VRAM profiler to binary-search the minimum
memory fraction a script needs.

- Maps to `--gpu-memory-utilization` in vLLM and `--mem-fraction-static` in sglang.
- For TensorRT-LLM, maps to `kv_cache_config.free_gpu_memory_fraction` via
  `--override-engine-args`.
- Launch scripts use `build_gpu_mem_args` to compute the default fraction;
  the override bypasses the estimator and splits the raw value between workers.
- Scripts that use `--kv-cache-memory-bytes` (vLLM) bypass the fraction-based KV
  cache sizing, making the profiler's fraction override ineffective for KV cache.
  Those scripts should warn when `_PROFILE_PYTEST_VRAM_FRAC_OVERRIDE` is set.
