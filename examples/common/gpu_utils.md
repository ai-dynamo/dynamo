# GPU Memory Control

How vLLM, SGLang, and TensorRT-LLM allocate GPU memory, and how we override
it for deterministic parallel test execution.

---

## Why absolute caps, not fractions

Memory fractions (`--gpu-memory-utilization`, `--mem-fraction-static`) are
unreliable for parallel / CI workloads:

- **Non-deterministic** — same fraction produces different KV cache sizes
  depending on what else is on the GPU at init time.
- **Profiling race** — concurrent engines each see "nearly all memory free",
  allocate based on that, and OOM.
- **Not portable** — a fraction tuned for 48 GiB is wrong on 24 or 80 GiB.
- **Different semantics** — vLLM/SGLang use fraction of *total* VRAM;
  TensorRT-LLM uses fraction of *free* VRAM after model load.

Instead, we use **absolute KV cache caps**:

| Engine | Deterministic override | Env var |
|--------|----------------------|---------|
| vLLM | `--kv-cache-memory-bytes N` | `_PROFILE_OVERRIDE_VLLM_KV_CACHE_BYTES` |
| SGLang | `--max-total-tokens N` | `_PROFILE_OVERRIDE_SGLANG_MAX_TOTAL_TOKENS` |
| TensorRT-LLM | `--override-engine-args '{"kv_cache_config":{"max_tokens":N}}'` | `_PROFILE_OVERRIDE_TRTLLM_MAX_TOTAL_TOKENS` |

---

## Quick Reference

| | vLLM | SGLang | TensorRT-LLM |
|---|---|---|---|
| Fraction flag | `--gpu-memory-utilization` | `--mem-fraction-static` | `free_gpu_memory_fraction` |
| Fraction base | Total VRAM | Total VRAM | Free VRAM (post-load) |
| Default | 0.90 | 0.90 | 0.90 |
| Max seq len | `--max-model-len` | `--context-length` | `max_seq_len` |
| KV cache override | `--kv-cache-memory-bytes` | `--max-total-tokens` | `KvCacheConfig.max_tokens` via `--override-engine-args` |

---

## Per-Engine Notes

### vLLM

`--gpu-memory-utilization` sets a budget as fraction of total VRAM.
KV cache = budget - weights - activations - overhead. Pool is fixed at startup.

`--kv-cache-memory-bytes` overrides automatic sizing and **skips memory
profiling** ([PR #21489]). The KV cache is pinned to the exact byte value —
no profiling race, no CUDAGraph estimation errors, safe for concurrent
instances ([#10643]). When set, `--gpu-memory-utilization` only affects
headroom for activations, not KV cache size.

`--max-model-len` caps sequence length. Reducing it is the fastest way to
cut VRAM when the model fits but KV cache doesn't.

[PR #21489]: https://github.com/vllm-project/vllm/pull/21489
[#10643]: https://github.com/vllm-project/vllm/issues/10643

### SGLang

`--mem-fraction-static` sets a budget as fraction of total VRAM.
KV cache pool = budget - weights. Activations and CUDA graph buffers are
*outside* this budget (unlike vLLM).

`--max-total-tokens` caps the KV token pool directly, regardless of fraction.
When set, the token cap is the binding constraint.

`--context-length` and `--max-running-requests` affect request scheduling
only — they do **not** change KV cache allocation.

### TensorRT-LLM

`free_gpu_memory_fraction` is a fraction of **free** VRAM after model load.
Set via YAML or `--override-engine-args '{"kv_cache_config":{"free_gpu_memory_fraction": 0.24}}'`.

Deterministic KV cache control uses `build_trtllm_override_args_with_mem` in
`gpu_utils.sh`, which builds JSON for `--override-engine-args`. Token-based
(`_PROFILE_OVERRIDE_TRTLLM_MAX_TOTAL_TOKENS`) or byte-based
(`_PROFILE_OVERRIDE_TRTLLM_MAX_GPU_TOTAL_BYTES`) caps are supported. If the
launch script already passes `--override-engine-args`, the function merges
the GPU config into the existing JSON via `--merge-with-json`.

---

## Engine-Specific GPU Memory Functions

Launch scripts source `gpu_utils.sh` and call engine-specific functions to pick
up env-var overrides during profiling and parallel execution:

```bash
source "$SCRIPT_DIR/../../../common/gpu_utils.sh"

# vLLM
GPU_MEM_ARGS=$(build_vllm_gpu_mem_args)
python -m dynamo.vllm --model "$MODEL" $GPU_MEM_ARGS &

# SGLang
GPU_MEM_ARGS=$(build_sglang_gpu_mem_args)
python -m dynamo.sglang --model-path "$MODEL" $GPU_MEM_ARGS &

# TRT-LLM (JSON merging, separate function)
OVERRIDE_JSON=$(build_trtllm_override_args_with_mem)
python -m dynamo.trtllm --model-path "$MODEL" ${OVERRIDE_JSON:+--override-engine-args "$OVERRIDE_JSON"} &
```

When the env var is set, the function returns the corresponding flag.
Otherwise it returns empty and the engine uses its default allocation.

| Env var | Function | Output |
|---------|----------|--------|
| `_PROFILE_OVERRIDE_VLLM_KV_CACHE_BYTES` | `build_vllm_gpu_mem_args` | `--kv-cache-memory-bytes N --gpu-memory-utilization 0.01` |
| `_PROFILE_OVERRIDE_SGLANG_MAX_TOTAL_TOKENS` | `build_sglang_gpu_mem_args` | `--max-total-tokens N` |
| `_PROFILE_OVERRIDE_TRTLLM_MAX_TOTAL_TOKENS` | `build_trtllm_override_args_with_mem` | `{"kv_cache_config": {"max_tokens": N}}` (JSON) |
| `_PROFILE_OVERRIDE_TRTLLM_MAX_GPU_TOTAL_BYTES` | `build_trtllm_override_args_with_mem` | `{"kv_cache_config": {"max_gpu_total_bytes": N}}` (JSON) |

All functions return per-process args. In multi-worker-per-GPU setups
(e.g. `disagg_same_gpu.sh`), each worker gets the same override value.
The profiler finds the per-worker budget directly.

**Profiler** (`profile_pytest.py`): binary-searches the KV cap to find the
minimum passing value, applies a 2x safety factor, outputs pytest markers
(`@pytest.mark.requested_vllm_kv_cache_bytes(N)`,
`@pytest.mark.requested_sglang_kv_tokens(N)`, or
`@pytest.mark.requested_trtllm_kv_tokens(N)`).

**Scheduler** (`pytest_parallel_gpu.py`): reads the markers at runtime and
sets the env var per-test. See `tests/README.md` for details.
