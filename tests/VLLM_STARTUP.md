# Fast vLLM startup in CI

This note separates model size, GPU memory, and startup latency. They are
related, but they are not interchangeable: for small models, Python imports,
model/tokenizer configuration, process creation, compilation, and kernel
warmup dominate checkpoint loading.

## CI policy

1. Give every parallel vLLM test an explicit
   `requested_vllm_kv_cache_bytes` marker. The launch helpers translate it to
   `--kv-cache-memory-bytes`, which skips vLLM memory profiling and makes GPU
   allocation deterministic.
2. For correctness and plumbing tests, keep compilation disabled. Existing
   launch scripts use `--enforce-eager`; vLLM 0.27.1 and newer can use
   `--optimization-level 0`. Use the default optimization level only when the
   test exercises compilation, CUDA graphs, or production-like performance.
3. Do not stagger explicitly capped vLLM launches. They do not run the memory
   profiler whose free-memory snapshot races with another process. The test
   scheduler and `tests/serve/common.py` therefore allow capped launches to
   start concurrently.
4. If a test needs compilation, persist `VLLM_CACHE_ROOT` across identical
   boots or bake a validated cache into the image. Consider
   `VLLM_FORCE_AOT_LOAD=1` in the cache-validation job so a cache miss cannot
   silently become a long compile.
5. If an explicit KV size is impossible, enable `VLLM_ENABLE_STARTUP_PLAN=1`.
   A later identical boot can reuse the recorded memory profile when its
   hardware/config fingerprint and free-memory baseline match.
6. Predownload models, then run with `HF_HUB_OFFLINE=1`. Dynamo's parallel GPU
   runner already does this. On network filesystems, retain vLLM's automatic
   safetensors prefetch or benchmark `--safetensors-load-strategy eager` when
   enough host RAM is available.

Do not globally use `--skip-tokenizer-init`: current Dynamo vLLM integration
forces tokenizer initialization because vLLM sampling parameters require it.
`--load-format dummy` is appropriate only for tests that intentionally do not
validate model output.

## H100 measurements

Measured on one H100 80 GB with `vllm/vllm-openai:v0.27.1-ubuntu2404`, a warm
container, model length 2,048, maximum 8 sequences, maximum 2,048 batched
tokens, and a 512 MiB explicit KV cache. Time-to-ready is from process launch
to the `HTTP server started` log. A four-to-eight-token chat request verified
each candidate returned HTTP 200; this benchmark did not attempt to rank model
quality.

| Model/configuration | Cache state | Ready (s) | Engine init (s) | GPU memory (MiB) | First request (s) |
|---|---:|---:|---:|---:|---:|
| Qwen3-0.6B, `-O0`, offline | warm files | 35.7 | 2.07 | 2,586 | 0.151 |
| Qwen3-0.6B, `--enforce-eager` | warm files | 40.1 | 2.26 | 2,586 | 0.085 |
| Qwen3-0.6B, default `-O2` | cold compile | 79.3 | 41.25 | — | — |
| Qwen3-0.6B, default `-O2` | warm AOT cache | 30.3 | 3.09 | — | — |
| Gemma 3 270M IT, `-O0`, offline | first load | 46.8 | 3.25 | 1,924 | 0.115 |
| Granite 4.0 H 350M, `-O0` | cold Mamba JIT | 103.1 | 61.91 | 2,036 | 0.922 |
| Granite 4.0 H 350M, `-O0` | warm kernel cache | 28.0 | 2.06 | 2,036 | 0.164 |
| LFM2.5 350M, `-O0` | first load | 48.0 | 7.82 | 2,102 | 0.452 |

The default Qwen cold compile spent 34.13 seconds in `torch.compile`; the same
AOT cache loaded in 0.28 seconds. Granite's first boot spent about 61 seconds
warming Mamba2 Triton kernels, even at `-O0`, then reused that kernel cache on
the next boot.

## Model choice

- **Best memory reduction with broad runtime coverage:** Gemma 3 270M IT used
  about 662 MiB less GPU memory than Qwen in this setup. It is a well-known,
  recent architecture and is registered for vLLM, SGLang, and TensorRT-LLM,
  but every CI environment must accept the Gemma license and provide access.
- **Best ungated, permissively licensed newer architecture:** Granite 4.0 H
  350M is Apache-2.0 and used about 550 MiB less GPU memory. It is attractive
  when the Mamba kernel cache persists, but a poor default for isolated cold
  containers.
- **Newest capability-oriented candidate:** LFM2.5 350M is newer, advertises
  tool use, and used about 484 MiB less GPU memory. It is limited to the vLLM
  and SGLang roles in the registry and uses the LFM 1.0 license.
- **Keep Qwen3-0.6B for architecture-sensitive coverage:** KV transfer,
  Qwen-specific parsing, Qwen LoRA, and cross-backend tests should remain
  pinned until a candidate is validated for those exact behaviors.

The measurements show that replacing Qwen can improve same-GPU packing by
roughly 19–26%, but does not automatically reduce cold time-to-ready. The
largest broadly applicable startup wins are explicit KV sizing, disabling or
reusing compilation, and removing artificial launch staggering.

## Reproducing a comparison

Resolve a compatible candidate from the registry, override only the desired
role, and run the same CI slice:

```bash
candidate=$(python3 -m tests.utils.model_registry \
  --kind llm \
  --require instruction_tuned \
  --require small_ci_candidate \
  --backend vllm \
  --max-parameters-millions 400)

DYN_CI_VLLM_SMOKE_MODEL="$candidate" python3 -m pytest tests/serve/test_vllm.py
```

Use the role override rather than a repository-wide replacement: the registry
will reject a model that lacks the role's declared backend or architectural
requirements.

## Upstream references

- [vLLM optimization levels and faster startup](https://docs.vllm.ai/en/latest/configuration/optimization/)
- [vLLM environment variables and startup plans](https://docs.vllm.ai/en/latest/configuration/env_vars/)
- [vLLM safetensors loading strategies](https://docs.vllm.ai/en/v0.27.1/api/vllm/config/load/)
- [Gemma 3 270M IT model card](https://huggingface.co/google/gemma-3-270m-it)
- [Granite 4.0 H 350M model card](https://huggingface.co/ibm-granite/granite-4.0-h-350m)
- [LFM2.5 350M model card](https://huggingface.co/LiquidAI/LFM2.5-350M)
- [TensorRT-LLM supported models](https://nvidia.github.io/TensorRT-LLM/latest/models/supported-models.html)
