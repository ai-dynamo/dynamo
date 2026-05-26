# Local-Dev Launch Flag Matrix

Per-backend reference for `python3 -m dynamo.<backend>` flags. Verify
the exact set against `python3 -m dynamo.<backend> --help` on the
installed version — flags drift across backend releases.

---

## vLLM (`dynamo.vllm`)

### Aggregated (default)

```bash
python3 -m dynamo.vllm --model Qwen/Qwen3-0.6B
```

Common flags:

| Flag | Purpose |
|---|---|
| `--tensor-parallel-size N` | TP across N GPUs |
| `--pipeline-parallel-size N` | PP across N GPUs (less common locally) |
| `--max-model-len N` | Override context window |
| `--max-num-seqs N` | Max concurrent requests |
| `--max-num-batched-tokens N` | Continuous-batching token budget |
| `--gpu-memory-utilization 0.85` | Default 0.9; lower for stable behavior on shared GPUs |
| `--enable-prefix-caching` | Prefix cache (interacts with KV-aware routing) |
| `--quantization fp8` | If the model is FP8 (per / `pyproject.toml [vllm]` pins NIXL via `nixl[cu12]==1.1.0`) |
| `--load-format auto` | Load whatever format the checkpoint provides |
| `--port 8000` | Default 8000 |

### Disaggregated (per)

Prefill (terminal 1):

```bash
python3 -m dynamo.vllm \
  --model Qwen/Qwen3-0.6B \
  --disaggregation-mode prefill \
  --kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_both"}'
```

Decode (terminal 2):

```bash
python3 -m dynamo.vllm \
  --model Qwen/Qwen3-0.6B \
  --disaggregation-mode decode \
  --port 8001
```

Note: the prefill worker requires the explicit `--kv-transfer-config`.
Without it, the prefill worker enters `CrashLoopBackOff` with
`--connector is deprecated and the default is no longer nixl`.

### vLLM-Omni (multimodal video, `dynamo.vllm.omni`)

```bash
python3 -m dynamo.vllm.omni \
  --model Wan-AI/Wan2.2-T2V-14B-Diffusers \
  --enforce-eager
```

The Wan2.2 video launchers force eager mode (the cherry-pick #9563 /
DYN-3014 fix; the JIT compile path crashes with CUDA illegal memory
access in cross-attention).

---

## TensorRT-LLM (`dynamo.trtllm`)

### Aggregated

```bash
python3 -m dynamo.trtllm --model Qwen/Qwen3-0.6B
```

Common flags:

| Flag | Purpose |
|---|---|
| `--tp-size N` | Tensor parallelism (TRT-LLM uses `--tp-size`, not `--tensor-parallel-size`) |
| `--pp-size N` | Pipeline parallelism |
| `--max-batch-size N` | Engine max batch |
| `--max-input-len N` | Static input length budget |
| `--max-output-len N` | Static output length budget |
| `--quant-config <path>` | Path to `quant_config.json` (per [dynamo-optimize](../../dynamo-optimize/references/modelopt-cli.md)) |
| `--engine-dir <path>` | Pre-built engine (skip the build step) |

TRT-LLM is more sensitive to build-vs-runtime separation than vLLM. The
first run of a model triggers an engine build (minutes to hours
depending on size); subsequent runs reuse `--engine-dir`.

### Disaggregated

Same `--disaggregation-mode prefill|decode` pattern as vLLM. The
`--kv-transfer-config` requirement from applies identically.

---

## SGLang (`dynamo.sglang`)

### Aggregated

```bash
python3 -m dynamo.sglang --model Qwen/Qwen3-0.6B
```

Common flags:

| Flag | Purpose |
|---|---|
| `--tp N` | Tensor parallelism |
| `--mem-fraction-static 0.85` | Fraction of GPU memory for static allocations |
| `--load-format gms` | Use the GPU Memory Service for weight loading (per Dynamo containers when `enable_gpu_memory_service: true` is set in `container/context.yaml`) |
| `--enable-memory-saver` | Required when `--load-format gms` (per and the RC5 fix #9772 / DYN-3047) |

### Disaggregated

```bash
python3 -m dynamo.sglang \
  --model Qwen/Qwen3-0.6B \
  --disaggregation-mode prefill \
  --kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_both"}'
```

---

## Mocker (`dynamo.mocker`)

For testing the request path without a real model.

```bash
python3 -m dynamo.mocker --port 8000
```

Useful for:

- Benchmarking the Frontend / Router / Planner without GPU.
- Local AIPerf iteration against a known-stable target.
- CI-style smoke tests.

Mocker does not load weights and ignores `--model`.

---

## All Backends: Auth and HF Token

For gated HuggingFace models, set the token before launch:

```bash
export HF_TOKEN=<token>
# or
huggingface-cli login
```

Without a token, the worker logs `401 Unauthorized` from huggingface.co
and fails to register the model with `/v1/models`. This is the
local-dev variant of (in K8s deployments, mount the
`hf-token-secret`).
