# Nemotron-Super-RL FP8 Recipes

Production-ready deployments for **nvidia/nemotron-super-rl-030326-FP8** (~124B hybrid Mamba/Attention/MoE) across multiple backends.

These recipes target **Dynamo 1.0**. See [Dynamo 0.9.1 Compatibility](#dynamo-091-compatibility) for notes on running with older containers.

## Available Configurations

| Configuration | GPUs | Backend | Mode | Description |
|--------------|------|---------|------|-------------|
| [**vllm/agg**](vllm/agg/) | 4x H100 | vLLM | Aggregated | TP=4, KV-aware routing |
| [**trtllm/agg**](trtllm/agg/) | 4x H100 | TensorRT-LLM | Aggregated | TP=4, KV-aware routing |
| [**sglang/agg**](sglang/agg/) | 4x H100 | SGLang | Aggregated | TP=4, KV-aware routing (not working on 0.9.1) |
| [**trtllm/disagg**](trtllm/disagg/) | 4x H100 | TensorRT-LLM | Disaggregated | TP=2 P/D split, UCX KV transfer |
| [**sglang/disagg**](sglang/disagg/) | 4x H100 | SGLang | Disaggregated | TP=2 P/D split, nixl KV transfer (not working on 0.9.1) |

## Prerequisites

1. **Dynamo Platform installed** -- See [Kubernetes Deployment Guide](../../docs/kubernetes/README.md)
2. **GPU cluster** with 4x H100 80GB (or H200) GPUs
3. **HuggingFace token** with access to NVIDIA models

## Quick Start

```bash
# Set namespace
export NAMESPACE=dynamo-demo
kubectl create namespace ${NAMESPACE}

# Create HuggingFace token secret
kubectl create secret generic hf-token-secret \
  --from-literal=HF_TOKEN="your-token-here" \
  -n ${NAMESPACE}

# Download model (update storageClassName in model-cache.yaml first!)
kubectl apply -f model-cache/ -n ${NAMESPACE}
kubectl wait --for=condition=Complete job/model-download -n ${NAMESPACE} --timeout=3600s

# Deploy (choose one configuration)
kubectl apply -f vllm/agg/deploy.yaml -n ${NAMESPACE}
# OR: kubectl apply -f trtllm/agg/deploy.yaml -n ${NAMESPACE}
# OR: kubectl apply -f trtllm/disagg/deploy.yaml -n ${NAMESPACE}
# OR: kubectl apply -f sglang/agg/deploy.yaml -n ${NAMESPACE}
```

## Test the Deployment

```bash
# Port-forward the frontend
kubectl port-forward svc/nemotron-super-fp8-vllm-agg-frontend 8000:8000 -n ${NAMESPACE}

# Basic chat (with reasoning)
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "nvidia/nemotron-super-rl-030326-FP8",
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 100
  }'

# Tool calling
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "nvidia/nemotron-super-rl-030326-FP8",
    "messages": [{"role": "user", "content": "What is the weather in SF?"}],
    "tools": [{"type": "function", "function": {"name": "get_weather", "parameters": {"type": "object", "properties": {"location": {"type": "string"}}, "required": ["location"]}}}],
    "max_tokens": 256
  }'

# Disable thinking (only works with nemotron_nano reasoning parser in 1.0+)
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "nvidia/nemotron-super-rl-030326-FP8",
    "messages": [{"role": "user", "content": "What is 2+2?"}],
    "chat_template_kwargs": {"enable_thinking": false},
    "max_tokens": 64
  }'
```

## Model Details

- **Model**: `nvidia/nemotron-super-rl-030326-FP8`
- **Architecture**: Nemotron-H (hybrid Mamba/Attention/MoE, 88 layers)
- **Parameters**: ~124B total (~119B FP8, ~4.7B BF16)
- **Quantization**: ModelOpt FP8 (F8_E4M3) with FP8 KV cache
- **Features**: Reasoning (`<think>` blocks) and tool calling (`<tool_call>` format)

## Hardware Requirements

| Configuration | GPU Memory | Notes |
|--------------|------------|-------|
| 4x H100 80GB | ~29 GiB weights + ~36 GiB KV cache | Recommended minimum |
| 4x H200 141GB | ~29 GiB weights + ~100 GiB KV cache | Higher throughput |

## Parser Configuration

All recipes include tool call and reasoning parsers:

- `--dyn-reasoning-parser nemotron_nano` -- Extracts `<think>...</think>` into `reasoning_content`. Correctly handles both `enable_thinking: true` and `enable_thinking: false`.
- `--dyn-tool-call-parser nemotron_nano` -- Parses `<tool_call><function=name>` into structured `tool_calls`.

To disable reasoning at request time, pass `"chat_template_kwargs": {"enable_thinking": false}`. The model also supports `"chat_template_kwargs": {"low_effort": true}` for lighter-weight reasoning.

## Routing

All recipes use **approximate KV-aware routing** (`--router-mode kv --no-kv-events` on the frontend). The frontend uses prefix hashing to route requests to workers most likely to have relevant KV cache blocks, improving cache hit rates. This is especially beneficial for workloads with shared system prompts or multi-turn conversations.

Approximate (hash-based) routing is used because none of the backends currently support publishing KV cache events for hybrid Mamba+Attention models. Exact KV-aware routing, which relies on real-time event streams from workers (`--kv-events-config` for vLLM/SGLang, `--publish-events-and-metrics` for TRT-LLM), would provide more accurate cache-aware decisions but is not yet available for this architecture.

## Backend-Specific Notes

### vLLM
- No connector flags needed in 1.0 (default is no connector)
- Requires `--is-decode-worker` to skip KV event publisher setup
- Requires `--mamba-cache-mode align` to work around [vllm#34865](https://github.com/vllm-project/vllm/issues/34865): prefix caching with the default `mamba_cache_mode="all"` produces NaN logprobs and garbage tokens for Nemotron-H. Fixed in vLLM 0.17.0 ([vllm#34874](https://github.com/vllm-project/vllm/pull/34874)); the 1.0 container ships vLLM 0.16.0, so the workaround is needed.
- **Attention backend**: On Hopper the default (`FLASH_ATTN`) is safe. On Blackwell, vLLM defaults to FlashInfer, which has a [stale NaN bug](https://github.com/vllm-project/vllm/issues/35138) with hybrid Mamba models ([vllm#35219](https://github.com/vllm-project/vllm/pull/35219)). For Blackwell, specify `--attention-backend FLASH_ATTN` or `--attention-backend TRITON_ATTN` to avoid the issue.
- Sets `VLLM_FLASHINFER_ALLREDUCE_BACKEND=trtllm` to avoid a [hang during CUDA graph capture](https://github.com/vllm-project/vllm/issues/35772) with TP>1. This is the [new default](https://github.com/vllm-project/vllm/pull/35793) in later vLLM versions but must be set explicitly in 0.16.0.
- Load time: ~2 minutes
- Attention KV cache: ~3.1M tokens per GPU

### TensorRT-LLM
- Uses PyTorch backend (`backend: pytorch` in engine config)
- 1.0 supports block reuse for Mamba hybrid cache (enabled by default). In 0.9.1, block reuse is not supported and `enable_block_reuse: false` must be set explicitly.
- **Disaggregated mode** requires `cache_transceiver_config: backend: UCX`. NIXL and MOONCAKE backends do not support hybrid models with Mamba SSM state — only UCX (or MPI) can transfer both attention KV cache and Mamba conv/SSM state between workers.
- Load time: ~7 minutes (CUDA graph compilation for 16 batch sizes)
- Attention KV cache: ~19.7M tokens per GPU

### SGLang
- Requires sglang >= v0.5.9 (1.0 ships v0.5.9; 0.9.1 ships v0.5.8 which has blocking bugs)
- Load time: ~1 minute (~31s weight load + ~25s CUDA graph capture for 21 batch sizes)
- Attention KV cache: ~9.4M tokens per GPU (aggregated)
- Mamba state cache: ~16 GiB per GPU, max 407 concurrent sequences (aggregated)
- No special flags needed for aggregated -- simplest configuration of the three backends
- **Disaggregated mode works** with nixl KV transfer (TP=2 per worker, 2 GPUs each). Mooncake (`--disaggregation-transfer-backend mooncake`) is also supported as an alternative transfer backend.
- Known issue: prefill warmup logs `Prefill warmup failed: 'SamplingParams' object is not subscriptable` -- non-blocking, does not affect functionality

## Dynamo 0.9.1 Compatibility

These recipes target Dynamo 1.0. To run on 0.9.1 containers, the following changes are needed:

### vLLM (`vllm-runtime:0.9.1`)
- Change image tags from `:1.0.0` to `:0.9.1`
- **Add** `--connector none` to worker args (required in 0.9.1 to disable nixl KV connector; rejected in 1.0)
- Change `--dyn-reasoning-parser` from `nemotron_nano` to `deepseek_r1` (nemotron_nano reasoning parser is broken in 0.9.1)
- `enable_thinking: false` will **not work** with `deepseek_r1` parser (response content goes to `reasoning_content`, `content` is null)
- `--mamba-cache-mode align` is still needed (0.9.1 ships vLLM 0.14.1, also affected by [vllm#34865](https://github.com/vllm-project/vllm/issues/34865))

### TensorRT-LLM (`tensorrtllm-runtime:0.9.1`)
- Change image tags from `:1.0.0` to `:0.9.1`
- Change `--dyn-reasoning-parser` from `nemotron_nano` to `deepseek_r1`
- Same `enable_thinking: false` caveat as vLLM above
- **Add** `enable_block_reuse: false` to `kv_cache_config` in the ConfigMap (required in 0.9.1; 1.0 supports block reuse natively)

### SGLang (`sglang-runtime:0.9.1`)
- **Not supported.** The bundled sglang v0.5.8 has two blocking bugs:
  1. FP8 quantization bug (`ModelOptFp8LinearMethod.create_weights()` signature mismatch)
  2. Config format mismatch (`hybrid_override_pattern` vs `layers_block_type`)
- Both are fixed in sglang v0.5.9 but the 0.9.1 container ships v0.5.8

## Notes

- **Disaggregated mode**: Supported with TRT-LLM via UCX (`trtllm/disagg`) and SGLang via nixl or mooncake (`sglang/disagg`). Not supported with vLLM due to hybrid KV cache incompatibilities. TRT-LLM disagg requires UCX because NIXL/MOONCAKE cannot transfer Mamba SSM state.
- **Storage class**: Update `storageClassName` in `model-cache/model-cache.yaml` before deploying.
- **Model size**: ~240GB download; expect 30-60 minutes depending on bandwidth.
