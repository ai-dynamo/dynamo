# Kimi-K2.5 Recipes

> 🚧 **Work-in-Progress — Experimental Recipe**
>
> The TensorRT-LLM Python package used for Dynamo's TRT-LLM integration does not yet include
> native Kimi K2.5 support. This recipe is an **experimental** effort to bring
> Kimi K2.5 to Dynamo ahead of upstream availability. It needs to patch the container image on top of released dynamo image.

Deployment recipe for **Kimi-K2.5** using TensorRT-LLM with Dynamo's KV-aware routing.

## Available Configurations

There are two model weight variants, each with its own model download and deploy manifests:

| Variant | Model | Deploy Configs | Notes |
|---------|-------|---------------|-------|
| **nvidia** 🚧 | `nvidia/Kimi-K2.5-NVFP4` | [`deploy.yaml`](trtllm/agg/nvidia/deploy.yaml), [`deploy-kvbm.yaml`](trtllm/agg/nvidia/deploy-kvbm.yaml) | Requires a [patched image](trtllm/agg/nvidia/patch/) |
| **baseten** | `baseten-admin/Kimi-2.5-text-nvfp4-v3` | [`deploy.yaml`](trtllm/agg/baseten/deploy.yaml) | Works with the stock image |

All configurations use TP8, EP8, aggregated mode with KV-aware routing.

The **nvidia** variant also has a KVBM (KV Block Manager) deploy that enables CPU-offloaded KV cache via `deploy-kvbm.yaml`.

## Prerequisites

1. **Dynamo Platform installed** — See [Kubernetes Deployment Guide](../../docs/kubernetes/README.md)
2. **GPU cluster** with B200 GPUs (8x per worker)
3. **HuggingFace token** with access to the model

## Quick Start (nvidia variant)

```bash
# Set namespace
export NAMESPACE=dynamo-demo
kubectl create namespace ${NAMESPACE}

# Create HuggingFace token secret
kubectl create secret generic hf-token-secret \
  --from-literal=HF_TOKEN="your-token-here" \
  -n ${NAMESPACE}

# Download model (update storageClassName in model-cache/model-cache.yaml first!)
kubectl apply -f model-cache/nvidia/ -n ${NAMESPACE}
kubectl wait --for=condition=Complete job/model-download -n ${NAMESPACE} --timeout=3600s

# Patch the container image (required for nvidia weights)
cd trtllm/agg/nvidia/patch
./patch-container.sh nvcr.io/nvidia/ai-dynamo/tensorrtllm-runtime:my-tag
cd -

# Deploy
kubectl apply -f trtllm/agg/nvidia/deploy.yaml -n ${NAMESPACE}
```

For baseten weights, use `model-cache/baseten/` and `trtllm/agg/baseten/deploy.yaml` instead — no image patch needed.

## Test the Deployment

```bash
# Port-forward the frontend
kubectl port-forward svc/kimi-k25-agg-frontend 8000:8000 -n ${NAMESPACE}

# Send a test request
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "nvidia/Kimi-K2.5-NVFP4",
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 100
  }'
```

## Model Details

- **Model**: `nvidia/Kimi-K2.5-NVFP4` (NV FP4 quantized, text-only)
- **Architecture**: MoE (Mixture-of-Experts), based on DeepSeek-V3 architecture
- **Backend**: TensorRT-LLM (PyTorch backend)
- **Parallelism**: TP8, EP8 (Expert Parallel)


## Hardware Requirements

| Configuration | GPUs |
|--------------|------|
| Aggregated | 8x B200 |

## Verifying Reasoning

The deployment uses `--dyn-reasoning-parser kimi_k25` to extract the model's chain-of-thought into a separate `reasoning_content` field. Verify that reasoning is properly separated from the final answer:

```bash
curl -s http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "nvidia/Kimi-K2.5-NVFP4",
    "messages": [{"role": "user", "content": "What is 2+2? Answer briefly."}],
    "max_tokens": 200
  }' | python3 -m json.tool
```

**Expected behavior:**

- `message.reasoning_content` contains the model's thinking process
- `message.content` contains only the final answer (e.g., `"4"`)
- No raw `</think>` tags appear in either field

**Example response:**

```json
{
  "choices": [{
    "message": {
      "content": "4",
      "role": "assistant",
      "reasoning_content": "The user is asking a simple math question: \"What is 2+2?\" and wants a brief answer.\n\n2+2 equals 4.\n\nI should answer briefly as requested."
    },
    "finish_reason": "stop"
  }]
}
```

If `reasoning_content` is `null` with raw `</think>` tags in `content`, the reasoning parser is not configured. Ensure the worker has `--dyn-reasoning-parser kimi_k25`.

## Verifying Tool Calling

The deployment uses `--dyn-tool-call-parser kimi_k2` to extract function calls into OpenAI-compatible structured `tool_calls`. Send a request with tool definitions:

```bash
curl -s http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "nvidia/Kimi-K2.5-NVFP4",
    "messages": [{"role": "user", "content": "What is the weather in San Francisco?"}],
    "tools": [{
      "type": "function",
      "function": {
        "name": "get_weather",
        "description": "Get the current weather for a location",
        "parameters": {
          "type": "object",
          "properties": {
            "location": {"type": "string", "description": "City name"}
          },
          "required": ["location"]
        }
      }
    }],
    "max_tokens": 300
  }' | python3 -m json.tool
```

**Expected behavior:**

- `message.tool_calls` contains a structured array with `name`, `arguments`, and `id`
- `message.content` contains only the natural language portion
- `message.reasoning_content` contains the model's reasoning about which tool to call
- `finish_reason` is `"tool_calls"`
- No raw `<|tool_calls_section_begin|>` tokens in `content`

**Example response:**

```json
{
  "choices": [{
    "message": {
      "content": "I'll check the weather in San Francisco for you.",
      "tool_calls": [{
        "id": "functions.get_weather:0",
        "type": "function",
        "function": {
          "name": "get_weather",
          "arguments": "{\"location\":\"San Francisco\"}"
        }
      }],
      "role": "assistant",
      "reasoning_content": "The user is asking for the weather in San Francisco. I have a function called get_weather that can retrieve weather information. I need to call this function with \"San Francisco\" as the location parameter."
    },
    "finish_reason": "tool_calls"
  }]
}
```

If `tool_calls` is missing with raw `<|tool_calls_section_begin|>` tokens in `content`, the tool call parser is not configured. Ensure the worker has `--dyn-tool-call-parser kimi_k2`.

## Notes

- Update `storageClassName` in `model-cache/model-cache.yaml` before deploying
- The nvidia variant requires a [patched TensorRT-LLM image](trtllm/agg/nvidia/patch/) until Kimi K2.5 support lands upstream
