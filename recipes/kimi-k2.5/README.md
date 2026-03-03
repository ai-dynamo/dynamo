# Kimi-K2.5 Recipes

Deployment recipe for **Kimi-K2.5** using TensorRT-LLM with Dynamo's KV-aware routing.

> **Note:** Support for the official **`nvidia/Kimi-K2.5-NVFP4`** checkpoint is in progress and will be added soon. The current recipe uses **`baseten-admin/Kimi-2.5-text-nvfp4-v3`**, a text-only variant where users can experience Kimi-K2.5 and its tool calling and reasoning capabilities.

## Available Configurations

| Configuration | GPUs | Mode | Description |
|--------------|------|------|-------------|
| [**trtllm/agg**](trtllm/agg/) | 8x GPU | Aggregated | TP8, EP8, KV-aware routing |

## Prerequisites

1. **Dynamo Platform installed** — See [Kubernetes Deployment Guide](../../docs/pages/kubernetes/README.md)
2. **GPU cluster** with B200 GPUs (8x per worker)
3. **HuggingFace token** with access to the model

## Quick Start

```bash
# Set namespace
export NAMESPACE=dynamo-demo
kubectl create namespace ${NAMESPACE}

# Create HuggingFace token secret
kubectl create secret generic hf-token-secret \
  --from-literal=HF_TOKEN="your-token-here" \
  -n ${NAMESPACE}

# Download model (update storageClassName in model-cache/model-cache.yaml first!)
kubectl apply -f model-cache/ -n ${NAMESPACE}
kubectl wait --for=condition=Complete job/model-download -n ${NAMESPACE} --timeout=3600s

# Deploy
kubectl apply -f trtllm/agg/deploy.yaml -n ${NAMESPACE}
```

## Test the Deployment

```bash
# Port-forward the frontend
kubectl port-forward svc/kimi-k25-agg-frontend 8000:8000 -n ${NAMESPACE}

# Send a test request
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "baseten-admin/Kimi-2.5-text-nvfp4-v3",
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 100
  }'
```

## Model Details

- **Model**: `baseten-admin/Kimi-2.5-text-nvfp4-v3` (NV FP4 quantized, text-only)
- **Architecture**: MoE (Mixture-of-Experts), based on DeepSeek-V3 architecture
- **Backend**: TensorRT-LLM (PyTorch backend)
- **Parallelism**: TP8, EP8 (Expert Parallel)
- **Features**: Reasoning (chain-of-thought), tool calling (function calling)

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
    "model": "baseten-admin/Kimi-2.5-text-nvfp4-v3",
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
    "model": "baseten-admin/Kimi-2.5-text-nvfp4-v3",
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