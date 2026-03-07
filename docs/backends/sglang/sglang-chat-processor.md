---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: SGLang Chat Processor
subtitle: SGLang-native preprocessing and postprocessing for chat completions
---

The SGLang chat processor enables SGLang-native preprocessing and postprocessing in the Dynamo frontend. It uses SGLang's tokenizer, chat templates, tool call parser, and reasoning parser directly -- bypassing the default Rust preprocessor for `v1/chat/completions` requests.

## When to Use

Use `--dyn-chat-processor sglang` when you need:

- **Tool calling** with SGLang-supported parsers (hermes, llama3, qwen25, etc.)
- **Reasoning parsing** for chain-of-thought models (qwen3, deepseek-r1, etc.)
- **SGLang's chat template rendering** for models with complex templates that the Rust preprocessor doesn't handle

The default Rust preprocessor is faster (no Python GIL overhead) but does not support tool call parsing or reasoning content extraction.

## Quick Start

```bash
# Frontend with SGLang processor, tool calling, and reasoning
python -m dynamo.frontend \
  --router-mode kv \
  --dyn-chat-processor sglang \
  --tool-call-parser hermes \
  --reasoning-parser qwen3

# Workers (unchanged)
CUDA_VISIBLE_DEVICES=0 python -m dynamo.sglang \
  --model-path Qwen/Qwen3-14B-FP8 \
  --served-model-name Qwen/Qwen3-14B-FP8 \
  --tp 1 --trust-remote-code \
  --kv-events-config '{"publisher":"zmq","topic":"kv-events","endpoint":"tcp://*:5557"}'
```

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Dynamo Frontend                          │
│                                                                 │
│  HTTP Request ──► SglangProcessor                               │
│                      │                                          │
│            ┌─────────┼──────────┐                               │
│            ▼         │          │                                │
│     ┌────────────┐   │   ┌──────────────┐                       │
│     │ Preprocess │   │   │ Postprocess  │                       │
│     │            │   │   │              │                       │
│     │ • Messages │   │   │ • Detokenize │                       │
│     │ • Template │   │   │ • Reasoning  │                       │
│     │ • Tokenize │   │   │ • Tool calls │                       │
│     │ • Tools    │   │   │ • SSE format │                       │
│     └─────┬──────┘   │   └──────▲───────┘                       │
│           │          │          │                                │
│           ▼          │          │                                │
│     ┌────────────────┴──────────┴──┐                            │
│     │    KvRouter / RoundRobin     │                            │
│     └──────────┬───────────────────┘                            │
│                │ token_ids                                       │
└────────────────┼────────────────────────────────────────────────┘
                 │
                 ▼
         ┌───────────────┐
         │ SGLang Worker  │
         │ (engine only)  │
         └───────────────┘
```

### Request Flow

1. **Preprocess** (`sglang_prepost.py`):
   - Converts OpenAI tool definitions to SGLang `Tool` objects
   - Applies the model's chat template via `tokenizer.apply_chat_template(tokenize=True, tools=...)`
   - Creates `FunctionCallParser` and `ReasoningParser` instances for streaming

2. **Route**: Token IDs are sent to a worker via `KvRouter` (KV-aware) or round-robin

3. **Postprocess** (`SglangStreamingPostProcessor`):
   - Incrementally detokenizes output token IDs using a 6-token sliding-window lookback
   - Extracts reasoning content via `ReasoningParser.parse_stream_chunk()`
   - Strips tool call markup and accumulates tool call deltas via `FunctionCallParser.parse_stream_chunk()`
   - Emits complete tool calls on `finish_reason` with full `name` and `arguments`

4. **SSE**: The Rust layer formats the Python output as Server-Sent Events and sends to the client. For non-streaming requests, the Rust `DeltaAggregator` folds SSE chunks into a single response.

## Frontend Arguments

These arguments are passed to the **frontend** (not the worker) when using `--dyn-chat-processor sglang`:

| Argument | Default | Description |
|----------|---------|-------------|
| `--dyn-chat-processor sglang` | (none) | Enable the SGLang chat processor |
| `--tool-call-parser` | `None` | Tool call parser name (`hermes`, `llama3`, `qwen25`, etc.) |
| `--reasoning-parser` | `None` | Reasoning parser name (`qwen3`, `deepseek-r1`, etc.) |

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `DYN_SGLANG_STREAM_INTERVAL` | `20` | Number of tokens to accumulate before detokenizing. Higher values improve throughput. The first chunk always emits immediately (interval=1) to minimize time-to-first-token. |

## Tool Calling

The processor supports all SGLang tool call formats. Pass `--tool-call-parser` on the frontend:

```bash
python -m dynamo.frontend \
  --dyn-chat-processor sglang \
  --tool-call-parser hermes
```

### Supported Formats

| Parser | Models | Format |
|--------|--------|--------|
| `hermes` | Qwen3, Hermes-based | `tool_call` tags with JSON name/arguments |
| `llama3` | Llama 3.x | `python_tag` delimiter / JSON |
| `qwen25` | Qwen 2.5 | `tool_call` tags with Qwen-specific wrapping |

### Example: Tool Call Request

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-14B-FP8",
    "messages": [{"role": "user", "content": "What is the weather in Paris?"}],
    "tools": [{
      "type": "function",
      "function": {
        "name": "get_weather",
        "description": "Get weather for a city",
        "parameters": {
          "type": "object",
          "properties": {"city": {"type": "string"}},
          "required": ["city"]
        }
      }
    }],
    "tool_choice": "auto"
  }'
```

Response:

```json
{
  "choices": [{
    "message": {
      "role": "assistant",
      "tool_calls": [{
        "id": "call_8cd24396f3671048",
        "type": "function",
        "function": {
          "name": "get_weather",
          "arguments": "{\"city\": \"Paris\"}"
        }
      }],
      "reasoning_content": "The user wants weather info for Paris..."
    },
    "finish_reason": "tool_calls"
  }]
}
```

## Reasoning Parsing

For models that produce chain-of-thought reasoning (e.g., Qwen3, DeepSeek-R1), pass `--reasoning-parser`:

```bash
python -m dynamo.frontend \
  --dyn-chat-processor sglang \
  --reasoning-parser qwen3
```

The parser separates think tag content into the `reasoning_content` field and regular content into the `content` field.

## Performance

The SGLang processor adds Python overhead compared to the default Rust preprocessor. The `DYN_SGLANG_STREAM_INTERVAL` environment variable controls the trade-off:

| `stream_interval` | Throughput vs Rust | TTFT Impact | Use Case |
|-------------------|--------------------|-------------|----------|
| 1 | ~50-60% | Minimal | Low-latency streaming |
| 20 (default) | ~85-100% | Minimal | General use |
| 50+ | ~95%+ | Moderate | Batch/throughput-oriented |

Higher values reduce Python GIL contention by batching tokens before detokenization. The first chunk always emits immediately regardless of the interval, so TTFT is unaffected. The default of 20 balances throughput and responsiveness.

### Preprocessing Offload

For high-concurrency deployments, offload preprocessing to a process pool:

```bash
python -m dynamo.frontend \
  --dyn-chat-processor sglang \
  --dyn-preprocess-workers 4
```

This moves `apply_chat_template` + tokenization to worker processes, freeing the async event loop. The post-processor remains in the main process.

## Migration from `--use-sglang-tokenizer`

`--use-sglang-tokenizer` on the **worker** is deprecated. Replace with `--dyn-chat-processor sglang` on the **frontend**:

```diff
  # Before (deprecated)
- python -m dynamo.sglang --model-path <model> --use-sglang-tokenizer
- python -m dynamo.frontend

  # After
  python -m dynamo.sglang --model-path <model>
+ python -m dynamo.frontend --dyn-chat-processor sglang
```

Key differences:

| | `--use-sglang-tokenizer` | `--dyn-chat-processor sglang` |
|---|---|---|
| Location | Worker flag | Frontend flag |
| KV router | Not supported | Supported |
| Tool calling | Not supported | Supported |
| Reasoning | Not supported | Supported |
| Endpoints | `v1/chat/completions` only | `v1/chat/completions` only |

## See Also

- **[Tool Calling](../../agents/tool-calling.md)**: General tool calling guide
- **[Reference Guide](sglang-reference-guide.md)**: Full SGLang backend reference
- **[Agentic Workloads](agents.md)**: Priority scheduling and cache pinning for agents
