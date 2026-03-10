---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: SGLang Chat Processor
subtitle: SGLang-native preprocessing and postprocessing for chat completions
---

The SGLang chat processor enables SGLang-native preprocessing and postprocessing in the Dynamo frontend. It uses SGLang's tokenizer, chat templates, tool call parser, and reasoning parser directly -- bypassing the default Rust preprocessor for `v1/chat/completions` requests.

## When to Use

Use `--dyn-chat-processor sglang` when Dynamo's built-in Rust preprocessor does not yet support a tool call parser or reasoning parser you need. The SGLang processor delegates to SGLang's Python implementations, so any parser SGLang supports works immediately.

Common cases:

- A **tool call format** not yet in the Rust `tool_calling` library
- A **reasoning parser** not yet supported natively
- A **chat template** that the Rust preprocessor doesn't handle correctly

If the parser you need is missing from the Rust preprocessor, consider [opening an issue or PR](https://github.com/ai-dynamo/dynamo/issues) to add native support -- native parsers avoid the Python GIL overhead entirely.

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

## Frontend Arguments

These arguments are passed to the **frontend** (not the worker) when using `--dyn-chat-processor sglang`:

| Argument | Default | Description |
|----------|---------|-------------|
| `--dyn-chat-processor sglang` | (none) | Enable the SGLang chat processor |
| `--tool-call-parser` | `None` | Tool call parser name (any SGLang-supported parser) |
| `--reasoning-parser` | `None` | Reasoning parser name (any SGLang-supported parser) |

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

Any parser supported by SGLang can be used. See the [SGLang documentation](https://docs.sglang.ai/) for the full list of available tool call parsers.

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
