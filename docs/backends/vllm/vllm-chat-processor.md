---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: vLLM Chat Processor
subtitle: vLLM-native preprocessing and postprocessing for chat completions
---

The vLLM chat processor enables vLLM-native preprocessing and postprocessing in the Dynamo frontend. It uses vLLM's tokenizer, chat templates, tool call parser, and reasoning parser directly -- bypassing the default Rust preprocessor for `v1/chat/completions` requests.

## When to Use

Use `--dyn-chat-processor vllm` when Dynamo's built-in Rust preprocessor does not yet support a tool call parser or reasoning parser you need. The vLLM processor delegates to vLLM's Python implementations, so any parser vLLM supports works immediately.

Common cases:

- A **tool call format** not yet in the Rust `tool_calling` library
- A **reasoning parser** not yet supported natively
- A **chat template** that the Rust preprocessor doesn't handle correctly

If the parser you need is missing from the Rust preprocessor, consider [opening an issue or PR](https://github.com/ai-dynamo/dynamo/issues) to add native support -- native parsers avoid the Python GIL overhead entirely.

## Quick Start

```bash
# Frontend with vLLM processor, tool calling, and reasoning
python -m dynamo.frontend \
  --router-mode kv \
  --dyn-chat-processor vllm \
  --enable-auto-tool-choice \
  --tool-call-parser hermes \
  --reasoning-parser qwen3

# Workers (unchanged)
CUDA_VISIBLE_DEVICES=0 python -m dynamo.vllm \
  --model Qwen/Qwen3-14B-FP8 \
  --served-model-name Qwen/Qwen3-14B-FP8 \
  --tensor-parallel-size 1 --trust-remote-code
```

## Frontend Arguments

These arguments are passed to the **frontend** (not the worker) when using `--dyn-chat-processor vllm`. The frontend forwards unknown arguments to vLLM's own CLI parser (`AsyncEngineArgs` and `FrontendArgs`), so any vLLM frontend or engine flag is accepted.

| Argument | Default | Description |
|----------|---------|-------------|
| `--dyn-chat-processor vllm` | (none) | Enable the vLLM chat processor |
| `--tool-call-parser` | `None` | Tool call parser name (any vLLM-supported parser) |
| `--reasoning-parser` | `None` | Reasoning parser name (any vLLM-supported parser) |
| `--enable-auto-tool-choice` | `False` | Allow the model to emit tool calls without an explicit `tool_choice` in the request |

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `DYN_VLLM_STREAM_INTERVAL` | `20` | Number of tokens to accumulate before detokenizing. Higher values improve throughput. The first chunk always emits immediately (interval=1) to minimize time-to-first-token. |

## Tool Calling

The processor supports all vLLM tool call formats. Pass `--tool-call-parser` (and typically `--enable-auto-tool-choice`) on the frontend:

```bash
python -m dynamo.frontend \
  --dyn-chat-processor vllm \
  --enable-auto-tool-choice \
  --tool-call-parser hermes
```

Any parser supported by vLLM can be used. See the [vLLM documentation](https://docs.vllm.ai/) for the full list of available tool call parsers.

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
  --dyn-chat-processor vllm \
  --reasoning-parser qwen3
```

The parser separates think tag content into the `reasoning_content` field and regular content into the `content` field.

## See Also

- **[Tool Calling](../../tool-calling/README.md)**: General tool calling guide
- **[Reference Guide](vllm-reference-guide.md)**: Full vLLM backend reference
- **[Examples](vllm-examples.md)**: vLLM deployment examples
