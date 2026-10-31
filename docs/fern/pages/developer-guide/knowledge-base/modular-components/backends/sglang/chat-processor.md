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
| `--isolate-tool-call-markers` | `false` | Tokenize message content that contains the parser's tool-call delimiters as plain sub-words (see [Echoed delimiters](#echoed-delimiters)) |

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

Any parser supported by SGLang can be used. See the [SGLang documentation](https://docs.sglang.io/) for the full list of available tool call parsers.

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

### Echoed delimiters

Most tool-call formats open and close a call with a delimiter that the tokenizer stores as one added token (`<tool_call>` for GLM, Qwen, Hermes). That token is not reserved: a literal `<tool_call>` inside a tool result or a pasted file encodes to the same id the model emits to open a call. SGLang's detectors work on text, so on their own they cannot tell a call the model issued from a delimiter it reproduced from its input.

The processor uses the token ids to keep the two apart:

- The tool-call parser is opened only by the opening delimiter's token id and closed by the closing id. Delimiter text the model spells out of ordinary sub-words is returned as `content`, never parsed as a call. A call is held from its opening id to its closing id and parsed whole, so a spelled-out delimiter inside the call cannot end it early or start a second call: a call that contains one is returned as `content`. This is always on when both delimiters of the configured parser are single tokens of the served model's tokenizer (Hunyuan's suffixed delimiters are resolved from that tokenizer, as the parser does).
- With `--isolate-tool-call-markers`, message content that contains a delimiter is tokenized without added-token matching, so the model sees `<`, `tool`, `_call`, `>` and has no single token to copy back. Delimiters the chat template itself emits (the tool instructions, earlier assistant `tool_calls`) keep their ids, and content without a delimiter tokenizes exactly as before. A request is rejected instead of being encoded with the delimiter ids when the template rewrites delimiter-bearing content in a way the processor cannot align, or when the parser's delimiters are not single tokens for the tokenizer (the gate above cannot work either). The option changes what the model sees for such content; evaluate it against your model before enabling it in production.

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

- **[Tool Calling](../../../../../use-cases/tool-calling-and-reasoning/tool-call-parsing.mdx)**: General tool calling guide
- **[Reference Guide](reference-guide.md)**: Full SGLang backend reference
- **[Agentic Workloads](agents-on-sglang.md)**: Priority scheduling and KV cache tuning
