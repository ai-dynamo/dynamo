---
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Troubleshooting Tool Calls
subtitle: Capture raw model output with logprobs so issues can be localized
---

When a tool call comes back wrong (`tool_calls` is `null`, the arguments
look malformed, raw `<tool_call>` markers appear in `message.content`, or
`finish_reason` is `"stop"` when you expected `"tool_calls"`), the request
and response alone usually do not say *where* the bug is. The model and the
parser produce indistinguishable failures from the response side.

Adding `"logprobs": true` to a single repro request makes the engine's raw
token output visible in the response. That is enough for someone on the
Dynamo team to identify whether the issue is in the model, the parser
configuration, or the parser itself. This page shows the field to add and
what the response will look like, so you can capture and share useful
diagnostic info.

> [!NOTE]
> Recipe applies to non-streaming requests against Dynamo's OpenAI
> `/v1/chat/completions` endpoint. For multi-channel reasoning models
> (`harmony`, `kimi_k2`, `kimi_k25`, `gemma4`), the recipe recovers only
> the assistant-content channel; the reasoning channel is not surfaced in
> `logprobs.content`.
>
> If the worker is the SGLang backend, `logprobs: true` is rejected by
> default because SGLang's tokenizer manager detokenizes top-k tokens
> serially, causing latency degradation. Launch the worker with
> `DYN_SGL_ALLOW_TOP_LOGPROBS=1` set in the environment to opt in for the
> duration of the repro request, then unset it afterward. Tracked at
> [sgl-project/sglang#24447](https://github.com/sgl-project/sglang/pull/24447).

## The request

Add `"logprobs": true` to your failing request:

```bash
curl -s http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "Qwen/Qwen2.5-7B-Instruct",
    "messages": [
      {"role": "user", "content": "What is the weather in NYC?"}
    ],
    "tools": [{
      "type": "function",
      "function": {
        "name": "get_weather",
        "parameters": {
          "type": "object",
          "properties": {
            "location": {"type": "string"},
            "unit": {"enum": ["celsius", "fahrenheit"]}
          },
          "required": ["location"]
        }
      }
    }],
    "tool_choice": "auto",
    "temperature": 0.0,
    "logprobs": true
  }'
```

## The response

You will get back the usual fields (`message.tool_calls`, `message.content`,
`finish_reason`) plus a new `choices[0].logprobs.content` field carrying the
engine's raw token stream:

```json
{
  "choices": [{
    "finish_reason": "tool_calls",
    "message": {
      "role": "assistant",
      "content": null,
      "tool_calls": [{
        "type": "function",
        "function": {
          "name": "get_weather",
          "arguments": "{\"location\":\"New York, NY\",\"unit\":\"fahrenheit\"}"
        }
      }]
    },
    "logprobs": {
      "content": [
        {"token": "<tool_call>", "bytes": [60, 116, 111, 111, 108, 95, 99, 97, 108, 108, 62]},
        {"token": "\n", "bytes": [10]},
        {"token": "{\"", "bytes": [123, 34]},
        {"token": "name", "bytes": [110, 97, 109, 101]},
        "...",
        {"token": "</tool_call>", "bytes": [60, 47, 116, 111, 111, 108, 95, 99, 97, 108, 108, 62]}
      ]
    }
  }]
}
```

Each entry in `logprobs.content` is one generated token with its exact UTF-8
`bytes`. Concatenating those bytes in order reconstructs the raw model
output, before any tool-call parser touched it. That is the key piece for
triage: it tells us what the model actually produced, separately from what
the parser made of it.

## Parser debug mode

When a parse failure shows up on only a small percentage of requests, the
decisive artifact is one raw model completion next to its parsed output.
Setting `DYN_PARSER_DEBUG=1` makes the frontend emit exactly that: one
structured log event per choice at stream end, pairing the raw pre-parse
completion text with the parsed `reasoning_content`, `content`, and
`tool_calls`.

> **Opt-in only.** Raw completions carry user data, so this mode is off by
> default. Enable it only in environments where logging request content is
> acceptable.

Run the frontend with both flags set (JSONL makes the event queryable):

```bash
DYN_PARSER_DEBUG=1 DYN_LOGGING_JSONL=true python -m dynamo.frontend ...
```

Each event carries `stream_id`, `choice_index`, `reasoning_parser`,
`tool_parser`, `raw`, `raw_chunks`, `reasoning_content`, `content`,
`tool_call_count`, `tool_call_names`, and `finish_reason` as separate
fields. `raw` is the full pre-parse completion text; `raw_chunks` is the
same text split exactly as the stream deltas arrived -- streaming parse
failures are often chunk-boundary-dependent, so the boundaries are what
lets you replay and reproduce the failure.

A real failure shape: the model emits a well-formed tool call, but a
delta boundary splits the closing `</think>` marker and the reasoning
parser swallows the entire tool call into `reasoning_content` --
`tool_calls` stays empty and `finish_reason` is `stop`. The event shows
`raw` containing the intact `<function=...>` markup the parsers
destroyed, and `raw_chunks` shows the exact split needed to reproduce:

```json
{"target":"parser_debug","fields":{"stream_id":"chatcmpl-abc","choice_index":0,"reasoning_parser":"nemotron_v3","tool_parser":"qwen3_coder","raw":"<think>pick a command</think>\n<tool_call>\n<function=terminal>...","raw_chunks":"[\"<think>pick a command<\", \"/think>\\n<tool_call>\\n<\", \"function=terminal>...\"]","reasoning_content":"pick a command</think>\n<tool_call>\n<function=terminal>...","content":"","tool_call_count":0,"tool_call_names":"[]","finish_reason":"Stop"}}
```

One grep over the frontend log surfaces every request where parsing
produced no tool calls despite tool-call-shaped raw text:

```bash
grep '"target":"parser_debug"' frontend.jsonl | jq 'select(.fields.tool_call_count == 0)'
```

## What to include when reporting an issue

Share these four things in the bug report or issue thread:

1. **The full request body** (model name, messages, tools, sampling params,
   and `logprobs: true`).
2. **The full response body.** Do not truncate `logprobs.content` -- the
   per-token entries are the part that matters.
3. **The Dynamo version and the backend** (vLLM, SGLang, TRT-LLM, including
   versions if known).
4. **The worker launch command**, especially the `--dyn-tool-call-parser`
   value if set.

With those four pieces, the Dynamo team can usually localize the bug
without standing up your model. The team will reconstruct the raw stream
from the `bytes` arrays and compare it against `message.content` and
`message.tool_calls` to decide whether the issue is in the model output,
the parser configuration, or the parser logic.

## See also

- [Tool Call Parsing (Dynamo)](README.md) -- Dynamo-native parser names and
  request examples
- [Parser Engine Fallback](engine-fallback.md) --
  `--dyn-chat-processor` fallback path to vLLM and SGLang parsers
- [Frontend Configuration Reference](../components/frontend/configuration.md)
  -- full CLI flag reference for the frontend and worker
