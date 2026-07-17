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
> serially, causing latency degradation. Set
> `DYN_SGL_ALLOW_TOP_LOGPROBS=1` in the SGLang worker's `env:` to opt in,
> then remove it and re-apply once the repro is captured. Tracked at
> [sgl-project/sglang#24447](https://github.com/sgl-project/sglang/pull/24447).

## The request

Add `"logprobs": true` to your failing request. Port-forward the Frontend Service first (`kubectl port-forward svc/<deployment-name>-frontend 8000:8000 -n ${NAMESPACE}`):

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

- [Tool Call Parsing (Dynamo)](README.mdx) -- Dynamo-native parser names and
  request examples
- [Chat Processors](chat-processors.mdx) --
  `--dyn-chat-processor` and the engine-fallback path to vLLM and SGLang parsers
- [Frontend Configuration Reference](../components/frontend/frontend-config-reference.mdx)
  -- full CLI flag reference for the frontend and worker
