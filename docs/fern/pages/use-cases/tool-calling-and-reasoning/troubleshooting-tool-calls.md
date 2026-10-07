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

> [!IMPORTANT]
>  Recipe applies to non-streaming requests against Dynamo's OpenAI `/v1/chat/completions` endpoint. For multi-channel reasoning models (`harmony`, `kimi_k2`, `kimi_k25`, `gemma4`), the recipe recovers only the assistant-content channel; the reasoning channel is not surfaced in `logprobs.content`. If the worker is the SGLang backend, `logprobs: true` is rejected by default because SGLang's tokenizer manager detokenizes top-k tokens serially, causing latency degradation. Set `DYN_SGL_ALLOW_TOP_LOGPROBS=1` in the SGLang worker's `env:` to opt in, then remove it and re-apply once the repro is captured. Tracked at [sgl-project/sglang#24447](https://github.com/sgl-project/sglang/pull/24447).

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

## Parser Generation Routing

To select a parser generation, set `DYN_PARSER_VERSION=1` or `DYN_PARSER_VERSION=2`. Leave it unset or use `auto` to keep the existing family defaults. The aliases `v1` and `v2` are not accepted.

| `DYN_PARSER_VERSION` | Selection |
| --- | --- |
| Unset or `auto` | Existing family defaults |
| `1` | V1 |
| `2` | V2 |

Unsupported values stop startup. Explicit selections validate the configured tool and reasoning parsers and reject combinations that the selected generation cannot support.

| Configured family | Unset or `auto` selection |
| --- | --- |
| Muse Glimmer | V2 |
| DeepSeek V4.1 | V2 when both `tool_call_parser` and `reasoning_parser` are `deepseek_v41` |
| DeepSeek V4 | Original V1 route |
| Gemma 4 | Original V1 route |
| GLM 4.7 | Original V1 route; its reasoning name may be `glm45` |
| Kimi K2 / K2.5 | Original V1 route; configure the matching tool and reasoning names |
| Kimi K3 | Original V1 route |
| Qwen3 Coder | Original V1 route; pair `qwen3_coder` with `qwen3` |
| Other parser names | Existing family-specific route |

Version `2` uses a compatible unified parser for streaming and batch when either the tool-call parser or reasoning parser identifies a supported family. When both are configured, they must resolve to the same family. The DeepSeek V4.1 pair uses V2 by default for streaming and batch. With unset settings or `auto`, Muse streaming requests with named/required tool choices or structural tags retain their existing V1 handling; Muse batch responses use V2. Set `DYN_PARSER_VERSION=2` to require V2 for supported request modes. Tool-only and reasoning-only V1 configurations keep their existing parsers under the original family defaults.

Explicit V1 and V2 apply the same generation selection to streaming and batch. Streaming passes output chunks to the selected parser as they arrive; partial tool-call deltas depend on the family and output shape. Batch parsing waits for the complete response before constructing `message.content`, `message.reasoning_content`, and `message.tool_calls`. `tool_choice: none` suppresses tool calls while the selected content decoder removes recognized native markup from visible content.

Forced tool choices can install a guided JSON constraint, and structural-tag requests keep that constraint when the configured family has a compatible unified parser. A request mode that requires the V1 tool-call jail cannot silently fall back when explicit V2 is selected; Dynamo rejects that unsupported parser configuration before serving requests.

On the default Kimi K3 route, a prompt ending with the XTML reasoning opener `<|open|>think<|sep|>` starts parsing inside the reasoning channel; whitespace between the markers is ignored. This also applies when `DYN_PARSER_VERSION` is unset.

## Known limitations and follow-up

This revision pins the latest published `dynamo-parsers-v2` crate, version 0.7.17. It includes the merged quoted-control and streaming fixes from [frontend-crates #326](https://github.com/ai-dynamo/frontend-crates/pull/326), along with the Kimi tool-ID and call-boundary fixes from [frontend-crates #332](https://github.com/ai-dynamo/frontend-crates/pull/332). The five Dynamo regression tests that were ignored for 0.7.13 are enabled and pass against 0.7.17. The targeted `cargo test --locked -p dynamo-llm quoted_ -- --nocapture` run passed 13 unit tests, one aggregator test, and two integration tests. The every-split Muse structured-response case also passes in `cargo test --locked -p dynamo-llm response_format_explicit_reasoning_without_prompt_prefill -- --nocapture`. These checks do not cover every parser path.

A separate quoted native tool-call opener case can still cut off the rest of an answer under V2, in streaming and batch requests. The exact regression remains follow-up work tracked with related native Qwen3 and Kimi streaming changes in [Dynamo #15577](https://github.com/ai-dynamo/dynamo/pull/15577). With generic `<think>` prompt prefill, splitting `thought</think>` after `thought<` can also leave the remaining tool JSON in reasoning; this behavior is present on the existing V1 route as well. Both cases still need qualification before they can be claimed as fixed.

For configurations supported by V1, select `DYN_PARSER_VERSION=1` to use the legacy parser.

## See also

- [Tool Call Parsing (Dynamo)](tool-call-parsing.mdx) -- Dynamo-native parser names and
  request examples
- [Chat Processors](chat-processors.mdx) --
  `--dyn-chat-processor` and the engine-fallback path to vLLM and SGLang parsers
- [Frontend Configuration Reference](../../reference/components/frontend-configuration.mdx)
  -- full CLI flag reference for the frontend and worker
