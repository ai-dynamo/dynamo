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
>  Recipe applies to non-streaming requests against Dynamo's OpenAI `/v1/chat/completions` endpoint. For multi-channel reasoning models (`harmony`, `kimi_k2`, `kimi_k25`, `gemma4`), the recipe recovers only the assistant-content channel; the reasoning channel is not surfaced in `logprobs.content`.

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

Set `DYN_PARSER_VERSION` on the frontend process that runs the `dynamo` chat processor, and consistently on worker processes that construct parser-dependent guidance. The worker's `--dyn-tool-call-parser` and `--dyn-reasoning-parser` publish model configuration; the frontend selects and runs the parser. Restart after changing the environment: generation selection is cached for the process lifetime. This revision pins `dynamo-parsers` `10.2.2` and `dynamo-parsers-v2` `0.7.35`, independently of the latest library release.

| `DYN_PARSER_VERSION` | Compatible Unified configuration | Unavailable or incompatible configuration |
| --- | --- | --- |
| Unset or `auto` | V2, except reasoning-only `qwen3` below | Existing V1 route |
| `1` | V1 if both configured names exist in V1 | Reject names unavailable in V1 |
| `2` | V2 | Reject; no V1 configuration fallback |

Only `auto`, `1`, and `2` are accepted; `v1` and `v2` stop startup. Auto configuration fallback does not make an unknown V1 parser valid. A runtime V2 parse error does not retry through V1.

The pinned Unified registry and Dynamo aliases resolve to the families below. Either configured selector can choose a family. When both selectors are set, both must resolve to the same family; the table describes routing availability, not GPU qualification or complete model/backend integration.

| Unified family | Accepted family names and Dynamo aliases |
| --- | --- |
| DeepSeek V4 | `deepseek_v4`, `deepseek-v4`, `deepseekv4` |
| DeepSeek V4.1 | `deepseek_v41` |
| Gemma 4 | `gemma4`, `gemma-4` |
| GLM 4.7 | `glm47`, `glm45` |
| Kimi K2 / K2.5 | `kimi_k2`, `kimi_k25` |
| Kimi K3 | `kimi_k3`, `kimi-k3` |
| Qwen3 Coder | `qwen3`, `qwen3_coder` |
| Muse Glimmer | `muse_glimmer`, `muse` |
| Nemotron V3 | `nemotron_v3`, `nemotron3` |
| Beam | `beam` |
| GPT-OSS / Harmony | `gpt_oss`, `harmony` |

For example, `qwen3_coder` + `qwen3`, `glm47` + `glm45`, and `kimi_k2` + `kimi_k25` are compatible pairs. `hermes` + `qwen3` is not a Unified pair and uses V1 under auto; explicit `2` rejects it. `qwen3_coder` + `deepseek_v4` is also incompatible. Set just `qwen3_coder` to choose its complete Unified grammar from the tool selector, or just `deepseek_v41` to choose that grammar from the reasoning selector. One-sided selection still uses the family's reasoning, text, and tool grammar; request tool policy remains a serving-layer responsibility.

**Reasoning-only Qwen exception:** with no tool selector and exactly `--dyn-reasoning-parser qwen3`, unset/auto stays on V1. Qwen thinking/QwQ models can use Hermes JSON tool syntax, which differs from Coder XML. Explicit `2` selects the Coder Unified grammar even for reasoning-only `qwen3`. Set the Coder tool selector only for a matching output format.

**Deprecated setting:** remove `DYN_ENABLE_EXPERIMENTAL_PARSERS_V2` when migrating to `DYN_PARSER_VERSION`. Legacy `true` maps to explicit V2; `false` maps to auto and now prefers compatible V2. `true` with version `1`, `false` with version `2`, and malformed booleans reject. For rollback, unset the deprecated variable and use `1` only when both selectors exist in V1; Unified-only families such as Muse Glimmer, Beam, and DeepSeek V4.1 cannot use V1.

Unified owns one ordered reasoning, text, and tool event stream. The legacy split route extracts reasoning before tool parsing and may buffer calls in the jail. Streaming feeds chunks incrementally; partial call deltas depend on family and request mode. Non-streaming responses aggregate the selected parser's output into `message.content`, `message.reasoning_content`, and `message.tool_calls`; they do not select a different generation. On a Unified route, `tool_choice: none` suppresses structured calls while the content decoder handles recognized markup. On V1, `none` disables the tool jail for parsers such as `qwen3_coder`, so native tool markup may remain in `content`.

On Unified routes, Dynamo resolves prompt-prefilled reasoning after template rendering, so an opener consumed in the prompt need not appear again in generated text. For example, Kimi K3's `<|open|>think<|sep|>` suffix starts reasoning, including supported whitespace between markers. This is distinct from force-reasoning defaults and engine-native grammar gates.

Legacy MiniMax M3 has a separate constrained-generation policy. With an explicit `guided_json`, `guided_regex`, `guided_choice`, or `guided_grammar` constraint and no explicit thinking control, Dynamo renders thinking disabled and skips reasoning parsing. If thinking is explicitly enabled, the reasoning parser remains unseeded for these constraints: a completed `</mm:think>` can separate reasoning, but output ending before that closer can be returned as `content`. Do not infer the starting channel solely from the prompt opener on this path.

Forced choices can install guided JSON or family-native structural tags. A mode requiring the V1 jail cannot silently fall back under explicit V2; unsupported combinations are rejected. See [Structural Tags](structural-tag.md) for builder availability and backend activation.

## Known limitations and follow-up

This revision pins `dynamo-parsers-v2` version 0.7.35. The earlier 0.7.17 update included the quoted-control and streaming fixes from [frontend-crates #326](https://github.com/ai-dynamo/frontend-crates/pull/326) and the Kimi tool-ID and call-boundary fixes from [frontend-crates #332](https://github.com/ai-dynamo/frontend-crates/pull/332). Targeted parser regressions do not establish qualification of every parser path.

The existing quoted-control regressions cover reasoning markers and leave literal native tool-call opener text as a qualification gap. The generic prompt-prefill regression also does not qualify V1 with a closing `</think>` split after its first `<`. Related work is in [Dynamo #15577](https://github.com/ai-dynamo/dynamo/pull/15577). These coverage gaps do not establish that either failure occurs on every current route, and passing nearby reasoning-marker tests does not establish that they are fixed.

For configurations supported by V1, select `DYN_PARSER_VERSION=1` to use the legacy parser.

## See also

- [Tool Call Parsing (Dynamo)](tool-call-parsing.mdx) -- Dynamo-native parser names and
  request examples
- [Chat Processors](chat-processors.mdx) --
  `--dyn-chat-processor` and the engine-fallback path to vLLM and SGLang parsers
- [Frontend Configuration Reference](../../reference/components/frontend-configuration.mdx)
  -- full CLI flag reference for the frontend and worker
