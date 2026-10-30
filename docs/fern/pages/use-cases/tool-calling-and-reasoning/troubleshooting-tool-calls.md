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

Set `DYN_PARSER_VERSION` on the frontend running the `dynamo` processor and on workers that build parser-dependent guidance. Workers publish `--dyn-tool-call-parser` and `--dyn-reasoning-parser`; the frontend runs parsing. Restart after changing the environment because version selection is cached. Dynamo uses the parser versions pinned in the workspace [Cargo.toml](https://github.com/ai-dynamo/dynamo/blob/main/Cargo.toml); [Cargo.lock](https://github.com/ai-dynamo/dynamo/blob/main/Cargo.lock) records the resolved versions. The frontend-crates library examples may target a newer release.

| `DYN_PARSER_VERSION` | Compatible Unified configuration | Unavailable or incompatible configuration |
| --- | --- | --- |
| Unset or `auto` | V2, except reasoning-only Qwen/Nemotron below | V1 if each configured name exists in its V1 registry; otherwise reject |
| `1` | V1 if each configured name exists in its V1 registry | Reject names unavailable in V1 |
| `2` | V2 | Reject; no V1 configuration fallback |

Only `auto`, `1`, and `2` are accepted; `v1` and `v2` stop startup. Invalid combinations are rejected during model setup with an error naming the configured selectors and the missing implementation. Runtime V2 parse errors never retry through V1.

V2 uses one family grammar for reasoning, text, and tools. Either selector can usually choose that family; when both are set, they must resolve to the same family or a supported shared grammar. Auto requires both selectors to be `beam` for Beam. V1 uses separate reasoning and tool-call parsers, so it can mix formats when both names exist in their respective V1 registries. This table lists parser availability, not model/backend or GPU qualification.

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
| Muse Spark | `muse_spark`, `muse_spark_open` |
| Nemotron V3 | `nemotron_v3`, `nemotron3` |
| Beam | `beam` |
| GPT-OSS / Harmony | `gpt_oss`, `harmony` |

Examples below list the tool-call selector first, then the reasoning selector:

- `qwen3_coder` + `qwen3`, `glm47` + `glm45`, and `kimi_k2` + `kimi_k25` each resolve to one Unified family, so auto uses V2. `qwen3_coder` + `nemotron_v3` also uses V2 through Nemotron’s shared Qwen XML tool grammar.
- `hermes` + `qwen3` and `qwen3_coder` + `deepseek_v4` mix different formats. Auto uses the V1 split pipeline; explicit `2` rejects them. Use `1` when you want to keep a valid V1 combination on V1.
- `muse_glimmer` + `qwen3` is rejected: it has no matching Unified family and V1 has no `muse_glimmer` tool parser. With `auto` or `2`, use `muse_glimmer` in both selectors, or configure only one selector, to select its Unified grammar.

A single tool selector such as `qwen3_coder`, or reasoning selector such as `deepseek_v41`, selects the whole Unified family grammar. The serving layer still enforces tool policy. Family aliases accepted by V2 are not necessarily registered in the corresponding V1 slot.

**Reasoning-only exceptions:** without a tool selector, auto stays on V1 for `qwen3`, `nemotron_v3`, or `nemotron3` reasoning. Qwen thinking/QwQ models can emit Hermes JSON tools rather than Coder XML; a Nemotron reasoning name alone does not identify its tool grammar. Explicit `2` selects the corresponding Unified grammar; use it only for matching output.

**Deprecated setting:** replace `DYN_ENABLE_EXPERIMENTAL_PARSERS_V2` with `DYN_PARSER_VERSION`. Legacy `true` means explicit V2; `false` means auto, which now prefers compatible V2. Conflicting pairs (`true` + `1`, `false` + `2`) and malformed booleans reject. For rollback, unset the old variable and use `1` if both selectors exist in V1. Muse Glimmer, Muse Spark, Beam, and DeepSeek V4.1 are Unified-only.

Unified emits reasoning, text, and tools in source order. V1 extracts reasoning first and may buffer calls in the tool jail. Streaming exposes incremental output where the family and request mode support it; batch aggregates output into `content`, `reasoning_content`, and `tool_calls`. Both use the same selected generation. With `tool_choice: none`, Unified suppresses calls and decodes recognized markup. V1 bypasses the tool jail for parsers such as `qwen3_coder`, so raw markup may remain in `content`.

Unified resolves prompt-prefilled reasoning after template rendering: an opener in the prompt need not recur in generated text. Kimi K3's `<|open|>think<|sep|>` suffix, including supported whitespace between markers, starts reasoning. This is separate from force-reasoning defaults and engine grammar gates.

For legacy MiniMax M3, explicit `guided_json`, `guided_regex`, `guided_choice`, or `guided_grammar` disables thinking and reasoning parsing when no explicit thinking control is supplied. When reasoning parsing remains enabled, it starts unseeded: `</mm:think>` can separate completed reasoning, but truncation before it may return reasoning as `content`. A prompt opener alone does not set the starting channel on this path.

Forced choices can install guided JSON or family-native structural tags. A mode requiring the V1 jail cannot silently fall back under explicit V2; unsupported combinations are rejected. See [Structural Tags](structural-tag.md) for builder availability and backend activation.

## Known limitations and follow-up

The pinned V2 release includes the quoted-control and streaming fixes from [frontend-crates #326](https://github.com/ai-dynamo/frontend-crates/pull/326) and Kimi tool-ID and call-boundary fixes from [#332](https://github.com/ai-dynamo/frontend-crates/pull/332).

Two cases remain unqualified: literal native tool-call openers inside quoted content, and V1 prompt prefill when `</think>` splits after its first `<`. Existing reasoning-marker tests do not cover them; see [Dynamo #15577](https://github.com/ai-dynamo/dynamo/pull/15577).

For configurations supported by V1, select `DYN_PARSER_VERSION=1` to use the legacy parser.

## See also

- [Tool Call Parsing (Dynamo)](tool-call-parsing.mdx) -- Dynamo-native parser names and
  request examples
- [Chat Processors](chat-processors.mdx) --
  `--dyn-chat-processor` and the engine-fallback path to vLLM and SGLang parsers
- [Frontend Configuration Reference](../../reference/components/frontend-configuration.mdx)
  -- full CLI flag reference for the frontend and worker
