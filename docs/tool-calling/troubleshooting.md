---
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Troubleshooting Tool Calls
subtitle: Inspect raw model output with logprobs to localize tool-call issues
---

When a tool call comes back wrong -- `tool_calls` is `null` when you expected
a function call, the arguments are malformed, or raw `<tool_call>` markers
appear inside `message.content` -- the request and response alone cannot tell
you whether the model produced bad text or a downstream parser mangled good
text. This page shows how to add `logprobs` to a single repro request so you
can see exactly what the engine emitted, independent of the parser.

For the happy-path setup and parser names, see
[Tool Call Parsing (Dynamo)](dynamo.md) and
[Tool Call Parsing (Engine Fallback)](engine-fallback.md).

> [!NOTE]
> **Scope.** This recipe applies to non-streaming requests against Dynamo's
> OpenAI `/v1/chat/completions` endpoint. Streaming responses, the Responses
> API, the Anthropic-API mode (`--enable-anthropic-api`), and the kserve gRPC
> frontend serialize logprobs differently and are not covered.
>
> For multi-channel reasoning models (`harmony`, `kimi_k2`, `kimi_k25`,
> `gemma4`), the recipe recovers only the assistant-content channel. The
> reasoning-content channel is not surfaced in `logprobs.content` because the
> OpenAI schema has no equivalent `reasoning_logprobs` field. If the model
> emits a tool call on a reasoning channel, the reconstructed stream will
> look truncated even when the model is fine.

## When to use this page

Re-run the failing request with `logprobs` enabled when any of these are true:

- `message.tool_calls[].function.arguments` is malformed JSON or omits
  expected fields.
- `message.content` contains raw `<tool_call>...</tool_call>` markers (or the
  family-equivalent wrapper) instead of being `null`.
- `finish_reason` is `"stop"` when you expected `"tool_calls"`.
- A customer has shared only the request and response and you need to triage
  remotely without standing up their model.

## The technique

Add one field to the request:

```json
{
  "logprobs": true
}
```

This makes the server include `choices[0].logprobs.content` in the response:
an array with one entry per generated token, each carrying the surface-form
`token` string and a `bytes` array (the exact UTF-8 sequence). Concatenating
those bytes reconstructs the raw output stream verbatim, before any
tool-call parser touched it.

The flag is intended to be observationally equivalent: on the same prompt
and sampling parameters, `finish_reason`, `content`, and `tool_calls` should
not change. In practice some backends take slightly different sampling
paths under `logprobs=true` (vLLM routes through a separate SamplingParams
branch; SGLang can disable certain CUDA-graph fast paths). If you suspect
logprobs is perturbing the bug itself, capture one request with and one
without, diff the non-logprobs fields, and check `usage`.

Added cost is response size, roughly 50-100 bytes per generated token, so
this is intended for one-off debug requests, not production traffic.

Edge case: at end-of-stream when Dynamo's parser jail drains undecided
bytes, the trailing tokens may have `logprobs: null` even when requested.
If the final entry of `logprobs.content` is missing, the wrapper-marker
close (for example, `</tool_call>`) may have been emitted past the jail
flush point. Read the captured tokens up to that point as still
authoritative.

## Worked example

This uses the same Qwen2.5 / `get_weather` setup as the
[Dynamo parser example](dynamo.md#examples), with `logprobs` added.

### Request

```bash
curl -s http://localhost:8080/v1/chat/completions \
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

### Response

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
        {"token": "\":", "bytes": [34, 58]},
        {"token": " \"get", "bytes": [32, 34, 103, 101, 116]},
        {"token": "_weather", "bytes": [95, 119, 101, 97, 116, 104, 101, 114]},
        {"token": "\",", "bytes": [34, 44]},
        "...",
        {"token": "</tool_call>", "bytes": [60, 47, 116, 111, 111, 108, 95, 99, 97, 108, 108, 62]}
      ]
    }
  }]
}
```

### Reconstruct the raw stream

```python
import json

response = json.load(open("response.json"))
entries = response["choices"][0]["logprobs"]["content"]

# Sanity check: confirm the backend populates `bytes`. If it's null or
# empty, the backend version does not surface byte-level logprobs; fall
# back to the `token` field (lossy for non-ASCII).
assert entries[0].get("bytes"), "backend did not populate logprobs.bytes"

raw_bytes = b"".join(bytes(e["bytes"]) for e in entries)
print(raw_bytes.decode("utf-8"))
```

Output:

```
<tool_call>
{"name": "get_weather", "arguments": {"location": "New York, NY", "unit": "fahrenheit"}}
</tool_call>
```

That string is the model's exact output -- bytes that the engine produced
before any tool-call parser stripped wrapper tokens or lifted JSON into
`message.tool_calls`. Compare it to the rendered `message.content` and
`message.tool_calls` to localize where things diverged.

## Before the ladder: quick pre-checks

Three classes of failure look like model or parser bugs but are not. Rule
them out first:

1. **Truncation.** If `finish_reason == "length"`, the model hit
   `max_tokens` or the context window mid-generation. The raw stream
   looking "garbled" is just incomplete output. Raise `max_tokens` or
   shorten the prompt and re-run.

2. **`tool_choice="required"` with a wrapped format.** When
   `tool_choice="required"`, Dynamo's preprocessor expects bare JSON, not
   `<tool_call>`-wrapped JSON. If the model emits wrapper markers anyway,
   parsing fails because the contract differs, not because the parser is
   broken. Set `tool_choice="auto"` if the model is trained to wrap.

3. **Hallucinated tool name.** If `tool_calls[].function.name` is populated
   but does not match any name in your request's `tools[]`, the model
   invented a function. That is a prompt or model issue, not a parser
   issue. Lower temperature, add few-shot examples, or constrain the model
   upstream.

If none of these apply, work the ladder below.

## Diagnostic ladder

Once you have the raw stream from `logprobs.content` and the parsed view from
`message.*`, work the three checks in order:

### 1. Does the detokenized stream itself look garbled?

If the raw bytes are nonsense -- truncated mid-token, off-grammar, or wrong
language -- the issue is upstream of any parser. The model produced bad
tokens.

- **Likely causes:** prompt formatting (missing chat template), sampling
  parameters (temperature too high, top_p too low, repetition penalty
  fighting the JSON syntax), checkpoint version mismatch, custom Jinja
  template not rendering tool definitions correctly.
- **Where to fix:** request side. Check the worker's `--custom-jinja-template`
  if you set one, drop temperature toward 0 for a deterministic repro, and
  verify the prompt rendering matches the model card.

### 2. Is the raw stream well-formed but `message.content` still has wrapper markers?

If `logprobs.content` decodes to a valid `<tool_call>...</tool_call>` (or
family-equivalent) block, but `message.content` contains those wrapper
markers as literal text and `message.tool_calls` is `null` or absent, no
parser is engaged.

- **Likely causes:** worker started without `--dyn-tool-call-parser`, or
  the wrong parser name for the model family.
- **Known trap:** if you are using engine fallback
  (`--dyn-chat-processor vllm` or `sglang`) AND disaggregated serving, the
  engine-fallback parser is silently not engaged. See the warning in
  [Tool Call Parsing (Engine Fallback)](engine-fallback.md). Do not proceed
  to step 3 in that case.
- **Where to fix:** restart the worker with the right parser:

  ```bash
  python -m dynamo.<backend> --model <name> \
    --dyn-tool-call-parser <family>
  ```

  See the parser-name table in
  [Tool Call Parsing (Dynamo)](dynamo.md#supported-tool-call-parsers).

### 3. Is the raw stream well-formed but `tool_calls.arguments` is mangled?

If `logprobs.content` decodes to a clean JSON object inside the wrapper but
`message.tool_calls[].function.arguments` differs from it -- fields dropped,
escapes wrong, sibling keys merged -- a parser ran and produced the wrong
output.

- **Likely causes:** parser bug specific to this model family, or a parser
  configured against the wrong family.
- **Where to localize:** diff the JSON visible in `logprobs.content` against
  `tool_calls[].function.arguments` character by character. The diff shows
  exactly what the parser dropped or rewrote.
- **Where to report:** open an issue with both blobs side by side. The Dynamo
  parser source lives at `lib/parsers/src/tool_calling/<family>/` and the
  cross-impl parity harness at
  [tests/parity/README.md](../../tests/parity/README.md) records the
  contract.

## Pipeline view

The recipe works because each stage in the response pipeline has its own
observable in the response, and `logprobs.content` sits **upstream** of the
parser:

```
model emits tokens
      |
      v
logprobs.content   <-- ground truth
                       (parser does not touch this)
      |
      v
tool-call parser
(consumes wrapper markers,
 extracts JSON into tool_calls)
      |
      v
message.content    <-- what parser left in text
                       (null/empty if parsed,
                        raw markers if no parser engaged)
message.tool_calls <-- what parser extracted as structure
                       (populated if parsed,
                        null/absent if no parser engaged)
finish_reason      <-- "tool_calls" iff extraction succeeded
```

Diffs between adjacent layers localize the bug to a single stage. The same
recipe applies whether the parser lives in Dynamo's Rust preprocessor (the
default), in the SGLang `FunctionCallParser`, or in vLLM's auto-tool-choice
path -- `logprobs.content` is OpenAI-standard and sits upstream of all
three.

## See also

- [Tool Call Parsing (Dynamo)](dynamo.md) -- Dynamo-native parser names and
  request examples
- [Tool Call Parsing (Engine Fallback)](engine-fallback.md) -- `--dyn-chat-processor`
  fallback path to vLLM / SGLang parsers
- [Frontend Configuration Reference](../components/frontend/configuration.md)
  -- Full CLI flag reference for the frontend and worker
- [Cross-impl parity test suite](../../tests/parity/README.md) -- Where the
  parser contract is recorded across Dynamo, vLLM, and SGLang
