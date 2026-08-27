---
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Structural Tag (Guided Decoding for Tool Calls)
subtitle: Constrain model output to valid tool call format using xgrammar structural tags
---

Structural tags use [xgrammar](https://xgrammar.mlc.ai/docs/latest/structural_tag/structural_tag_api.html)
guided decoding to constrain model output to a valid tool call format at the
token level. Instead of hoping the model produces well-formed tool calls,
structural tags enforce the expected format by restricting the decoding
vocabulary at each generation step.

Benefits:

- **Format guarantee** — model output always matches the parser's expected
  tool call syntax (begin/end tags, parameter structure).
- **Schema enforcement** — tool arguments can be constrained to the function's
  JSON schema.
- **Single-call enforcement** — `parallel_tool_calls=false` is enforced via
  `stop_after_first` in the grammar, not just by convention.
- **Tool call ban** — when `tool_choice="none"`, parser-specific strings can be
  excluded so the model cannot complete native tool-call syntax (see
  [trade-offs](#tool_choicenone-and-marker-exclusion)).

## Prerequisites

- A backend engine with xgrammar support.
- A Dynamo tool call parser that provides a structural tag config (see
  [Supported Parsers](#supported-parsers) below).

## Quick Start

Enable structural tags on the **worker** with `--dyn-structural-tag`, alongside the tool-call parser. The Frontend needs no extra flags:

```yaml
apiVersion: nvidia.com/v1beta1
kind: DynamoGraphDeployment
metadata:
  name: qwen35-structural-tag
spec:
  components:
  - name: Frontend
    type: frontend
    replicas: 1
    podTemplate:
      spec:
        containers:
        - name: main
          image: ${RUNTIME_IMAGE}
  - name: SGLangWorker
    type: worker
    replicas: 1
    podTemplate:
      spec:
        containers:
        - name: main
          image: ${RUNTIME_IMAGE}
          envFrom:
          - secretRef:
              name: hf-token-secret
          command:
          - python3
          - -m
          - dynamo.sglang
          args:
          - --model-path
          - Qwen/Qwen3.5-4B
          - --served-model-name
          - Qwen/Qwen3.5-4B
          - --dyn-tool-call-parser
          - qwen3_coder
          - --dyn-structural-tag
```

Eligible tool-calling requests will now use xgrammar structural tags for guided
decoding. See [Activation Scope](#activation-scope) for the exact policy.

## CLI Flags

| Flag | Values | Default | Description |
|---|---|---|---|
| `--dyn-structural-tag` | optional JSON object | unset | Enable structural tags, optionally with advanced configuration. |

The flag without a value uses the defaults below. Every field is optional:

```json
{
  "scope": "always",
  "schema": "strict",
  "allow_tool_calls_with_structured_output": true,
  "exclude_special_tokens": false,
  "reasoning_boundary": "backend",
  "tool_arguments_any_order": true
}
```

| Field | Values | Default | Description |
|---|---|---|---|
| `scope` | `auto`, `always` | `auto` | Selects eligible tool-calling requests. |
| `schema` | `auto`, `strict` | `auto` | Selects which tool argument schemas are enforced. |
| `allow_tool_calls_with_structured_output` | boolean | `false` | Lets `tool_choice="auto"` choose between tool calls and a schema-constrained final response. Requires parsers v2. |
| `exclude_special_tokens` | boolean, `null` | `null` | Controls reasoning and tool-call marker exclusions. `null` preserves the model-family default. Requires parsers v2. |
| `reasoning_boundary` | `structural_tag`, `backend` | `structural_tag` | Selects whether the structural tag closes prompt-opened reasoning or the inference engine activates the post-reasoning grammar. `backend` requires parsers v2 and backend support. |
| `tool_arguments_any_order` | boolean | `false` | Allows tool argument properties in any order. This weakens required-property and duplicate-key validation and requires parsers v2. Structured-output schemas are unaffected. |

`DYN_STRUCTURAL_TAG` accepts `true`, `false`, or the same JSON object. Unknown
fields and invalid values are rejected during worker startup.

## Supported Parsers

Not all parsers support structural tags. Parsers without a structural tag
config fall back to standard behaviour (a warning is logged if structural
tags are enabled but the parser does not support them).

Currently tested and supported:

- `qwen3_coder`, `nemotron_nano`
- `hermes`, `qwen25`
- `deepseek_v3_2`, `deepseek_v4`
- `glm47` with parsers v2

The parsers-v2 builders for `qwen3_coder`, `deepseek_v4`, and `glm47` require
`DYN_ENABLE_EXPERIMENTAL_PARSERS_V2=1` in the worker and frontend processes.

Contributions adding structural tag support for new parsers are welcome.

## Activation Scope

The `scope` field controls when structural tags are used
based on the request's `tool_choice`:

### `auto` (default)

| `tool_choice` | Structural tag? |
|---|---|
| `required` / `named` | Always |
| `auto` | Only when any tool has `strict: true` or `parallel_tool_calls` is `false` |
| `none` | Exclusion tag only (excludes tool-call markers, see [below](#tool_choicenone-and-marker-exclusion)) |

### `always`

| `tool_choice` | Structural tag? |
|---|---|
| `required` / `named` | Always |
| `auto` | Always |
| `none` | Exclusion tag only |


## Schema Modes

The `schema` field controls what JSON schema is used for
tool arguments inside the structural tag:

### `auto` (default)

- Tools with `strict: true` — their actual parameter schema is used.
- Tools without `strict` — an unconstrained schema is used, allowing
  the model to generate any valid content in the parser's native format.

### `strict`

- All tools use their actual parameter schema regardless of the `strict`
  flag.

## `tool_choice="none"` and Marker Exclusion

When `tool_choice="none"` and structural tags are enabled, Dynamo injects an
exclusion structural tag that excludes parser-specific tool-call markers (for
example `<tool_call>`) so the model cannot complete native tool-call syntax.

**Quality trade-off**. If tools remain in the prompt on `none` (often via
`--no-exclude-tools-when-tool-choice-none` to keep the chat prefix stable for KV
reuse) while bans block tool-call tokens, the model still sees tools but cannot
complete valid tool-call text.

Answers may suffer: awkward phrasing, tool-like fragments, or other artifacts.

You choose between a stable shared prefix with KV reuse versus omitting tools from the prompt on `none` (default), which usually yields cleaner chat output but changes the prefix and weakens KV reuse when `tool_choice` varies. How much this matters depends on the model and workload.

This interacts with the `--exclude-tools-when-tool-choice-none` flag (default:
`true`), which strips tool definitions from the chat template when
`tool_choice="none"`:

| `exclude-tools-when-tool-choice-none` | Structural tag | Effect |
|---|---|---|
| `true` (default) | off | Tools removed from prompt. Model doesn't know about tools. Prompt changes break KV cache prefix sharing. |
| `true` | on | Tools removed from prompt; tool-call markers are also excluded. Prompt changes break KV cache prefix sharing. |
| `false` | on | Tools stay in prompt; guided decoding excludes tool-call markers. Model sees tools but cannot complete a native tool-call opening. Stable KV cache prefix across different `tool_choice` values. |
| `false` | off | Tools stay in prompt; no token ban. Same response shaping as above: no structured `tool_calls` for explicit `none`. Tool-like text may still appear in `content`. |

For multi-turn conversations where `tool_choice` changes between turns,
consider `--no-exclude-tools-when-tool-choice-none` combined with
`--dyn-structural-tag` to keep the prompt stable and benefit from
KV cache reuse.

## Example

To pin the scope and schema, pass a JSON value to `--dyn-structural-tag`:

```yaml
  - name: SGLangWorker
    type: worker
    replicas: 1
    podTemplate:
      spec:
        containers:
        - name: main
          image: ${RUNTIME_IMAGE}
          envFrom:
          - secretRef:
              name: hf-token-secret
          command:
          - python3
          - -m
          - dynamo.sglang
          args:
          - --model-path
          - Qwen/Qwen3.5-4B
          - --served-model-name
          - Qwen/Qwen3.5-4B
          - --dyn-tool-call-parser
          - qwen3_coder
          - --dyn-structural-tag
          - '{"scope":"always","schema":"strict","allow_tool_calls_with_structured_output":true}'
```

## See Also

- [Tool Call Parsing (Dynamo)](tool-call-parsing.mdx) — parser names and basic tool calling setup
- [Chat Processors](chat-processors.mdx) — chat processor and engine-fallback parsers
- [xgrammar Structural Tag Documentation](https://xgrammar.mlc.ai/docs/latest/structural_tag/structural_tag_api.html) — xgrammar format specification
