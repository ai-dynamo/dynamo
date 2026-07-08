---
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Structural Tag (Guided Decoding for Tool Calls)
subtitle: Constrain model output to valid tool call format using xgrammar structural tags
---

Structural tags use [xgrammar](https://xgrammar.mlc.ai/docs/structural_tag/structural_tag_api.html)
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
- **Tool call ban** — when `tool_choice="none"`, specific tokenizer tokens can be
  banned so the model cannot start native tool-call syntax (see
  [trade-offs](#tool_choicenone-and-token-banning)).

## Prerequisites

- A backend engine with xgrammar support.
- A Dynamo tool call parser that provides a structural tag config (see
  [Supported Parsers](#supported-parsers) below).

## Quick Start

Enable structural tags on the **worker** with `--dyn-enable-structural-tag`, alongside the tool-call parser. The Frontend needs no extra flags:

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
          - --dyn-enable-structural-tag
```

Eligible tool-calling requests will now use xgrammar structural tags for guided
decoding. See [Activation Scope](#activation-scope) for the exact policy.

## CLI Flags

| Flag | Values | Default | Description |
|---|---|---|---|
| `--dyn-enable-structural-tag` | bool | `false` | Master switch. When disabled, tool calling works the same as without structural tags. |
| `--dyn-structural-tag-scope` | `auto`, `always` | `auto` | Controls when structural tags are activated (see [Activation Scope](#activation-scope)). |
| `--dyn-structural-tag-schema` | `auto`, `strict` | `auto` | Controls parameter schema strictness inside structural tags (see [Schema Modes](#schema-modes)). |

## Supported Parsers

Not all parsers support structural tags. Parsers without a structural tag
config fall back to standard behaviour (a warning is logged if structural
tags are enabled but the parser does not support them).

Currently tested and supported:

- `qwen3_coder`, `nemotron_nano`
- `hermes`, `qwen25`
- `deepseek_v3_2`, `deepseek_v4`

Contributions adding structural tag support for new parsers are welcome.

## Activation Scope

The `--dyn-structural-tag-scope` flag controls when structural tags are used
based on the request's `tool_choice`:

### `auto` (default)

| `tool_choice` | Structural tag? |
|---|---|
| `required` / `named` | Always |
| `auto` | Only when any tool has `strict: true` or `parallel_tool_calls` is `false` |
| `none` | Exclusion tag only (bans tool call tokens, see [below](#tool_choicenone-and-token-banning)) |

### `always`

| `tool_choice` | Structural tag? |
|---|---|
| `required` / `named` | Always |
| `auto` | Always |
| `none` | Exclusion tag only |


## Schema Modes

The `--dyn-structural-tag-schema` flag controls what JSON schema is used for
tool arguments inside the structural tag:

### `auto` (default)

- Tools with `strict: true` — their actual parameter schema is used.
- Tools without `strict` — an unconstrained schema is used, allowing
  the model to generate any valid content in the parser's native format.

### `strict`

- All tools use their actual parameter schema regardless of the `strict`
  flag.

## `tool_choice="none"` and Token Banning

When `tool_choice="none"` and structural tags are enabled, Dynamo injects an
exclusion structural tag that bans parser-specific tool-call start tokens (for
example `<tool_call>`) so the model cannot start native tool-call syntax.

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
| `true` | on | Tools removed from prompt; tokens also banned. Prompt changes break KV cache prefix sharing. |
| `false` | on | Tools stay in prompt; guided decoding bans tokens. Model sees tools but cannot emit banned openings. Stable KV cache prefix across different `tool_choice` values. |
| `false` | off | Tools stay in prompt; no token ban. Same response shaping as above: no structured `tool_calls` for explicit `none`. Tool-like text may still appear in `content`. |

For multi-turn conversations where `tool_choice` changes between turns,
consider `--no-exclude-tools-when-tool-choice-none` combined with
`--dyn-enable-structural-tag` to keep the prompt stable and benefit from
KV cache reuse.

## Example

To pin the scope and schema, add `--dyn-structural-tag-scope` and `--dyn-structural-tag-schema` to the worker `args:` alongside the parser and master switch:

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
          - --dyn-enable-structural-tag
          - --dyn-structural-tag-scope
          - always
          - --dyn-structural-tag-schema
          - strict
```

## See Also

- [Tool Call Parsing (Dynamo)](README.md) — parser names and basic tool calling setup
- [Parser Engine Fallback](engine-fallback.md) — upstream engine parsers
- [xgrammar Structural Tag Documentation](https://xgrammar.mlc.ai/docs/structural_tag/structural_tag_api.html) — xgrammar format specification
