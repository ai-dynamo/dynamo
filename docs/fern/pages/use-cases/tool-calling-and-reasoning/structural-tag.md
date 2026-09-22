---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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


## Request Validation

Dynamo validates supplied function parameter schemas with explicit `strict: true` before inference on `/v1/chat/completions` and `/v1/responses`. Invalid schemas now return HTTP 400 where earlier versions could accept them. This validation runs even when structural tags are disabled or `tool_choice` is `none`. Responses checks all submitted functions, including namespace members and functions excluded by `allowed_tools`.

Use an object-only root. Close each object with `additionalProperties: false` and include every named property in `required`. These object checks also apply to nullable nested objects and schemas that declare `properties` or `patternProperties`. For example:

```json
{
  "type": "object",
  "properties": {
    "query": {"type": "string", "minLength": 1}
  },
  "required": ["query"],
  "additionalProperties": false
}
```

Preflight checks schema shapes, object constraints, nesting depth, and size budgets. It rejects root `anyOf`, the composition keywords `allOf`, `oneOf`, `not`, `dependentRequired`, `dependentSchemas`, `if`, `then`, and `else`, and the array keywords `uniqueItems`, `contains`, `minContains`, `maxContains`, and `unevaluatedItems` at any schema location. String patterns and formats, numeric bounds, `minItems`, and `maxItems` can pass preflight.

Each supplied schema can declare at most 5,000 properties and 1,000 enum entries. Property names, definition names, string enum values, and string const values together can contain at most 120,000 Unicode characters. An all-string enum with more than 250 entries has a separate 15,000-character limit. Unused definitions count toward these budgets; repeated references do not add counts.

Object schemas can be nested at most 10 levels below the root object. Only object schemas add a level; array, `anyOf`, and other schema-bearing keywords add none of their own. Dynamo counts literal nesting in the submitted document: a recursive reference does not add levels, and an object under `$defs` starts one level below the root, like a root property. OpenAI documents the same ten-level limit without stating whether the root counts; Dynamo does not count the root.

Dynamo supports `#` and URI-fragment JSON Pointers to schema locations in the same document, including recursive object schemas. Remote references, named-anchor references, `$dynamicRef`, `$recursiveRef`, reference-only cycles, and references in schemas with nested identifier scopes are outside Dynamo's reference support. Root types that require broader composition analysis are also unsupported. These are Dynamo support limits, not a claim that OpenAI rejects those representations.

Omitted, `null`, or `false` strictness retains existing behavior. Omitted or `null` parameters also retain existing behavior; an explicit `{}` is a supplied schema and fails the object-root check. Dynamo does not normalize or fill in the submitted schema.

Passing preflight does not establish complete OpenAI compatibility or guarantee backend enforcement of every constraint. Structural-tag activation and the deployment schema mode remain separate controls.

The keyword and reference exclusions above are the complete list of keywords rejected solely because they are present. Traversing a keyword such as `propertyNames`, `unevaluatedProperties`, or `prefixItems` checks its nested schemas; it does not establish OpenAI support for that keyword. Backend schema compilation can still reject a schema that passes these checks.

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

- [Tool Call Parsing (Dynamo)](tool-call-parsing.mdx) — parser names and basic tool calling setup
- [Chat Processors](chat-processors.mdx) — chat processor and engine-fallback parsers
- [xgrammar Structural Tag Documentation](https://xgrammar.mlc.ai/docs/latest/structural_tag/structural_tag_api.html) — xgrammar format specification
