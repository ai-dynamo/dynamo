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
- **Tool call ban** — when `tool_choice="none"`, specific tokenizer tokens can be
  banned so the model cannot start native tool-call syntax (see
  [trade-offs](#tool_choicenone-and-token-banning)).

## Prerequisites

- A backend engine that accepts structural-tag/xgrammar guided decoding.
- A Dynamo tool call parser that provides a structural tag config (see
  [Supported Parsers](#supported-parsers) below).

## Quick Start

Structural tags are enabled by default. Configure the tool-call parser on the
**worker**; the Frontend needs no extra flags:

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
```

Eligible tool-calling requests will now use xgrammar structural tags for guided
decoding. See [Activation Scope](#activation-scope) for the exact policy.

## CLI Flags

| Flag | Values | Default | Description |
|---|---|---|---|
| `--dyn-enable-structural-tag` | bool | `true` | Master switch for the Rust, Python vLLM, and Python SGLang frontend preprocessing paths. Explicitly disabling it prevents tool-call structural-tag injection. |
| `--dyn-structural-tag-scope` | `auto`, `always` | `always` | Controls when structural tags are activated (see [Activation Scope](#activation-scope)). |
| `--dyn-structural-tag-schema` | `auto`, `strict` | `auto` | Controls parameter schema strictness inside structural tags (see [Schema Modes](#schema-modes)). |

## Supported Parsers

Not all parsers support structural tags. Parsers without a structural-tag
builder fall back to their existing best-effort tool-calling behavior without
rejecting the request.

Currently tested and supported:

- `qwen3_coder`, `nemotron_nano`
- `hermes`, `qwen25`
- `deepseek_v3_2`, `deepseek_v4`
- `kimi_k2`, `kimi_k3`, `kimi-k3`
- `inkling`

Contributions adding structural tag support for new parsers are welcome.

This list describes Dynamo's Rust parser registry. The Python vLLM and SGLang
frontend processors apply the same mode, scope, and schema policy through the
tool parser supplied by their installed engine version. Parser availability can
therefore differ by backend and engine version.

> [!NOTE]
> Native Rust sidecars retain their existing conservative `off`/`auto`
> settings and are not included in this default change. The default-on policy
> applies when regular vLLM or SGLang workers publish the deployment runtime
> configuration consumed by frontend preprocessing.

## Activation Scope

The `--dyn-structural-tag-scope` flag controls when structural tags are used
based on the request's `tool_choice`:

### `always` (default)

| `tool_choice` | Structural tag? |
|---|---|
| `required` / `named` | Always |
| `auto` | Always |
| `none` | Exclusion tag on the Rust path only |

### `auto` (legacy conditional activation)

| `tool_choice` | Structural tag? |
|---|---|
| `required` / `named` | Always |
| `auto` | Only when any tool has `strict: true` or `parallel_tool_calls` is `false` |
| `none` | Exclusion tag on the Rust path only (see [below](#tool_choicenone-and-token-banning)) |

## Schema Modes

The `--dyn-structural-tag-schema` flag controls what JSON schema is used for
tool arguments inside the structural tag:

### `auto` (default)

- Tools with omitted `strict` or `strict: true` — their declared parameter
  schema is used.
- Tools with `strict: false` — the native tool envelope remains constrained,
  but argument content is schema-relaxed.

### `strict`

- All tools use their actual parameter schema regardless of the `strict`
  flag.

If a model-native builder cannot safely represent part of a schema, it keeps
the strongest safe tool envelope and relaxes that argument section. If the
builder cannot produce a structural tag at all, Dynamo uses the existing
compatibility path rather than introducing a new request error; automatic tool
choice may therefore remain unconstrained for that parser/schema combination.

## `tool_choice="none"` and Token Banning

On the Rust frontend preprocessing path, when `tool_choice="none"` and
structural tags are enabled, Dynamo injects an exclusion structural tag that
bans parser-specific tool-call start tokens (for example `<tool_call>`) so the
model cannot start native tool-call syntax. The Python vLLM and SGLang frontend
processors continue to handle `none` through their existing prompt and response
shaping; this release does not add token banning to those paths.

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
