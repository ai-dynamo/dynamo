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
- **Tool call ban** — when `tool_choice="none"`, parser-specific strings can be
  excluded so the model cannot complete native tool-call syntax (see
  [trade-offs](#tool_choicenone-and-token-banning)).

## Prerequisites

- A backend engine that accepts structural-tag/xgrammar guided decoding.
- A Dynamo tool call parser that provides a structural tag config (see
  [Supported Parsers](#supported-parsers) below).

## Quick Start

Structural tags are enabled by default. Configure the tool-call parser on the
**worker**; the Frontend needs no extra flags:

> [!NOTE]
> TensorRT-LLM guided decoding remains **opt-in** pending
> [TensorRT-LLM #19913](https://github.com/NVIDIA/TensorRT-LLM/issues/19913).
> Its `--guided-decoding-backend` default is unset, so default deployments ignore
> structural tags and do not enforce their tool argument schemas. For affected
> deployments using `--guided-decoding-backend xgrammar`, pass
> `--no-dyn-enable-structural-tag` until a supported upgrade resolves the issue.

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
| `--dyn-structural-tag` | boolean or JSON object | enabled | Configure structural tags, or pass `false` to disable optional guidance. |

Custom `--dyn-structural-tag` config example:

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

The flag without a value uses the defaults below. Every field is optional:

| Field | Values | Default | Description |
|---|---|---|---|
| `scope` | `auto`, `always` | `always` | Selects eligible tool-calling requests. |
| `schema` | `auto`, `strict` | `auto` | Selects which tool argument schemas are enforced. |
| `allow_tool_calls_with_structured_output` | boolean | `false` | Lets `tool_choice="auto"` choose between tool calls and a schema-constrained final response. Requires parsers v2. |
| `exclude_special_tokens` | boolean, `null` | `null` | Controls reasoning and tool-call marker exclusions. `null` preserves the model-family default. Requires parsers v2. |
| `reasoning_boundary` | `auto`, `structural_tag`, `backend` | `auto` | Selects whether the structural tag closes prompt-opened reasoning or the inference engine activates the post-reasoning grammar. `auto` follows the backend's advertised policy. `backend` requires parsers v2 and a reasoning parser. |
| `tool_arguments_any_order` | boolean | `false` | Allows tool argument properties in any order. This weakens required-property and duplicate-key validation and requires parsers v2 with XGrammar >= 0.2.3. Structured-output schemas are unaffected. |

If the backend advertises that it owns reasoning-aware grammar activation,
explicitly selecting `structural_tag` is rejected to avoid applying both
reasoning gates. Explicit `backend` selection assumes the inference engine is
configured accordingly.

`DYN_STRUCTURAL_TAG` accepts `true`, `false`, or the same JSON object. Unknown
fields and invalid values are rejected during worker startup.

## Preserve Previous Behavior

To disable optional structural guidance, pass `false` to the worker:

```bash
python3 -m dynamo.vllm ... --dyn-structural-tag false
python3 -m dynamo.sglang ... --dyn-structural-tag false
python3 -m dynamo.trtllm ... --dyn-structural-tag false
```

Or set the equivalent environment variable before starting the worker:

```bash
export DYN_STRUCTURAL_TAG=false
```

Rust preprocessing still applies native tags to Kimi K2 `required`/named
choices and Kimi K3 named choices when optional structural tags are disabled.
A lookahead pattern can still be rejected by XGrammar
0.2.1 or 0.2.7 unless the tool sets `strict: false` under schema mode `auto`.
Global schema mode `strict` overrides that workaround. Kimi K2/K3 automatic
choices and Kimi K3 `required` do not use this exception. Python vLLM and SGLang
preprocessing respect the opt-out for all tool choices.

To keep structural tags enabled but preserve the previous conditional activation
policy, set the scope to `auto`:

```bash
export DYN_STRUCTURAL_TAG='{"scope":"auto"}'
```

With this scope, required and named tool choices remain eligible. Automatic tool
choice uses structural tags only when a tool sets `strict: true` or the request
sets `parallel_tool_calls` to `false`.

The legacy `--dyn-enable-structural-tag`, `--no-dyn-enable-structural-tag`,
`--dyn-structural-tag-scope`, and `--dyn-structural-tag-schema` flags and their
environment variables remain accepted with a deprecation warning.

Setting `strict: false` on a tool relaxes its argument schema. On vLLM
0.30.0, pinned by Dynamo's CUDA image, auto choice with every tool explicitly
non-strict also gets no structural tag from the registry; see
[Schema Modes](#schema-modes). Use the global opt-out above to disable optional
structural guidance.

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
- `glm47` with parsers v2

The parsers-v2 builders for `qwen3_coder`, `deepseek_v4`, and `glm47` require
`DYN_PARSER_VERSION=2` in the worker and frontend processes.

Contributions adding structural tag support for new parsers are welcome.

This list describes Dynamo's Rust parser registry. The Python vLLM and SGLang
frontend processors apply the same mode, scope, and schema policy through the
tool parser supplied by their installed engine version. Parser availability can
therefore differ by backend and engine version.

> [!NOTE]
> Native Rust sidecars retain their existing conservative `off`/`auto`
> settings, including the native Kimi forced-tool exception described above,
> and are not included in this default change. The default-on policy
> applies when regular vLLM, SGLang, or TensorRT-LLM workers publish the
> deployment runtime configuration consumed by frontend preprocessing.

## Activation Scope

The `scope` field controls when structural tags are used
based on the request's `tool_choice`:

### `always` (default)

| `tool_choice` | Structural tag? |
|---|---|
| `required` / `named` | Always |
| `auto` | Always attempted; the parser may return no tag |
| `none` | Exclusion tag on the Rust path only |

### `auto` (legacy conditional activation)

| `tool_choice` | Structural tag? |
|---|---|
| `required` / `named` | Always |
| `auto` | Only when any tool has `strict: true` or `parallel_tool_calls` is `false` |
| `none` | Exclusion tag only when tools remain in the prompt (see [below](#tool_choicenone-and-token-banning)) |

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

Dynamo supports `#` and URI-fragment JSON Pointers to schema locations in the same document, including recursive object schemas. A `$ref` that also declares an object type must reach a target that closes the object with `additionalProperties: false`. Remote references, named-anchor references, `$dynamicRef`, `$recursiveRef`, direct reference cycles (a `$ref` chain that returns to itself without passing through a child schema), and references in schemas with nested identifier scopes are outside Dynamo's reference support. Root types that require broader composition analysis are also unsupported. These are Dynamo support limits, not a claim that OpenAI rejects those representations.

Omitted, `null`, or `false` strictness retains existing behavior. Omitted or `null` parameters also retain existing behavior; an explicit `{}` is a supplied schema and fails the object-root check. Dynamo does not normalize or fill in the submitted schema.

Passing preflight does not establish complete OpenAI compatibility or guarantee backend enforcement of every constraint. Structural-tag activation and the deployment schema mode remain separate controls.

The keyword and reference exclusions above are the complete list of keywords rejected solely because they are present. Traversing a keyword such as `propertyNames`, `unevaluatedProperties`, or `prefixItems` checks its nested schemas; it does not establish OpenAI support for that keyword. Backend schema compilation can still reject a schema that passes these checks.

## Schema Modes

The `schema` field controls what JSON schema is used for
tool arguments inside the structural tag:

### `auto` (default)

- Tools with omitted `strict` or `strict: true` — their declared parameter
  schema is used.
- Tools with `strict: false` — argument content is schema-relaxed when the
  parser builds a structural tag.

### `strict`

- All tools use their actual parameter schema regardless of the `strict`
  flag.

If a model-native builder cannot safely represent part of a schema, it keeps
the strongest safe tool envelope and relaxes that argument section. If the
builder cannot produce an automatically added tool-call tag, Dynamo uses the existing
compatibility path rather than introducing a new request error; automatic tool
choice may therefore remain unconstrained for that parser/schema combination.

> [!WARNING]
> Valid tool schemas containing constructs unsupported by the backend's
> XGrammar version, such as regex lookahead, can cause request rejection.
> Backend compilation errors do not trigger Dynamo's builder fallback.
> With XGrammar 0.2.1 or 0.2.7 and schema mode `auto`, setting the tool
> function's `strict: false` avoids this lookahead rejection by disabling
> argument-schema enforcement. This workaround is not guaranteed on older
> versions. The deployment opt-out still retains Rust Kimi K3 named-call tags.

In vLLM 0.30.0, pinned by Dynamo's CUDA image, `tool_choice="auto"` returns no
structural tag when every tool is explicitly `strict: false`. The `always`
activation scope still attempts the tag, but cannot enforce the native tool
envelope for that request. Set schema mode to `strict` to override the opt-out
and let vLLM build the tag.

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
| `true` | on | Tools removed from prompt. Prompt changes break KV cache prefix sharing. |
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
          - '{"scope":"always","schema":"strict"}'
```

## See Also

- [Tool Call Parsing (Dynamo)](tool-call-parsing.mdx) — parser names and basic tool calling setup
- [Chat Processors](chat-processors.mdx) — chat processor and engine-fallback parsers
- [xgrammar Structural Tag Documentation](https://xgrammar.mlc.ai/docs/latest/structural_tag/structural_tag_api.html) — xgrammar format specification
