---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Structural Tag (Guided Decoding for Tool Calls)
subtitle: Constrain model output to valid tool call format using xgrammar structural tags
---

Structural tags constrain a model's tool calls while the model generates them.
For each tool-calling request, Dynamo builds an
[XGrammar structural tag](https://xgrammar.mlc.ai/docs/latest/structural_tag/structural_tag_api.html)
that describes the model's native tool-call format and the request's tools. The
inference engine uses it to block tokens that would break that format, so the
tool-call parser receives well-formed calls. For most parsers, the tag also
limits calls to the request's declared tools; see
[Supported Parsers](#supported-parsers) for exceptions.

## How Structural Tags Work

Suppose a model served with `--dyn-tool-call-parser qwen3_coder` receives a
request that declares one tool and lets the model decide whether to call it:

```json
{
  "tools": [{
    "type": "function",
    "function": {
      "name": "get_weather",
      "parameters": {
        "type": "object",
        "properties": {"city": {"type": "string"}},
        "required": ["city"]
      }
    }
  }],
  "tool_choice": "auto"
}
```

Without a structural tag, the model can emit a tool name the request did not
declare, an unclosed tool-call marker, or a call without `city`. The parser can
only report or drop that output after the model generates it.

With a structural tag, the model can still answer in plain text. Once it starts
a tool call with `<tool_call>`, the engine allows only `get_weather`, the closing markers,
and, by default, arguments that follow the declared schema. With
`parallel_tool_calls: false`, the tag also stops after the first call.

The **native tool-call format** is the markup a model family uses, for example
`<tool_call>...</tool_call>`. The **chat processor** renders prompts and builds
tags for that format. The `dynamo` processor uses Dynamo's parsers; the `vllm`
and `sglang` [engine-fallback processors](chat-processors.mdx) use the engine's
own tool parser.

### Settings and Defaults

| Setting | Where to set it | Default and purpose |
|---|---|---|
| `--dyn-tool-call-parser` | Worker | Unset. Selects the native tool-call format for the `dynamo` chat processor. |
| `--dyn-structural-tag` or `DYN_STRUCTURAL_TAG` | Worker | Enabled on regular workers. Accepts a boolean or JSON object. Sidecar defaults differ; see [Backend and Path Compatibility](#backend-and-path-compatibility). |
| `scope` field in `--dyn-structural-tag` | Worker JSON, for example `'{"scope":"always"}'` | `always`. Selects which request `tool_choice` values get a tag; see [Tool Choice and Scope](#tool-choice-and-scope). |
| `schema` field in `--dyn-structural-tag` | Worker JSON, for example `'{"schema":"auto"}'` | `auto`. Selects which tool argument schemas the tag enforces; see [Argument Schemas](#argument-schemas). |
| `--dyn-chat-processor` or `DYN_CHAT_PROCESSOR` | Frontend | `dynamo`. Selects Dynamo's processor or an engine-fallback processor. |
| `DYN_PARSER_VERSION` | Frontend and workers that build parser-dependent guidance | `auto`. Prefers a compatible V2 parser route; see [Supported Parsers](#supported-parsers). |
| `tool_choice` | Request body | `auto` when tools are present. Lets the model choose whether to call a tool. |
| `tools[].function.strict` | Request body, per tool | Omitted. Under schema mode `auto`, Dynamo enforces the declared schema with the regex exceptions below. |

Scope `auto`, schema mode `auto`, parser version `auto`, and request
`tool_choice="auto"` are separate settings.

## Quick Start

Structural tags are enabled by default. Set `--dyn-tool-call-parser` on the
**worker** to a parser listed in [Supported Parsers](#supported-parsers). The
frontend needs no extra flags.

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

On TensorRT-LLM workers, also set `--guided-decoding-backend xgrammar`. Without
it, TensorRT-LLM ignores structural tags. See
[Backend and Path Compatibility](#backend-and-path-compatibility).

## How Requests Are Constrained

### Tool Choice and Scope

Set scope through the worker's `--dyn-structural-tag` JSON, for example
`'{"scope":"auto"}'`. The table compares that worker setting with the
request's `tool_choice`:

| Request `tool_choice` | Worker scope `always` (default) | Worker scope `auto` (legacy) |
|---|---|---|
| `required` or named | Tag | Tag |
| `auto` | Tag attempted; the model can still answer without calling a tool | Tag only when a tool sets `strict: true` or the request sets `parallel_tool_calls: false` |
| `none` | Exclusion tag with the `dynamo` chat processor, only when tools stay in the prompt | Same as `always` |

A tag is attempted only when the parser has structural-tag support. The
[`none` section](#tool_choicenone-and-token-banning) explains when the exclusion
tag applies.

### `tool_choice="none"` and Token Banning

By default, Dynamo honors `tool_choice="none"` by removing tool definitions from
the prompt and returning no structured `tool_calls`. The
`--exclude-tools-when-tool-choice-none` worker flag (default `true`) controls
whether tool definitions stay in the prompt.

With the `dynamo` chat processor, when structural tags are enabled and tool
definitions stay in the prompt, `none` also gets an exclusion tag. The tag bans
parser-specific tool-call start tokens, for example `<tool_call>`, so the model
cannot start native tool-call syntax. This applies in both scopes. The `vllm`
and `sglang` chat processors never add the exclusion tag; they handle `none`
through prompt and response shaping only.

| `exclude-tools-when-tool-choice-none` | Structural tags | Effect |
|---|---|---|
| `true` (default) | off | Tool definitions removed from the prompt. Earlier tool calls in the conversation can still lead the model to attempt one. The prompt differs from turns that include tools, which reduces KV cache prefix reuse. |
| `true` | on | Same as above; no exclusion tag is added. |
| `false` | on | Tool definitions stay in the prompt; the exclusion tag blocks native tool-call markers. More of the prompt prefix matches turns that include tools. |
| `false` | off | Tool definitions stay in the prompt; no token ban. Responses have no structured `tool_calls` for explicit `none`, but tool-like text can appear in `content`. |

Keeping tool definitions in the prompt can improve KV cache reuse when
`tool_choice` changes between turns, but it does not guarantee cache hits. With
the exclusion tag, the model still sees the tools but cannot start a tool call,
so answers can contain awkward phrasing or tool-like fragments. Removing tool
definitions, the default, usually gives cleaner answers. The effect depends on
the model and workload.

For multi-turn conversations where `tool_choice` changes between turns and KV
cache reuse matters, consider `--no-exclude-tools-when-tool-choice-none` with
structural tags enabled, and check answer quality for your model.

### Argument Schemas

Set schema mode through the worker's `--dyn-structural-tag` JSON, for example
`'{"schema":"auto"}'`. In schema mode `auto` (the default):

- A tool with omitted `strict` has its declared parameter schema enforced,
  except that Dynamo leaves out regex constraints the backend cannot compile.
  An unsupported `pattern` is removed from the argument grammar; an unsupported
  `patternProperties` key relaxes that tool's entire argument schema. See
  [Regex constructs the backend cannot compile](#regex-constructs-the-backend-cannot-compile).
- A tool with `strict: true` keeps its declared parameter schema, including
  regex constraints. A backend compilation error can reject the request.
- A tool with `strict: false` keeps the tool-call envelope, but its argument
  content is relaxed instead of following the declared schema. The `vllm`
  chat processor is an exception; see
  [Backend and Path Compatibility](#backend-and-path-compatibility).

Schema mode `strict` enforces every tool's declared parameter schema, keeps
regex constraints, and ignores `strict: false`. Per-tool `strict: false` does
not disable structural tags; see
[Disable Structural Tags](#disable-structural-tags).

If part of a schema cannot be represented safely in the native format, Dynamo
keeps the strongest safe tool-call envelope and relaxes that argument section.

Separately, the frontend validates `strict: true` schemas against OpenAI's strict
subset before inference. See
[Strict Function Schemas](tool-call-parsing.mdx#strict-function-schemas).

### Structured Output in the Same Request

With the `dynamo` chat processor, structured-output fields change the policy
above:

- For parsers other than Beam and Mistral 4, a JSON `response_format` with
  `tool_choice="auto"` gets a tool-call tag only when
  `allow_tool_calls_with_structured_output` is `true`; otherwise only the
  response-format constraint applies. With `tool_choice="none"`, no exclusion
  tag is added.
- Explicit `guided_json`, `guided_regex`, `guided_grammar`, or nonempty
  `guided_choice` constraints cannot be combined with `required` or named tool
  choice (HTTP 400). With `auto`, they suppress additional tool-call structural
  tags regardless of `allow_tool_calls_with_structured_output`. Mistral 4 can
  merge `guided_json` into its native grammar. Null values and an empty
  `guided_choice` array do not activate this rule.

### When Dynamo Cannot Build a Tag

A parser without structural-tag support keeps its existing best-effort
tool-calling behavior, and the request is not rejected. If Dynamo cannot build
an automatically added tag for a request's schemas, it uses the same
compatibility path, so automatic tool choice can remain unconstrained for that
parser and schema combination.

With the `dynamo` chat processor, two families reject the request instead:

- Muse Spark `required` and named requests return HTTP 400 if their native tag
  cannot be built.
- Beam returns HTTP 400 when it cannot install its grammar for a request that
  asks for enforcement: a forced tool choice, `parallel_tool_calls: false`, or
  a `strict: true` tool that the tool choice allows. For example, the schema is
  outside Beam's grammar, the prompt leaves the final channel open, or explicit
  guided decoding owns the output.

This fallback covers only tags Dynamo cannot build. Dynamo filters unsupported
regex constraints from automatically enforced argument schemas, but a tag the
engine still cannot compile can cause the request to be rejected. See
[Regex constructs the backend cannot compile](#regex-constructs-the-backend-cannot-compile).

## Configure Structural Tags

Set `--dyn-structural-tag` or `DYN_STRUCTURAL_TAG` on the worker:

| Value | Effect |
|---|---|
| Flag omitted, or `true` | Enabled with scope `always` and schema mode `auto`. Some sidecars default to off when the flag is omitted; see [Backend and Path Compatibility](#backend-and-path-compatibility). |
| `false` | Optional structural tags disabled; see [Disable Structural Tags](#disable-structural-tags) |
| JSON object | Enabled with the given options. Omitted fields keep their defaults. |

For example, to enforce every declared argument schema:

```bash
python3 -m dynamo.sglang ... --dyn-structural-tag '{"schema":"strict"}'
```

To keep structural tags enabled with the previous conditional activation policy,
set the scope to `auto`:

```bash
export DYN_STRUCTURAL_TAG='{"scope":"auto"}'
```

Unknown fields and invalid values are rejected during worker startup. The
[Parsing reference](../../reference/components/runtime-configuration.mdx#parsing)
lists every field, including `allow_tool_calls_with_structured_output`,
`exclude_special_tokens`, `reasoning_boundary`, and `tool_arguments_any_order`.

The legacy `--dyn-enable-structural-tag`, `--no-dyn-enable-structural-tag`,
`--dyn-structural-tag-scope`, and `--dyn-structural-tag-schema` flags and their
environment variables remain accepted on Python workers with a deprecation
warning. The native sidecars do not accept them.

### Disable Structural Tags

To disable optional structural tags, pass `false` to the worker:

```bash
python3 -m dynamo.vllm ... --dyn-structural-tag false
python3 -m dynamo.sglang ... --dyn-structural-tag false
python3 -m dynamo.trtllm ... --dyn-structural-tag false
dynamo-sglang-sidecar ... --dyn-structural-tag false
dynamo-vllm-sidecar ... --dyn-structural-tag false
```

Or set the equivalent environment variable before starting the worker:

```bash
export DYN_STRUCTURAL_TAG=false
```

With the `dynamo` chat processor, including deployments that use the native
sidecars, some model families keep their native tags when structural tags are
disabled, because their native formats have no generic fallback:

- Kimi K2 `required` and named choices
- Kimi K3 named choices
- Muse Spark `required` and named choices

These tags use the same schema policy, including the automatic regex adjustment
when `strict` is omitted. Explicit strictness can still cause a backend
compilation error; see
[Regex constructs the backend cannot compile](#regex-constructs-the-backend-cannot-compile).
Automatic choices for these families, and Kimi K3 `required`, follow the
opt-out. The `vllm` and `sglang` chat processors respect the opt-out for all tool
choices.

## Supported Parsers

The `dynamo` chat processor builds structural tags for these
`--dyn-tool-call-parser` values:

**V2 (Unified)** parses reasoning, text, and tool calls with one model-family
grammar. **V1** uses separate reasoning and tool-call parsers.
`DYN_PARSER_VERSION=auto` (the default) prefers V2 when the configured tool and
reasoning selectors form a compatible Unified configuration. If they do not,
`auto` uses V1 only when every configured selector is available in V1;
otherwise model setup is rejected. Set `DYN_PARSER_VERSION=1` to select V1, or
`2` to require V2 without configuration fallback.

Set `DYN_PARSER_VERSION` on the frontend and on workers that build
parser-dependent guidance. Restart those processes after changing it because
version selection is cached. A **V2 parser route only** entry below is
unavailable with explicit `1`, or with a mixed tool/reasoning pair that selects
V1. See
[Parser Generation Routing](troubleshooting-tool-calls.md#parser-generation-routing)
for compatible pairs, aliases, and reasoning-only exceptions.

| Parser | Notes |
|---|---|
| `beam` | V2 parser route only. Under `auto`, a call can also name an undeclared tool with free-form arguments. Can return HTTP 400 when it cannot enforce a requested constraint; see [When Dynamo Cannot Build a Tag](#when-dynamo-cannot-build-a-tag). |
| `deepseek_v3_2`, `deepseek_v4`, `deepseek_v41` | `deepseek_v41`: V2 parser route only |
| `glm47` | V1 and V2 parser routes |
| `gpt_oss`, `harmony` | V2 parser route only. With structural tags enabled, a `required` request that allows parallel calls gets a tag whose payload is one JSON array of calls, not native Harmony calls. |
| `hermes`, `qwen25` | |
| `inkling` | |
| `kimi_k2`, `kimi_k3`, `kimi-k3` | Some choices keep their tags when structural tags are disabled; see [Disable Structural Tags](#disable-structural-tags) |
| `minimax_m3`, `minimax-m3` | V2 parser route only |
| `muse_spark`, `muse_spark_open` | V2 parser route only. Some choices keep their tags when structural tags are disabled. |
| `nemotron_nano`, `qwen3_coder` | |

When the selected V2 family has no structural-tag support, Dynamo uses the tool
parser's V1 support if it has one; this does not change which parser reads the
response. The V2 library has no structural-tag support for `muse_glimmer`,
`gemma4`, or `nemotron_v3`.

The `vllm` and `sglang` chat processors use the tool parser of their installed
engine version, so their parser support can differ from this table.

Contributions adding structural tag support for new parsers are welcome.

### Check Availability from Python

Output-parser availability does not guarantee structural-tag support. To check
a parser name, query the registered structural-tag support:

```python
from dynamo._core import get_structural_tag_parser_names

parser_name = "deepseek_v41"
if parser_name not in get_structural_tag_parser_names():
    raise ValueError(f"No structural-tag grammar registered for {parser_name}")
```

`get_structural_tag_parser_names()` returns a sorted, deduplicated list across
the V1 and V2 parsers. It does not select a parser route, validate a tool and
reasoning parser pair, or check the backend's XGrammar version.

## Backend and Path Compatibility

Dynamo builds the tag; the inference engine compiles it with XGrammar and
enforces it. Support therefore depends on the worker type, the chat processor,
and the engine version. Versions below are the ones Dynamo's images pin.

| Worker | Default | Notes |
|---|---|---|
| vLLM (`dynamo.vllm`) | On | vLLM 0.33.0 bundles XGrammar 0.2.8. |
| SGLang (`dynamo.sglang`) | On | SGLang 0.5.23 bundles XGrammar 0.2.7. |
| TensorRT-LLM (`dynamo.trtllm`) | On, but ignored unless `--guided-decoding-backend xgrammar` is set | The backend default is unset, so default deployments ignore structural tags and do not enforce tool argument schemas. TensorRT-LLM 1.4.0rc3 bundles XGrammar 0.2.8. Releases that bundle XGrammar 0.1.32 reject several tool-call tag formats ([TensorRT-LLM #19913](https://github.com/NVIDIA/TensorRT-LLM/issues/19913)); on those releases, pass `--dyn-structural-tag false`. |
| vLLM sidecar (`dynamo-vllm-sidecar`) | On for vLLM 0.33 or later, including release candidates and development builds; off for older vLLM | Older vLLM gRPC servers reject these tags. |
| SGLang sidecar (`dynamo-sglang-sidecar`) | On | |
| TensorRT-LLM sidecar (`dynamo-trtllm-sidecar`) | Off | Enable with `--dyn-structural-tag` or `DYN_STRUCTURAL_TAG=true`. |
| LPX sidecar (`dynamo-lpx-sidecar`) | Not supported | Rejects any `--dyn-structural-tag` value other than `false`. |

The `vllm` and `sglang` chat processors apply the same scope and schema policy,
with these differences:

- Neither adds the `tool_choice="none"` exclusion tag.
- The `vllm` chat processor builds tags through vLLM's tool parser registry. In
  vLLM 0.33.0, that registry returns no structural tag for `tool_choice="auto"`
  when every tool is explicitly `strict: false`. The `always` scope still
  attempts the tag, but cannot enforce the native tool envelope for that
  request. To override the opt-out and let vLLM build the tag, set schema mode
  to `strict`.
- The `sglang` chat processor keeps the tool-call envelope for `strict: false`
  tools. A parameter-level tool strictness setting in SGLang enforces every
  tool's arguments regardless of `strict`.

## Limitations

### Regex constructs the backend cannot compile

A valid JSON Schema can contain regex syntax that XGrammar cannot compile.
Before building a tool-call tag, Dynamo checks `pattern` values and
`patternProperties` keys. The Python vLLM and SGLang preprocessors check each
regex with the installed XGrammar's regex converter. The Rust preprocessor
checks for constructs that XGrammar rejects in at least one tag format:

- lookahead `(?=` `(?!` and lookbehind `(?<=` `(?<!`
- backreferences `\1`–`\9` and `\k<name>`
- group modifiers other than `(?:` and `(?<name>`, such as `(?i)`, `(?>`, and
  `(?P<name>`
- word boundaries `\b` `\B` and Unicode property escapes `\p{…}` `\P{…}`
  outside a character class
- the empty class `[]`, a `{` that does not open `{n}`, `{n,}`, or `{n,m}`
  with counts up to 10^15, and a quantifier followed by another quantifier
  other than one lazy `?`

The adjustment depends on the worker's schema mode and the tool's `strict`
field:

| Worker schema mode | Tool `strict` | Effect of a detected unsupported regex |
|---|---|---|
| `auto` | Omitted | Remove an unsupported `pattern` from the argument grammar and retain the remaining schema. If a `patternProperties` key is unsupported, relax that tool's entire argument schema, as with `strict: false`. |
| `auto` | `true` | Keep the declared schema; backend compilation can reject the request. |
| `auto` | `false` | Arguments are already schema-relaxed. |
| `strict` | Any value or omitted | Keep the declared schema; backend compilation can reject the request. |

These adjustments affect grammar construction; the request's schema stays
unchanged. Other tools retain their own schema policy. When the parser builds
a tag, its tool-call envelope and tool-name policy remain. The `vllm`
chat processor can return no tag when every tool becomes `strict: false`; see
[Backend and Path Compatibility](#backend-and-path-compatibility).
The same policy applies to the native tags retained by
[Disable Structural Tags](#disable-structural-tags). Adjustments are logged at
debug level.

The regex check is conservative across tag formats. XGrammar 0.2.7 and 0.2.8
accept some leading lookaheads, such as `^(?!…)` or `^(?=…)`, when the pattern
stays inside a JSON Schema. Their regex converter rejects lookahead, and
lookahead elsewhere in a JSON Schema pattern can still fail. Do not assume
that the backend ignores every lookahead. Automatic guidance can therefore
omit a regex that a particular tag format would accept.

If a `strict: true` tool fails backend compilation, either rewrite the regex
for the installed XGrammar or set that tool's `strict` to `false` under schema
mode `auto`, if relaxing argument validation is acceptable. Schema mode
`strict` overrides this opt-out, as does a parameter-level tool strictness
setting with the `sglang` chat processor. The workaround is not guaranteed on
XGrammar 0.1.32. Disabling structural tags does not remove the native tags
listed in [Disable Structural Tags](#disable-structural-tags).

## See Also

- [Tool Call Parsing (Dynamo)](tool-call-parsing.mdx) — parser names, basic tool calling setup, and strict function schemas
- [Chat Processors](chat-processors.mdx) — chat processor and engine-fallback parsers
- [Runtime Configuration](../../reference/components/runtime-configuration.mdx#parsing) — `--dyn-structural-tag` field reference
- [Troubleshooting Tool Calls](troubleshooting-tool-calls.md) — parser route selection and issue reports
- [xgrammar Structural Tag Documentation](https://xgrammar.mlc.ai/docs/latest/structural_tag/structural_tag_api.html) — xgrammar format specification
