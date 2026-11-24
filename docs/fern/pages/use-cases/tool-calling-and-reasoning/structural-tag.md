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

This page uses these terms:

| Term | Meaning |
|---|---|
| Native tool-call format | The markup a model family uses for tool calls, for example `<tool_call>...</tool_call>`. The `--dyn-tool-call-parser` you set on the worker selects it. |
| Scope | Which `tool_choice` values get a structural tag. The default is `always`. See [Tool Choice and Scope](#tool-choice-and-scope). |
| Schema mode | Which tool argument schemas the tag enforces. The default is `auto`. See [Argument Schemas](#argument-schemas). |
| Chat processor | The frontend component that renders prompts and builds tags. The default is `dynamo`. The `vllm` and `sglang` [engine-fallback processors](chat-processors.mdx) build tags through the engine's own tool parser. |

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

| `tool_choice` | Scope `always` (default) | Scope `auto` (legacy) |
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

Schema mode `auto` (default):

- A tool with omitted `strict` or `strict: true` has its declared parameter
  schema enforced.
- A tool with `strict: false` keeps the tool-call envelope, but its argument
  content is relaxed instead of following the declared schema. `strict: false`
  does not disable structural tags; see
  [Disable Structural Tags](#disable-structural-tags). The `vllm` chat processor
  is an exception; see
  [Backend and Path Compatibility](#backend-and-path-compatibility).

Schema mode `strict` enforces every tool's declared parameter schema and
ignores `strict: false`.

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

This fallback covers only tags Dynamo cannot build. A tag the engine cannot
compile can cause the request to be rejected; see
[Unsupported Schema Constructs](#unsupported-schema-constructs).

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

These tags still enforce tool argument schemas, so they are subject to
[Unsupported Schema Constructs](#unsupported-schema-constructs). Automatic
choices for these families, and Kimi K3 `required`, follow the opt-out. The
`vllm` and `sglang` chat processors respect the opt-out for all tool choices.

## Supported Parsers

The `dynamo` chat processor builds structural tags for these
`--dyn-tool-call-parser` values:

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

The default `DYN_PARSER_VERSION=auto` selects the V2 parser route for compatible
parser names. V2 support needs a compatible V2 route on both the frontend and
the worker; see
[Parser Generation Routing](troubleshooting-tool-calls.md#parser-generation-routing).
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

### Unsupported Schema Constructs

A valid tool schema can contain constructs that the backend's XGrammar version
cannot compile, such as a regex lookahead. The engine can then reject the
request. Dynamo's fallback does not apply, because the tag was built
successfully.

XGrammar 0.2.7 and 0.2.8 reject a lookahead when a native-format tag turns a
string `pattern` into a grammar regex. When the pattern stays inside a JSON
Schema, these versions warn and ignore the lookahead instead.

To avoid the rejection, set `strict: false` on the tool under schema mode
`auto`; this works for parsers that honor per-tool strictness. Schema mode
`strict` overrides this workaround, and so does a parameter-level tool
strictness setting with the `sglang` chat processor. The workaround is not
guaranteed on XGrammar 0.1.32. Disabling structural tags does not remove the
native tags listed in [Disable Structural Tags](#disable-structural-tags).

## See Also

- [Tool Call Parsing (Dynamo)](tool-call-parsing.mdx) — parser names, basic tool calling setup, and strict function schemas
- [Chat Processors](chat-processors.mdx) — chat processor and engine-fallback parsers
- [Runtime Configuration](../../reference/components/runtime-configuration.mdx#parsing) — `--dyn-structural-tag` field reference
- [Troubleshooting Tool Calls](troubleshooting-tool-calls.md) — parser route selection and issue reports
- [xgrammar Structural Tag Documentation](https://xgrammar.mlc.ai/docs/latest/structural_tag/structural_tag_api.html) — xgrammar format specification
