# Reasoning Parser Corner Cases

Reference taxonomy for unit testing **reasoning** parsers under
`src/reasoning/` (Granite, GPT-OSS, Gemma, Qwen3 think-tag, Minimax,
DeepSeek V3 think-tag, etc.). Sibling files cover adjacent stages:

- **Tool-call parsers** (`src/tool_calling/`): see `PARSER_CASES.md`.
- **Frontend gating**: see
  `components/src/dynamo/frontend/tests/FRONTEND_CASES.md`.
- **Pipeline boundary**: see `PIPELINE_CASES.md`.

The reasoning taxonomy mirrors the parser taxonomy on stage / mode /
format axes:

- **Stage** — `REASONING.*` (this file).
- **Mode** — `batch` (entire model output as one string) or `stream`
  (incremental `delta_text`).
- **Format** — `PARSER.fmt|xml|harmony.*` tags from `PARSER_CASES.md`
  also attach to reasoning tests when the reasoning parser consumes
  the corresponding format. The format tag describes the *grammar*,
  not the parser stage.

Each `#[test]` carries one or more `// REASONING.<tag>` annotations
and (where applicable) format tags. `N/A` is called out explicitly.

## Quick reference

### Reasoning parser family groups

Several "model parsers" share the same reasoning grammar. Group them
first, otherwise we copy the same delimiter cases into many model
rows and miss the contract that is actually being tested.

| Group | Dynamo parser families / aliases | Grammar / contract | Notes |
|---|---|---|---|
| Explicit `<think>` block | `qwen3`, `deepseek_v4`, `nemotron_deci`, `glm45`, `basic` | Reasoning is inside `<think>...</think>`. Text outside the markers is normal text. | Most reusable cases are `REASONING.batch.{1,2,3,4,5,6}` and `REASONING.stream.{1,2,3,4}`. |
| Force-reasoning `<think>` block | `deepseek_r1`, `deepseek_v3`, `deepseek_v3_1`, `deepseek_v3_2`, `step3`, `nemotron_nano`, `nemotron3`, `nemotron_v3` | Same delimiter, but generation may start already inside reasoning. Marker-free text can be reasoning until an end marker or stop condition. | Reuses the explicit `<think>` cases, but the plain-text/no-start behavior is different. |
| Kimi K2.5 / K2.6 force-reasoning block | `kimi_k25` | `<think>` reasoning with a tool-section marker that can end an open reasoning span. | Keep separate from generic force-reasoning because tool-section boundaries are part of the reasoning contract. |
| Kimi Unicode delimiter block | `kimi` | `◁think▷...◁/think▷` | Same block semantics as `<think>`, different marker spelling and more Unicode boundary risk. |
| Mistral bracket block | `mistral` | `[THINK]...[/THINK]` | Same paired-marker cases with bracket tokens. |
| Gemma channel block | `gemma4` | `<|channel>thought...<channel|>` | Needs prefix-disambiguation coverage: `thought` vs similar prefixes and partial channel markers. |
| Granite phrase block | `granite` | `Here is my thought process:` followed by `Here is my response:` | IBM Granite 3.x / Granite 3.2 language model format; phrase-prefix buffering matters because partial phrases look like ordinary text until enough bytes arrive. |
| Harmony channel block | `gpt_oss` | Harmony channels, especially `analysis`, `final`, `commentary`, and tool-call envelopes. | Some upstream tests are token/tag-level rather than plain text; convert what is portable and mark token-ID-only cases separately. |
| Append-think content wrapper | `minimax_append_think` | Does not expose reasoning in the same way; vLLM prepends/preserves `<think>` as output content. | Treat as its own contract instead of forcing it into the normal "separate reasoning text" model. |

### Reasoning mode and frontend gates

Reasoning parser behavior is not determined by the parser family alone.
There are static parser choices, request-time gates, and frontend-specific
knobs that decide whether the reasoning parser runs at all.

```text
OpenAI chat request
  -> parser selection
       --dyn-reasoning-parser <name>     Dynamo-native Rust parser
       --reasoning-parser <name>         vLLM/SGLang fallback parser
       unset                             no reasoning parser
  -> request gates
       tool_choice=required/named        skip reasoning parser before tool call jail
       SGLang separate_reasoning=false   leave thinking text in normal content
  -> thinking controls
       enable_thinking=false             disables several explicit-marker families
       thinking=false                    disables Kimi / DeepSeek-style paths
       thinking_mode="chat"              disables DeepSeek V4-style thinking
       thinking=true                     enables opt-in families such as DeepSeek V3
  -> parser state
       explicit markers                  wait for <think> / family marker
       force reasoning                   start inside reasoning until end marker
       prompt-injected marker            set_in_reasoning(true)
  -> output split
       reasoning_text                    hidden reasoning content
       normal_text                       visible content or downstream tool call parser input
```

The follow-up parity table folds this into the **Reasoning Parser Family**
popup. That popup should show the implementation, public parser family,
row-level parser mode, and static parser configuration. Exact
parser-level harness flags should live in each result-cell tooltip;
production request gates should live in the serving-flow panel above
the table instead of being repeated in every row.

| Mode | Static parser behavior | Production caveat |
|---|---|---|
| Explicit markers | Parser waits for an opening marker such as `<think>`, Harmony `analysis`, Gemma channel markers, or Granite phrases. | Frontend/template flags may decide whether the model emits markers or whether the parser starts with `set_in_reasoning(true)`. |
| Force reasoning | Parser begins inside reasoning; marker-free text may be `reasoning_text` until an end marker, stop condition, or family-specific boundary. | Request flags can still disable some force-reasoning paths before the parser runs. |
| Mostly static custom parsers | Parser has parser-specific contract, such as Harmony channels, Granite phrases, or MiniMax append-think behavior. | Generic gates still apply: omit the parser, use guided tool choice, or use SGLang `separate_reasoning=false`. |

For exact parity-result reproduction once the follow-up harness lands,
use the generated table tooltip on each cell. The tooltip should list
the parser-level harness flags actually used for Dynamo, vLLM, and
SGLang. Frontend gates such as `tool_choice` and `separate_reasoning`
are not part of these parser-level fixtures unless the tooltip says so.

### Reasoning, batch mode

The numbering is organized by behavior, not by the order cases were
added. Keep no-op / empty input first, keep all ordinary reasoning
extraction under `REASONING.batch.2.*`, and keep batch tool call boundary
cases third. Do not split `2.a` / `2.c` away from `2.b` / `2.d` / `2.e`
/ `2.f`; they are the same extraction bucket with different surrounding
content.

- **`REASONING.batch.1.a`** Empty input — empty model text. Parser must not crash and must produce coherent empty output.
- **`REASONING.batch.1.b`** Plain text, no reasoning markers — non-empty model text with no reasoning wrapper. Parser must not invent `reasoning_text`.
- **`REASONING.batch.1.c`** Whitespace-only input — whitespace with no reasoning wrapper. Parser must preserve or normalize according to the family contract.
- **`REASONING.batch.1.d`** Null / missing upstream content — upstream content is absent rather than an empty string. This is a frontend/wrapper dispatch case unless the fixture schema explicitly allows `model_text: null`.
- **`REASONING.batch.2.a`** Reasoning-only span — `<think>...</think>` (or analog) present, with no visible answer text outside the span.
- **`REASONING.batch.2.b`** Visible prefix before reasoning — user-visible text appears before the reasoning span.
- **`REASONING.batch.2.c`** Reasoning followed by visible answer text — reasoning content goes to `reasoning_text`; final answer text goes to `normal_text`.
- **`REASONING.batch.2.d`** Visible text around reasoning — user-visible text appears on both sides of the reasoning span.
- **`REASONING.batch.2.e`** Empty reasoning span — `<think></think>` (or analog with zero bytes between markers). Must still register as a reasoning span if the family exposes empty spans.
- **`REASONING.batch.2.f`** Complex reasoning body — large blocks, multi-paragraph, special characters, Unicode, newlines, and marker-looking strings inside the reasoning body.
- **`REASONING.batch.3.a`** Closed reasoning followed by downstream tool-call text — reasoning parser must extract the think content **and leave a valid paired-parser tool-call payload intact in `normal_text`** for the downstream tool-call parser to consume.
- **`REASONING.batch.3.b`** Open reasoning interrupted by downstream tool-call text — tool-call markers can terminate or escape an open reasoning span for families that define that boundary.
- **`REASONING.batch.4`** Malformed reasoning marker syntax — dangling close marker, invalid channel marker, or marker that does not belong to the family grammar.
- **`REASONING.batch.5`** Missing end-marker recovery — engine hit `max_tokens` mid-think; pin behavior: recover partial reasoning, surface as truncated, or treat all as `normal_text`.
- **`REASONING.batch.6.a`** Multiple reasoning spans, adjacent or separated by normal text — e.g., two `<think>...</think>` blocks in one response. Behavior is impl-defined: concatenate, surface only the first, or document a divergence.
- **`REASONING.batch.6.b`** Multiple reasoning spans with malformed later span — first span is valid, later span is incomplete or malformed.

### Reasoning, stream mode

Stream buckets follow the same shape as batch: no-op / empty stream
first, ordinary reasoning extraction second, multi-span extraction next,
then chunk-boundary cases.

- **`REASONING.stream.1.a`** Empty stream chunk — no emitted content. Parser must not crash, buffer forever, or invent reasoning text.
- **`REASONING.stream.1.b`** Plain text stream, no reasoning markers — chunks contain user-visible text only. Explicit-marker parsers should pass it through as `normal_text`; force-reasoning parsers may keep it in `reasoning_text`.
- **`REASONING.stream.2.a`** Single reasoning block across N chunks — incremental assembly of `<think>...</think>` content over multiple chunks of any sizing.
- **`REASONING.stream.2.b`** Multiple reasoning spans across chunks — same contract as `REASONING.batch.6.*`, but state must reset between spans.
- **`REASONING.stream.3.a`** Start-marker chunk-boundary split — opening `<think>` (or analog) straddles a chunk boundary; partial-token matching must keep buffering instead of flushing as plain text.
- **`REASONING.stream.3.b`** End-marker chunk-boundary split — closing `</think>` straddles a chunk boundary; same buffering contract as `REASONING.stream.3.a`.
- **`REASONING.stream.3.c`** Partial marker prefix later proves not to be a marker — emit the buffered text according to the parser's current state.

### Upstream test inventory

There are reasoning tests in both vLLM and SGLang. Reuse the cases,
not the Python harnesses: translate the input / expected output into
language-neutral YAML fixtures, then preserve the upstream URL in the
fixture `ref` field.

| Source | What exists upstream | Reusable Dynamo categories |
|---|---|---|
| [vLLM `tests/reasoning/`](https://github.com/vllm-project/vllm/tree/main/tests/reasoning) | Per-family parser tests for Qwen3, DeepSeek R1/V3, Gemma 4, GPT-OSS, Granite, Kimi K2, MiniMax M2, Mistral, Nemotron V3, and shared base thinking parsers. | Strong source for `REASONING.batch.{1,2,4,5,6}` and `REASONING.stream.{1,2,3,4}`. |
| [vLLM chat reasoning + tool tests](https://github.com/vllm-project/vllm/blob/main/tests/entrypoints/openai/chat_completion/test_chat_with_tool_reasoning.py) | End-to-end streaming and batch chat tests where reasoning and tool calls appear in the same response. | Directly maps to `REASONING.batch.3.*` and streaming tool-boundary cases. |
| [SGLang unit reasoning parser tests](https://github.com/sgl-project/sglang/blob/main/test/registered/unit/parser/test_reasoning_parser.py) | Broad detector coverage: base reasoning, Qwen3, forced Qwen3, Kimi, Kimi K2, GLM45, Hunyuan, Nemotron3, Gemma4, GPT-OSS, MiniMax append-think, buffer-loss regressions, tool-call interaction, and continue-final-message behavior. | Strong source for delimiter grouping, partial marker splits, tool-interrupt cases, and no-stream behavior. |
| [SGLang reasoning kit](https://github.com/sgl-project/sglang/blob/main/python/sglang/test/kits/reasoning_kit.py) and [registered reasoning tests](https://github.com/sgl-project/sglang/tree/main/test/registered/reasoning) | API-level reasoning-content behavior, streaming response checks, and model-facing integration tests. | Useful for expected OpenAI response shape, less reusable as pure parser fixtures. |

### Reusable upstream case shapes

| Upstream shape | Seen in | Dynamo bucket |
|---|---|---|
| Plain text, empty input, whitespace, or missing content with no reasoning markers | vLLM shared parser tests; SGLang buffer-loss and no-reasoning regressions | `REASONING.batch.1.*` |
| Basic complete reasoning span | vLLM Qwen3, DeepSeek, Mistral, Gemma, Granite; SGLang base/Qwen/Kimi/Gemma detectors | `REASONING.batch.2.a` |
| Reasoning block followed or preceded by normal answer text | vLLM Qwen3/Kimi/Mistral/Granite; SGLang base detector | `REASONING.batch.2.{b,c,d}` |
| Reasoning followed by tool-call markers, with the tool marker preserved for the downstream parser | vLLM chat reasoning+tool tests; vLLM Qwen3/Kimi K2; SGLang Kimi K2/GLM45/Hunyuan/GPT-OSS tool-call tests | `REASONING.batch.3.a` |
| Open reasoning span interrupted by a tool-section marker | vLLM Qwen3/Kimi K2; SGLang Kimi K2/GLM45/Hunyuan | `REASONING.batch.3.b` |
| No end marker / truncated reasoning | vLLM Qwen3, DeepSeek, Mistral; SGLang forced-reasoning detectors | `REASONING.batch.5` |
| Empty reasoning span | vLLM Kimi K2/MiniMax; SGLang base and tool-call interaction tests | `REASONING.batch.2.e` |
| Special characters, code blocks, nested marker-looking text | vLLM MiniMax/Kimi/Mistral; SGLang parser-advanced tests | `REASONING.batch.2.f` |
| Multiple reasoning spans or repeated start/end patterns | vLLM shared base tests; SGLang integration scenarios | `REASONING.batch.6.*` |
| Empty stream chunk and marker-free stream | vLLM shared parser tests; SGLang no-reasoning and buffer-loss regressions | `REASONING.stream.1.*` |
| Single reasoning block across stream chunks | vLLM Qwen3/Kimi/Gemma/Granite; SGLang base streaming tests | `REASONING.stream.2.a` |
| Start/end marker split across stream chunks | vLLM Qwen3/Kimi/Gemma/Granite; SGLang buffer-loss and partial-marker tests | `REASONING.stream.3.{a,b}` |
| Partial marker prefix later diverges into normal text | vLLM Granite/Gemma prefix tests; SGLang buffer-loss regressions | `REASONING.stream.3.c` |
| Token-ID-dependent buffering before text | vLLM Kimi K2 and GPT-OSS token/tag tests | Portable only when fixtures include `delta_token_ids`; otherwise mark as unavailable rather than inventing IDs. |

### Format-conditional tags (cross-stage)

The `PARSER.fmt|xml|harmony.*` tags from `PARSER_CASES.md` also apply
to reasoning parsers when they consume the corresponding format. For
example, the GPT-OSS reasoning parser exercises Harmony's
`<|channel|>analysis<|message|>...<|end|>` envelope, so its tests
carry both `REASONING.batch.2.a` and `PARSER.harmony.1`.

### Categories that are N/A for reasoning parsers

- `PARSER.batch.2` — "multiple tool calls" maps to `REASONING.batch.6.*`
  (multiple reasoning spans), not directly. The tool-call-shaped tag
  doesn't apply.
- `PARSER.batch.8` — interleaved tool-call markers and normal text;
  reasoning's analog is `REASONING.batch.2.{b,c,d}` (reasoning +
  normal text) or `REASONING.batch.3.*` (reasoning + downstream tool
  call).
- `FRONTEND.tool_choice` — request-time gating, see
  `FRONTEND_CASES.md`.
- `PIPELINE.finish_reason` — pipeline-boundary contract, see
  `PIPELINE_CASES.md`.

---

## `REASONING.batch.1.*` — No reasoning content

No reasoning wrapper is present. This includes plain text, empty input,
whitespace-only input, and null / missing upstream content.

- Applies to every reasoning parser.
- Parser must not invent `reasoning_text`.
- This bucket is first because no-op behavior is the baseline for
  routing and fallback.

## `REASONING.batch.2.*` — Reasoning extraction

`<think>...</think>` (or analog: `<seed:think>`, harmony's
`<|channel|>analysis`) is present and the content is not part of a
tool call.

- Applies to every reasoning parser.
- Parser populates `reasoning_text` with the content between the
  markers.
- `normal_text` carries only user-visible text outside the reasoning
  block.
- Empty and complex content stay in this family because they are still
  ordinary extraction cases.

## `REASONING.batch.3.*` — Reasoning + downstream tool call

Model emits reasoning content and then tool-call tokens:
`<think>...</think><|tool_call_begin|>...`. The reasoning parser must
extract the reasoning content and preserve the tool-call-shaped text in
`normal_text` so the downstream tool-call parser can consume it.

- `3.a` applies to paired parser families where the reasoning span is
  closed before downstream tool-call text begins.
- `3.b` applies only to parser families that intentionally configure a
  downstream boundary while reasoning is still open.
- Failure mode: greedy reasoning parser eats the tool-call content
  that follows. Pin the boundary explicitly.

## `REASONING.batch.4` — Malformed marker syntax

Dangling close marker, invalid channel marker, or family-specific
syntax that does not form a valid reasoning span.

## `REASONING.batch.5` — Missing end-marker recovery

Open reasoning marker without the matching close marker. This is the
reasoning analog of model output truncated by `max_tokens`.

## `REASONING.batch.6.*` — Multiple reasoning spans

Two or more reasoning spans appear in one model response. The contract
is family-specific: concatenate, surface only the first span, or
document a divergence.

---

## `REASONING.stream.1.*` — No reasoning stream content

No complete reasoning wrapper is present. This includes plain text,
empty chunks, and marker-looking prefixes that ultimately do not match
the family grammar.

- Explicit-marker parsers should not invent reasoning content.
- Force-reasoning parsers may keep marker-free text in `reasoning_text`
  until an end marker or tool boundary appears.
- Empty chunks should be stable no-ops.

## `REASONING.stream.2.*` — Streaming reasoning extraction

One complete reasoning block delivered across multiple SSE chunks of
any sizing. Parser incrementally reconstructs `reasoning_text`.

- Applies to every reasoning parser. Dominant production path.

## `REASONING.stream.3.*` — Split reasoning markers

The opening reasoning marker (`<think>`, `<|channel|>analysis`, etc.)
straddles a chunk boundary. Partial-token matching must return "keep
buffering" rather than flushing the partial bytes as `normal_text` and
completing the match when the next chunk lands.

The closing reasoning marker (`</think>`, `<|end|>`, etc.) straddles a
chunk boundary. Same buffering contract as the start marker.

- Applies to every reasoning parser.

## `REASONING.stream.4.*` — Downstream parser boundary

A downstream tool-call section marker appears while the reasoning
parser is still processing reasoning. The reasoning parser must stop at
the right boundary and preserve the downstream parser's input.

- Applies only to reasoning parser families with an explicit downstream
  boundary. For other families, this bucket is an N/A stub until the
  production path wires such a boundary.

---

## Customer-incident regression tests

Same convention as `PARSER_CASES.md`: include the originating
reference inline in the `#[test]` comment.

```rust
#[test] // REASONING.batch.3.b (PR #1234)
fn test_unclosed_think_tag_no_longer_swallows_tool_call() { ... }
```

---

## Adding a new reasoning parser: target complete coverage

The current fixture PR is seed coverage: it establishes the YAML
contract, provenance, and representative cases, but it does not claim
that every family already has the complete taxonomy below. Before a
parser family is considered complete, add:

1. `REASONING.batch.1.*` — no reasoning content: plain text, empty
   input, whitespace, and null / missing upstream content.
2. `REASONING.batch.2.*` — baseline reasoning extraction, narration
   split, empty reasoning span, and complex reasoning body.
3. `REASONING.batch.3.*` — boundary contract with downstream tool-call
   parser. Non-negotiable for any reasoning parser used alongside
   tool calling.
4. `REASONING.batch.{4, 5}` — malformed input + missing-end-marker
   recovery. Pin behavior; silent drop is the failure mode.
5. `REASONING.batch.6.*` — multiple reasoning spans. Document the
   contract.
6. `REASONING.stream.{1, 2, 3, 4}` — streaming. Add `stream.4.*`
   before declaring complete coverage for any parser with an explicit
   downstream tool-call boundary.
7. Format tags where applicable: `PARSER.harmony.1` for parsers
   consuming Harmony-format input, etc.
