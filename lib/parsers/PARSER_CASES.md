# Tool-Call / Reasoning Parser Corner Cases

Reference taxonomy for unit testing tool-call and reasoning parsers. Each parser
added under `src/tool_calling/` or `src/reasoning/` should cover the generic
`PARSER.<n>` categories; family-specific parsers also cover their respective
`PARSER.xml<n>` / `PARSER.harmony<n>` categories, plus any applicable
`PARSER.fmt<n>` format variants. `N/A` should be called out explicitly in the
test file rather than silently omitted.

Category layout:
- **`PARSER.1`–`PARSER.15`** — **Behavior contract**. Universal semantic invariants. Apply to every parser regardless of grammar.
- **`PARSER.fmt1`–`PARSER.fmt4`** — **Format variants**. Grammar-conditional syntactic invariants (name conventions, whitespace, alternate spellings, empty wrappers).
- **`PARSER.xml1`–`PARSER.xml2`** — XML family only (hermes, glm47, qwen3_coder, minimax_m2, kimi_k2).
- **`PARSER.harmony1`–`PARSER.harmony2`** — Harmony family only (gpt-oss).

Tests that exist because of a specific customer-reported bug should
include the originating ticket / PR / issue reference inline in the
`#[test]` comment — e.g. `#[test] // PARSER.5 (PR #8208)`. The PARSER
label is the categorical claim; the parenthetical is the audit trail.
No separate "regression" taxonomy is needed.

White-box helper unit tests of `detect_tool_call_start_*` /
`find_tool_call_end_position_*` are tagged `// helper` in the test code;
they don't carry a numbered category since they have no cross-impl
analogue and exist to pin internal Rust function behavior only.

Per-model gap tracking lives elsewhere (not in this repo).

## Quick reference

### Generic — behavior contract

- **`PARSER.1`** Single tool call — happy path (one complete, well-formed call).
- **`PARSER.2`** Multiple tool calls — sequential or parallel (2+ in one response).
- **`PARSER.3`** No tool call (response is text only).
- **`PARSER.4`** Malformed / partial JSON args (truncated, missing close brace, invalid syntax).
- **`PARSER.5`** Missing end-token recovery (recover calls when `section_end` is absent due to max_tokens / EOS).
- **`PARSER.6`** Empty args (`arguments={}` / no-arg call).
- **`PARSER.7`** Complex arg types (nested objects, arrays, bool, number, Unicode / newlines in values).
- **`PARSER.8`** Streaming — token-by-token assembly + chunk-boundary splits.
- **`PARSER.9`** Paired reasoning + tool in same response.
- **`PARSER.10`** Reasoning only (think tags, no tool call).
- **`PARSER.11`** `tool_choice` = auto / required / named / none.
- **`PARSER.12`** `finish_reason` semantics (`stop` / `tool_calls` / `length` mapping).
- **`PARSER.13`** Normal text interleaved with tool calls.
- **`PARSER.14`** Empty content / empty `tool_calls` array / null response.
- **`PARSER.15`** Duplicate tool calls (same name twice). No test anywhere in the repo; universal gap.

### Format variants — grammar-conditional

- **`PARSER.fmt1`** Function-name conventions — allowed identifier chars (hyphens, underscores, dots), prefix variants (`functions.NAME` vs bare `NAME`), and rejection of malformed function IDs. Models differ on what they emit; parsers must take a position.
- **`PARSER.fmt2`** Whitespace / formatting tolerance — whitespace inside or between tokens (newlines after `<|tool_call_begin|>`, spaces around the function ID, etc.). Parser must accept the same call regardless of formatting.
- **`PARSER.fmt3`** Token format variants — multiple acceptable spellings for the same semantic (e.g., Kimi K2's singular `<|tool_call_section_*|>` vs plural `<|tool_calls_section_*|>` section tokens). Parser must accept all configured variants.
- **`PARSER.fmt4`** Empty section / no-content wrappers — start+end fences with nothing between them (`<|tool_calls_section_begin|><|tool_calls_section_end|>`). Must produce zero calls and preserve any surrounding text.

### XML-family (`PARSER.xml*`)

- **`PARSER.xml1`** XML entity / HTML unescape handling (`&lt;`, `&amp;`, `&quot;` in parameter values).
- **`PARSER.xml2`** Schema-aware type coercion (string → number/bool/array based on declared parameter schema).

### Harmony (`PARSER.harmony*`)

- **`PARSER.harmony1`** Channel / recipient parsing (analysis / commentary / final channels).
- **`PARSER.harmony2`** Tool-call envelope variants — `<|channel|>commentary to=functions.X<|message|>...<|call|>` and related multi-tag forms specific to Harmony's grammar.

### Universal gaps (no test anywhere, not promoted to numbered categories)

- Unicode in function names (non-ASCII tool names, emoji).
- Numeric overflow in args (very large int / float outside JSON spec range).
- Empty function name (`"name": ""`).
- Concurrent parallel requests (process-level contention during parse).
- Guided-decoding ↔ tool-call interaction (constrained generation emits malformed args).
- Extremely long output (≥10 KB tool-call JSON in a single call).
- Mid-stream error injection / interruption (worker kill, network drop mid-parse).
- Schema arg-count mismatch (model emits extra or missing args vs declared schema).

---

## `PARSER.1` — Single tool call, happy path

One complete, well-formed call in the response.

- Applies to every tool-call parser.
- Baseline correctness check. If `PARSER.1` fails, nothing else below matters.

## `PARSER.2` — Multiple tool calls (sequential or parallel)

Two or more calls in one response, in the same block or back-to-back.

- Applies to every tool-call parser.
- Some grammars emit parallel calls in one block (DSML, XML); others emit
  sequential top-level sentinels (JSON dialects). Either way, extract all.

## `PARSER.3` — No tool call

Response is plain text, no tool-call grammar present.

- Applies to every tool-call parser.
- Must return empty `Vec<ToolCall>` and the input as `normal_text`. Zero false
  positives.

## `PARSER.4` — Malformed / partial JSON args

Truncated JSON, missing close brace, invalid syntax inside the arguments
payload.

- Applies to every tool-call parser. For parsers whose grammar never embeds
  JSON (none today — all top-N families embed JSON somewhere), mark explicit
  `N/A`.
- Behavior must be documented: either graceful fallback to string (DSML's
  current behavior via `serde_json::from_str(...).unwrap_or_else(|_| String(...))`)
  or explicit error. Silent drop is the failure mode.

## `PARSER.5` — Missing end-token recovery

The model's response is truncated before the closing fence arrives
(`<|tool_calls_section_end|>` for Kimi, `</｜DSML｜tool_calls>` for DeepSeek
DSML, etc.) — typically because the engine hit `max_tokens` or the model
emitted EOS mid-generation.

- Applies to every tool-call parser with paired start/end fences.
- Customer-facing bug class: silent drop of the in-flight call looks like a
  successful HTTP 200 with no tool_calls and no error.
- Two acceptable resolutions: (a) recover completed invokes even without the
  outer close fence (Kimi K2 does this post-fix), or (b) return an explicit
  error. Either way, pin the behavior with a test so a future change is
  intentional.

## `PARSER.6` — Empty args

Tool call with `arguments={}`, or a no-parameter invoke.

- Applies to every tool-call parser.
- Must still return the call — empty args is a valid call, not a missing one.

## `PARSER.7` — Complex argument types

Nested objects, arrays, booleans, numbers, mixed types, Unicode values, and
newlines inside argument values.

- Applies to every tool-call parser.
- For grammars that carry type hints (DSML's `string="true|false"`), verify
  JSON round-tripping. For XML grammars without hints, the type-coercion
  half of the test is covered under `PARSER.xml2` instead — here just verify
  that complex values make it through without truncation or escape bugs.

## `PARSER.8` — Streaming

Chunked input arriving over SSE. Covers two concerns that tend to fail
together:

1. **Token-by-token assembly** — the parser incrementally reconstructs the
   tool-call structure across many small chunks.
2. **Chunk-boundary splits** — start fence, end fence, or parameter name /
   value straddles a chunk boundary. Partial-token matching must return
   `true` (keep buffering, don't flush as plain text) and complete the
   match on the next chunk.

- Applies to every tool-call parser. Dominant production path.

## `PARSER.9` — Paired reasoning + tool in same response

Model emits `<think>...</think>` (or analog) followed by a tool call. Both
must be extracted: `reasoning_content` populated AND `tool_calls` populated.

- Applies to every (tool, reasoning) parser pair.
- Watch for the "unclosed think-tag swallows tool call" bug — if the reasoning
  parser is greedy it may eat the tool-call content that follows.

## `PARSER.10` — Reasoning only

`<think>...</think>` or analog present, no tool call. Parser must populate
`reasoning_content` and leave `tool_calls` empty.

- Applies to every reasoning parser.

## `PARSER.11` — `tool_choice` = auto / required / named / none

Each of the four OpenAI `tool_choice` modes exercised per parser.

- Applies to every tool-call parser.
- Cross-parser suites at `lib/llm/tests/tool_choice.rs` /
  `parallel_tool_call_integration.rs` / `tool_choice_finish_reasons.rs`
  run `hermes` only today. Adding a new parser requires parametrizing those
  suites or adding a per-parser equivalent.
- Universal gap across most parsers in the repo as of 2026-04.

## `PARSER.12` — `finish_reason` semantics

`stop` vs `tool_calls` vs `length` mapping, in both streaming and
non-streaming paths.

- Applies to every tool-call parser.
- When a tool call lands, `finish_reason` must become `tool_calls`. When
  `max_tokens` truncates mid-stream, `length` must propagate — this is
  often the signal that should trigger `PARSER.5` recovery on the parser side.

## `PARSER.13` — Normal text interleaved with tool calls

Model emits narration text before / after / between tool-call blocks. Parser
must split content correctly: text → `normal_content`, calls → `tool_calls`.

- Applies to every tool-call parser.

## `PARSER.14` — Empty content / empty `tool_calls` array / null response

Engine emits a chunk with `delta.content = ""`, or a final response with
`tool_calls: []`, or `null` values inside arguments.

- Applies to every tool-call parser.
- Null-value handling inside parameters is parser-level (`parse_parameters`
  in DSML handles it via `serde_json::Value::Null`). Empty-choices /
  empty-stream handling is typically at the e2e integration layer.

## `PARSER.15` — Duplicate tool calls (same name twice)

Two calls to the same function name in one response, possibly with the same
arguments.

- Applies to every tool-call parser.
- **Zero coverage across the entire repo as of 2026-04.** Universal gap.
- Expected behavior: both calls must appear in `tool_calls` with distinct
  IDs. (The runtime / client is responsible for deciding whether duplicate
  invocation is intended.)

## Customer-incident regression tests

When a test exists because a specific customer ticket / PR / GH issue
uncovered a bug, include that reference inline in the `#[test]` comment:

```rust
#[test] // PARSER.5 (PR #8208)
fn test_parse_malformed_no_section_end() { ... }
```

The PARSER label still names the category being exercised; the
parenthetical names the originating incident. Greppable from both
directions: `grep -r 'PARSER.5'` finds all PARSER.5 tests; `grep -r '#8208'`
finds every test tied to that incident across layers.

---

## `PARSER.fmt1` — Function-name conventions

What identifier characters the parser accepts in tool function names, and
which prefix variants it recognizes (`functions.NAME` vs bare `NAME`).

- Grammar-conditional: applies to parsers that emit named tool-call IDs and
  perform their own validation. Most XML and harmony parsers do.
- Models differ on what they emit. The parser must take a position and pin
  it with a test so a future tokenizer change doesn't silently start dropping
  valid calls.

## `PARSER.fmt2` — Whitespace / formatting tolerance

The parser accepts the same logical call regardless of incidental whitespace
inside or between grammar tokens (newlines after `<|tool_call_begin|>`,
spaces around the function ID, padding inside arg JSON, etc.).

- Grammar-conditional. Applies to any parser whose grammar permits
  whitespace variation between tokens.
- Rejecting whitespace strictly is also a valid choice — pin the behavior
  either way.

## `PARSER.fmt3` — Token format variants

Multiple acceptable spellings for the same semantic — e.g., Kimi K2's
singular `<|tool_call_section_*|>` vs plural `<|tool_calls_section_*|>`
section tokens. The parser must accept all configured variants.

- Grammar-conditional. Applies only when the parser's config explicitly
  enumerates more than one token-form alias.

## `PARSER.fmt4` — Empty section / no-content wrappers

Start + end fences with nothing between them
(`<|tool_calls_section_begin|><|tool_calls_section_end|>`). The parser must
produce zero calls and preserve any surrounding text as `normal_text`.

- Grammar-conditional. Applies to parsers with paired start/end fences.

---

## `PARSER.xml1` — XML entity / HTML unescape handling

Parameter values contain XML-encoded entities (`&lt;`, `&amp;`, `&quot;`,
`&apos;`, numeric entities like `&#38;`) that must be decoded before the
value is surfaced to the client.

- Applies only to XML-family tool-call parsers: `hermes`, `glm47`,
  `qwen3_coder`, `minimax_m2`, `kimi_k2` (despite its special-token outer
  fence, the inner parameter payload is XML-ish).
- **N/A for DSML** — the `string="true|false"` attribute tells the parser
  whether to JSON-decode or pass through verbatim; no entity decoding pass.
- **N/A for JSON-family and Harmony** — JSON has its own escape semantics
  handled by `serde_json`.

## `PARSER.xml2` — Schema-aware type coercion

Parser uses the declared tool schema to coerce string args to
number / bool / array based on the declared parameter type.

- Applies only to XML-family parsers without explicit type annotations in
  the wire format. `xml/parser.rs`, `glm47_parser.rs` do this.
- **N/A for DSML** — the `string="true|false"` attribute carries the type
  intent per parameter, so no schema lookup is needed.
- **N/A for JSON-family** — JSON has native types.
- **N/A for Harmony** — payload is JSON inside the channel envelope.

---

## `PARSER.harmony1` — Channel / recipient parsing

OpenAI Harmony's token stream carries channel metadata
(`<|channel|>analysis|commentary|final<|message|>`) and recipient targets
(`to=functions.foo`). Parser must route the `commentary` channel content
into tool-call extraction while surfacing `analysis` as reasoning and
`final` as the user-visible output.

- **Harmony only.** N/A for every other family.

## `PARSER.harmony2` — Envelope tag grammar

Harmony wraps tool calls and reasoning in multi-tag envelopes:
`<|channel|>commentary to=functions.X <|constrain|>json<|message|>{...}<|call|>`,
mirrored by `<|channel|>analysis<|message|>...<|end|>` for reasoning.
The parser must walk the envelope correctly across its legal variations:

- **Complete envelope** — all tags present (the happy path).
- **Missing `<|start|>` / assistant prefix** — model output lands inside an existing turn.
- **Missing `<|call|>` (truncation recovery)** — engine hit `max_tokens` mid-emit; pin behavior (recover or surface error explicitly).
- **Reasoning + tool in same turn** — `<|channel|>analysis ... <|end|><|start|>assistant<|channel|>commentary ...` chains.
- **Streaming chunk boundaries through the envelope** — chunks split inside `<|constrain|>`, `<|message|>`, `to=functions.X`, etc. The streaming parser must keep buffering until the next tag completes.

Cross-cuts other PARSER.* cases when they're exercised on harmony format text
(e.g. `// PARSER.5, PARSER.harmony2` for harmony-flavored truncation recovery).

- **Harmony only.** N/A for every other family.

---

## Applicability summary

| Category block | Parsers | Notes |
| -- | -- | -- |
| `PARSER.1`–`PARSER.15` (behavior contract) | All | Required contract for every parser |
| `PARSER.fmt1`–`PARSER.fmt4` (format variants) | Grammar-conditional | Each variant required only where the grammar permits it |
| `PARSER.xml1`–`PARSER.xml2` | XML-family only | Entity decoding + schema-aware coercion |
| `PARSER.harmony1`–`PARSER.harmony2` | Harmony only | Channel routing + envelope variants |

## Adding a new parser: what you must include

Minimum viable set for a new tool-call parser:

1. `PARSER.1`, `PARSER.2`, `PARSER.3` — baseline correctness.
2. `PARSER.4` or explicit N/A justification — handle or refuse malformed input.
3. `PARSER.5` — pin behavior when the outer fence is missing. Silent drop is a
   regression waiting to happen.
4. `PARSER.6`, `PARSER.7` — empty and complex args.
5. `PARSER.8` — streaming. Essentially non-negotiable for any parser that sits
   behind a streaming frontend.
6. `PARSER.13` — interleaved text.
7. `PARSER.15` — document whether duplicate calls are supported. Flat gap
   today; landing a test with the parser establishes the contract.
8. Format variants where applicable (`PARSER.fmt1`–`PARSER.fmt4`): cover any
   that the parser's grammar permits. Mark `N/A` for those that don't apply.
9. Family-specific categories where applicable: `PARSER.xml1` / `PARSER.xml2`
   for XML grammars, `PARSER.harmony1` / `PARSER.harmony2` for Harmony.

For reasoning parsers, replace `PARSER.4` / `PARSER.5` / `PARSER.8`-assembly with
`PARSER.8`-partial-close-tag and `PARSER.10` (reasoning-only).
