# Tool-Call Parser Corner Cases

Reference taxonomy for unit testing **tool-call** parsers under
`src/tool_calling/`. Sibling files cover adjacent stages:

- **Reasoning parsers** (`src/reasoning/`): see `REASONING_CASES.md`.
- **Frontend gating** (request-time `tool_choice`, etc.): see
  `components/src/dynamo/frontend/tests/FRONTEND_CASES.md`.
- **Pipeline boundary** (`finish_reason` independence, etc.): see
  `PIPELINE_CASES.md`.

The taxonomy splits on three orthogonal axes:

1. **Stage** — which module is under test (`PARSER.*` here, `REASONING.*`,
   `FRONTEND.*`, `PIPELINE.*` in their own files).
2. **Mode** — invocation surface: `batch` (entire model output as one
   string) or `stream` (incremental `delta_text` + `delta_token_ids`).
   Each mode has its own contiguous case numbers; a "single tool call"
   case in batch is `PARSER.batch.1`, in stream is `PARSER.stream.1` —
   the numbers don't share semantics across modes.
3. **Format scope** — applicability narrower than "all parsers":
   `fmt`, `xml`, `harmony`.

Each `#[test]` carries one or more `// PARSER.<tag>` annotations naming
the categories it exercises. `N/A` is called out explicitly in the
test file rather than silently omitted.

Tests that exist because of a specific customer ticket / PR / GH issue
include the originating reference inline in the comment — e.g.
`// PARSER.batch.5 (PR #8208)`. The category label is the categorical
claim; the parenthetical is the audit trail. Greppable from both
directions: `grep -r 'PARSER.batch.5'` finds all such tests;
`grep -r '#8208'` finds every test tied to the incident across layers.

White-box helper unit tests of `detect_tool_call_start_*` /
`find_tool_call_end_position_*` are tagged `// helper`; they don't
carry a numbered category since they have no cross-impl analogue and
exist to pin internal Rust function behavior only.

## Quick reference

### Parser, batch mode

Universal behavior contract. Applies to every tool-call parser when
fed the full model output as a single string.

- **`PARSER.batch.1`** Single tool call — happy path (one complete, well-formed call).
- **`PARSER.batch.2`** Multiple tool calls — sequential or parallel (2+ in one response).
- **`PARSER.batch.3`** No tool call (response is text only).
- **`PARSER.batch.4`** Malformed / partial JSON args (truncated, missing close brace, invalid syntax). Recovery contract is impl-defined; document divergences rather than asserting one truth.
- **`PARSER.batch.5`** Missing end-token recovery (recover calls when the closing fence is absent due to `max_tokens` / EOS).
- **`PARSER.batch.6`** Empty args (`arguments={}` / no-arg call).
- **`PARSER.batch.7`** Complex arg types (nested objects, arrays, bool, number, Unicode / newlines in values).
- **`PARSER.batch.8`** Normal text interleaved with tool calls.
- **`PARSER.batch.9`** Empty content / empty `tool_calls` array / null response.
- **`PARSER.batch.10`** Duplicate tool calls (same name twice, possibly with same args).

### Parser, stream mode

Token-by-token assembly from a streaming engine. Same logical cases
as batch mode, but driven through `parse_streaming_increment(delta)`.
This requires a separate test harness; case numbers are independent
of batch.

- **`PARSER.stream.1`** Single tool call across N chunks (any chunk boundary).
- **`PARSER.stream.2`** Multiple tool calls, each spanning multiple chunks.
- **`PARSER.stream.3`** Partial-token chunking (chunk boundary splits a grammar token mid-string). Partial-token matching must return `keep buffering`, not flush as plain text.
- **`PARSER.stream.4`** Streaming termination — final chunk arrives with `finish_reason=tool_calls` / EOS; parser flushes any in-flight call.

### Format-conditional (scope narrower than "all parsers")

Same `PARSER.fmt|xml|harmony` tags also appear in `src/reasoning/`
tests when a reasoning parser consumes the corresponding format.
The tag describes the *grammar*, not the parser stage.

#### `PARSER.fmt.*` — format variants

- **`PARSER.fmt.1`** Function-name conventions — allowed identifier chars (hyphens, underscores, dots), prefix variants (`functions.NAME` vs bare `NAME`), and rejection of malformed function IDs.
- **`PARSER.fmt.2`** Whitespace / formatting tolerance — whitespace inside or between grammar tokens.
- **`PARSER.fmt.3`** Token format variants — multiple acceptable spellings for the same semantic (e.g., Kimi K2's singular `<|tool_call_section_*|>` vs plural `<|tool_calls_section_*|>` section tokens).
- **`PARSER.fmt.4`** Empty section / no-content wrappers — start+end fences with nothing between them.

#### `PARSER.xml.*` — XML-family only

(`hermes`, `glm47`, `qwen3_coder`, `minimax_m2`, `kimi_k2`.)

- **`PARSER.xml.1`** XML entity / HTML unescape handling (`&lt;`, `&amp;`, `&quot;`, etc.).
- **`PARSER.xml.2`** Schema-aware type coercion (string → number/bool/array based on declared parameter schema).

#### `PARSER.harmony.*` — Harmony only (gpt-oss)

- **`PARSER.harmony.1`** Channel / recipient parsing (analysis / commentary / final channels; `to=functions.X` recipients).
- **`PARSER.harmony.2`** Envelope tag grammar — `<|channel|>commentary to=functions.X <|constrain|>json<|message|>{...}<|call|>` and its legal variations.

### Universal gaps (no test anywhere)

- Unicode in function names (non-ASCII tool names, emoji).
- Numeric overflow in args (very large int / float outside JSON spec range).
- Empty function name (`"name": ""`).
- Concurrent parallel requests (process-level contention during parse).
- Guided-decoding ↔ tool-call interaction (constrained generation emits malformed args).
- Extremely long output (≥10 KB tool-call JSON in a single call).
- Mid-stream error injection / interruption (worker kill, network drop mid-parse).
- Schema arg-count mismatch (model emits extra or missing args vs declared schema).

---

## `PARSER.batch.1` — Single tool call, happy path

One complete, well-formed call in the response.

- Applies to every tool-call parser.
- Baseline correctness check. If `PARSER.batch.1` fails, nothing else
  below matters.

## `PARSER.batch.2` — Multiple tool calls (sequential or parallel)

Two or more calls in one response, in the same block or back-to-back.

- Applies to every tool-call parser.
- Some grammars emit parallel calls in one block (DSML, XML); others
  emit sequential top-level sentinels (JSON dialects). Either way,
  extract all.

## `PARSER.batch.3` — No tool call

Response is plain text, no tool-call grammar present.

- Applies to every tool-call parser.
- Must return empty `Vec<ToolCall>` and the input as `normal_text`.
  Zero false positives.

## `PARSER.batch.4` — Malformed / partial JSON args

Truncated JSON, missing close brace, invalid syntax inside the
arguments payload.

- Applies to every tool-call parser. For parsers whose grammar never
  embeds JSON (none today), mark explicit `N/A`.
- Behavior is **impl-defined**: graceful fallback to string (DSML's
  current behavior via
  `serde_json::from_str(...).unwrap_or_else(|_| String(...))`),
  drop-on-error, or explicit error are all valid choices. Cross-impl
  parity tests should record divergences in their `KNOWN_DIVERGENCES`
  registry rather than asserting one truth.

## `PARSER.batch.5` — Missing end-token recovery

Model response is truncated before the closing fence arrives
(`<|tool_calls_section_end|>` for Kimi K2, `</｜DSML｜tool_calls>` for
DeepSeek DSML, etc.) — typically because the engine hit `max_tokens`
or the model emitted EOS mid-generation.

- Applies to every tool-call parser with paired start/end fences.
- Customer-facing bug class: silent drop of the in-flight call looks
  like a successful HTTP 200 with no `tool_calls` and no error.
- Two acceptable resolutions: (a) recover completed invokes even
  without the outer close fence (Kimi K2 does this post-fix), or (b)
  return an explicit error. Either way, pin the behavior with a test
  so a future change is intentional.

## `PARSER.batch.6` — Empty args

Tool call with `arguments={}`, or a no-parameter invoke.

- Applies to every tool-call parser.
- Must still return the call — empty args is a valid call, not a
  missing one.

## `PARSER.batch.7` — Complex argument types

Nested objects, arrays, booleans, numbers, mixed types, Unicode
values, and newlines inside argument values.

- Applies to every tool-call parser.
- For grammars that carry type hints (DSML's `string="true|false"`),
  verify JSON round-tripping. For XML grammars without hints, the
  type-coercion half lives under `PARSER.xml.2` instead — here just
  verify that complex values make it through without truncation or
  escape bugs.

## `PARSER.batch.8` — Normal text interleaved with tool calls

Model emits narration text before / after / between tool-call blocks.
Parser must split content correctly: text → `normal_text`, calls →
`tool_calls`.

- Applies to every tool-call parser.
- When the narration is reasoning content (`<think>...</think>` or
  analog), the test additionally exercises the reasoning-parser
  handoff. Cross-tag with `REASONING.batch.2` if the reasoning parser
  also runs over the input.

## `PARSER.batch.9` — Empty content / empty `tool_calls` array / null response

Engine emits a chunk with `delta.content = ""`, or a final response
with `tool_calls: []`, or `null` values inside arguments.

- Applies to every tool-call parser.
- Null-value handling inside parameters is parser-level
  (`parse_parameters` in DSML handles it via
  `serde_json::Value::Null`). Empty-choices / empty-stream handling
  is typically at the e2e integration layer.

## `PARSER.batch.10` — Duplicate tool calls (same name twice)

Two calls to the same function name in one response, possibly with
the same arguments.

- Applies to every tool-call parser.
- Expected behavior: both calls must appear in `tool_calls` with
  distinct IDs. (The runtime / client is responsible for deciding
  whether duplicate invocation is intended.)

---

## `PARSER.stream.1` — Single tool call across N chunks

One complete tool call delivered across multiple SSE chunks of any
sizing. Parser incrementally reconstructs the call.

- Applies to every tool-call parser. Dominant production path.

## `PARSER.stream.2` — Multiple tool calls, each across N chunks

Two or more calls in the response, each spanning multiple chunks.
Parser must emit each call as its complete invocation lands; must not
mix arguments across calls.

- Applies to every tool-call parser.

## `PARSER.stream.3` — Partial-token chunking

Chunk boundary splits a grammar token mid-string (start fence, end
fence, or parameter name / value straddles a chunk boundary). Partial
matches must return "keep buffering" rather than flushing as plain
text and completing on a later chunk.

- Applies to every tool-call parser.

## `PARSER.stream.4` — Streaming termination

Final chunk arrives with `finish_reason=tool_calls` (or `length` /
`stop`). Parser flushes any in-flight call (or surfaces the truncation
explicitly per `PARSER.batch.5`).

- Applies to every tool-call parser.

---

## `PARSER.fmt.1` — Function-name conventions

Identifier characters the parser accepts in tool function names, and
which prefix variants it recognizes (`functions.NAME` vs bare `NAME`).

- Grammar-conditional: applies to parsers that emit named tool-call
  IDs and perform their own validation. Most XML and harmony parsers
  do.
- Models differ on what they emit. The parser must take a position
  and pin it with a test so a future tokenizer change doesn't
  silently start dropping valid calls.

## `PARSER.fmt.2` — Whitespace / formatting tolerance

Parser accepts the same logical call regardless of incidental
whitespace inside or between grammar tokens (newlines after
`<|tool_call_begin|>`, spaces around the function ID, padding inside
arg JSON, etc.).

- Grammar-conditional. Applies to any parser whose grammar permits
  whitespace variation between tokens.
- Rejecting whitespace strictly is also a valid choice — pin the
  behavior either way.

## `PARSER.fmt.3` — Token format variants

Multiple acceptable spellings for the same semantic — e.g., Kimi K2's
singular `<|tool_call_section_*|>` vs plural
`<|tool_calls_section_*|>` section tokens. Parser must accept all
configured variants.

- Grammar-conditional. Applies only when the parser's config
  explicitly enumerates more than one token-form alias.

## `PARSER.fmt.4` — Empty section / no-content wrappers

Start + end fences with nothing between them
(`<|tool_calls_section_begin|><|tool_calls_section_end|>`). Parser
must produce zero calls and preserve any surrounding text as
`normal_text`.

- Grammar-conditional. Applies to parsers with paired start/end
  fences.

---

## `PARSER.xml.1` — XML entity / HTML unescape handling

Parameter values contain XML-encoded entities (`&lt;`, `&amp;`,
`&quot;`, `&apos;`, numeric entities like `&#38;`) that must be
decoded before the value is surfaced to the client.

- Applies only to XML-family tool-call parsers: `hermes`, `glm47`,
  `qwen3_coder`, `minimax_m2`, `kimi_k2` (despite its special-token
  outer fence, the inner parameter payload is XML-ish).
- **N/A for DSML** — the `string="true|false"` attribute tells the
  parser whether to JSON-decode or pass through verbatim; no entity
  decoding pass.
- **N/A for JSON-family and Harmony** — JSON has its own escape
  semantics handled by `serde_json`.

## `PARSER.xml.2` — Schema-aware type coercion

Parser uses the declared tool schema to coerce string args to
number / bool / array based on the declared parameter type.

- Applies only to XML-family parsers without explicit type
  annotations in the wire format. `xml/parser.rs`,
  `glm47_parser.rs` do this.
- **N/A for DSML** — the `string="true|false"` attribute carries
  the type intent per parameter, so no schema lookup is needed.
- **N/A for JSON-family** — JSON has native types.
- **N/A for Harmony** — payload is JSON inside the channel envelope.

---

## `PARSER.harmony.1` — Channel / recipient parsing

OpenAI Harmony's token stream carries channel metadata
(`<|channel|>analysis|commentary|final<|message|>`) and recipient
targets (`to=functions.foo`). Parser must route the `commentary`
channel content into tool-call extraction while surfacing `analysis`
as reasoning and `final` as the user-visible output.

- **Harmony only.** N/A for every other family.

## `PARSER.harmony.2` — Envelope tag grammar

Harmony wraps tool calls and reasoning in multi-tag envelopes:
`<|channel|>commentary to=functions.X <|constrain|>json<|message|>{...}<|call|>`,
mirrored by `<|channel|>analysis<|message|>...<|end|>` for reasoning.
The parser must walk the envelope correctly across its legal
variations:

- **Complete envelope** — all tags present (the happy path).
- **Missing `<|start|>` / assistant prefix** — model output lands
  inside an existing turn.
- **Missing `<|call|>` (truncation recovery)** — engine hit
  `max_tokens` mid-emit; pin behavior (recover or surface error
  explicitly).
- **Reasoning + tool in same turn** —
  `<|channel|>analysis ... <|end|><|start|>assistant<|channel|>commentary ...`
  chains.
- **Streaming chunk boundaries through the envelope** — chunks split
  inside `<|constrain|>`, `<|message|>`, `to=functions.X`, etc. The
  streaming parser must keep buffering until the next tag completes.

Cross-cuts other categories when exercised on harmony format text
(e.g. `// PARSER.batch.5, PARSER.harmony.2` for harmony-flavored
truncation recovery).

- **Harmony only.** N/A for every other family.

---

## Customer-incident regression tests

When a test exists because a specific customer ticket / PR / GH
issue uncovered a bug, include that reference inline in the
`#[test]` comment:

```rust
#[test] // PARSER.batch.5 (PR #8208)
fn test_parse_malformed_no_section_end() { ... }
```

The category label still names the category being exercised; the
parenthetical names the originating incident. No separate
"regression" taxonomy is needed.

---

## Applicability summary

| Category block | Parsers | Notes |
| -- | -- | -- |
| `PARSER.batch.{1..10}` | All | Universal behavior contract |
| `PARSER.stream.{1..4}` | All | Streaming surface; same logical cases via different harness |
| `PARSER.fmt.{1..4}` | Grammar-conditional | Each variant required only where the grammar permits it |
| `PARSER.xml.{1,2}` | XML-family only | Entity decoding + schema-aware coercion |
| `PARSER.harmony.{1,2}` | Harmony only | Channel routing + envelope variants |

## Adding a new tool-call parser: what you must include

Minimum viable set:

1. `PARSER.batch.{1, 2, 3}` — baseline correctness.
2. `PARSER.batch.4` (or explicit `N/A`) — handle or refuse malformed
   input.
3. `PARSER.batch.5` — pin behavior when the outer fence is missing.
   Silent drop is a regression waiting to happen.
4. `PARSER.batch.{6, 7}` — empty and complex args.
5. `PARSER.stream.{1, 2, 3, 4}` — streaming. Essentially
   non-negotiable for any parser that sits behind a streaming
   frontend.
6. `PARSER.batch.8` — interleaved text.
7. `PARSER.batch.{9, 10}` — empty/null and duplicate calls.
8. Format variants where applicable (`PARSER.fmt.{1..4}`): cover
   any that the parser's grammar permits. Mark `N/A` for those that
   don't apply.
9. Family-specific categories where applicable: `PARSER.xml.{1, 2}`
   for XML grammars, `PARSER.harmony.{1, 2}` for Harmony.

For reasoning parsers, see `REASONING_CASES.md`.
