# Tool-Call / Reasoning Parser Corner Cases

Reference taxonomy for unit testing tool-call and reasoning parsers. Each parser
added under `src/tool_calling/` or `src/reasoning/` should cover the generic
`CASE.<n>` categories; family-specific parsers also cover their respective
`CASE.xml<n>` / `CASE.harmony<n>` categories. `N/A` should be called out
explicitly in the test file rather than silently omitted.

Category layout:
- **`CASE.1`–`CASE.16`** — **Generic**. Apply to every parser regardless of grammar.
- **`CASE.xml1`–`CASE.xml2`** — XML-family only (hermes, glm47, qwen3_coder, minimax_m2, kimi_k2).
- **`CASE.harmony1`** — Harmony only (gpt-oss).

Per-model gap tracking lives elsewhere (not in this repo).

## Quick reference

### Generic (all parsers)

- **`CASE.1`** Single tool call — happy path (one complete, well-formed call). Ex: `xml::test_parse_simple_tool_call`.
- **`CASE.2`** Multiple tool calls — sequential or parallel (2+ in one response). Ex: `tool_choice::test_streaming_required_tool_parallel`.
- **`CASE.3`** No tool call (response is text only). Ex: `xml::test_parse_no_tool_calls`.
- **`CASE.4`** Malformed / partial JSON args (truncated, missing close brace, invalid syntax). Ex: `deepseek_v3::test_parse_..._with_invalid_json`, `xml::test_parse_missing_..._closing_tag`.
- **`CASE.5`** Missing end-token recovery (recover calls when `section_end` is absent due to max_tokens / EOS). Ex: `kimi_k2::test_parse_malformed_no_section_end`.
- **`CASE.6`** Empty args (`arguments={}` / no-arg call). Ex: `kimi_k2::test_parse_no_arg_call`.
- **`CASE.7`** Complex arg types (nested objects, arrays, bool, number, Unicode / newlines in values). Ex: `kimi_k2::test_parse_complex_json_arguments`.
- **`CASE.8`** Streaming — token-by-token assembly + chunk-boundary splits. Ex: `basic::test_buffer_state_persistence_across_calls`, `basic::test_partial_token_matching_closing_tag`, `basic::test_kimi_k2_one_shot_split`.
- **`CASE.9`** Paired reasoning + tool in same response. Ex: `test_reasoning_parser::test_nemotron_with_reasoning_and_tool_calls`.
- **`CASE.10`** Reasoning only (think tags, no tool call). Ex: `basic::test_detect_and_parse_reasoning_reasoning`.
- **`CASE.11`** `tool_choice` = auto / required / named / none. Ex: `tool_choice::test_named_tool_choice_parses_json` (**hermes only today**).
- **`CASE.12`** `finish_reason` semantics (`stop` / `tool_calls` / `length` mapping). Ex: `tool_choice_finish_reasons::*` (hermes only); `test_streaming_tool_parsers::test_qwen_finish_reason_length_vllm`.
- **`CASE.13`** Normal text interleaved with tool calls. Ex: `xml::test_parse_with_normal_text`.
- **`CASE.14`** Empty content / empty `tool_calls` array / null response. Ex: `parallel_tool_call_integration::test_empty_tool_calls`.
- **`CASE.15`** Duplicate tool calls (same name twice). No test anywhere in the repo; universal gap.
- **`CASE.16`** Regression for a specific customer bug (ticket ID referenced in test name or body). Ex: `kimi_k2::test_parse_malformed_no_section_end`.

### XML-family (`CASE.xml*`)

- **`CASE.xml1`** XML entity / HTML unescape handling (`&lt;`, `&amp;`, `&quot;` in parameter values). Ex: `xml::test_html_unescape`, `glm47::test_xml_entity_decoding`.
- **`CASE.xml2`** Schema-aware type coercion (string → number/bool/array based on declared parameter schema). Ex: `xml::test_schema_aware_type_conversion`, `glm47::test_type_coercion_*`.

### Harmony (`CASE.harmony*`)

- **`CASE.harmony1`** Channel / recipient parsing (analysis / commentary / final channels). Ex: `harmony_parser::test_parse_tool_calls_harmony_*`.

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

## `CASE.1` — Single tool call, happy path

One complete, well-formed call in the response.

- Applies to every tool-call parser.
- Baseline correctness check. If `CASE.1` fails, nothing else below matters.
- Example: `dsml/parser.rs::test_parse_single_tool_call_string_param`.

## `CASE.2` — Multiple tool calls (sequential or parallel)

Two or more calls in one response, in the same block or back-to-back.

- Applies to every tool-call parser.
- Some grammars emit parallel calls in one block (DSML, XML); others emit
  sequential top-level sentinels (JSON dialects). Either way, extract all.
- Example: `dsml/parser.rs::test_parse_multiple_tool_calls`,
  `tool_choice::test_streaming_required_tool_parallel`.

## `CASE.3` — No tool call

Response is plain text, no tool-call grammar present.

- Applies to every tool-call parser.
- Must return empty `Vec<ToolCall>` and the input as `normal_text`. Zero false
  positives.
- Example: `dsml/parser.rs::test_parse_no_tool_calls`.

## `CASE.4` — Malformed / partial JSON args

Truncated JSON, missing close brace, invalid syntax inside the arguments
payload.

- Applies to every tool-call parser. For parsers whose grammar never embeds
  JSON (none today — all top-N families embed JSON somewhere), mark explicit
  `N/A`.
- Behavior must be documented: either graceful fallback to string (DSML's
  current behavior via `serde_json::from_str(...).unwrap_or_else(|_| String(...))`)
  or explicit error. Silent drop is the failure mode.
- Example: `dsml/parser.rs::test_parse_deepseek_v4_malformed_json_value_falls_back_to_string`,
  `deepseek_v3_parser.rs::test_parse_tool_calls_deepseek_v3_with_invalid_json`.

## `CASE.5` — Missing end-token recovery

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
- Example (post-recovery): `kimi_k2_parser.rs::test_parse_malformed_no_section_end`.
- Example (behavior-pinning, pre-recovery): `dsml/parser.rs::test_parse_deepseek_v4_missing_end_token`.

## `CASE.6` — Empty args

Tool call with `arguments={}`, or a no-parameter invoke.

- Applies to every tool-call parser.
- Must still return the call — empty args is a valid call, not a missing one.
- Example: `kimi_k2_parser.rs::test_parse_no_arg_call`,
  `dsml/parser.rs::test_parse_deepseek_v4_no_parameters`.

## `CASE.7` — Complex argument types

Nested objects, arrays, booleans, numbers, mixed types, Unicode values, and
newlines inside argument values.

- Applies to every tool-call parser.
- For grammars that carry type hints (DSML's `string="true|false"`), verify
  JSON round-tripping. For XML grammars without hints, the type-coercion
  half of the test is covered under `CASE.xml2` instead — here just verify
  that complex values make it through without truncation or escape bugs.
- Example: `dsml/parser.rs::test_parse_mixed_types_realistic`,
  `kimi_k2_parser.rs::test_parse_complex_json_arguments`.

## `CASE.8` — Streaming

Chunked input arriving over SSE. Covers two concerns that tend to fail
together:

1. **Token-by-token assembly** — the parser incrementally reconstructs the
   tool-call structure across many small chunks.
2. **Chunk-boundary splits** — start fence, end fence, or parameter name /
   value straddles a chunk boundary. Partial-token matching must return
   `true` (keep buffering, don't flush as plain text) and complete the
   match on the next chunk.

- Applies to every tool-call parser. Dominant production path.
- Example: `basic::test_buffer_state_persistence_across_calls`,
  `basic::test_partial_token_matching_closing_tag`,
  `basic::test_kimi_k2_one_shot_split`,
  `test_streaming_tool_parsers::test_deepseek_v4_e2e_fragmented_tokens_vllm`.

## `CASE.9` — Paired reasoning + tool in same response

Model emits `<think>...</think>` (or analog) followed by a tool call. Both
must be extracted: `reasoning_content` populated AND `tool_calls` populated.

- Applies to every (tool, reasoning) parser pair.
- Watch for the "unclosed think-tag swallows tool call" bug — if the reasoning
  parser is greedy it may eat the tool-call content that follows.
- Example: `test_reasoning_parser::test_nemotron_with_reasoning_and_tool_calls`,
  `test_reasoning_parser::test_kimi_k25_with_reasoning_and_tool_calls`.

## `CASE.10` — Reasoning only

`<think>...</think>` or analog present, no tool call. Parser must populate
`reasoning_content` and leave `tool_calls` empty.

- Applies to every reasoning parser.
- Example: `reasoning/base_parser.rs::test_detect_and_parse_reasoning_reasoning`,
  `reasoning/mod.rs::test_deepseek_v4_detect_and_parse`.

## `CASE.11` — `tool_choice` = auto / required / named / none

Each of the four OpenAI `tool_choice` modes exercised per parser.

- Applies to every tool-call parser.
- Cross-parser suites at `lib/llm/tests/tool_choice.rs` /
  `parallel_tool_call_integration.rs` / `tool_choice_finish_reasons.rs`
  run `hermes` only today. Adding a new parser requires parametrizing those
  suites or adding a per-parser equivalent.
- Universal gap across most parsers in the repo as of 2026-04.
- Example: `tool_choice::test_named_tool_choice_parses_json`,
  `tool_choice::test_required_tool_choice_parses_json_array`.

## `CASE.12` — `finish_reason` semantics

`stop` vs `tool_calls` vs `length` mapping, in both streaming and
non-streaming paths.

- Applies to every tool-call parser.
- When a tool call lands, `finish_reason` must become `tool_calls`. When
  `max_tokens` truncates mid-stream, `length` must propagate — this is
  often the signal that should trigger `CASE.5` recovery on the parser side.
- Example: `tool_choice_finish_reasons::test_named_tool_choice_normal_stop_becomes_tool_calls`,
  `test_streaming_tool_parsers::test_qwen_finish_reason_length_vllm`.

## `CASE.13` — Normal text interleaved with tool calls

Model emits narration text before / after / between tool-call blocks. Parser
must split content correctly: text → `normal_content`, calls → `tool_calls`.

- Applies to every tool-call parser.
- Example: `dsml/parser.rs::test_parse_with_normal_text`,
  `test_streaming_tool_parsers.rs::test_deepseek_v4_e2e_content_before_tool_vllm`.

## `CASE.14` — Empty content / empty `tool_calls` array / null response

Engine emits a chunk with `delta.content = ""`, or a final response with
`tool_calls: []`, or `null` values inside arguments.

- Applies to every tool-call parser.
- Null-value handling inside parameters is parser-level (`parse_parameters`
  in DSML handles it via `serde_json::Value::Null`). Empty-choices /
  empty-stream handling is typically at the e2e integration layer.
- Example: `dsml/parser.rs::test_parse_null_parameter`,
  `parallel_tool_call_integration::test_empty_tool_calls`.

## `CASE.15` — Duplicate tool calls (same name twice)

Two calls to the same function name in one response, possibly with the same
arguments.

- Applies to every tool-call parser.
- **Zero coverage across the entire repo as of 2026-04.** Universal gap.
- Expected behavior: both calls must appear in `tool_calls` with distinct
  IDs. (The runtime / client is responsible for deciding whether duplicate
  invocation is intended.)

## `CASE.16` — Regression for a specific customer bug

Test named after (or containing) a ticket reference, pinning the fix for
a customer-reported failure.

- Applies per-incident. Not a category every parser needs to cover in
  advance; populated as bugs are reported and fixed.
- Existing example: `kimi_k2_parser.rs::test_parse_malformed_no_section_end`.

---

## `CASE.xml1` — XML entity / HTML unescape handling

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
- Example: `xml/parser.rs::test_html_unescape`,
  `glm47_parser.rs::test_xml_entity_decoding`.

## `CASE.xml2` — Schema-aware type coercion

Parser uses the declared tool schema to coerce string args to
number / bool / array based on the declared parameter type.

- Applies only to XML-family parsers without explicit type annotations in
  the wire format. `xml/parser.rs`, `glm47_parser.rs` do this.
- **N/A for DSML** — the `string="true|false"` attribute carries the type
  intent per parameter, so no schema lookup is needed.
- **N/A for JSON-family** — JSON has native types.
- **N/A for Harmony** — payload is JSON inside the channel envelope.
- Example: `xml/parser.rs::test_schema_aware_type_conversion`,
  `glm47_parser.rs::test_type_coercion_array_comma_separated`.

---

## `CASE.harmony1` — Channel / recipient parsing

OpenAI Harmony's token stream carries channel metadata
(`<|channel|>analysis|commentary|final<|message|>`) and recipient targets
(`to=functions.foo`). Parser must route the `commentary` channel content
into tool-call extraction while surfacing `analysis` as reasoning and
`final` as the user-visible output.

- **Harmony only.** N/A for every other family.
- Example: `harmony/harmony_parser.rs::test_parse_tool_calls_harmony_with_multi_args`,
  `harmony/harmony_parser.rs::test_parse_tool_calls_harmony_with_normal_text`.

---

## Applicability summary

| Category block | Parsers | Notes |
| -- | -- | -- |
| `CASE.1`–`CASE.16` (generic) | All | Required contract for every parser |
| `CASE.xml1`–`CASE.xml2` | XML-family only | Entity decoding + schema-aware coercion |
| `CASE.harmony1` | Harmony only | Channel routing |

## Adding a new parser: what you must include

Minimum viable set for a new tool-call parser:

1. `CASE.1`, `CASE.2`, `CASE.3` — baseline correctness.
2. `CASE.4` or explicit N/A justification — handle or refuse malformed input.
3. `CASE.5` — pin behavior when the outer fence is missing. Silent drop is a
   regression waiting to happen.
4. `CASE.6`, `CASE.7` — empty and complex args.
5. `CASE.8` — streaming. Essentially non-negotiable for any parser that sits
   behind a streaming frontend.
6. `CASE.13` — interleaved text.
7. `CASE.15` — document whether duplicate calls are supported. Flat gap
   today; landing a test with the parser establishes the contract.
8. Family-specific categories where applicable: `CASE.xml1` / `CASE.xml2`
   for XML grammars, `CASE.harmony1` for Harmony.

For reasoning parsers, replace `CASE.4` / `CASE.5` / `CASE.8`-assembly with
`CASE.8`-partial-close-tag and `CASE.10` (reasoning-only).
