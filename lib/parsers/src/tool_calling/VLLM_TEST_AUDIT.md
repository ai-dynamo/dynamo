# vLLM Tool Parser Test Audit

Source: vLLM `main` at `b53c507bc91f87e28b03e9b54bbff7c76e97d58b` (`https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b`).

Scope: `vllm/tool_parsers/*`, `tests/tool_parsers/*`, `tests/tool_use/*`, and `tests/entrypoints/openai/tool_parsers/*`.

This report expands 493 test rows: 421 explicit test functions plus 72 inherited common-suite rows from `ToolParserTests`.

Bucket references use `lib/parsers/PARSER_CASES.md` (post-#9127 taxonomy; see also `REASONING_CASES.md`, `PIPELINE_CASES.md`, `components/src/dynamo/frontend/tests/FRONTEND_CASES.md` for sibling-doc concerns). A row can have multiple buckets when one test covers several behaviors. Parametrized tests are listed once by test function; inspect the linked source for per-parameter cases.

## Bucket Summary

Counts represent how many rows carry each tag (a row can carry several). Per the
`PARSER_CASES.md:35-38` rule, white-box helper tests are tagged `// helper`
**only** and don't carry a numbered category. Rows previously double-tagged as
`PARSER.<bucket>.<n>` + `// helper` had the numbered tag stripped on
2026-05-08; this is what reduced `PARSER.batch.7` from 86 → 58.

| Bucket | Rows |
| -- | --: |
| `PARSER.batch.1` (single tool call — happy path) | 51 |
| `PARSER.batch.2` *category: multiple tool calls — sequential or parallel* | — |
| `PARSER.batch.2.a` (parallel calls, canonical batched) | 16 |
| `PARSER.batch.2.b` (multi-invoke close-together — same delta or rapid chunks) | 8 |
| `PARSER.batch.2.c` (multi-invoke with surrounding content) | 3 |
| `PARSER.batch.2.d` (id / index distinctness) | 3 |
| `PARSER.batch.3` *category: no tool call (plain text response — parser-side concerns only)* | — |
| `PARSER.batch.3.a` (canonical "no tool call in plain text" — inherited common-suite shape) | 41 |
| `PARSER.batch.3.d` (streaming text-only edge cases — non-duplication / interval / hermes-specific) | 6 |
| `PARSER.batch.4` *category: malformed / partial JSON args (impl-defined recovery)* | — |
| `PARSER.batch.4.a` (generic catch-all — impl-defined no-crash) | 10 |
| `PARSER.batch.4.b` (invalid JSON syntax) | 5 |
| `PARSER.batch.4.c` (missing structural keys) | 4 |
| `PARSER.batch.4.d` (malformed wrapper / XML structure) | 11 |
| `PARSER.batch.5` *category: missing end-token recovery (max_tokens / EOS truncation)* | — |
| `PARSER.batch.5.a` (missing closing tag / fallback no-tags) | 6 |
| `PARSER.batch.5.b` (missing opening tag) | 1 |
| `PARSER.batch.5.c` (truncation / EOS mid-call) | 2 |
| `PARSER.batch.6` *category: empty arguments (parameter-less or `{}`)* | — |
| `PARSER.batch.6.a` (canonical empty `{}`) | 11 |
| `PARSER.batch.6.b` (zero-arg formatting variants — inline / newline / streaming) | 6 |
| `PARSER.batch.6.c` (no-args-key / parameterless) | 4 |
| `PARSER.batch.7` *category: complex argument types (nested / typed / Unicode)* | — |
| `PARSER.batch.7.a` (standard scalar/container types) | 13 |
| `PARSER.batch.7.b` (escaped strings / Unicode / special chars) | 14 |
| `PARSER.batch.7.c` (schema mismatch — string value where schema declares typed primitive) | 15 |
| `PARSER.batch.7.d` (multi-arg / nested / quoted / split-across-chunks) | 16 |
| `PARSER.batch.8` *category: normal text interleaved with tool calls (positional)* | — |
| `PARSER.batch.8.a` (narration before tool call only) | 18 |
| `PARSER.batch.8.b` (narration after tool call only) | 1 |
| `PARSER.batch.8.c` (narration sandwich — both before and after) | 15 |
| `PARSER.batch.8.d` (narration between multiple tool calls) | 5 |
| `PARSER.batch.9` (empty content / `tool_calls=[]` / null response) | 8 |
| `PARSER.batch.10` (duplicate tool calls — same name twice) | 0 |
| `PARSER.stream.1` (single tool call across N chunks) | 177 |
| `PARSER.stream.2` (multiple tool calls, each across N chunks) | 21 |
| `PARSER.stream.3` (partial-token chunking — boundary mid-grammar-token) | 58 |
| `PARSER.stream.4` (streaming termination — final chunk + flush) | 8 |
| `PARSER.fmt.1` (function-name conventions — hyphens, prefixes, validation) | 21 |
| `PARSER.fmt.2` (whitespace / formatting tolerance) | 8 |
| `PARSER.fmt.3` (token / wire-format variants — multiple acceptable spellings) | 23 |
| `PARSER.fmt.4` (empty section / no-content wrappers) | 2 |
| `PARSER.fmt.5` (argument-shape conventions — JSON envelope / native ID / aliasing) | 7 |
| `PARSER.xml.1` (XML entity / HTML unescape handling) | 0 |
| `PARSER.xml.2` (schema-aware type coercion — string → typed via schema) | 6 |
| `PARSER.harmony.1` (channel / recipient parsing — analysis / commentary / final) | 11 |
| `PARSER.harmony.2` (envelope tag grammar — `<\|channel\|>...<\|call\|>` legal variations) | 0 |
| **Sibling-doc buckets:** | |
| `REASONING.batch.1` | 2 |
| `REASONING.batch.2` | 36 |
| `FRONTEND.3/.6` | 43 |
| `FRONTEND.3` | 20 |
| `FRONTEND.1/.3` | 22 |
| `PIPELINE.finish_reason` | 9 |
| `// helper` | 44 |
| `(inline regression annotation)` | 24 |
| `(dissolved; see PARSER.batch.4 impl-defined)` | 25 |
| **Total (distinct test rows)** | 493 |


## Sub-case Taxonomy Proposals (part-2 rollout)

Following the `batch.8.{a,b,c,d}` pilot (PR #9338), this section proposes
sub-case splits for top-level cases where vLLM's existing tests organically
partition along a meaningful axis (i.e., bucket size is shape-variation, not
just parser-distribution). Five buckets qualify; the rest stay flat in
part-2:

- **Split now (part-2):** `batch.2`, `batch.4`, `batch.5`, `batch.6`, `batch.7`.
- **Audit-resolved, fixture rollout deferred to part-3:** `batch.3` —
  proposed `.b/.c/.d` reduced to `.a + .d` after FRONTEND co-tag review
  pulled 9 rows out of `PARSER.batch.3` entirely (now FRONTEND-only). `.a`
  is identical to the existing bare-batch.3 shape (no diagnostic value in
  splitting alone); `.d` is streaming-specific and needs `parse_streaming_increment()`
  harness wiring. Both deferred. See `PARSER.batch.3` section below.
- **Don't split:** `batch.1`, `batch.9`, `batch.10`. Their row counts come
  from the inherited common-suite repeating across N parsers, not from
  genuine shape variation. A split would produce mostly-empty cells.
- **Defer to part-3:** `stream.*`, `fmt.*`, `xml.*`, `harmony.*` —
  smaller buckets with stream-specific or wire-format-specific axes; need
  their own design pass.

Sub-case definitions below quote actual vLLM test names from the per-family
sections to ground each axis. The same `.a/.b/.c/[.d]` letter convention
established by `batch.8` applies — sub-case files in
`tests/parity/parser/fixtures/<family>/PARSER.batch.<n>.yaml` will carry
case IDs `PARSER.batch.<n>.a` through `.d`.

### `PARSER.batch.2` — Multiple tool calls (sequential or parallel)

| Sub-case | Axis | Representative vLLM tests |
| -- | -- | -- |
| `.a` Parallel calls (canonical batched) | All calls present together; harness checks each is extracted | `*.test_parallel_tool_calls` (9 inherited common-suite), `test_extract_tool_calls_with_multiple_tools`, `test_multiple_tool_calls`, `test_parallel_tool_calls_with_results` |
| `.b` Multi-invoke close-together | Calls arrive in same delta or rapid chunks; loop must emit all | `TestMultipleInvokes.test_two_invokes_in_single_delta`, `test_two_invokes_incremental`, `test_streaming_multi_token_per_step`, `test_multiple_tools_chunked` |
| `.c` Multi-invoke with surrounding content | Mixed normal-text + multiple calls | `test_extract_tool_calls_mixed_content`, `test_extract_tool_calls_non_streaming_mixed_content_and_multiple_tool_calls`, `test_extract_tool_calls_streaming_multiple_tool_calls_no_content_between` |
| `.d` ID / index distinctness | Each call gets a unique synthesized id / sequential index | `TestExtractToolCallsStreaming.test_unique_tool_call_ids`, `TestDeltaMessageFormat.test_multi_invoke_indices`, `test_parallel_tool_calls_false` (negative case) |

### `PARSER.batch.3` — No tool call (plain text response)

**Audit resolved (FRONTEND co-tag review):** the originally-proposed `.b`
and `.c` sub-cases turned out to be FRONTEND/PIPELINE-primary, not
parser-side. 9 of 11 rows carry `FRONTEND.*` co-tags and live in
`tests/tool_use/` (request-validation suites), not `tests/tool_parsers/`.
Those rows had their `PARSER.batch.3` tag removed and now live under
`FRONTEND.*` only. The 2 actually-parser-side outliers were reclassified:
`test_extract_tool_calls_streaming_v11_no_tools` → `.d` (streaming),
`TestAdjustRequest.test_no_change_when_no_tools` → `.a` (canonical).

| Sub-case | Axis | Representative vLLM tests |
| -- | -- | -- |
| `.a` Canonical "no tool call in plain text" | Model emits text-only response; parser must return `calls=[]` and the input as `normal_text`. The inherited common-suite baseline. | `*.test_no_tool_calls` (11 inherited), `test_extract_tool_calls_no_tools` (10 across model-specific files), `*.test_no_tool_call`, `test_plain_content`, `test_plain_content_no_tool`, `TestAdjustRequest.test_no_change_when_no_tools` |
| `.d` Streaming text-only edge cases | Streaming: text not duplicated after a tool call, content-at-interval flushing, hermes-specific just-forward-text behavior, v11-no-tools streaming. | `TestExtractToolCallsStreaming.test_streaming_does_not_duplicate_plain_text_after_tool_call`, `test_no_tool_call_streaming`, `test_plain_text_at_interval`, `test_hermes_streaming_just_forward_text_with_stream_interval`, `test_hermes_parser_streaming_just_forward_text`, `test_extract_tool_calls_streaming_v11_no_tools` |

**Implementation status:**
- **`.a` (41 rows)** — exactly mirrors the existing bare `PARSER.batch.3` fixture
  shape across all 19 families (one input: plain-text response, one tool
  declared but unused). Renaming bare `batch.3` → `batch.3.a` is mostly
  cosmetic since there's no shape variation to surface; deferred until
  `batch.3.d` is wired up so the split has diagnostic value.
- **`.d` (6 rows)** — streaming-mode edge cases. Out of scope for the
  current batch-mode harness (`test_parity_parser.py` runs only `parse()`,
  not `parse_streaming_increment()`). Defer to **part-3** alongside
  `PARSER.stream.*` rollout.

**Conclusion:** no fixture work for `batch.3` in part-2. The audit-doc
cleanup (this section + per-row tag fixups) is the deliverable. The 9
formerly-mis-bucketed rows are now correctly tagged FRONTEND-only, ready
for a future `FRONTEND_CASES.md` audit.

### `PARSER.batch.4` — Malformed / partial JSON args

| Sub-case | Axis | Representative vLLM tests |
| -- | -- | -- |
| `.a` Generic catch-all (no-crash) | Impl-defined: parser must not crash on garbage | `*.test_malformed_input` (8 inherited common-suite), `test_malformed_tool_call_no_regex_match`, `test_extract_tool_calls_pre_v11_regex_fallback_fails` |
| `.b` Invalid JSON syntax | Bad quote, extra comma, leaked delimiter chars | `test_invalid_json`, `TestExtractToolCalls.test_invalid_json_still_extracted`, `test_extract_tool_calls_invalid_json`, `TestStreamingExtraction.test_streaming_split_delimiter_no_invalid_json`, `test_hermes_parser_non_streaming_tool_call_invalid_json` |
| `.c` Missing structural keys | name/arguments/parameters key missing in body | `test_extract_tool_calls_missing_name_key`, `test_extract_tool_calls_missing_parameters_and_arguments_key`, `test_extract_tool_calls_missing_name_or_arguments`, `test_responses_request_named_tool_choice_missing_name` |
| `.d` Malformed wrapper / XML structure | Unclosed/missing tags, missing delimiter | `test_extract_tool_calls_malformed_xml`, `test_malformed_xml_no_gt_delimiter`, `test_extract_tool_calls_incomplete_tool_call`, `test_extract_tool_calls_multiline_json_not_supported`, `TestExtractToolCalls.test_invalid_funcall_id_skipped`, `TestExtractToolCallsStreaming.test_no_emission_while_incomplete` |

### `PARSER.batch.5` — Missing end-token recovery

| Sub-case | Axis | Representative vLLM tests |
| -- | -- | -- |
| `.a` Missing closing tag | Open `<tool_call>` / `<parameter>` without matching close | `test_extract_tool_calls_missing_closing_parameter_tag`, `test_extract_tool_calls_streaming_missing_closing_tag`, `test_extract_tool_calls_fallback_no_tags` |
| `.b` Missing opening tag | Closing tag with no corresponding open (less common) | `test_extract_tool_calls_streaming_missing_opening_tag` |
| `.c` Truncation / EOS mid-call | Stream ends mid-payload (max_tokens hit, model stopped) | `TestStreamingEdgeCases.test_truncated_tool_call_no_end_marker`, `test_hermes_parser_non_streaming_tool_call_until_eos` |

(No `.d` — `batch.5` is small (9 rows) and 3 axes cover the population.)

### `PARSER.batch.6` — Empty args

| Sub-case | Axis | Representative vLLM tests |
| -- | -- | -- |
| `.a` Canonical empty `{}` | Empty object literal in arguments | `*.test_empty_arguments` (8 inherited common-suite), `TestExtractToolCallsStreaming.test_empty_arguments_streaming`, `test_extract_tool_calls_empty_arguments`, `TestStreamingExtraction.test_streaming_empty_args` |
| `.b` Zero-arg formatting variants | Inline vs newline emission, streaming vs non-streaming | `TestGlm47ExtractToolCalls.test_zero_arg_inline`, `test_zero_arg_newline`, `TestGlm47Streaming.test_no_args`, `TestHYV3ExtractToolCalls.test_zero_arg_inline`/`_newline`, `TestHYV3ExtractToolCallsStreaming.test_zero_arg_streaming` |
| `.c` No-args-key / parameterless | `arguments` key absent entirely (vs `{}`) | `TestExtractToolCalls.test_single_tool_no_params`, `TestExtractToolCalls.test_no_arguments`, `test_zero_argument_tool_call`, `test_extract_tool_calls_v11_without_args_skipped` |

(No `.d` — 3 axes cover the 21-row population.)

### `PARSER.batch.7` — Complex argument types

| Sub-case | Axis | Representative vLLM tests |
| -- | -- | -- |
| `.a` Standard scalar/container types | int, float, bool, null, list, object — canonical type matrix | `*.test_various_data_types` (8 inherited common-suite), `test_extract_tool_calls_numeric_deserialization`, `TestExtractToolCalls.test_tool_call_with_number_and_boolean`, `TestStreamingExtraction.test_streaming_numeric_args` |
| `.b` Escaped / Unicode / special chars | String escapes, Unicode preservation, HTML-in-args | `*.test_escaped_strings` (8 inherited common-suite), `test_unicode_characters_preserved`, `test_extract_tool_calls_special_characters`, `test_streaming_json_escape_in_string`, `TestStreamingExtraction.test_streaming_html_argument_does_not_duplicate_tag_prefixes`, `test_streaming_incremental_string_value` |
| `.c` Schema mismatch — string value where schema declares typed primitive | Input has `{"celsius": "20"}` while schema says `celsius: integer`. The contract pinned is *value-preservation* (Dynamo's behavior); a few parsers (vLLM's deepseek_v3_2) coerce at the parser layer and surface as divergences. | `TestExtractToolCalls.test_type_conversion_in_non_streaming`, `TestExtractToolCallsStreaming.test_type_conversion_in_streaming`, `test_convert_param_value_single_types`, `test_convert_param_value_multi_typed_values`, `test_convert_param_value_stricter_type_checking`, `test_convert_param_value_edge_cases`, `test_extract_tool_calls_type_conversion` |
| `.d` Multi-arg / nested / quoted | Multiple parameters, deep nesting, quotes/brackets inside strings, primitives split across stream chunks | `TestExtractToolCalls.test_multiple_arguments`, `test_nested_arguments`, `TestStreamingExtraction.test_streaming_multi_arg`, `test_extract_tool_calls_deeply_nested_json`, `test_extract_tool_calls_with_quotes_and_brackets_in_string`, `test_extract_tool_calls_complex_type_with_single_quote`, `TestStreamingExtraction.test_streaming_boolean_split_across_chunks`, `test_streaming_false_split_across_chunks`, `test_streaming_number_split_across_chunks`, `test_streaming_trailing_bare_bool_not_duplicated`, `test_streaming_multi_token_with_multiple_args`, `test_hermes_streaming_boolean_args_with_stream_interval` |

### Buckets explicitly NOT split in part-2

- **`batch.1` (51 rows)** — Distribution: 8× `*_simple_args` + 11× `test_extract_tool_calls` + 4× `test_tool_call` + 3× `test_single_tool_call` are the same canonical happy-path tests inherited across N parsers. Splitting would produce a 4×N matrix where each cell is the same shape repeated.
- **`batch.9` (8 rows)** — Already trivially partitioned in spec body (`""` / `null` / `[]`) — no new axis to add.
- **`batch.10` (0 rows)** — Effectively no vLLM coverage (Dynamo-only authored).

(`batch.3` was initially listed here but a closer audit promoted it to a
part-3 candidate with a proposed `.a/.b/.c/.d` split — see the `PARSER.batch.3`
section above.)

### Buckets deferred to part-3

`PARSER.stream.*` (264 rows total — heavy), `PARSER.fmt.*` (61 rows),
`PARSER.xml.*` (6 rows), `PARSER.harmony.*` (11 rows). Stream sub-cases need
their own design pass (chunk-boundary axes are different from batch axes).
The fmt/xml/harmony buckets are smaller and may not pay off; assess in
part-3.


## Model / Parser Sections

### DeepSeek V3.1

Model-specific / endpoint-specific rows:

| Test | Bucket(s) | Link | Notes |
| -- | -- | -- | -- |
| `test_extract_tool_calls_with_tool` | `PARSER.batch.1` | [source](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/test_deepseekv31_tool_parser.py#L24) | test extract tool calls with tool |
| `test_extract_tool_calls_with_multiple_tools` | `PARSER.batch.2.a` | [source](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/test_deepseekv31_tool_parser.py#L39) | test extract tool calls with multiple tools |

### DeepSeek V3.2

Model-specific / endpoint-specific rows:

| Test | Bucket(s) | Link | Notes |
| -- | -- | -- | -- |
| `TestConvertParamValue.test_null` | `// helper` | [source](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/test_deepseekv32_tool_parser.py#L78) | TestConvertParamValue.test null |
| `TestConvertParamValue.test_string` | `// helper` | [source](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/test_deepseekv32_tool_parser.py#L82) | TestConvertParamValue.test string |
| `TestConvertParamValue.test_integer_valid` | `// helper` | [source](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/test_deepseekv32_tool_parser.py#L85) | TestConvertParamValue.test integer valid |
| `TestConvertParamValue.test_integer_invalid_falls_back_to_str` | `// helper` | [source](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/test_deepseekv32_tool_parser.py#L88) | TestConvertParamValue.test integer invalid falls back to str |
| `TestConvertParamValue.test_number_float` | `// helper` | [source](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/test_deepseekv32_tool_parser.py#L91) | TestConvertParamValue.test number float |
| `TestConvertParamValue.test_number_whole_returns_int` | `// helper` | [source](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/test_deepseekv32_tool_parser.py#L94) | TestConvertParamValue.test number whole returns int |
| `TestConvertParamValue.test_boolean_true` | `// helper` | [source](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/test_deepseekv32_tool_parser.py#L98) | TestConvertParamValue.test boolean true |
| `TestConvertParamValue.test_boolean_false` | `// helper` | [source](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/test_deepseekv32_tool_parser.py#L102) | TestConvertParamValue.test boolean false |
| `TestConvertParamValue.test_object_valid_json` | `// helper` | [source](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/test_deepseekv32_tool_parser.py#L106) | TestConvertParamValue.test object valid json |
| `TestConvertParamValue.test_object_invalid_json_falls_back` | `// helper` | [source](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/test_deepseekv32_tool_parser.py#L109) | TestConvertParamValue.test object invalid json falls back |
| `TestConvertParamValue.test_array_valid_json` | `// helper` | [source](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/test_deepseekv32_tool_parser.py#L112) | TestConvertParamValue.test array valid json |
| `TestConvertParamValue.test_unknown_type_tries_json_then_string` | `// helper` | [source](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/test_deepseekv32_tool_parser.py#L115) | TestConvertParamValue.test unknown type tries json then string |
| `TestExtractToolCalls.test_no_tool_call` | `PARSER.batch.3.a` | [source](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/test_deepseekv32_tool_parser.py#L130) | TestExtractToolCalls.test no tool call |
| `TestExtractToolCalls.test_single_tool_no_params` | `PARSER.batch.1`, `PARSER.batch.6.c` | [source](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/test_deepseekv32_tool_parser.py#L136) | TestExtractToolCalls.test single tool no params |
| `TestExtractToolCalls.test_single_tool_with_params` | `PARSER.batch.1` | [source](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/test_deepseekv32_tool_parser.py#L144) | TestExtractToolCalls.test single tool with params |
| `TestExtractToolCalls.test_content_before_tool_call` | `PARSER.batch.8.a` | [source](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/test_deepseekv32_tool_parser.py#L158) | TestExtractToolCalls.test content before tool call |
| `TestExtractToolCalls.test_no_content_prefix_returns_none` | `PARSER.batch.1` | [source](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/test_deepseekv32_tool_parser.py#L166) | TestExtractToolCalls.test no content prefix returns none |
| `TestExtractToolCalls.test_multiple_tools` | `PARSER.batch.2.a` | [source](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/test_deepseekv32_tool_parser.py#L172) | TestExtractToolCalls.test multiple tools |
| `TestExtractToolCalls.test_type_conversion_in_non_streaming` | `PARSER.batch.7.c` | [source](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/test_deepseekv32_tool_parser.py#L191) | Non-streaming extraction must convert params using the tool schema. |
| `TestExtractToolCallsStreaming.test_plain_content_no_tool` | `PARSER.batch.3.a`, `PARSER.stream.1` | [source](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/test_deepseekv32_tool_parser.py#L277) | TestExtractToolCallsStreaming.test plain content no tool |
| `TestExtractToolCallsStreaming.test_single_tool_streaming` | `PARSER.batch.1`, `PARSER.stream.1` | [source](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/test_deepseekv32_tool_parser.py#L284) | TestExtractToolCallsStreaming.test single tool streaming |
| `TestExtractToolCallsStreaming.test_tool_name_emitted` | `PARSER.stream.1` | [source](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/test_deepseekv32_tool_parser.py#L290) | TestExtractToolCallsStreaming.test tool name emitted |
| `TestExtractToolCallsStreaming.test_content_before_tool_call_streaming` | `PARSER.stream.1`, `PARSER.batch.8.a` | [source](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/test_deepseekv32_tool_parser.py#L302) | TestExtractToolCallsStreaming.test content before tool call streaming |
| `TestExtractToolCallsStreaming.test_type_conversion_in_streaming` | `PARSER.batch.7.c`, `PARSER.stream.1` | [source](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/test_deepseekv32_tool_parser.py#L308) | TestExtractToolCallsStreaming.test type conversion in streaming |
| `TestExtractToolCallsStreaming.test_multiple_tools_streaming` | `PARSER.batch.2.b`, `PARSER.stream.2` | [source](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/test_deepseekv32_tool_parser.py#L327) | TestExtractToolCallsStreaming.test multiple tools streaming |
| `TestExtractToolCallsStreaming.test_state_reset_on_new_stream` | `PARSER.stream.4` | [source](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/test_deepseekv32_tool_parser.py#L354) | A second stream (previous_text == '') must reset state cleanly. |
| `TestExtractToolCallsStreaming.test_empty_arguments_streaming` | `PARSER.batch.6.a`, `PARSER.stream.1` | [source](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/test_deepseekv32_tool_parser.py#L363) | Invoke block with zero parameters should produce empty JSON. |
| `TestExtractToolCallsStreaming.test_unique_tool_call_ids` | `PARSER.batch.2.d`, `PARSER.stream.1`, `PARSER.stream.2` | [source](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/test_deepseekv32_tool_parser.py#L370) | Each tool call in a parallel stream gets a distinct synthesized id. DSv3.2 has no native call-ID surface — this is parallel-call distinctness, not fmt.5 native-ID preservation. |
| `TestExtractToolCallsStreaming.test_eos_after_tool_calls` | `PARSER.stream.3`, `PARSER.stream.4`, `PIPELINE.finish_reason` | [source](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/test_deepseekv32_tool_parser.py#L393) | EOS token (empty delta_text, non-empty delta_token_ids) returns a non-None DeltaMessage so the serving framework can finalize. |
| `TestExtractToolCallsStreaming.test_streaming_matches_non_streaming` | `PARSER.stream.1` | [source](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/test_deepseekv32_tool_parser.py#L413) | Streaming and non-streaming must produce the same result. |
| `TestExtractToolCallsStreaming.test_single_tool_chunked_stream_interval` | `PARSER.batch.1`, `PARSER.stream.1`, `PARSER.stream.3` | [source](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/test_deepseekv32_tool_parser.py#L466) | Start token split across chunks (stream interval > 1). |
| `TestExtractToolCallsStreaming.test_content_before_tool_chunked` | `PARSER.stream.1`, `PARSER.stream.3`, `PARSER.batch.8.a` | [source](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/test_deepseekv32_tool_parser.py#L474) | Content before tool call with chunked streaming. |
| `TestExtractToolCallsStreaming.test_multiple_tools_chunked` | `PARSER.batch.2.b`, `PARSER.stream.2`, `PARSER.stream.3` | [source](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/test_deepseekv32_tool_parser.py#L483) | Multiple tools with chunked streaming. |
| `TestExtractToolCallsStreaming.test_no_emission_while_incomplete` | `PARSER.batch.4.d`, `PARSER.stream.4` | [source](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/test_deepseekv32_tool_parser.py#L499) | No tool calls should be emitted until an invoke block completes. |
| `TestExtractToolCallsStreaming.test_no_marker_leak_chunked` | `PARSER.stream.3`, `(inline regression annotation)` | [source](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/test_deepseekv32_tool_parser.py#L511) | Chunked streaming must NOT leak DSML start-marker fragments as content (GitHub #40801). |
| `TestExtractToolCallsStreaming.test_no_marker_leak_with_prefix_chunked` | `PARSER.stream.3`, `(inline regression annotation)` | [source](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/test_deepseekv32_tool_parser.py#L521) | Content before a tool call must not include start-marker fragments when chunked (GitHub #40801). |
| `TestExtractToolCallsStreaming.test_no_marker_leak_char_by_char` | `PARSER.stream.3`, `(inline regression annotation)` | [source](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/test_deepseekv32_tool_parser.py#L533) | Character-by-character streaming must not leak marker fragments (GitHub #40801). |
| `TestExtractToolCallsStreaming.test_no_marker_leak_all_split_points` | `PARSER.stream.3`, `(inline regression annotation)` | [source](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/test_deepseekv32_tool_parser.py#L543) | Start token split at every possible boundary must not leak (GitHub #40801). |
| `TestExtractToolCallsStreaming.test_false_partial_marker_emitted` | `PARSER.stream.3` | [source](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/test_deepseekv32_tool_parser.py#L555) | Text ending with a prefix of the start token that turns out NOT to be a marker must still be emitted as content. |
| `TestDelimiterPreservation.test_delimiter_preserved_fast_detokenization` | `PARSER.stream.3`, `FRONTEND.3/.6` | [source](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/test_deepseekv32_tool_parser.py#L571) | DSML delimiters as literal text must still be detected. |
| `TestDelimiterPreservation.test_tool_detection_skip_special_tokens_false` | `PARSER.stream.3`, `(inline regression annotation)`, `FRONTEND.3/.6` | [source](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/test_deepseekv32_tool_parser.py#L599) | Regression: skip_special_tokens must be False when tools are enabled. |
| `test_convert_param_value_single_types` | `PARSER.batch.7.c` | [source](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/test_deepseekv32_tool_parser.py#L656) | Test _convert_param_value with single type parameters. |
| `test_convert_param_value_multi_typed_values` | `PARSER.batch.7.c` | [source](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/test_deepseekv32_tool_parser.py#L703) | Test _convert_param_value with multi-typed values (list of types). |
| `test_convert_param_value_stricter_type_checking` | `PARSER.batch.7.c` | [source](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/test_deepseekv32_tool_parser.py#L746) | Test stricter type checking in the updated implementation. |
| `test_convert_param_value_edge_cases` | `PARSER.batch.7.c` | [source](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/test_deepseekv32_tool_parser.py#L779) | Test edge cases for _convert_param_value. |
| `test_convert_param_value_checked_helper` | `// helper` | [source](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/test_deepseekv32_tool_parser.py#L805) | Test the _convert_param_value_checked helper function indirectly. |

### DeepSeek V3

Common suite config: [TestDeepSeekV3ToolParser](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/test_deepseekv3_tool_parser.py#L20).

Inherited common `ToolParserTests` rows:

| Test | Bucket(s) | Link | Notes |
| -- | -- | -- | -- |
| `TestDeepSeekV3ToolParser.test_no_tool_calls` | `PARSER.batch.3.a`, `PARSER.stream.1` | [source](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/common_tests.py#L125) | Plain text produces no tool calls; common suite runs streaming and non-streaming. Model config: tests/tool_parsers/test_deepseekv3_tool_parser.py:20. |
| `TestDeepSeekV3ToolParser.test_single_tool_call_simple_args` | `PARSER.batch.1`, `PARSER.stream.1` | [source](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/common_tests.py#L145) | One simple call in streaming and non-streaming. Model config: tests/tool_parsers/test_deepseekv3_tool_parser.py:20. |
| `TestDeepSeekV3ToolParser.test_parallel_tool_calls` | `PARSER.batch.2.a`, `PARSER.stream.2`, `PARSER.fmt.1` | [source](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/common_tests.py#L185) | Multiple calls plus unique ID assertion. Model config: tests/tool_parsers/test_deepseekv3_tool_parser.py:20. |
| `TestDeepSeekV3ToolParser.test_various_data_types` | `PARSER.batch.7.a`, `PARSER.stream.1` | [source](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/common_tests.py#L220) | All JSON scalar/container argument types. Model config: tests/tool_parsers/test_deepseekv3_tool_parser.py:20. |
| `TestDeepSeekV3ToolParser.test_empty_arguments` | `PARSER.batch.6.a`, `PARSER.stream.1` | [source](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/common_tests.py#L260) | Parameterless call. Model config: tests/tool_parsers/test_deepseekv3_tool_parser.py:20. |
| `TestDeepSeekV3ToolParser.test_surrounding_text` | `PARSER.batch.8.c`, `PARSER.stream.1` | [source](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/common_tests.py#L282) | Tool call surrounded by normal text. Model config: tests/tool_parsers/test_deepseekv3_tool_parser.py:20. |
| `TestDeepSeekV3ToolParser.test_escaped_strings` | `PARSER.batch.7.b`, `PARSER.stream.1` | [source](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/common_tests.py#L302) | Escaped strings / Unicode-like values. Model config: tests/tool_parsers/test_deepseekv3_tool_parser.py:20. |
| `TestDeepSeekV3ToolParser.test_malformed_input` | `PARSER.batch.4.a`, `PARSER.stream.1`, `(dissolved; see PARSER.batch.4 impl-defined)` | [source](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/common_tests.py#L322) | Malformed inputs must not crash. Model config: tests/tool_parsers/test_deepseekv3_tool_parser.py:20. |
| `TestDeepSeekV3ToolParser.test_streaming_reconstruction` | `PARSER.stream.1` | [source](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/common_tests.py#L340) | Streaming output should match non-streaming output. Model config: tests/tool_parsers/test_deepseekv3_tool_parser.py:20. |

### DeepSeek V4

Model-specific / endpoint-specific rows:

| Test | Bucket(s) | Link | Notes |
| -- | -- | -- | -- |
| `test_registered` | `PARSER.fmt.3` | [source](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/test_deepseekv4_tool_parser.py#L121) | test registered |
| `test_extract_tool_calls` | `PARSER.batch.1` | [source](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/test_deepseekv4_tool_parser.py#L125) | test extract tool calls |
| `test_function_calls_block_is_not_accepted` | `PARSER.fmt.3` | [source](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/test_deepseekv4_tool_parser.py#L144) | test function calls block is not accepted |
| `test_streaming_extracts_complete_invokes` | `PARSER.stream.1` | [source](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/test_deepseekv4_tool_parser.py#L156) | test streaming extracts complete invokes |
| `test_get_vllm_registry_structural_tag_returns_structural_tag` | `PARSER.fmt.3`, `FRONTEND.3/.6` | [source](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/test_deepseekv4_tool_parser.py#L172) | test get vllm registry structural tag returns structural tag |

### ERNIE 4.5 MoE

Model-specific / endpoint-specific rows:

| Test | Bucket(s) | Link | Notes |
| -- | -- | -- | -- |
| `test_extract_tool_calls_no_tools` | `PARSER.batch.3.a` | [source](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/test_ernie45_moe_tool_parser.py#L53) | test extract tool calls no tools |
| `test_extract_tool_calls` | `PARSER.batch.1` | [source](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/test_ernie45_moe_tool_parser.py#L161) | test extract tool calls Parametrized. |
| `test_extract_tool_calls_streaming_incremental` | `PARSER.stream.3` | [source](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/test_ernie45_moe_tool_parser.py#L323) | Verify the Ernie45 Parser streaming behavior by verifying each chunk is as expected. Parametrized. |

### FunctionGemma

Model-specific / endpoint-specific rows:

| Test | Bucket(s) | Link | Notes |
| -- | -- | -- | -- |
| `TestExtractToolCalls.test_no_tool_calls` | `PARSER.batch.3.a` | [source](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/test_functiongemma_tool_parser.py#L34) | TestExtractToolCalls.test no tool calls |
| `TestExtractToolCalls.test_single_tool_call` | `PARSER.batch.1` | [source](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/test_functiongemma_tool_parser.py#L42) | TestExtractToolCalls.test single tool call |
| `TestExtractToolCalls.test_multiple_arguments` | `PARSER.batch.7.d` | [source](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/test_functiongemma_tool_parser.py#L54) | TestExtractToolCalls.test multiple arguments |
| `TestExtractToolCalls.test_text_before_tool_call` | `PARSER.batch.8.a` | [source](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/test_functiongemma_tool_parser.py#L70) | TestExtractToolCalls.test text before tool call |
| `TestExtractToolCalls.test_multiple_tool_calls` | `PARSER.batch.2.a` | [source](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/test_functiongemma_tool_parser.py#L81) | Two parallel calls (`get_weather` + `get_time`). |
| `TestParseArguments.test_empty_arguments` | `// helper` | [source](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/test_functiongemma_tool_parser.py#L97) | TestParseArguments.test empty arguments |
| `TestParseArguments.test_single_string_argument` | `// helper` | [source](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/test_functiongemma_tool_parser.py#L101) | TestParseArguments.test single string argument |
| `TestParseArguments.test_multiple_arguments` | `// helper` | [source](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/test_functiongemma_tool_parser.py#L105) | TestParseArguments.test multiple arguments |
| `TestParseArguments.test_numeric_argument` | `// helper` | [source](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/test_functiongemma_tool_parser.py#L110) | TestParseArguments.test numeric argument |
| `TestParseArguments.test_boolean_argument` | `// helper` | [source](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/test_functiongemma_tool_parser.py#L114) | TestParseArguments.test boolean argument |
| `TestParseArguments.test_argument_with_spaces` | `// helper` | [source](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/test_functiongemma_tool_parser.py#L118) | TestParseArguments.test argument with spaces |
| `TestAdjustRequest.test_skip_special_tokens_disabled` | `FRONTEND.3/.6` | [source](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/test_functiongemma_tool_parser.py#L124) | TestAdjustRequest.test skip special tokens disabled |
| `TestAdjustRequest.test_skip_special_tokens_when_tool_choice_none` | `FRONTEND.3/.6` | [source](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/test_functiongemma_tool_parser.py#L132) | TestAdjustRequest.test skip special tokens when tool choice none |
| `TestBufferDeltaText.test_regular_text_not_buffered` | `// helper` | [source](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/test_functiongemma_tool_parser.py#L142) | TestBufferDeltaText.test regular text not buffered |
| `TestBufferDeltaText.test_complete_tag_flushed` | `// helper` | [source](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/test_functiongemma_tool_parser.py#L147) | TestBufferDeltaText.test complete tag flushed |

### Gemma 4

Model-specific / endpoint-specific rows:

| Test | Bucket(s) | Link | Notes |
| -- | -- | -- | -- |
| `TestParseGemma4Args.test_empty_string` | `// helper` | [source](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/test_gemma4_tool_parser.py#L52) | TestParseGemma4Args.test empty string |
| `TestParseGemma4Args.test_whitespace_only` | `// helper` | [source](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/test_gemma4_tool_parser.py#L55) | TestParseGemma4Args.test whitespace only |
| `TestParseGemma4Args.test_single_string_value` | `// helper` | [source](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/test_gemma4_tool_parser.py#L58) | TestParseGemma4Args.test single string value |
| `TestParseGemma4Args.test_string_value_with_comma` | `// helper` | [source](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/test_gemma4_tool_parser.py#L62) | TestParseGemma4Args.test string value with comma |
| `TestParseGemma4Args.test_multiple_string_values` | `// helper` | [source](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/test_gemma4_tool_parser.py#L66) | TestParseGemma4Args.test multiple string values |
| `TestParseGemma4Args.test_integer_value` | `// helper` | [source](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/test_gemma4_tool_parser.py#L72) | TestParseGemma4Args.test integer value |
| `TestParseGemma4Args.test_float_value` | `// helper` | [source](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/test_gemma4_tool_parser.py#L76) | TestParseGemma4Args.test float value |
| `TestParseGemma4Args.test_boolean_true` | `// helper` | [source](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/test_gemma4_tool_parser.py#L80) | TestParseGemma4Args.test boolean true |
| `TestParseGemma4Args.test_boolean_false` | `// helper` | [source](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/test_gemma4_tool_parser.py#L84) | TestParseGemma4Args.test boolean false |
| `TestParseGemma4Args.test_null_value` | `// helper` | [source](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/test_gemma4_tool_parser.py#L88) | TestParseGemma4Args.test null value |
| `TestParseGemma4Args.test_mixed_types` | `// helper` | [source](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/test_gemma4_tool_parser.py#L96) | TestParseGemma4Args.test mixed types |
| `TestParseGemma4Args.test_nested_object` | `// helper` | [source](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/test_gemma4_tool_parser.py#L107) | TestParseGemma4Args.test nested object |
| `TestParseGemma4Args.test_array_of_strings` | `// helper` | [source](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/test_gemma4_tool_parser.py#L111) | TestParseGemma4Args.test array of strings |
| `TestParseGemma4Args.test_unterminated_string` | `// helper` | [source](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/test_gemma4_tool_parser.py#L115) | Unterminated strings should take everything after the delimiter. |
| `TestParseGemma4Args.test_empty_value` | `// helper` | [source](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/test_gemma4_tool_parser.py#L120) | Key with no value after colon. |
| `TestParseGemma4Args.test_empty_value_partial_withheld` | `// helper`, `(dissolved; see PARSER.batch.4 impl-defined)` | [source](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/test_gemma4_tool_parser.py#L125) | Key with no value is withheld in partial mode to avoid premature emission. |
| `TestParseGemma4Args.test_empty_value_after_other_keys_partial_withheld` | `// helper`, `(dissolved; see PARSER.batch.4 impl-defined)` | [source](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/test_gemma4_tool_parser.py#L133) | Trailing key with no value is withheld; earlier keys are kept. |
| `TestParseGemma4Array.test_string_array` | `// helper` | [source](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/test_gemma4_tool_parser.py#L140) | TestParseGemma4Array.test string array |
| `TestParseGemma4Array.test_empty_array` | `// helper` | [source](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/test_gemma4_tool_parser.py#L144) | TestParseGemma4Array.test empty array |
| `TestParseGemma4Array.test_bare_values` | `// helper` | [source](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/test_gemma4_tool_parser.py#L148) | TestParseGemma4Array.test bare values |
| `TestExtractToolCalls.test_no_tool_calls` | `PARSER.batch.3.a` | [source](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/test_gemma4_tool_parser.py#L159) | TestExtractToolCalls.test no tool calls |
| `TestExtractToolCalls.test_single_tool_call` | `PARSER.batch.1` | [source](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/test_gemma4_tool_parser.py#L167) | TestExtractToolCalls.test single tool call |
| `TestExtractToolCalls.test_multiple_arguments` | `PARSER.batch.7.d` | [source](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/test_gemma4_tool_parser.py#L179) | TestExtractToolCalls.test multiple arguments |
| `TestExtractToolCalls.test_text_before_tool_call` | `PARSER.batch.8.a` | [source](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/test_gemma4_tool_parser.py#L194) | TestExtractToolCalls.test text before tool call |
| `TestExtractToolCalls.test_multiple_tool_calls` | `PARSER.batch.2.a` | [source](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/test_gemma4_tool_parser.py#L207) | Two parallel calls (`get_weather` + `get_time`). |
| `TestExtractToolCalls.test_nested_arguments` | `PARSER.batch.7.d` | [source](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/test_gemma4_tool_parser.py#L221) | TestExtractToolCalls.test nested arguments |
| `TestExtractToolCalls.test_tool_call_with_number_and_boolean` | `PARSER.batch.7.a` | [source](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/test_gemma4_tool_parser.py#L236) | TestExtractToolCalls.test tool call with number and boolean |
| `TestExtractToolCalls.test_incomplete_tool_call` | `PARSER.batch.4.d` | [source](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/test_gemma4_tool_parser.py#L252) | TestExtractToolCalls.test incomplete tool call |
| `TestExtractToolCalls.test_hyphenated_function_name` | `PARSER.fmt.1` | [source](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/test_gemma4_tool_parser.py#L260) | Ensure function names with hyphens are parsed correctly. |
| `TestExtractToolCalls.test_dotted_function_name` | `PARSER.fmt.1` | [source](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/test_gemma4_tool_parser.py#L270) | Ensure function names with dots are parsed correctly. |
| `TestExtractToolCalls.test_no_arguments` | `PARSER.batch.6.c` | [source](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/test_gemma4_tool_parser.py#L280) | Tool calls with empty arguments. |
| `TestStreamingExtraction.test_basic_streaming_single_tool` | `PARSER.batch.1`, `PARSER.stream.1` | [source](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/test_gemma4_tool_parser.py#L372) | Simulate the exact streaming scenario from the bug report. Model generates: <\|tool_call>call:get_weather{location:<\|"\|>Paris, France<\|"\|>}<tool_call\|> Expected: arguments should be |
| `TestStreamingExtraction.test_streaming_multi_arg` | `PARSER.batch.7.d`, `PARSER.stream.1` | [source](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/test_gemma4_tool_parser.py#L401) | Streaming with multiple arguments. |
| `TestStreamingExtraction.test_streaming_no_extra_brace` | `PARSER.stream.1`, `(inline regression annotation)`, `(dissolved; see PARSER.batch.4 impl-defined)` | [source](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/test_gemma4_tool_parser.py#L421) | Verify the closing } is NOT leaked into arguments (Bug #2). |
| `TestStreamingExtraction.test_streaming_no_unquoted_keys` | `PARSER.stream.1`, `(inline regression annotation)`, `(dissolved; see PARSER.batch.4 impl-defined)` | [source](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/test_gemma4_tool_parser.py#L443) | Verify keys are properly quoted in JSON (Bug #1). |
| `TestStreamingExtraction.test_streaming_name_no_call_prefix` | `PARSER.stream.1`, `PARSER.fmt.1` | [source](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/test_gemma4_tool_parser.py#L463) | Verify function name has no 'call:' prefix. |
| `TestStreamingExtraction.test_streaming_text_before_tool_call` | `PARSER.stream.1`, `PARSER.batch.8.a` | [source](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/test_gemma4_tool_parser.py#L477) | Text before tool call should be emitted as content. |
| `TestStreamingExtraction.test_streaming_numeric_args` | `PARSER.batch.7.a`, `PARSER.stream.1` | [source](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/test_gemma4_tool_parser.py#L498) | Streaming with numeric and boolean argument values. |
| `TestStreamingExtraction.test_streaming_boolean_split_across_chunks` | `PARSER.batch.7.d`, `PARSER.stream.3`, `(dissolved; see PARSER.batch.4 impl-defined)` | [source](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/test_gemma4_tool_parser.py#L515) | Boolean value split across token boundaries must not corrupt JSON. |
| `TestStreamingExtraction.test_streaming_false_split_across_chunks` | `PARSER.batch.7.d`, `PARSER.stream.3` | [source](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/test_gemma4_tool_parser.py#L530) | Boolean false split across chunks. |
| `TestStreamingExtraction.test_streaming_number_split_across_chunks` | `PARSER.batch.7.d`, `PARSER.stream.3` | [source](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/test_gemma4_tool_parser.py#L545) | Number split across chunks must not change type. |
| `TestStreamingExtraction.test_streaming_empty_args` | `PARSER.batch.6.a`, `PARSER.stream.1` | [source](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/test_gemma4_tool_parser.py#L560) | Tool call with no arguments. |
| `TestStreamingExtraction.test_streaming_split_delimiter_no_invalid_json` | `PARSER.batch.4.b`, `PARSER.stream.3`, `(dissolved; see PARSER.batch.4 impl-defined)` | [source](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/test_gemma4_tool_parser.py#L572) | Partial <\|"\|> delimiter chars must not leak into streamed JSON. Reproduces the bug from https://github.com/vllm-project/vllm/issues/38946 where a token boundary splits the string d |
| `TestStreamingExtraction.test_streaming_does_not_duplicate_plain_text_after_tool_call` | `PARSER.batch.3.d`, `PARSER.stream.1`, `(dissolved; see PARSER.batch.4 impl-defined)` | [source](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/test_gemma4_tool_parser.py#L601) | Buffered plain text after a tool call must not corrupt current_text. |
| `TestStreamingExtraction.test_streaming_html_argument_does_not_duplicate_tag_prefixes` | `PARSER.batch.7.b`, `PARSER.stream.1`, `(dissolved; see PARSER.batch.4 impl-defined)` | [source](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/test_gemma4_tool_parser.py#L630) | HTML content inside tool arguments must not be duplicated. |
| `TestStreamingExtraction.test_streaming_trailing_bare_bool_not_duplicated` | `PARSER.batch.7.d`, `PARSER.stream.4` | [source](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/test_gemma4_tool_parser.py#L661) | Trailing bare boolean must not be streamed twice. |

### GigaChat 3

Model-specific / endpoint-specific rows:

| Test | Bucket(s) | Link | Notes |
| -- | -- | -- | -- |
| `test_no_tool_call` | `PARSER.batch.3.a` | [source](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/test_gigachat3_tool_parser.py#L118) | test no tool call Parametrized. |
| `test_tool_call` | `PARSER.batch.1` | [source](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/test_gigachat3_tool_parser.py#L277) | test tool call Parametrized. |
| `test_streaming_tool_call_with_large_steps` | `PARSER.stream.1` | [source](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/test_gigachat3_tool_parser.py#L332) | Test that the closing braces are streamed correctly. Parametrized. |

### GLM 4.7 MoE

Model-specific / endpoint-specific rows:

| Test | Bucket(s) | Link | Notes |
| -- | -- | -- | -- |
| `TestGlm47ExtractToolCalls.test_no_tool_call` | `PARSER.batch.3.a` | [source](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/test_glm47_moe_tool_parser.py#L62) | TestGlm47ExtractToolCalls.test no tool call |
| `TestGlm47ExtractToolCalls.test_zero_arg_inline` | `PARSER.batch.6.b` | [source](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/test_glm47_moe_tool_parser.py#L68) | TestGlm47ExtractToolCalls.test zero arg inline |
| `TestGlm47ExtractToolCalls.test_zero_arg_newline` | `PARSER.batch.6.b`, `PARSER.fmt.2` | [source](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/test_glm47_moe_tool_parser.py#L76) | TestGlm47ExtractToolCalls.test zero arg newline |
| `TestGlm47ExtractToolCalls.test_args_same_line` | `PARSER.fmt.2` | [source](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/test_glm47_moe_tool_parser.py#L82) | TestGlm47ExtractToolCalls.test args same line |
| `TestGlm47ExtractToolCalls.test_args_with_newlines` | `PARSER.fmt.2` | [source](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/test_glm47_moe_tool_parser.py#L88) | TestGlm47ExtractToolCalls.test args with newlines |
| `TestGlm47ExtractToolCalls.test_content_before` | `PARSER.batch.8.a` | [source](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/test_glm47_moe_tool_parser.py#L94) | TestGlm47ExtractToolCalls.test content before |
| `TestGlm47ExtractToolCalls.test_multiple` | `PARSER.batch.1` | [source](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/test_glm47_moe_tool_parser.py#L100) | TestGlm47ExtractToolCalls.test multiple |
| `TestGlm47ExtractToolCalls.test_empty_content_none` | `PARSER.batch.9` | [source](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/test_glm47_moe_tool_parser.py#L108) | TestGlm47ExtractToolCalls.test empty content none |
| `TestGlm47ExtractToolCalls.test_whitespace_content_none` | `PARSER.batch.9`, `PARSER.fmt.2` | [source](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/test_glm47_moe_tool_parser.py#L113) | TestGlm47ExtractToolCalls.test whitespace content none |
| `TestGlm47Streaming.test_no_args` | `PARSER.batch.6.b`, `PARSER.stream.1` | [source](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/test_glm47_moe_tool_parser.py#L129) | TestGlm47Streaming.test no args |
| `TestGlm47Streaming.test_with_args` | `PARSER.stream.1` | [source](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/test_glm47_moe_tool_parser.py#L146) | TestGlm47Streaming.test with args |

### GLM 4 MoE

Model-specific / endpoint-specific rows:

| Test | Bucket(s) | Link | Notes |
| -- | -- | -- | -- |
| `test_extract_tool_calls_no_tools` | `PARSER.batch.3.a` | [source](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/test_glm4_moe_tool_parser.py#L72) | test extract tool calls no tools |
| `test_extract_tool_calls` | `PARSER.batch.1` | [source](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/test_glm4_moe_tool_parser.py#L236) | test extract tool calls Parametrized. |
| `test_extract_tool_calls_with_thinking_tags` | `REASONING.batch.2`, `REASONING.batch.2` | [source](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/test_glm4_moe_tool_parser.py#L252) | Test tool extraction when thinking tags are present. |
| `test_extract_tool_calls_malformed_xml` | `PARSER.batch.4.d` | [source](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/test_glm4_moe_tool_parser.py#L279) | Test that malformed XML is handled gracefully. |
| `test_extract_tool_calls_empty_arguments` | `PARSER.batch.6.a` | [source](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/test_glm4_moe_tool_parser.py#L299) | Test tool calls with no arguments. |
| `test_extract_tool_calls_mixed_content` | `PARSER.batch.2.c`, `PARSER.batch.8.c` | [source](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/test_glm4_moe_tool_parser.py#L315) | Test extraction with mixed content and multiple tool calls. |
| `test_streaming_basic_functionality` | `PARSER.stream.1` | [source](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/test_glm4_moe_tool_parser.py#L358) | Test basic streaming functionality. |
| `test_streaming_no_tool_calls` | `PARSER.batch.3.a`, `PARSER.stream.1` | [source](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/test_glm4_moe_tool_parser.py#L383) | Test streaming when there are no tool calls. |
| `test_streaming_with_content_before_tool_calls` | `PARSER.stream.1`, `PARSER.batch.8.a` | [source](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/test_glm4_moe_tool_parser.py#L404) | Test streaming when there's content before tool calls. |
| `test_extract_tool_calls_special_characters` | `PARSER.batch.7.b` | [source](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/test_glm4_moe_tool_parser.py#L425) | Test tool calls with special characters and unicode. |
| `test_extract_tool_calls_incomplete_tool_call` | `PARSER.batch.4.d` | [source](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/test_glm4_moe_tool_parser.py#L450) | Test incomplete tool calls (missing closing tag). |
| `test_streaming_incremental_string_value` | `PARSER.batch.7.b`, `PARSER.stream.1`, `PARSER.stream.3` | [source](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/test_glm4_moe_tool_parser.py#L478) | Test incremental streaming of string argument values. |
| `test_streaming_empty_tool_call` | `PARSER.stream.1`, `PARSER.fmt.4` | [source](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/test_glm4_moe_tool_parser.py#L528) | Test that empty tool calls don't cause infinite loops. |
| `test_streaming_prev_tool_call_arr_updates` | `PARSER.stream.3` | [source](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/test_glm4_moe_tool_parser.py#L548) | Test that prev_tool_call_arr is populated incrementally. |
| `test_streaming_multiple_tool_calls_sequential` | `PARSER.stream.2` | [source](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/test_glm4_moe_tool_parser.py#L588) | Test streaming multiple sequential tool calls. |
| `test_streaming_json_escape_in_string` | `PARSER.batch.7.b`, `PARSER.stream.1` | [source](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/test_glm4_moe_tool_parser.py#L624) | Test that special characters in string values are properly escaped. |
| `test_streaming_long_content_incremental` | `PARSER.stream.3`, `(inline regression annotation)`, `(dissolved; see PARSER.batch.4 impl-defined)` | [source](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/test_glm4_moe_tool_parser.py#L656) | Test incremental streaming of long content (Issue #32829). This is the core fix: for long string values like code (4000+ chars), the parser should stream incrementally rather than  |
| `test_extract_tool_calls_numeric_deserialization` | `PARSER.batch.7.a` | [source](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/test_glm4_moe_tool_parser.py#L765) | Test that numeric arguments are deserialized as numbers, not strings. |
| `test_zero_argument_tool_call` | `PARSER.batch.6.c`, `(inline regression annotation)` | [source](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/test_glm4_moe_tool_parser.py#L804) | Regression: zero-argument tool call crash (PR #32321). |
| `test_malformed_tool_call_no_regex_match` | `PARSER.batch.4.a`, `(inline regression annotation)` | [source](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/test_glm4_moe_tool_parser.py#L820) | Regression: malformed tool_call with no regex match (PR #32321). |
| `test_delimiter_preserved_transformers_5x` | `PARSER.stream.3`, `(inline regression annotation)`, `FRONTEND.3/.6` | [source](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/test_glm4_moe_tool_parser.py#L832) | Regression: adjust_request sets skip_special_tokens=False (PR #31622). |
| `test_unicode_characters_preserved` | `PARSER.batch.7.b`, `(inline regression annotation)` | [source](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/test_glm4_moe_tool_parser.py#L884) | Regression: Unicode chars must not be escaped to \uXXXX (PR #30920). |
| `test_streaming_multi_token_chunks` | `PARSER.stream.3` | [source](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/test_glm4_moe_tool_parser.py#L909) | Test that multi-token chunks (stream_interval > 1) are handled correctly. With stream_interval > 1 or MTP, multiple XML tags arrive in one delta. The old buffer-based parser could  |
| `test_streaming_entire_tool_call_at_once` | `PARSER.stream.1` | [source](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/test_glm4_moe_tool_parser.py#L944) | Test that a complete tool call arriving in one delta works. This simulates the extreme MTP case where all tokens arrive at once. |
| `test_streaming_content_between_tool_calls_multi_token` | `PARSER.stream.1`, `PARSER.stream.3`, `PARSER.batch.8.d` | [source](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/test_glm4_moe_tool_parser.py#L978) | Test content between tool calls with multi-token chunks. |
| `test_streaming_multi_token_with_multiple_args` | `PARSER.batch.7.d`, `PARSER.stream.1` | [source](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/test_glm4_moe_tool_parser.py#L1031) | Test multi-token streaming with multiple arguments of mixed types. |
| `test_stream_interval_single_tool_call` | `PARSER.batch.1`, `PARSER.stream.1`, `PARSER.stream.3` | [source](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/test_glm4_moe_tool_parser.py#L1136) | Tool call streaming produces correct name + args at any interval. Parametrized. |
| `test_stream_interval_multiple_tool_calls` | `PARSER.stream.2`, `PARSER.stream.3`, `PARSER.fmt.1` | [source](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/test_glm4_moe_tool_parser.py#L1172) | Multiple sequential tool calls with correct indices at any interval. Parametrized. |
| `test_stream_interval_content_then_tool_call` | `PARSER.stream.3`, `PARSER.batch.8.a` | [source](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/test_glm4_moe_tool_parser.py#L1212) | Content before a tool call is fully emitted before tool deltas. Parametrized. |
| `test_stream_interval_extreme_single_chunk` | `PARSER.stream.1`, `PARSER.stream.3` | [source](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/test_glm4_moe_tool_parser.py#L1252) | Extreme MTP: entire output arrives in one chunk (interval=9999). |
| `test_stream_interval_content_between_tool_calls` | `PARSER.stream.3`, `PARSER.batch.8.d` | [source](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/test_glm4_moe_tool_parser.py#L1289) | Content between tool calls must be emitted, not silently dropped. Parametrized. |

### Granite 4

Model-specific / endpoint-specific rows:

| Test | Bucket(s) | Link | Notes |
| -- | -- | -- | -- |
| `test_tool_call_parser_complex` | `PARSER.batch.1` | [source](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/test_granite4_tool_parser.py#L69) | test tool call parser complex Parametrized. |

### Granite 20B FC

Common suite config: [TestGranite20bFcToolParser](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/test_granite_20b_fc_tool_parser.py#L14).

Inherited common `ToolParserTests` rows:

| Test | Bucket(s) | Link | Notes |
| -- | -- | -- | -- |
| `TestGranite20bFcToolParser.test_no_tool_calls` | `PARSER.batch.3.a`, `PARSER.stream.1` | [source](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/common_tests.py#L125) | Plain text produces no tool calls; common suite runs streaming and non-streaming. Model config: tests/tool_parsers/test_granite_20b_fc_tool_parser.py:14. |
| `TestGranite20bFcToolParser.test_single_tool_call_simple_args` | `PARSER.batch.1`, `PARSER.stream.1` | [source](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/common_tests.py#L145) | One simple call in streaming and non-streaming. Model config: tests/tool_parsers/test_granite_20b_fc_tool_parser.py:14. |
| `TestGranite20bFcToolParser.test_parallel_tool_calls` | `PARSER.batch.2.a`, `PARSER.stream.2`, `PARSER.fmt.1` | [source](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/common_tests.py#L185) | Multiple calls plus unique ID assertion. Model config: tests/tool_parsers/test_granite_20b_fc_tool_parser.py:14. |
| `TestGranite20bFcToolParser.test_various_data_types` | `PARSER.batch.7.a`, `PARSER.stream.1` | [source](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/common_tests.py#L220) | All JSON scalar/container argument types. Model config: tests/tool_parsers/test_granite_20b_fc_tool_parser.py:14. |
| `TestGranite20bFcToolParser.test_empty_arguments` | `PARSER.batch.6.a`, `PARSER.stream.1` | [source](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/common_tests.py#L260) | Parameterless call. Model config: tests/tool_parsers/test_granite_20b_fc_tool_parser.py:14. |
| `TestGranite20bFcToolParser.test_surrounding_text` | `PARSER.batch.8.c`, `PARSER.stream.1` | [source](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/common_tests.py#L282) | Tool call surrounded by normal text. Model config: tests/tool_parsers/test_granite_20b_fc_tool_parser.py:14. |
| `TestGranite20bFcToolParser.test_escaped_strings` | `PARSER.batch.7.b`, `PARSER.stream.1` | [source](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/common_tests.py#L302) | Escaped strings / Unicode-like values. Model config: tests/tool_parsers/test_granite_20b_fc_tool_parser.py:14. |
| `TestGranite20bFcToolParser.test_malformed_input` | `PARSER.batch.4.a`, `PARSER.stream.1`, `(dissolved; see PARSER.batch.4 impl-defined)` | [source](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/common_tests.py#L322) | Malformed inputs must not crash. Model config: tests/tool_parsers/test_granite_20b_fc_tool_parser.py:14. |
| `TestGranite20bFcToolParser.test_streaming_reconstruction` | `PARSER.stream.1` | [source](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/common_tests.py#L340) | Streaming output should match non-streaming output. Model config: tests/tool_parsers/test_granite_20b_fc_tool_parser.py:14. |

### Granite

Common suite config: [TestGraniteToolParser](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/test_granite_tool_parser.py#L16).

Inherited common `ToolParserTests` rows:

| Test | Bucket(s) | Link | Notes |
| -- | -- | -- | -- |
| `TestGraniteToolParser.test_no_tool_calls` | `PARSER.batch.3.a`, `PARSER.stream.1` | [source](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/common_tests.py#L125) | Plain text produces no tool calls; common suite runs streaming and non-streaming. Model config: tests/tool_parsers/test_granite_tool_parser.py:16. |
| `TestGraniteToolParser.test_single_tool_call_simple_args` | `PARSER.batch.1`, `PARSER.stream.1` | [source](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/common_tests.py#L145) | One simple call in streaming and non-streaming. Model config: tests/tool_parsers/test_granite_tool_parser.py:16. |
| `TestGraniteToolParser.test_parallel_tool_calls` | `PARSER.batch.2.a`, `PARSER.stream.2`, `PARSER.fmt.1` | [source](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/common_tests.py#L185) | Multiple calls plus unique ID assertion. Model config: tests/tool_parsers/test_granite_tool_parser.py:16. |
| `TestGraniteToolParser.test_various_data_types` | `PARSER.batch.7.a`, `PARSER.stream.1` | [source](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/common_tests.py#L220) | All JSON scalar/container argument types. Model config: tests/tool_parsers/test_granite_tool_parser.py:16. |
| `TestGraniteToolParser.test_empty_arguments` | `PARSER.batch.6.a`, `PARSER.stream.1` | [source](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/common_tests.py#L260) | Parameterless call. Model config: tests/tool_parsers/test_granite_tool_parser.py:16. |
| `TestGraniteToolParser.test_surrounding_text` | `PARSER.batch.8.c`, `PARSER.stream.1` | [source](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/common_tests.py#L282) | Tool call surrounded by normal text. Model config: tests/tool_parsers/test_granite_tool_parser.py:16. |
| `TestGraniteToolParser.test_escaped_strings` | `PARSER.batch.7.b`, `PARSER.stream.1` | [source](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/common_tests.py#L302) | Escaped strings / Unicode-like values. Model config: tests/tool_parsers/test_granite_tool_parser.py:16. |
| `TestGraniteToolParser.test_malformed_input` | `PARSER.batch.4.a`, `PARSER.stream.1`, `(dissolved; see PARSER.batch.4 impl-defined)` | [source](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/common_tests.py#L322) | Malformed inputs must not crash. Model config: tests/tool_parsers/test_granite_tool_parser.py:16. |
| `TestGraniteToolParser.test_streaming_reconstruction` | `PARSER.stream.1` | [source](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/common_tests.py#L340) | Streaming output should match non-streaming output. Model config: tests/tool_parsers/test_granite_tool_parser.py:16. |

Model-specific / endpoint-specific rows:

| Test | Bucket(s) | Link | Notes |
| -- | -- | -- | -- |
| `TestGraniteToolParser.test_granite_token_prefix_format` | `PARSER.fmt.3` | [source](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/test_granite_tool_parser.py#L93) | Verify parser handles Granite 3.0 <\|tool_call\|> token format. Parametrized. |
| `TestGraniteToolParser.test_granite_string_prefix_format` | `PARSER.fmt.3` | [source](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/test_granite_tool_parser.py#L107) | Verify parser handles Granite 3.1 <tool_call> string format. Parametrized. |

### Hermes

Model-specific / endpoint-specific rows:

| Test | Bucket(s) | Link | Notes |
| -- | -- | -- | -- |
| `test_hermes_parser_streaming_just_forward_text` | `PARSER.batch.3.d`, `PARSER.stream.1` | [source](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/test_hermes_tool_parser.py#L46) | test hermes parser streaming just forward text |
| `test_hermes_parser_streaming_failure_case_bug_19056` | `PARSER.stream.1`, `(inline regression annotation)` | [source](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/test_hermes_tool_parser.py#L78) | test hermes parser streaming failure case bug 19056 |
| `test_hermes_parser_streaming` | `PARSER.stream.1` | [source](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/test_hermes_tool_parser.py#L112) | test hermes parser streaming |
| `test_hermes_streaming_tool_call_with_stream_interval` | `PARSER.stream.3` | [source](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/test_hermes_tool_parser.py#L191) | Tool call streaming must produce correct name + args at any interval. Parametrized. |
| `test_hermes_streaming_content_then_tool_call_with_stream_interval` | `PARSER.stream.3`, `PARSER.batch.8.a` | [source](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/test_hermes_tool_parser.py#L221) | Content before a tool call must be fully streamed, then tool call. Parametrized. |
| `test_hermes_streaming_multiple_tool_calls_with_stream_interval` | `PARSER.stream.2`, `PARSER.stream.3` | [source](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/test_hermes_tool_parser.py#L252) | Multiple sequential tool calls must each be streamed correctly. Parametrized. |
| `test_hermes_streaming_boolean_args_with_stream_interval` | `PARSER.batch.7.d`, `PARSER.stream.3`, `(inline regression annotation)` | [source](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/test_hermes_tool_parser.py#L284) | Regression test for bug #19056 with stream_interval > 1. Parametrized. |
| `test_hermes_streaming_just_forward_text_with_stream_interval` | `PARSER.batch.3.d`, `PARSER.stream.3` | [source](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/test_hermes_tool_parser.py#L307) | Plain text with no tool calls must be fully forwarded. Parametrized. |
| `test_hermes_parser_non_streaming_no_tool_call` | `PARSER.batch.3.a` | [source](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/test_hermes_tool_parser.py#L324) | test hermes parser non streaming no tool call |
| `test_hermes_parser_non_streaming_tool_call_between_tags` | `PARSER.batch.1` | [source](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/test_hermes_tool_parser.py#L338) | test hermes parser non streaming tool call between tags |
| `test_hermes_parser_non_streaming_tool_call_until_eos` | `PARSER.batch.5.c` | [source](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/test_hermes_tool_parser.py#L356) | test hermes parser non streaming tool call until eos |
| `test_hermes_parser_non_streaming_tool_call_invalid_json` | `PARSER.batch.4.b` | [source](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/test_hermes_tool_parser.py#L376) | test hermes parser non streaming tool call invalid json |
| `test_hermes_streaming_content_and_tool_call_in_single_chunk` | `PARSER.stream.1`, `PARSER.stream.3` | [source](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/test_hermes_tool_parser.py#L392) | Content + complete tool call in one chunk must both be emitted. |

### Hunyuan A13B

Model-specific / endpoint-specific rows:

| Test | Bucket(s) | Link | Notes |
| -- | -- | -- | -- |
| `test_hunyuan_a13b_tool_parser_extract` | `PARSER.batch.1` | [source](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/test_hunyuan_a13b_tool_parser.py#L87) | test hunyuan a13b tool parser extract Parametrized. |
| `test_hunyuan_a13b_tool_parser_streaming` | `PARSER.stream.1` | [source](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/test_hunyuan_a13b_tool_parser.py#L165) | test hunyuan a13b tool parser streaming Parametrized. |

### HY V3

Model-specific / endpoint-specific rows:

| Test | Bucket(s) | Link | Notes |
| -- | -- | -- | -- |
| `TestHYV3ExtractToolCalls.test_no_tool_call` | `PARSER.batch.3.a` | [source](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/test_hy_v3_tool_parser.py#L59) | TestHYV3ExtractToolCalls.test no tool call |
| `TestHYV3ExtractToolCalls.test_zero_arg_inline` | `PARSER.batch.6.b` | [source](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/test_hy_v3_tool_parser.py#L65) | TestHYV3ExtractToolCalls.test zero arg inline |
| `TestHYV3ExtractToolCalls.test_zero_arg_newline` | `PARSER.batch.6.b`, `PARSER.fmt.2` | [source](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/test_hy_v3_tool_parser.py#L75) | TestHYV3ExtractToolCalls.test zero arg newline |
| `TestHYV3ExtractToolCalls.test_args_same_line` | `PARSER.fmt.2` | [source](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/test_hy_v3_tool_parser.py#L81) | TestHYV3ExtractToolCalls.test args same line |
| `TestHYV3ExtractToolCalls.test_args_with_newlines` | `PARSER.fmt.2` | [source](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/test_hy_v3_tool_parser.py#L93) | TestHYV3ExtractToolCalls.test args with newlines |
| `TestHYV3ExtractToolCalls.test_content_before` | `PARSER.batch.8.a` | [source](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/test_hy_v3_tool_parser.py#L105) | TestHYV3ExtractToolCalls.test content before |
| `TestHYV3ExtractToolCalls.test_multiple` | `PARSER.batch.1` | [source](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/test_hy_v3_tool_parser.py#L111) | TestHYV3ExtractToolCalls.test multiple |
| `TestHYV3ExtractToolCalls.test_empty_content_none` | `PARSER.batch.9` | [source](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/test_hy_v3_tool_parser.py#L121) | TestHYV3ExtractToolCalls.test empty content none |
| `TestHYV3ExtractToolCallsStreaming.test_no_tool_call_streaming` | `PARSER.batch.3.d`, `PARSER.stream.1` | [source](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/test_hy_v3_tool_parser.py#L184) | TestHYV3ExtractToolCallsStreaming.test no tool call streaming |
| `TestHYV3ExtractToolCallsStreaming.test_zero_arg_streaming` | `PARSER.batch.6.b`, `PARSER.stream.1` | [source](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/test_hy_v3_tool_parser.py#L191) | TestHYV3ExtractToolCallsStreaming.test zero arg streaming |
| `TestHYV3ExtractToolCallsStreaming.test_args_streaming` | `PARSER.stream.1` | [source](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/test_hy_v3_tool_parser.py#L206) | TestHYV3ExtractToolCallsStreaming.test args streaming |
| `TestHYV3ExtractToolCallsStreaming.test_content_before_streaming` | `PARSER.stream.1`, `PARSER.batch.8.a` | [source](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/test_hy_v3_tool_parser.py#L227) | TestHYV3ExtractToolCallsStreaming.test content before streaming |
| `TestHYV3ExtractToolCallsStreaming.test_multiple_streaming` | `PARSER.stream.1` | [source](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/test_hy_v3_tool_parser.py#L242) | TestHYV3ExtractToolCallsStreaming.test multiple streaming |
| `TestHYV3ExtractToolCallsStreaming.test_all_in_one_delta_streaming` | `PARSER.stream.1` | [source](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/test_hy_v3_tool_parser.py#L269) | TestHYV3ExtractToolCallsStreaming.test all in one delta streaming |

### InternLM2

Common suite config: [TestInternLM2ToolParser](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/test_internlm2_tool_parser.py#L33).

Inherited common `ToolParserTests` rows:

| Test | Bucket(s) | Link | Notes |
| -- | -- | -- | -- |
| `TestInternLM2ToolParser.test_no_tool_calls` | `PARSER.batch.3.a`, `PARSER.stream.1` | [source](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/common_tests.py#L125) | Plain text produces no tool calls; common suite runs streaming and non-streaming. Model config: tests/tool_parsers/test_internlm2_tool_parser.py:33. |
| `TestInternLM2ToolParser.test_single_tool_call_simple_args` | `PARSER.batch.1`, `PARSER.stream.1` | [source](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/common_tests.py#L145) | One simple call in streaming and non-streaming. Model config: tests/tool_parsers/test_internlm2_tool_parser.py:33. |
| `TestInternLM2ToolParser.test_parallel_tool_calls` | `PARSER.batch.2.a`, `PARSER.stream.2`, `PARSER.fmt.1` | [source](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/common_tests.py#L185) | Multiple calls plus unique ID assertion. Model config: tests/tool_parsers/test_internlm2_tool_parser.py:33. |
| `TestInternLM2ToolParser.test_various_data_types` | `PARSER.batch.7.a`, `PARSER.stream.1` | [source](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/common_tests.py#L220) | All JSON scalar/container argument types. Model config: tests/tool_parsers/test_internlm2_tool_parser.py:33. |
| `TestInternLM2ToolParser.test_empty_arguments` | `PARSER.batch.6.a`, `PARSER.stream.1` | [source](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/common_tests.py#L260) | Parameterless call. Model config: tests/tool_parsers/test_internlm2_tool_parser.py:33. |
| `TestInternLM2ToolParser.test_surrounding_text` | `PARSER.batch.8.c`, `PARSER.stream.1` | [source](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/common_tests.py#L282) | Tool call surrounded by normal text. Model config: tests/tool_parsers/test_internlm2_tool_parser.py:33. |
| `TestInternLM2ToolParser.test_escaped_strings` | `PARSER.batch.7.b`, `PARSER.stream.1` | [source](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/common_tests.py#L302) | Escaped strings / Unicode-like values. Model config: tests/tool_parsers/test_internlm2_tool_parser.py:33. |
| `TestInternLM2ToolParser.test_malformed_input` | `PARSER.batch.4.a`, `PARSER.stream.1`, `(dissolved; see PARSER.batch.4 impl-defined)` | [source](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/common_tests.py#L322) | Malformed inputs must not crash. Model config: tests/tool_parsers/test_internlm2_tool_parser.py:33. |
| `TestInternLM2ToolParser.test_streaming_reconstruction` | `PARSER.stream.1` | [source](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/common_tests.py#L340) | Streaming output should match non-streaming output. Model config: tests/tool_parsers/test_internlm2_tool_parser.py:33. |

### Jamba

Model-specific / endpoint-specific rows:

| Test | Bucket(s) | Link | Notes |
| -- | -- | -- | -- |
| `test_extract_tool_calls_no_tools` | `PARSER.batch.3.a` | [source](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/test_jamba_tool_parser.py#L94) | test extract tool calls no tools |
| `test_extract_tool_calls` | `PARSER.batch.1` | [source](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/test_jamba_tool_parser.py#L164) | test extract tool calls Parametrized. |
| `test_extract_tool_calls_streaming` | `PARSER.stream.1` | [source](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/test_jamba_tool_parser.py#L239) | test extract tool calls streaming Parametrized. |

### Kimi K2

Model-specific / endpoint-specific rows:

| Test | Bucket(s) | Link | Notes |
| -- | -- | -- | -- |
| `TestExtractToolCalls.test_no_tools` | `PARSER.batch.3.a` | [source](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/test_kimi_k2_tool_parser.py#L49) | TestExtractToolCalls.test no tools |
| `TestExtractToolCalls.test_extract_tool_calls` | `PARSER.batch.1` | [source](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/test_kimi_k2_tool_parser.py#L136) | TestExtractToolCalls.test extract tool calls Parametrized. |
| `TestExtractToolCalls.test_invalid_json_still_extracted` | `PARSER.batch.4.b` | [source](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/test_kimi_k2_tool_parser.py#L151) | Tool calls with invalid JSON are still returned (arguments as-is). |
| `TestExtractToolCalls.test_invalid_funcall_id_skipped` | `PARSER.batch.4.d`, `PARSER.fmt.1` | [source](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/test_kimi_k2_tool_parser.py#L165) | Tool calls with malformed id (no colon+digit) are skipped — function-name-surface validation, fmt.1 not fmt.5. |
| `TestExtractToolCalls.test_native_id_extracted` | `(inline regression annotation)`, `PARSER.fmt.5` | [source](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/test_kimi_k2_tool_parser.py#L178) | Regression: parser extracts native ID onto ToolCall (PR #32768). |
| `TestExtractToolCalls.test_multi_turn_native_id_continuity` | `(inline regression annotation)`, `PARSER.fmt.5` | [source](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/test_kimi_k2_tool_parser.py#L189) | Regression: native IDs from turn 1 preserved across turns (PR #32768). |
| `TestStreamingHappyPath.test_single_tool_call` | `PARSER.batch.1`, `PARSER.stream.1` | [source](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/test_kimi_k2_tool_parser.py#L238) | Verify DeltaToolCall output: name, id, arguments for one tool. |
| `TestStreamingHappyPath.test_multiple_tool_calls` | `PARSER.stream.1`, `PARSER.stream.2`, `PARSER.fmt.1` | [source](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/test_kimi_k2_tool_parser.py#L252) | Two tool calls emitted with correct indices, names, arguments. |
| `TestStreamingHappyPath.test_content_before_tools` | `PARSER.stream.1`, `PARSER.batch.8.a` | [source](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/test_kimi_k2_tool_parser.py#L272) | Content before section is streamed; markers/args don't leak. |
| `TestStreamingHappyPath.test_no_tool_calls` | `PARSER.batch.3.a`, `PARSER.stream.1` | [source](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/test_kimi_k2_tool_parser.py#L287) | Plain text streaming returns content only. |
| `TestStreamingHappyPath.test_incremental_arguments` | `PARSER.stream.1`, `PARSER.stream.3` | [source](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/test_kimi_k2_tool_parser.py#L295) | Arguments split across small chunks accumulate correctly. |
| `TestStreamingHappyPath.test_streaming_matches_nonstreaming` | `PARSER.stream.1` | [source](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/test_kimi_k2_tool_parser.py#L338) | Streaming reconstruction matches non-streaming extraction. Parametrized. |
| `TestStreamingEdgeCases.test_marker_suppression` | `PARSER.stream.1` | [source](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/test_kimi_k2_tool_parser.py#L356) | No special-token markers appear in reconstructed content. |
| `TestStreamingEdgeCases.test_noise_between_markers_suppressed` | `PARSER.stream.1` | [source](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/test_kimi_k2_tool_parser.py#L370) | Text between section_begin and tool_call_begin doesn't leak. |
| `TestStreamingEdgeCases.test_empty_tool_section` | `PARSER.stream.1`, `PARSER.fmt.4` | [source](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/test_kimi_k2_tool_parser.py#L388) | Empty section (begin immediately followed by end) doesn't crash. |
| `TestStreamingEdgeCases.test_three_different_tools` | `PARSER.stream.1` | [source](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/test_kimi_k2_tool_parser.py#L395) | Three tool calls with different functions stream correctly. |
| `TestStreamingEdgeCases.test_truncated_tool_call_no_end_marker` | `PARSER.batch.5.c`, `PARSER.stream.4` | [source](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/test_kimi_k2_tool_parser.py#L413) | Stream ending mid-tool-call (max_tokens) doesn't crash. |
| `TestStreamingEdgeCases.test_content_after_tool_section` | `PARSER.stream.4`, `PARSER.batch.8.b` | [source](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/test_kimi_k2_tool_parser.py#L435) | Trailing text after section_end doesn't crash or leak markers. |
| `TestAdjustRequest.test_sets_skip_special_tokens_false` | `FRONTEND.3/.6` | [source](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/test_kimi_k2_tool_parser.py#L462) | TestAdjustRequest.test sets skip special tokens false |
| `TestAdjustRequest.test_no_change_when_tool_choice_none` | `PARSER.batch.1` | [source](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/test_kimi_k2_tool_parser.py#L471) | TestAdjustRequest.test no change when tool choice none |
| `TestAdjustRequest.test_no_change_when_no_tools` | `PARSER.batch.3.a` | [source](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/test_kimi_k2_tool_parser.py#L480) | TestAdjustRequest.test no change when no tools |
| `TestStreamingIntervals.test_single_tool_call_at_interval` | `PARSER.batch.1`, `PARSER.stream.1`, `PARSER.stream.3` | [source](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/test_kimi_k2_tool_parser.py#L508) | TestStreamingIntervals.test single tool call at interval Parametrized. |
| `TestStreamingIntervals.test_content_then_tool_call_at_interval` | `PARSER.stream.3`, `PARSER.batch.8.a` | [source](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/test_kimi_k2_tool_parser.py#L521) | TestStreamingIntervals.test content then tool call at interval Parametrized. |
| `TestStreamingIntervals.test_multiple_tool_calls_at_interval` | `PARSER.stream.2`, `PARSER.stream.3` | [source](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/test_kimi_k2_tool_parser.py#L540) | TestStreamingIntervals.test multiple tool calls at interval Parametrized. |
| `TestStreamingIntervals.test_plain_text_at_interval` | `PARSER.batch.3.d`, `PARSER.stream.3` | [source](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/test_kimi_k2_tool_parser.py#L558) | TestStreamingIntervals.test plain text at interval Parametrized. |
| `TestStreamingIntervals.test_content_and_tool_call_in_single_chunk` | `PARSER.stream.1`, `PARSER.stream.3` | [source](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/test_kimi_k2_tool_parser.py#L569) | Content + complete tool call in one chunk must both be emitted. |

### Llama 3 JSON

Model-specific / endpoint-specific rows:

| Test | Bucket(s) | Link | Notes |
| -- | -- | -- | -- |
| `test_extract_tool_calls_simple` | `PARSER.batch.1` | [source](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/test_llama3_json_tool_parser.py#L25) | test extract tool calls simple |
| `test_extract_tool_calls_with_arguments` | `PARSER.batch.1` | [source](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/test_llama3_json_tool_parser.py#L42) | test extract tool calls with arguments |
| `test_extract_tool_calls_no_json` | `PARSER.batch.1` | [source](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/test_llama3_json_tool_parser.py#L56) | test extract tool calls no json |
| `test_extract_tool_calls_invalid_json` | `PARSER.batch.4.b` | [source](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/test_llama3_json_tool_parser.py#L66) | test extract tool calls invalid json |
| `test_extract_tool_calls_with_arguments_key` | `PARSER.fmt.5` | [source](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/test_llama3_json_tool_parser.py#L76) | test extract tool calls with arguments key |
| `test_extract_tool_calls_multiple_json` | `PARSER.batch.1` | [source](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/test_llama3_json_tool_parser.py#L87) | test extract tool calls multiple json |
| `test_extract_tool_calls_multiple_json_with_whitespace` | `PARSER.fmt.2` | [source](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/test_llama3_json_tool_parser.py#L112) | test extract tool calls multiple json with whitespace |
| `test_extract_tool_calls_multiple_json_with_surrounding_text` | `PARSER.batch.8.c` | [source](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/test_llama3_json_tool_parser.py#L128) | test extract tool calls multiple json with surrounding text |
| `test_extract_tool_calls_deeply_nested_json` | `PARSER.batch.7.d` | [source](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/test_llama3_json_tool_parser.py#L146) | test extract tool calls deeply nested json |
| `test_extract_tool_calls_multiple_with_deep_nesting` | `PARSER.batch.1` | [source](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/test_llama3_json_tool_parser.py#L170) | test extract tool calls multiple with deep nesting |
| `test_extract_tool_calls_with_quotes_and_brackets_in_string` | `PARSER.batch.7.d` | [source](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/test_llama3_json_tool_parser.py#L195) | test extract tool calls with quotes and brackets in string |
| `test_extract_tool_calls_with_escaped_quotes_in_nested_json` | `PARSER.batch.7.b` | [source](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/test_llama3_json_tool_parser.py#L217) | test extract tool calls with escaped quotes in nested json |
| `test_extract_tool_calls_missing_name_key` | `PARSER.batch.4.c` | [source](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/test_llama3_json_tool_parser.py#L234) | test extract tool calls missing name key |
| `test_extract_tool_calls_missing_parameters_and_arguments_key` | `PARSER.batch.4.c`, `PARSER.fmt.5` | [source](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/test_llama3_json_tool_parser.py#L244) | test extract tool calls missing parameters and arguments key |
| `test_regex_timeout_handling` | `(dissolved; see PARSER.batch.4 impl-defined)` | [source](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/test_llama3_json_tool_parser.py#L254) | Test regex timeout is handled gracefully |

### Llama 4 Pythonic

Model-specific / endpoint-specific rows:

| Test | Bucket(s) | Link | Notes |
| -- | -- | -- | -- |
| `test_no_tool_call` | `PARSER.batch.3.a` | [source](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/test_llama4_pythonic_tool_parser.py#L67) | test no tool call Parametrized. |
| `test_tool_call` | `PARSER.batch.1` | [source](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/test_llama4_pythonic_tool_parser.py#L207) | test tool call Parametrized. |
| `test_streaming_tool_call_with_large_steps` | `PARSER.stream.1` | [source](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/test_llama4_pythonic_tool_parser.py#L227) | test streaming tool call with large steps |
| `test_regex_timeout_handling` | `(dissolved; see PARSER.batch.4 impl-defined)` | [source](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/test_llama4_pythonic_tool_parser.py#L249) | test regex timeout is handled gracefully Parametrized. |

### LongCat

Common suite config: [TestLongCatToolParser](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/test_longcat_tool_parser.py#L32).

Inherited common `ToolParserTests` rows:

| Test | Bucket(s) | Link | Notes |
| -- | -- | -- | -- |
| `TestLongCatToolParser.test_no_tool_calls` | `PARSER.batch.3.a`, `PARSER.stream.1` | [source](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/common_tests.py#L125) | Plain text produces no tool calls; common suite runs streaming and non-streaming. Model config: tests/tool_parsers/test_longcat_tool_parser.py:32. |
| `TestLongCatToolParser.test_single_tool_call_simple_args` | `PARSER.batch.1`, `PARSER.stream.1` | [source](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/common_tests.py#L145) | One simple call in streaming and non-streaming. Model config: tests/tool_parsers/test_longcat_tool_parser.py:32. |
| `TestLongCatToolParser.test_parallel_tool_calls` | `PARSER.batch.2.a`, `PARSER.stream.2`, `PARSER.fmt.1` | [source](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/common_tests.py#L185) | Multiple calls plus unique ID assertion. Model config: tests/tool_parsers/test_longcat_tool_parser.py:32. |
| `TestLongCatToolParser.test_various_data_types` | `PARSER.batch.7.a`, `PARSER.stream.1` | [source](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/common_tests.py#L220) | All JSON scalar/container argument types. Model config: tests/tool_parsers/test_longcat_tool_parser.py:32. |
| `TestLongCatToolParser.test_empty_arguments` | `PARSER.batch.6.a`, `PARSER.stream.1` | [source](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/common_tests.py#L260) | Parameterless call. Model config: tests/tool_parsers/test_longcat_tool_parser.py:32. |
| `TestLongCatToolParser.test_surrounding_text` | `PARSER.batch.8.c`, `PARSER.stream.1` | [source](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/common_tests.py#L282) | Tool call surrounded by normal text. Model config: tests/tool_parsers/test_longcat_tool_parser.py:32. |
| `TestLongCatToolParser.test_escaped_strings` | `PARSER.batch.7.b`, `PARSER.stream.1` | [source](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/common_tests.py#L302) | Escaped strings / Unicode-like values. Model config: tests/tool_parsers/test_longcat_tool_parser.py:32. |
| `TestLongCatToolParser.test_malformed_input` | `PARSER.batch.4.a`, `PARSER.stream.1`, `(dissolved; see PARSER.batch.4 impl-defined)` | [source](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/common_tests.py#L322) | Malformed inputs must not crash. Model config: tests/tool_parsers/test_longcat_tool_parser.py:32. |
| `TestLongCatToolParser.test_streaming_reconstruction` | `PARSER.stream.1` | [source](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/common_tests.py#L340) | Streaming output should match non-streaming output. Model config: tests/tool_parsers/test_longcat_tool_parser.py:32. |

### MiniMax M2

Model-specific / endpoint-specific rows:

| Test | Bucket(s) | Link | Notes |
| -- | -- | -- | -- |
| `TestContentStreaming.test_plain_content` | `PARSER.batch.3.a`, `PARSER.stream.1` | [source](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/test_minimax_m2_tool_parser.py#L120) | No tool call tokens — all text is streamed as content. |
| `TestContentStreaming.test_content_before_tool_call` | `PARSER.stream.1`, `PARSER.batch.8.a` | [source](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/test_minimax_m2_tool_parser.py#L126) | Text before <minimax:tool_call> is streamed as content. |
| `TestContentStreaming.test_empty_delta_no_crash` | `PARSER.stream.1`, `PARSER.batch.9`, `(dissolved; see PARSER.batch.4 impl-defined)` | [source](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/test_minimax_m2_tool_parser.py#L140) | Empty delta_text with no token IDs returns None. |
| `TestSingleInvoke.test_incremental_chunks` | `PARSER.stream.1`, `PARSER.stream.3` | [source](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/test_minimax_m2_tool_parser.py#L154) | Each XML element arrives in a separate chunk. |
| `TestSingleInvoke.test_single_chunk_complete` | `PARSER.stream.1`, `PARSER.stream.3` | [source](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/test_minimax_m2_tool_parser.py#L171) | Entire tool call arrives in one delta. |
| `TestSingleInvoke.test_multiple_params` | `PARSER.batch.7.d` | [source](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/test_minimax_m2_tool_parser.py#L185) | Multiple parameters in one invoke. |
| `TestMultipleInvokes.test_two_invokes_incremental` | `PARSER.batch.2.b`, `PARSER.stream.3` | [source](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/test_minimax_m2_tool_parser.py#L207) | Two invokes arriving one chunk at a time. |
| `TestMultipleInvokes.test_two_invokes_in_single_delta` | `PARSER.batch.2.b`, `PARSER.stream.1` | [source](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/test_minimax_m2_tool_parser.py#L229) | Both invokes close in the same delta — loop must emit both. |
| `TestMultipleInvokes.test_different_functions` | `PARSER.batch.2.b` | [source](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/test_minimax_m2_tool_parser.py#L245) | Parallel calls to different functions. |
| `TestInternalState.test_prev_tool_call_arr_single` | `PARSER.stream.1` | [source](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/test_minimax_m2_tool_parser.py#L273) | TestInternalState.test prev tool call arr single |
| `TestInternalState.test_prev_tool_call_arr_multiple` | `PARSER.stream.1` | [source](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/test_minimax_m2_tool_parser.py#L286) | prev_tool_call_arr records each invoke with correct arguments. |
| `TestDeltaMessageFormat.test_tool_call_fields` | `PARSER.stream.1`, `PARSER.fmt.1`, `PARSER.fmt.3` | [source](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/test_minimax_m2_tool_parser.py#L312) | Each emitted tool call has id, name, arguments, type, index. |
| `TestDeltaMessageFormat.test_multi_invoke_indices` | `PARSER.batch.2.d`, `PARSER.stream.1`, `PARSER.fmt.1`, `PARSER.fmt.3` | [source](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/test_minimax_m2_tool_parser.py#L331) | Multiple invokes get sequential indices. |
| `TestEOSHandling.test_eos_after_tool_calls` | `PARSER.stream.4`, `PIPELINE.finish_reason` | [source](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/test_minimax_m2_tool_parser.py#L355) | EOS token (empty delta, non-special token id) returns content=''. |
| `TestEOSHandling.test_end_token_ignored` | `PARSER.fmt.3` | [source](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/test_minimax_m2_tool_parser.py#L370) | </minimax:tool_call> special token should NOT trigger EOS. |
| `TestSpecialTokenDetection.test_start_token_via_id` | `PARSER.stream.3`, `PARSER.fmt.3` | [source](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/test_minimax_m2_tool_parser.py#L393) | <minimax:tool_call> detected via delta_token_ids, not text. |
| `TestLargeChunks.test_header_and_params_in_separate_chunks` | `PARSER.stream.1`, `PARSER.stream.3` | [source](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/test_minimax_m2_tool_parser.py#L421) | Header in chunk 1, all params + close in chunk 2, then EOS. |
| `TestAnyOfNullableParam.test_anyof_nullable_param_non_null_value` | `PARSER.batch.7.c`, `PARSER.batch.9`, `PARSER.xml.2` | [source](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/test_minimax_m2_tool_parser.py#L454) | A valid non-null string should be preserved, not collapsed to None. |
| `TestAnyOfNullableParam.test_anyof_nullable_param_null_value` | `PARSER.batch.7.c`, `PARSER.batch.9`, `PARSER.xml.2` | [source](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/test_minimax_m2_tool_parser.py#L486) | An actual null-like value should be returned as None/null. |
| `TestAnyOfNullableParam.test_anyof_nullable_param_object_value` | `PARSER.batch.7.c`, `PARSER.batch.9`, `PARSER.xml.2` | [source](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/test_minimax_m2_tool_parser.py#L518) | A valid object value in anyOf with null should parse as dict. |

### MiniMax

Model-specific / endpoint-specific rows:

| Test | Bucket(s) | Link | Notes |
| -- | -- | -- | -- |
| `test_extract_tool_calls_no_tools` | `PARSER.batch.3.a` | [source](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/test_minimax_tool_parser.py#L86) | test extract tool calls no tools |
| `test_extract_tool_calls` | `PARSER.batch.1` | [source](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/test_minimax_tool_parser.py#L219) | test extract tool calls Parametrized. |
| `test_preprocess_model_output_with_thinking_tags` | `REASONING.batch.2`, `REASONING.batch.2`, `// helper` | [source](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/test_minimax_tool_parser.py#L232) | Test that tool calls within thinking tags are removed during preprocessing. |
| `test_extract_tool_calls_with_thinking_tags` | `REASONING.batch.2`, `REASONING.batch.2` | [source](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/test_minimax_tool_parser.py#L253) | Test tool extraction when thinking tags contain tool calls that should be ignored. |
| `test_extract_tool_calls_invalid_json` | `PARSER.batch.4.b` | [source](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/test_minimax_tool_parser.py#L281) | Test that invalid JSON in tool calls is handled gracefully. |
| `test_extract_tool_calls_missing_name_or_arguments` | `PARSER.batch.4.c` | [source](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/test_minimax_tool_parser.py#L300) | Test that tool calls missing name or arguments are filtered out. |
| `test_streaming_basic_functionality` | `PARSER.stream.1` | [source](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/test_minimax_tool_parser.py#L320) | Test basic streaming functionality. |
| `test_streaming_with_content_before_tool_calls` | `PARSER.stream.1`, `PARSER.batch.8.a` | [source](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/test_minimax_tool_parser.py#L350) | Test streaming when there's content before tool calls. |
| `test_streaming_no_tool_calls` | `PARSER.batch.3.a`, `PARSER.stream.1` | [source](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/test_minimax_tool_parser.py#L376) | Test streaming when there are no tool calls. |
| `test_streaming_with_thinking_tags` | `PARSER.stream.1`, `REASONING.batch.2`, `REASONING.batch.2` | [source](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/test_minimax_tool_parser.py#L396) | Test streaming with thinking tags that contain tool calls. |
| `test_extract_tool_calls_multiline_json_not_supported` | `PARSER.batch.4.d` | [source](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/test_minimax_tool_parser.py#L423) | Test that multiline JSON in tool calls is not currently supported. |
| `test_streaming_arguments_incremental_output` | `PARSER.stream.1`, `PARSER.stream.3` | [source](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/test_minimax_tool_parser.py#L446) | Test that streaming arguments are returned incrementally, not cumulatively. |
| `test_streaming_arguments_delta_only` | `PARSER.stream.1` | [source](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/test_minimax_tool_parser.py#L526) | Test that each streaming call returns only the delta (new part) of arguments. |
| `test_streaming_openai_compatibility` | `PARSER.stream.1` | [source](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/test_minimax_tool_parser.py#L610) | Test that streaming behavior with buffering works correctly. |
| `test_streaming_thinking_tag_buffering` | `PARSER.stream.1`, `REASONING.batch.2`, `REASONING.batch.2` | [source](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/test_minimax_tool_parser.py#L697) | Test that tool calls within thinking tags are properly handled during streaming. |
| `test_streaming_complex_scenario_with_multiple_tools` | `PARSER.batch.2.b`, `PARSER.stream.2` | [source](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/test_minimax_tool_parser.py#L796) | Test complex streaming scenario: tools inside <think> tags and multiple tool calls in one group. |
| `test_streaming_character_by_character_output` | `PARSER.stream.1` | [source](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/test_minimax_tool_parser.py#L968) | Test character-by-character streaming output to simulate real streaming scenarios. |
| `test_streaming_character_by_character_simple_tool_call` | `PARSER.stream.1` | [source](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/test_minimax_tool_parser.py#L1124) | Test character-by-character streaming for a simple tool call scenario. |
| `test_streaming_character_by_character_with_buffering` | `PARSER.stream.1` | [source](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/test_minimax_tool_parser.py#L1185) | Test character-by-character streaming with edge cases that trigger buffering. |

### Mistral

Model-specific / endpoint-specific rows:

| Test | Bucket(s) | Link | Notes |
| -- | -- | -- | -- |
| `test_extract_tool_calls_no_tools` | `PARSER.batch.3.a` | [source](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/test_mistral_tool_parser.py#L235) | test extract tool calls no tools Parametrized. |
| `test_extract_tool_calls_pre_v11_tokenizer` | `PARSER.fmt.3`, `PARSER.fmt.5` | [source](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/test_mistral_tool_parser.py#L361) | Parametrized over 8 ids including `argument_before_name` and `argument_before_name_and_name_in_argument` — JSON field-order swap (fmt.5 sub-axis 2). |
| `test_extract_tool_calls_pre_v11_multiple_bot_tokens_raises` | `PARSER.fmt.3` | [source](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/test_mistral_tool_parser.py#L374) | test extract tool calls pre v11 multiple bot tokens raises |
| `test_extract_tool_calls_pre_v11_regex_fallback` | `PARSER.fmt.3` | [source](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/test_mistral_tool_parser.py#L387) | The regex fallback path finds valid JSON via regex when the primary raw_decode fails on leading junk. It should re-serialize arguments and return a valid tool call. |
| `test_extract_tool_calls_pre_v11_regex_fallback_fails` | `PARSER.batch.4.a`, `PARSER.fmt.3` | [source](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/test_mistral_tool_parser.py#L405) | test extract tool calls pre v11 regex fallback fails |
| `test_extract_tool_calls` | `PARSER.batch.1` | [source](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/test_mistral_tool_parser.py#L501) | test extract tool calls Parametrized. |
| `test_extract_tool_calls_v11_without_args_skipped` | `PARSER.batch.6.c`, `PARSER.fmt.3` | [source](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/test_mistral_tool_parser.py#L514) | test extract tool calls v11 without args skipped |
| `test_extract_tool_calls_streaming_pre_v11_tokenizer` | `PARSER.stream.1`, `PARSER.fmt.3`, `PARSER.fmt.5` | [source](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/test_mistral_tool_parser.py#L740) | Streaming variant of pre-v11 parametrize set; includes `argument_before_name*` (fmt.5 sub-axis 2). |
| `test_extract_tool_calls_streaming` | `PARSER.stream.1` | [source](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/test_mistral_tool_parser.py#L818) | test extract tool calls streaming Parametrized. |
| `test_extract_tool_calls_streaming_v11_no_tools` | `PARSER.batch.3.d`, `PARSER.stream.1`, `PARSER.fmt.3` | [source](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/test_mistral_tool_parser.py#L835) | test extract tool calls streaming v11 no tools |
| `test_extract_tool_calls_streaming_one_chunk` | `PARSER.stream.3`, `PARSER.fmt.5` | [source](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/test_mistral_tool_parser.py#L1103) | Parametrized over both pre-v11 and v11 IDs incl. `argument_before_name*` (fmt.5 sub-axis 2). |
| `test_fast_detokenization_text_detection` | `(inline regression annotation)` | [source](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/test_mistral_tool_parser.py#L1159) | Regression: bot_token in text but not token_ids (PR #37209). Parametrized. |
| `test_extract_tool_calls_streaming_exception_returns_none` | `PARSER.stream.1`, `(dissolved; see PARSER.batch.4 impl-defined)` | [source](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/test_mistral_tool_parser.py#L1224) | test extract tool calls streaming exception returns none Parametrized. |
| `test_adjust_request_grammar_factory` | `FRONTEND.3/.6` | [source](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/test_mistral_tool_parser.py#L1323) | test adjust request grammar factory Parametrized. |
| `test_adjust_request_unsupported_grammar_for_tokenizer` | `FRONTEND.3/.6` | [source](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/test_mistral_tool_parser.py#L1351) | test adjust request unsupported grammar for tokenizer |
| `test_adjust_request_non_mistral_tokenizer` | `FRONTEND.3/.6` | [source](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/test_mistral_tool_parser.py#L1369) | test adjust request non mistral tokenizer Parametrized. |
| `test_adjust_request_unsupported_structured_outputs` | `FRONTEND.3/.6` | [source](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/test_mistral_tool_parser.py#L1390) | test adjust request unsupported structured outputs Parametrized. |
| `test_adjust_request_unsupported_response_format` | `PARSER.fmt.3`, `FRONTEND.3/.6` | [source](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/test_mistral_tool_parser.py#L1402) | test adjust request unsupported response format |
| `test_adjust_request_structured_outputs_generates_grammar` | `FRONTEND.3/.6` | [source](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/test_mistral_tool_parser.py#L1427) | test adjust request structured outputs generates grammar Parametrized. |
| `test_adjust_request_response_format_generates_grammar` | `PARSER.fmt.3`, `FRONTEND.3/.6` | [source](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/test_mistral_tool_parser.py#L1472) | test adjust request response format generates grammar Parametrized. |
| `test_adjust_request_tool_choice_with_json_schema_factory_routing` | `FRONTEND.3/.6` | [source](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/test_mistral_tool_parser.py#L1503) | test adjust request tool choice with json schema factory routing Parametrized. |
| `test_grammar_from_tool_parser_default_false` | `FRONTEND.3/.6` | [source](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/test_mistral_tool_parser.py#L1546) | test grammar from tool parser default false |
| `test_grammar_from_tool_parser_set_by_adjust_request` | `FRONTEND.3/.6` | [source](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/test_mistral_tool_parser.py#L1551) | test grammar from tool parser set by adjust request |
| `test_build_non_streaming_tool_calls` | `// helper` | [source](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/test_mistral_tool_parser.py#L1576) | test build non streaming tool calls Parametrized. |
| `TestExtractMaybeReasoningAndToolStreaming.test_no_reasoning_tools_called` | `PARSER.stream.1`, `REASONING.batch.2`, `REASONING.batch.1`, `REASONING.batch.2` | [source](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/test_mistral_tool_parser.py#L1642) | TestExtractMaybeReasoningAndToolStreaming.test no reasoning tools called |
| `TestExtractMaybeReasoningAndToolStreaming.test_no_reasoning_no_tools` | `PARSER.batch.3.a`, `PARSER.stream.1`, `REASONING.batch.2`, `REASONING.batch.1`, `REASONING.batch.2` | [source](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/test_mistral_tool_parser.py#L1666) | TestExtractMaybeReasoningAndToolStreaming.test no reasoning no tools |
| `TestExtractMaybeReasoningAndToolStreaming.test_mistral_reasoning_parser_no_think_token` | `PARSER.stream.1`, `REASONING.batch.2`, `REASONING.batch.2` | [source](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/test_mistral_tool_parser.py#L1683) | TestExtractMaybeReasoningAndToolStreaming.test mistral reasoning parser no think token |
| `TestExtractMaybeReasoningAndToolStreaming.test_mistral_reasoning_parser_with_think_token` | `PARSER.stream.1`, `REASONING.batch.2`, `REASONING.batch.2` | [source](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/test_mistral_tool_parser.py#L1709) | TestExtractMaybeReasoningAndToolStreaming.test mistral reasoning parser with think token |
| `TestExtractMaybeReasoningAndToolStreaming.test_non_mistral_reasoning_parser_always_expects_thinking` | `PARSER.stream.1`, `REASONING.batch.2`, `REASONING.batch.2` | [source](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/test_mistral_tool_parser.py#L1736) | TestExtractMaybeReasoningAndToolStreaming.test non mistral reasoning parser always expects thinking |
| `TestExtractMaybeReasoningAndToolStreaming.test_reasoning_already_ended_no_reset` | `PARSER.stream.1`, `REASONING.batch.2`, `REASONING.batch.2` | [source](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/test_mistral_tool_parser.py#L1763) | TestExtractMaybeReasoningAndToolStreaming.test reasoning already ended no reset |
| `TestExtractMaybeReasoningAndToolStreaming.test_pre_v15_ignores_prompt_reasoning_end` | `PARSER.stream.1`, `REASONING.batch.2`, `REASONING.batch.2` | [source](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/test_mistral_tool_parser.py#L1793) | TestExtractMaybeReasoningAndToolStreaming.test pre v15 ignores prompt reasoning end |
| `TestExtractMaybeReasoningAndToolStreaming.test_non_pre_v15_prompt_reasoning_end` | `PARSER.stream.1`, `REASONING.batch.2`, `REASONING.batch.2` | [source](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/test_mistral_tool_parser.py#L1825) | TestExtractMaybeReasoningAndToolStreaming.test non pre v15 prompt reasoning end |
| `TestExtractMaybeReasoningAndToolStreaming.test_reasoning_end_transition_with_content` | `PARSER.stream.1`, `REASONING.batch.2`, `PARSER.batch.8.c`, `REASONING.batch.2` | [source](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/test_mistral_tool_parser.py#L1858) | When reasoning ends and the delta has content, that content is cleared from delta_message and used as current_text for tool parsing. |
| `TestExtractMaybeReasoningAndToolStreaming.test_reasoning_end_transition_without_content` | `PARSER.stream.1`, `REASONING.batch.2`, `REASONING.batch.2` | [source](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/test_mistral_tool_parser.py#L1899) | When reasoning ends but the delta has no content, current_text is set to empty string. |

### OLMo 3

Model-specific / endpoint-specific rows:

| Test | Bucket(s) | Link | Notes |
| -- | -- | -- | -- |
| `test_no_tool_call` | `PARSER.batch.3.a` | [source](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/test_olmo3_tool_parser.py#L72) | test no tool call Parametrized. |
| `test_tool_call` | `PARSER.batch.1` | [source](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/test_olmo3_tool_parser.py#L187) | test tool call Parametrized. |
| `test_streaming_tool_call_with_large_steps` | `PARSER.stream.1` | [source](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/test_olmo3_tool_parser.py#L208) | test streaming tool call with large steps |
| `test_regex_timeout_handling` | `(dissolved; see PARSER.batch.4 impl-defined)` | [source](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/test_olmo3_tool_parser.py#L231) | test regex timeout is handled gracefully Parametrized. |

### OpenAI Harmony / gpt-oss

Model-specific / endpoint-specific rows:

| Test | Bucket(s) | Link | Notes |
| -- | -- | -- | -- |
| `test_extract_tool_calls_no_tools` | `PARSER.batch.3.a`, `PARSER.harmony.1` | [source](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/test_openai_tool_parser.py#L55) | test extract tool calls no tools |
| `test_extract_tool_calls_single_tool` | `PARSER.batch.1`, `PARSER.harmony.1` | [source](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/test_openai_tool_parser.py#L92) | test extract tool calls single tool Parametrized. |
| `test_extract_tool_calls_multiple_tools` | `PARSER.batch.2.a`, `PARSER.harmony.1` | [source](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/test_openai_tool_parser.py#L130) | test extract tool calls multiple tools |
| `test_extract_tool_calls_with_content` | `PARSER.batch.8.c`, `PARSER.harmony.1` | [source](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/test_openai_tool_parser.py#L220) | test extract tool calls with content |

### Phi-4 Mini

Common suite config: [TestPhi4MiniToolParser](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/test_phi4mini_tool_parser.py#L32).

Inherited common `ToolParserTests` rows:

| Test | Bucket(s) | Link | Notes |
| -- | -- | -- | -- |
| `TestPhi4MiniToolParser.test_no_tool_calls` | `PARSER.batch.3.a`, `PARSER.stream.1` | [source](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/common_tests.py#L125) | Plain text produces no tool calls; common suite runs streaming and non-streaming. Model config: tests/tool_parsers/test_phi4mini_tool_parser.py:32. |
| `TestPhi4MiniToolParser.test_single_tool_call_simple_args` | `PARSER.batch.1`, `PARSER.stream.1` | [source](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/common_tests.py#L145) | One simple call in streaming and non-streaming. Model config: tests/tool_parsers/test_phi4mini_tool_parser.py:32. |
| `TestPhi4MiniToolParser.test_parallel_tool_calls` | `PARSER.batch.2.a`, `PARSER.stream.2`, `PARSER.fmt.1` | [source](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/common_tests.py#L185) | Multiple calls plus unique ID assertion. Model config: tests/tool_parsers/test_phi4mini_tool_parser.py:32. |
| `TestPhi4MiniToolParser.test_various_data_types` | `PARSER.batch.7.a`, `PARSER.stream.1` | [source](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/common_tests.py#L220) | All JSON scalar/container argument types. Model config: tests/tool_parsers/test_phi4mini_tool_parser.py:32. |
| `TestPhi4MiniToolParser.test_empty_arguments` | `PARSER.batch.6.a`, `PARSER.stream.1` | [source](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/common_tests.py#L260) | Parameterless call. Model config: tests/tool_parsers/test_phi4mini_tool_parser.py:32. |
| `TestPhi4MiniToolParser.test_surrounding_text` | `PARSER.batch.8.c`, `PARSER.stream.1` | [source](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/common_tests.py#L282) | Tool call surrounded by normal text. Model config: tests/tool_parsers/test_phi4mini_tool_parser.py:32. |
| `TestPhi4MiniToolParser.test_escaped_strings` | `PARSER.batch.7.b`, `PARSER.stream.1` | [source](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/common_tests.py#L302) | Escaped strings / Unicode-like values. Model config: tests/tool_parsers/test_phi4mini_tool_parser.py:32. |
| `TestPhi4MiniToolParser.test_malformed_input` | `PARSER.batch.4.a`, `PARSER.stream.1`, `(dissolved; see PARSER.batch.4 impl-defined)` | [source](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/common_tests.py#L322) | Malformed inputs must not crash. Model config: tests/tool_parsers/test_phi4mini_tool_parser.py:32. |
| `TestPhi4MiniToolParser.test_streaming_reconstruction` | `PARSER.stream.1` | [source](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/common_tests.py#L340) | Streaming output should match non-streaming output. Model config: tests/tool_parsers/test_phi4mini_tool_parser.py:32. |

### Pythonic generic

Model-specific / endpoint-specific rows:

| Test | Bucket(s) | Link | Notes |
| -- | -- | -- | -- |
| `test_no_tool_call` | `PARSER.batch.3.a` | [source](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/test_pythonic_tool_parser.py#L64) | test no tool call Parametrized. |
| `test_tool_call` | `PARSER.batch.1` | [source](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/test_pythonic_tool_parser.py#L167) | test tool call Parametrized. |
| `test_streaming_tool_call_with_large_steps` | `PARSER.stream.1` | [source](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/test_pythonic_tool_parser.py#L188) | test streaming tool call with large steps |
| `test_regex_timeout_handling` | `(dissolved; see PARSER.batch.4 impl-defined)` | [source](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/test_pythonic_tool_parser.py#L211) | test regex timeout is handled gracefully Parametrized. |

### Qwen3 Coder / Qwen3 XML

Model-specific / endpoint-specific rows:

| Test | Bucket(s) | Link | Notes |
| -- | -- | -- | -- |
| `test_extract_tool_calls_no_tools` | `PARSER.batch.3.a` | [source](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/test_qwen3coder_tool_parser.py#L202) | test extract tool calls no tools |
| `test_extract_tool_calls` | `PARSER.batch.1` | [source](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/test_qwen3coder_tool_parser.py#L384) | test extract tool calls Parametrized. |
| `test_extract_tool_calls_fallback_no_tags` | `PARSER.batch.5.a` | [source](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/test_qwen3coder_tool_parser.py#L401) | Test fallback parsing when XML tags are missing |
| `test_extract_tool_calls_type_conversion` | `PARSER.batch.7.c`, `PARSER.xml.2` | [source](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/test_qwen3coder_tool_parser.py#L424) | Test parameter type conversion based on tool schema |
| `test_extract_tool_calls_streaming` | `PARSER.stream.3` | [source](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/test_qwen3coder_tool_parser.py#L652) | Test incremental streaming behavior including typed parameters Parametrized. |
| `test_extract_tool_calls_missing_closing_parameter_tag` | `PARSER.batch.4.d`, `PARSER.batch.5.a` | [source](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/test_qwen3coder_tool_parser.py#L729) | Test handling of missing closing </parameter> tag |
| `test_extract_tool_calls_streaming_missing_closing_tag` | `PARSER.batch.4.d`, `PARSER.batch.5.a`, `PARSER.stream.1` | [source](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/test_qwen3coder_tool_parser.py#L771) | Test streaming with missing closing </parameter> tag |
| `test_extract_tool_calls_streaming_incremental` | `PARSER.stream.3` | [source](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/test_qwen3coder_tool_parser.py#L846) | Test that streaming is truly incremental |
| `test_extract_tool_calls_complex_type_with_single_quote` | `PARSER.batch.7.d` | [source](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/test_qwen3coder_tool_parser.py#L904) | Test parameter type conversion based on tool schema |
| `test_extract_tool_calls_streaming_missing_opening_tag` | `PARSER.batch.5.b`, `PARSER.stream.1` | [source](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/test_qwen3coder_tool_parser.py#L943) | Test streaming with missing opening <tool_call> tag This tests that the streaming parser correctly handles tool calls that start directly with <function=...> |
| `test_malformed_xml_no_gt_delimiter` | `PARSER.batch.4.d`, `PARSER.stream.1`, `PARSER.stream.3`, `(inline regression annotation)` | [source](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/test_qwen3coder_tool_parser.py#L1023) | Regression: malformed XML without '>' must not crash (PR #36774). |
| `test_none_tool_calls_filtered` | `PARSER.batch.9`, `(inline regression annotation)` | [source](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/test_qwen3coder_tool_parser.py#L1040) | Regression: None tool calls filtered from output (PR #36774). |
| `test_anyof_parameter_not_double_encoded` | `PARSER.batch.7.c`, `(inline regression annotation)`, `PARSER.xml.2` | [source](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/test_qwen3coder_tool_parser.py#L1066) | Regression: anyOf parameters must not be double-encoded (PR #36032). |
| `test_streaming_multi_param_single_chunk` | `PARSER.batch.7.d`, `PARSER.stream.1`, `PARSER.stream.3`, `(inline regression annotation)` | [source](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/test_qwen3coder_tool_parser.py#L1105) | Regression: speculative decode delivering multiple params at once (PR #35615). |
| `test_no_double_serialization_string_args` | `(inline regression annotation)` | [source](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/test_qwen3coder_tool_parser.py#L1139) | Regression: string arguments must not be double-serialized (PR #35615). |
| `test_get_vllm_registry_structural_tag_returns_structural_tag` | `PARSER.fmt.3`, `FRONTEND.3/.6` | [source](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/test_qwen3coder_tool_parser.py#L1175) | test get vllm registry structural tag returns structural tag |
| `test_adjust_request_auto_uses_vllm_registry_structural_tag` | `PARSER.fmt.3`, `FRONTEND.3/.6` | [source](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/test_qwen3coder_tool_parser.py#L1213) | test adjust request auto uses vllm registry structural tag Parametrized. |
| `test_adjust_request_required_prefers_structural_tag` | `PARSER.fmt.3`, `FRONTEND.3/.6` | [source](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/test_qwen3coder_tool_parser.py#L1239) | test adjust request required prefers structural tag |

### Qwen3 XML

Common suite config: [TestQwen3xmlToolParser](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/test_qwen3xml_tool_parser.py#L15).

Inherited common `ToolParserTests` rows:

| Test | Bucket(s) | Link | Notes |
| -- | -- | -- | -- |
| `TestQwen3xmlToolParser.test_no_tool_calls` | `PARSER.batch.3.a`, `PARSER.stream.1` | [source](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/common_tests.py#L125) | Plain text produces no tool calls; common suite runs streaming and non-streaming. Model config: tests/tool_parsers/test_qwen3xml_tool_parser.py:15. |
| `TestQwen3xmlToolParser.test_single_tool_call_simple_args` | `PARSER.batch.1`, `PARSER.stream.1` | [source](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/common_tests.py#L145) | One simple call in streaming and non-streaming. Model config: tests/tool_parsers/test_qwen3xml_tool_parser.py:15. |
| `TestQwen3xmlToolParser.test_parallel_tool_calls` | `PARSER.batch.2.a`, `PARSER.stream.2`, `PARSER.fmt.1` | [source](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/common_tests.py#L185) | Multiple calls plus unique ID assertion. Model config: tests/tool_parsers/test_qwen3xml_tool_parser.py:15. |
| `TestQwen3xmlToolParser.test_various_data_types` | `PARSER.batch.7.a`, `PARSER.stream.1` | [source](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/common_tests.py#L220) | All JSON scalar/container argument types. Model config: tests/tool_parsers/test_qwen3xml_tool_parser.py:15. |
| `TestQwen3xmlToolParser.test_empty_arguments` | `PARSER.batch.6.a`, `PARSER.stream.1` | [source](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/common_tests.py#L260) | Parameterless call. Model config: tests/tool_parsers/test_qwen3xml_tool_parser.py:15. |
| `TestQwen3xmlToolParser.test_surrounding_text` | `PARSER.batch.8.c`, `PARSER.stream.1` | [source](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/common_tests.py#L282) | Tool call surrounded by normal text. Model config: tests/tool_parsers/test_qwen3xml_tool_parser.py:15. |
| `TestQwen3xmlToolParser.test_escaped_strings` | `PARSER.batch.7.b`, `PARSER.stream.1` | [source](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/common_tests.py#L302) | Escaped strings / Unicode-like values. Model config: tests/tool_parsers/test_qwen3xml_tool_parser.py:15. |
| `TestQwen3xmlToolParser.test_malformed_input` | `PARSER.batch.4.a`, `PARSER.stream.1`, `(dissolved; see PARSER.batch.4 impl-defined)` | [source](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/common_tests.py#L322) | Malformed inputs must not crash. Model config: tests/tool_parsers/test_qwen3xml_tool_parser.py:15. |
| `TestQwen3xmlToolParser.test_streaming_reconstruction` | `PARSER.stream.1` | [source](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/common_tests.py#L340) | Streaming output should match non-streaming output. Model config: tests/tool_parsers/test_qwen3xml_tool_parser.py:15. |

### Seed OSS

Model-specific / endpoint-specific rows:

| Test | Bucket(s) | Link | Notes |
| -- | -- | -- | -- |
| `test_extract_tool_calls_no_tools` | `PARSER.batch.3.a` | [source](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/test_seed_oss_tool_parser.py#L95) | test extract tool calls no tools |
| `test_extract_tool_calls` | `PARSER.batch.1` | [source](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/test_seed_oss_tool_parser.py#L220) | test extract tool calls Parametrized. |
| `test_streaming_tool_calls_no_tools` | `PARSER.batch.3.a`, `PARSER.stream.1` | [source](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/test_seed_oss_tool_parser.py#L238) | test streaming tool calls no tools |
| `test_streaming_tool_calls` | `PARSER.stream.3` | [source](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/test_seed_oss_tool_parser.py#L425) | Test incremental streaming behavior Parametrized. |

### Step3

Common suite config: [TestStep3ToolParser](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/test_step3_tool_parser.py#L20).

Inherited common `ToolParserTests` rows:

| Test | Bucket(s) | Link | Notes |
| -- | -- | -- | -- |
| `TestStep3ToolParser.test_no_tool_calls` | `PARSER.batch.3.a`, `PARSER.stream.1` | [source](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/common_tests.py#L125) | Plain text produces no tool calls; common suite runs streaming and non-streaming. Model config: tests/tool_parsers/test_step3_tool_parser.py:20. |
| `TestStep3ToolParser.test_single_tool_call_simple_args` | `PARSER.batch.1`, `PARSER.stream.1` | [source](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/common_tests.py#L145) | One simple call in streaming and non-streaming. Model config: tests/tool_parsers/test_step3_tool_parser.py:20. |
| `TestStep3ToolParser.test_parallel_tool_calls` | `PARSER.batch.2.a`, `PARSER.stream.2`, `PARSER.fmt.1` | [source](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/common_tests.py#L185) | Multiple calls plus unique ID assertion. Model config: tests/tool_parsers/test_step3_tool_parser.py:20. |
| `TestStep3ToolParser.test_various_data_types` | `PARSER.batch.7.a`, `PARSER.stream.1` | [source](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/common_tests.py#L220) | All JSON scalar/container argument types. Model config: tests/tool_parsers/test_step3_tool_parser.py:20. |
| `TestStep3ToolParser.test_empty_arguments` | `PARSER.batch.6.a`, `PARSER.stream.1` | [source](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/common_tests.py#L260) | Parameterless call. Model config: tests/tool_parsers/test_step3_tool_parser.py:20. |
| `TestStep3ToolParser.test_surrounding_text` | `PARSER.batch.8.c`, `PARSER.stream.1` | [source](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/common_tests.py#L282) | Tool call surrounded by normal text. Model config: tests/tool_parsers/test_step3_tool_parser.py:20. |
| `TestStep3ToolParser.test_escaped_strings` | `PARSER.batch.7.b`, `PARSER.stream.1` | [source](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/common_tests.py#L302) | Escaped strings / Unicode-like values. Model config: tests/tool_parsers/test_step3_tool_parser.py:20. |
| `TestStep3ToolParser.test_malformed_input` | `PARSER.batch.4.a`, `PARSER.stream.1`, `(dissolved; see PARSER.batch.4 impl-defined)` | [source](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/common_tests.py#L322) | Malformed inputs must not crash. Model config: tests/tool_parsers/test_step3_tool_parser.py:20. |
| `TestStep3ToolParser.test_streaming_reconstruction` | `PARSER.stream.1` | [source](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/common_tests.py#L340) | Streaming output should match non-streaming output. Model config: tests/tool_parsers/test_step3_tool_parser.py:20. |

### Step3.5

Model-specific / endpoint-specific rows:

| Test | Bucket(s) | Link | Notes |
| -- | -- | -- | -- |
| `test_extract_tool_calls_no_tools` | `PARSER.batch.3.a` | [source](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/test_step3p5_tool_parser.py#L168) | test extract tool calls no tools |
| `test_extract_tool_calls` | `PARSER.batch.1` | [source](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/test_step3p5_tool_parser.py#L350) | test extract tool calls Parametrized. |
| `test_extract_tool_calls_fallback_no_tags` | `PARSER.batch.5.a` | [source](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/test_step3p5_tool_parser.py#L368) | Test fallback parsing when XML tags are missing |
| `test_extract_tool_calls_type_conversion` | `PARSER.batch.7.c`, `PARSER.xml.2` | [source](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/test_step3p5_tool_parser.py#L389) | Test parameter type conversion based on tool schema |
| `test_extract_tool_calls_streaming` | `PARSER.stream.3` | [source](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/test_step3p5_tool_parser.py#L617) | Test incremental streaming behavior including typed parameters Parametrized. |
| `test_extract_tool_calls_missing_closing_parameter_tag` | `PARSER.batch.4.d`, `PARSER.batch.5.a` | [source](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/test_step3p5_tool_parser.py#L692) | Test handling of missing closing </parameter> tag |
| `test_extract_tool_calls_streaming_missing_closing_tag` | `PARSER.batch.4.d`, `PARSER.batch.5.a`, `PARSER.stream.1` | [source](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/test_step3p5_tool_parser.py#L734) | Test streaming with missing closing </parameter> tag |
| `test_extract_tool_calls_streaming_incremental` | `PARSER.stream.3` | [source](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/test_step3p5_tool_parser.py#L808) | Test that streaming is truly incremental |
| `test_extract_tool_calls_complex_type_with_single_quote` | `PARSER.batch.7.d` | [source](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/test_step3p5_tool_parser.py#L866) | Test parameter type conversion based on tool schema |
| `test_extract_tool_calls_streaming_mixed_content_and_multiple_tool_calls` | `PARSER.stream.2`, `PARSER.batch.8.d`, `PARSER.fmt.1` | [source](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/test_step3p5_tool_parser.py#L903) | Test mixed content with multiple complete tool calls. Scenario: Model outputs "hello" + complete tool call + "hi" + complete tool call. Expected: "hello" as content, first tool cal |
| `test_extract_tool_calls_non_streaming_mixed_content_and_multiple_tool_calls` | `PARSER.batch.2.c`, `PARSER.batch.8.d`, `PARSER.fmt.1` | [source](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/test_step3p5_tool_parser.py#L1009) | Test non-streaming extraction with mixed content and multiple tool calls. Scenario: Model outputs "hello" + complete tool call + "hi" + complete tool call. Expected: "hello" as con |
| `test_extract_tool_calls_streaming_full_input_mixed_content_and_multiple_tool_calls` | `PARSER.stream.1`, `PARSER.stream.2`, `PARSER.batch.8.d`, `PARSER.fmt.1` | [source](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/test_step3p5_tool_parser.py#L1086) | Test streaming with entire input as single delta_text. Scenario: Model outputs "hello" + complete tool call + "hi" + complete tool call. This test simulates the case where the enti |
| `test_extract_tool_calls_streaming_multiple_tool_calls_no_content_between` | `PARSER.batch.2.c`, `PARSER.stream.2`, `PARSER.batch.8.a`, `PARSER.fmt.1` | [source](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/test_step3p5_tool_parser.py#L1217) | Test multiple tool calls with no content between them. Scenario: Model outputs "hello" + tool call + tool call Expected: "hello" as content, first tool call parsed (index=0), secon |
| `test_extract_tool_calls_streaming_multi_token_chunk_boundary` | `PARSER.stream.3` | [source](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/test_step3p5_tool_parser.py#L1315) | Ensure fallback doesn't close a new tool_call when boundary is in one chunk. |
| `test_extract_tool_calls_non_streaming_multiple_tool_calls_no_content_between` | `PARSER.batch.8.a`, `PARSER.fmt.1` | [source](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/test_step3p5_tool_parser.py#L1366) | Test non-streaming extraction with tool calls and no content between them. Scenario: Model outputs "hello" + tool call + tool call. Expected: "hello" as content, first tool call pa |
| `test_streaming_mtp_variable_chunks` | `PARSER.stream.1`, `PARSER.stream.3`, `(inline regression annotation)` | [source](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/test_step3p5_tool_parser.py#L1465) | Regression: MTP variable-size chunks spanning param boundaries (PR #33690). |
| `test_streaming_multi_token_per_step` | `PARSER.batch.2.b`, `PARSER.stream.1`, `PARSER.stream.3`, `(inline regression annotation)` | [source](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/test_step3p5_tool_parser.py#L1497) | Regression: MTP large chunks spanning multiple tool calls (PR #33690). |

### xLAM

Model-specific / endpoint-specific rows:

| Test | Bucket(s) | Link | Notes |
| -- | -- | -- | -- |
| `test_extract_tool_calls_no_tools` | `PARSER.batch.3.a` | [source](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/test_xlam_tool_parser.py#L99) | test extract tool calls no tools |
| `test_extract_tool_calls` | `PARSER.batch.1` | [source](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/test_xlam_tool_parser.py#L223) | test extract tool calls Parametrized. |
| `test_extract_tool_calls_list_structure` | `PARSER.batch.1` | [source](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/test_xlam_tool_parser.py#L260) | Test extraction of tool calls when the model outputs a list-structured tool call. Parametrized. |
| `test_preprocess_model_output` | `// helper` | [source](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/test_xlam_tool_parser.py#L275) | test preprocess model output |
| `test_streaming_with_list_structure` | `PARSER.stream.1` | [source](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/test_xlam_tool_parser.py#L318) | test streaming with list structure |
| `test_extract_tool_calls_streaming_incremental` | `PARSER.stream.3` | [source](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/test_xlam_tool_parser.py#L479) | Verify the XLAM Parser streaming behavior by verifying each chunk is as expected. Parametrized. |

### Mistral endpoint tool-use

Model-specific / endpoint-specific rows:

| Test | Bucket(s) | Link | Notes |
| -- | -- | -- | -- |
| `test_tool_call_with_tool_choice` | `FRONTEND.3` | [source](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_use/mistral/test_mistral_tool_calls.py#L180) | test tool call with tool choice |
| `test_tool_call_auto_or_required` | `FRONTEND.1/.3` | [source](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_use/mistral/test_mistral_tool_calls.py#L227) | test tool call auto or required Parametrized. |
| `test_tool_call_none_with_tools` | `FRONTEND.1/.3` | [source](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_use/mistral/test_mistral_tool_calls.py#L282) | test tool call none with tools |
| `test_chat_without_tools` | `PIPELINE.finish_reason` | [source](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_use/mistral/test_mistral_tool_calls.py#L341) | test chat without tools |
| `test_tool_call_with_results` | `PARSER.batch.3.a`, `PIPELINE.finish_reason`, `PARSER.batch.8.c` | [source](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_use/mistral/test_mistral_tool_calls.py#L386) | test tool call with results |
| `test_tool_call_parallel` | `PARSER.batch.2.a` | [source](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_use/mistral/test_mistral_tool_calls.py#L443) | test tool call parallel |

### API request validation

Model-specific / endpoint-specific rows:

| Test | Bucket(s) | Link | Notes |
| -- | -- | -- | -- |
| `test_chat_completion_request_with_no_tools` | `FRONTEND.1/.3` | [source](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_use/test_chat_completion_request_validations.py#L9) | test chat completion request with no tools |
| `test_chat_completion_request_with_tool_choice_but_no_tools` | `FRONTEND.3`, `FRONTEND.1/.3` | [source](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_use/test_chat_completion_request_validations.py#L41) | test chat completion request with tool choice but no tools Parametrized. |

### Chat Completions endpoint

Model-specific / endpoint-specific rows:

| Test | Bucket(s) | Link | Notes |
| -- | -- | -- | -- |
| `test_chat_completion_without_tools` | `PIPELINE.finish_reason` | [source](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_use/test_chat_completions.py#L20) | test chat completion without tools |
| `test_chat_completion_with_tools` | `FRONTEND.1/.3` | [source](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_use/test_chat_completions.py#L91) | test chat completion with tools |
| `test_response_format_with_tool_choice_required` | `PARSER.batch.7.c`, `FRONTEND.3`, `PARSER.fmt.3`, `FRONTEND.3/.6` | [source](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_use/test_chat_completions.py#L166) | Test that combining response_format: json_object with tool_choice: required doesn't crash the engine. Before the fix, this would cause a validation error: "You can only use one kin |

### Gemma 4 Responses API

Model-specific / endpoint-specific rows:

| Test | Bucket(s) | Link | Notes |
| -- | -- | -- | -- |
| `test_gemma4_adjust_request_sets_skip_special_tokens_on_responses` | `PARSER.stream.3`, `FRONTEND.1/.3`, `FRONTEND.3/.6` | [source](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_use/test_gemma4_responses_adjust_request.py#L68) | ``Gemma4ToolParser.adjust_request`` must flip ``skip_special_tokens=False`` for both ``ChatCompletionRequest`` and ``ResponsesRequest`` so that ``<\|tool_call>`` delimiters reach th |
| `test_tool_parser_adjust_request_builds_valid_response_text_config` | `PARSER.batch.7.c`, `PARSER.stream.1`, `FRONTEND.1/.3`, `PARSER.fmt.3`, `FRONTEND.3/.6` | [source](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_use/test_gemma4_responses_adjust_request.py#L90) | ``ToolParser.adjust_request`` must produce a ``ResponseTextConfig`` whose dumped form contains the JSON schema under the ``schema`` alias and does not leak the unrelated ``"Respons |

### Generic endpoint parallel tool calls

Model-specific / endpoint-specific rows:

| Test | Bucket(s) | Link | Notes |
| -- | -- | -- | -- |
| `test_parallel_tool_calls` | `PARSER.batch.2.a` | [source](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_use/test_parallel_tool_calls.py#L24) | test parallel tool calls |
| `test_parallel_tool_calls_with_results` | `PARSER.batch.2.a`, `PARSER.batch.3.a`, `PIPELINE.finish_reason`, `PARSER.batch.8.c` | [source](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_use/test_parallel_tool_calls.py#L153) | test parallel tool calls with results |
| `test_parallel_tool_calls_false` | `PARSER.batch.2.d` | [source](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_use/test_parallel_tool_calls.py#L223) | Ensure only one tool call is returned when parallel_tool_calls is False. |

### Responses API request validation

Model-specific / endpoint-specific rows:

| Test | Bucket(s) | Link | Notes |
| -- | -- | -- | -- |
| `test_responses_request_with_no_tools` | `FRONTEND.1/.3`, `FRONTEND.3/.6` | [source](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_use/test_responses_request_validations.py#L26) | test responses request with no tools |
| `test_responses_request_no_tools_tool_choice_none` | `FRONTEND.3`, `FRONTEND.1/.3`, `FRONTEND.3/.6` | [source](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_use/test_responses_request_validations.py#L38) | test responses request no tools tool choice none |
| `test_responses_request_no_tools_tool_choice_auto` | `FRONTEND.3`, `FRONTEND.1/.3`, `FRONTEND.3/.6` | [source](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_use/test_responses_request_validations.py#L45) | test responses request no tools tool choice auto |
| `test_responses_request_required_without_tools` | `PIPELINE.finish_reason`, `FRONTEND.1/.3`, `FRONTEND.3/.6` | [source](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_use/test_responses_request_validations.py#L53) | test responses request required without tools Parametrized. |
| `test_responses_request_named_tool_choice_without_tools` | `FRONTEND.3`, `PIPELINE.finish_reason`, `FRONTEND.1/.3`, `FRONTEND.3/.6` | [source](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_use/test_responses_request_validations.py#L63) | test responses request named tool choice without tools |
| `test_responses_request_with_tools_default_tool_choice` | `FRONTEND.3`, `FRONTEND.1/.3`, `FRONTEND.3/.6` | [source](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_use/test_responses_request_validations.py#L74) | test responses request with tools default tool choice |
| `test_responses_request_with_tools_tool_choice_none` | `FRONTEND.3`, `FRONTEND.1/.3`, `FRONTEND.3/.6` | [source](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_use/test_responses_request_validations.py#L81) | test responses request with tools tool choice none |
| `test_responses_request_named_tool_choice_matching` | `FRONTEND.3`, `FRONTEND.1/.3`, `FRONTEND.3/.6` | [source](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_use/test_responses_request_validations.py#L93) | test responses request named tool choice matching |
| `test_responses_request_named_tool_choice_not_matching` | `FRONTEND.3`, `FRONTEND.1/.3`, `FRONTEND.3/.6` | [source](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_use/test_responses_request_validations.py#L106) | test responses request named tool choice not matching |
| `test_responses_request_with_tools_tool_choice_auto` | `FRONTEND.3`, `FRONTEND.1/.3`, `FRONTEND.3/.6` | [source](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_use/test_responses_request_validations.py#L118) | test responses request with tools tool choice auto |
| `test_responses_request_with_tools_tool_choice_required` | `FRONTEND.3`, `FRONTEND.1/.3`, `FRONTEND.3/.6` | [source](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_use/test_responses_request_validations.py#L130) | test responses request with tools tool choice required |
| `test_responses_request_empty_tools_tool_choice_none` | `FRONTEND.3`, `FRONTEND.1/.3`, `FRONTEND.3/.6` | [source](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_use/test_responses_request_validations.py#L142) | test responses request empty tools tool choice none |
| `test_responses_request_empty_tools_tool_choice_auto` | `FRONTEND.3`, `FRONTEND.1/.3`, `FRONTEND.3/.6` | [source](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_use/test_responses_request_validations.py#L149) | test responses request empty tools tool choice auto |
| `test_responses_request_named_tool_choice_missing_name` | `PARSER.batch.4.c`, `FRONTEND.3`, `FRONTEND.1/.3`, `FRONTEND.3/.6` | [source](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_use/test_responses_request_validations.py#L163) | test responses request named tool choice missing name Parametrized. |
| `test_responses_request_empty_tools_named_tool_choice` | `FRONTEND.3`, `FRONTEND.1/.3`, `FRONTEND.3/.6` | [source](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_use/test_responses_request_validations.py#L175) | test responses request empty tools named tool choice |

### Generic endpoint tool calls

Model-specific / endpoint-specific rows:

| Test | Bucket(s) | Link | Notes |
| -- | -- | -- | -- |
| `test_tool_call_and_choice` | `PARSER.batch.1` | [source](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_use/test_tool_calls.py#L21) | test tool call and choice |
| `test_tool_call_with_results` | `PARSER.batch.3.a`, `PIPELINE.finish_reason`, `PARSER.batch.8.c` | [source](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_use/test_tool_calls.py#L150) | test tool call with results |

### Tool-choice required / structured outputs

Model-specific / endpoint-specific rows:

| Test | Bucket(s) | Link | Notes |
| -- | -- | -- | -- |
| `test_structured_outputs_json` | `FRONTEND.3`, `FRONTEND.3/.6` | [source](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_use/test_tool_choice_required.py#L194) | test structured outputs json Parametrized. |
| `test_structured_outputs_json_without_parameters` | `FRONTEND.3`, `FRONTEND.3/.6` | [source](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_use/test_tool_choice_required.py#L263) | test structured outputs json without parameters Parametrized. |
| `test_streaming_output_valid` | `PARSER.stream.1`, `FRONTEND.3`, `FRONTEND.3/.6` | [source](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_use/test_tool_choice_required.py#L283) | test streaming output valid Parametrized. |
| `test_streaming_output_valid_with_trailing_extra_data` | `PARSER.stream.4`, `FRONTEND.3`, `FRONTEND.3/.6` | [source](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_use/test_tool_choice_required.py#L334) | test streaming output valid with trailing extra data |

### Granite 4 endpoint

Model-specific / endpoint-specific rows:

| Test | Bucket(s) | Link | Notes |
| -- | -- | -- | -- |
| `test_stop_sequence_interference` | `FRONTEND.3/.6` | [source](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/entrypoints/openai/tool_parsers/test_granite4_tool_parser.py#L210) | test stop sequence interference |

### Hermes + Granite4 endpoint

Model-specific / endpoint-specific rows:

| Test | Bucket(s) | Link | Notes |
| -- | -- | -- | -- |
| `test_non_streaming_tool_call` | `PARSER.batch.1` | [source](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/entrypoints/openai/tool_parsers/test_hermes_tool_parser.py#L150) | Test tool call in non-streaming mode. |
| `test_streaming_tool_call` | `PARSER.stream.1` | [source](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/entrypoints/openai/tool_parsers/test_hermes_tool_parser.py#L183) | Test tool call in streaming mode. |
| `test_non_streaming_product_tool_call` | `PARSER.batch.7.a` | [source](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/entrypoints/openai/tool_parsers/test_hermes_tool_parser.py#L230) | Test tool call integer and boolean parameters in non-streaming mode. |
| `test_streaming_product_tool_call` | `PARSER.batch.7.a`, `PARSER.stream.1` | [source](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/entrypoints/openai/tool_parsers/test_hermes_tool_parser.py#L272) | Test tool call integer and boolean parameters in streaming mode. |

### OpenAI Harmony / gpt-oss endpoint

Model-specific / endpoint-specific rows:

| Test | Bucket(s) | Link | Notes |
| -- | -- | -- | -- |
| `test_calculator_tool_call_and_argument_accuracy` | `PARSER.batch.1`, `PARSER.harmony.1` | [source](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/entrypoints/openai/tool_parsers/test_openai_tool_parser.py#L166) | Verify calculator tool call is made and arguments are accurate. |
| `test_streaming_tool_call_get_time_with_reasoning` | `PARSER.stream.1`, `REASONING.batch.2`, `REASONING.batch.2`, `PARSER.harmony.1` | [source](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/entrypoints/openai/tool_parsers/test_openai_tool_parser.py#L206) | Verify streamed reasoning and tool call behavior for get_time. |
| `test_streaming_multiple_tools` | `PARSER.batch.2.b`, `PARSER.stream.2`, `REASONING.batch.2`, `REASONING.batch.2`, `PARSER.harmony.1` | [source](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/entrypoints/openai/tool_parsers/test_openai_tool_parser.py#L234) | Test streamed multi-tool response with reasoning. |
| `test_invalid_tool_call` | `PARSER.batch.3.a`, `PARSER.harmony.1` | [source](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/entrypoints/openai/tool_parsers/test_openai_tool_parser.py#L260) | Verify that ambiguous instructions that should not trigger a tool do not produce any tool calls. |
| `test_tool_call_with_temperature` | `PARSER.harmony.1`, `(dissolved; see PARSER.batch.4 impl-defined)` | [source](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/entrypoints/openai/tool_parsers/test_openai_tool_parser.py#L285) | Verify model produces valid tool or text output under non-deterministic sampling. |
| `test_tool_response_schema_accuracy` | `PARSER.batch.7.c`, `PARSER.harmony.1` | [source](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/entrypoints/openai/tool_parsers/test_openai_tool_parser.py#L309) | Validate that tool call arguments adhere to their declared JSON schema. |
| `test_semantic_consistency_with_temperature` | `REASONING.batch.2`, `REASONING.batch.2`, `PARSER.harmony.1`, `(dissolved; see PARSER.batch.4 impl-defined)` | [source](https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/entrypoints/openai/tool_parsers/test_openai_tool_parser.py#L342) | Test that temperature variation doesn't cause contradictory reasoning. |
