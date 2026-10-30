# SGLang Tool Parser Test Audit

Source: local SGLang checkout at `612785ffdcaf35552f1ed433a981d596ca9fe900` under `/sgl-workspace/sglang`.

Rust gateway surface: cached LightSeek `smg` checkout at `e3eccacf96bc6e041a7ec6623e2251dba1129f28` under `/home/dynamo/.cargo/git/checkouts/smg-b0cecffeb94ac563/e3eccac`; SGLang `sgl-model-gateway/Cargo.toml` pins `tool-parser = "=1.0.0"`.

Scope: parser/detector tests that directly exercise tool-call extraction or the parser state machines.

- SGLang Python rows: `test/registered/unit/function_call/*`, `test/registered/function_call/test_kimik2_detector.py`, and `test/registered/unit/parser/test_harmony_parser.py`.
- Rust gateway rows: `crates/tool_parser/src/tests.rs` and `crates/tool_parser/tests/*.rs` from the cached LightSeek checkout consumed by SGLang gateway.

Excluded from parser-taxonomy changes:

- SGLang `openai_server/function_call/*`, `test_json_schema_constraint.py`, and Ascend API tests: request shaping, guided schema constraints, `tool_choice`, and API wiring. These are `FRONTEND.*`, guided-decoding, or endpoint concerns unless a row directly exposes parser extraction behavior.
- SGLang gateway `model_gateway/tests/api/*` and `e2e_test/*`: endpoint/request lifecycle coverage. The parser corpus used by that surface is the LightSeek `tool-parser` crate audited here.
- Generic parser tests for code completion, Jinja templates, conversations, debug dimension parsers, and reasoning-parser-only files. Reasoning rows are included only where they transition into tool-call parsing.

This report expands 728 test rows: 391 SGLang Python rows plus 337 Rust `tool-parser` rows. Parametrized tests are counted once by test function. A row can carry multiple bucket tags when one test exercises several behaviors.

## Bucket Summary

Counts represent rows carrying each tag. Helper-only rows are tagged `// helper`; they do not carry numbered parser buckets.

| Bucket | Rows |
| -- | --: |
| `PARSER.batch.1` | 112 |
| `PARSER.batch.2` | 2 |
| `PARSER.batch.2.a` | 49 |
| `PARSER.batch.2.b` | 23 |
| `PARSER.batch.2.d` | 1 |
| `PARSER.batch.3` | 29 |
| `PARSER.batch.4.b` | 33 |
| `PARSER.batch.4.c` | 4 |
| `PARSER.batch.4.d` | 16 |
| `PARSER.batch.4.e` | 5 |
| `PARSER.batch.5.a` | 11 |
| `PARSER.batch.5.c` | 31 |
| `PARSER.batch.6.a` | 12 |
| `PARSER.batch.6.b` | 4 |
| `PARSER.batch.6.c` | 4 |
| `PARSER.batch.7.a` | 17 |
| `PARSER.batch.7.b` | 31 |
| `PARSER.batch.7.c` | 22 |
| `PARSER.batch.7.d` | 97 |
| `PARSER.batch.8.a` | 33 |
| `PARSER.batch.8.b` | 10 |
| `PARSER.batch.8.c` | 3 |
| `PARSER.batch.8.d` | 12 |
| `PARSER.batch.9` | 6 |
| `PARSER.batch.11` | 3 |
| `PARSER.batch.12` | 12 |
| `PARSER.batch.13` | 6 |
| `PARSER.stream.1` | 162 |
| `PARSER.stream.2` | 25 |
| `PARSER.stream.3` | 59 |
| `PARSER.stream.4` | 11 |
| `PARSER.fmt.1` | 9 |
| `PARSER.fmt.2` | 19 |
| `PARSER.fmt.3` | 114 |
| `PARSER.fmt.5` | 11 |
| `PARSER.xml.1` | 4 |
| `PARSER.xml.2` | 2 |
| `PARSER.harmony.1` | 52 |
| `PARSER.harmony.2` | 31 |
| `REASONING.batch.1` | 6 |
| `REASONING.batch.2` | 9 |
| `FRONTEND.3` | 7 |
| `FRONTEND.3/.6` | 10 |
| `PIPELINE.finish_reason` | 9 |
| `// helper` | 71 |
| **Total distinct rows** | 728 |

## Research Completeness Notes

- The prior SGLang note was a conclusion summary. It covered 141 lines and no row-level evidence. This rewrite follows the vLLM audit shape: bucket counts plus per-parser source-linked rows.
- The Python detector surface adds concrete SGLang-only cases already reflected in the parity work: malformed-prefix resync (`PARSER.batch.4.e`), delimiter-state cases (`PARSER.batch.11` / `.12`), and unknown-tool-name handling (`PARSER.batch.13`).
- The cached Rust `tool-parser` surface is materially broader than the Python-only audit. It has tests for MiniMax M2, Step3, Cohere, DeepSeek 3.1 / DSML / V4, Qwen XML, generic JSON, fallback behavior, partial JSON repair, and mixed-format edge cases.
- Several LightSeek rows hit categories that `PARSER_CASES.md` still lists as universal gaps for Dynamo, including empty function names, Unicode function names, extremely long function names, very large payloads, duplicate JSON keys, and partial-token buffer-boundary cases. These should become Dynamo fixture candidates or be explicitly tracked as out of scope.
- Rust `tool-parser` XML rows give concrete coverage for `PARSER.xml.1` HTML/entity decoding. The earlier SGLang Python-only audit did not expose that bucket.
- Streaming coverage is much deeper than the current Dynamo YAML batch harness can express. The rows here should feed a future Dynamo+SGLang streaming harness, while vLLM streaming remains special because it usually needs token IDs.

## Existing-Test Overlap Decision

- `PARSER.batch.11` overlaps with `PARSER.batch.7.b` only at the input-character level. It is kept separate because delimiter-aware parsers must not split on separator-looking characters while inside a string value.
- `PARSER.batch.12` is the same delimiter-state check on the multi-call path. It overlaps with `PARSER.batch.2.a`, but multi-call extraction alone does not prove the delimiter state machine ignores separator-looking characters inside one argument.
- `PARSER.batch.13` is not malformed syntax. The call is grammatically valid; the tested behavior is registry lookup after parse, with SGLang defaulting to drop for some detectors and forwarding for others.
- The LightSeek Rust rows for long names, Unicode names, duplicate keys, and very large payloads currently map to existing `fmt` / `batch.7` buckets for this audit, but they are strong candidates for explicit sub-case labels if we want parity fixtures to preserve those axes.

## SGLang-Only Taxonomy Additions

### `PARSER.batch.4.e`

SGLang and LightSeek both include malformed-prefix recovery rows where a bad tool-looking fragment is followed by a later valid call. Dynamo and vLLM usually treat the whole string as normal text. This is impl-defined malformed-input behavior, but it needs its own subcase because it tests resynchronization rather than no-crash handling.

Representative rows:

- `TestLlama32Detector.test_invalid_then_valid_json`
- `TestJsonArrayParser.test_malformed_json_recovery`
- `tool_parser_mixed_edge_cases.rs::test_parser_recovery_after_invalid_input`
- `tool_parser_fallback.rs::test_mixed_valid_and_invalid_content`

### `PARSER.batch.11` / `PARSER.batch.12`

SGLang Python and LightSeek Rust both test parser state around separators, braces, brackets, and format markers inside string values. Existing Dynamo fixtures cover the Llama semicolon shape; the Rust rows show this should be applied broadly to JSON-array, XML, Mistral, Qwen, and generic fallback grammars.

Representative rows:

- `TestJsonArrayParser.test_braces_in_strings`
- `TestJsonArrayParser.test_separator_in_same_chunk`
- `TestJsonArrayParser.test_separator_in_separate_chunk`
- `tool_parser_mixed_edge_cases.rs::test_format_markers_in_string_content`
- `tool_parser_mistral.rs::test_mistral_with_brackets_in_strings`
- `tool_parser_qwen.rs::test_qwen_with_newlines_in_strings`

### `PARSER.batch.13`

SGLang has explicit unknown-tool-name rows. Python detectors often drop unknown tools by default unless forwarding is enabled; some detector families forward because they do not route through the same base JSON lookup. LightSeek adds similar unknown/invalid-function-name coverage for XML and MiniMax-style grammars.

Representative rows:

- `test_unknown_tool_name_dropped_default`
- `test_unknown_tool_name_forwarded`
- `TestHunyuanDetectorDetectAndParse.test_unknown_tool_skipped`
- `TestLfm2Detector.test_detect_and_parse_unknown_function`

### `PARSER.xml.1`

The Python-only audit did not exercise XML entity decoding, but the Rust `tool-parser` surface does. This should be considered covered by SGLang gateway research and should feed Dynamo XML-family fixture planning.

Representative rows:

- `tool_parser_minimax_m2.rs::test_minimax_xml_entities`
- `tool_parser_qwen_xml.rs::test_qwen_xml_html_entity_decoding`
- `tool_parser_qwen_xml.rs::test_qwen_xml_html_numeric_entities`
- `tool_parser_qwen_xml.rs::test_qwen_xml_mixed_html_and_json`

## Fixture Impact

| Finding | Dynamo action |
| -- | -- |
| `PARSER.batch.4.e` malformed-prefix recovery | Keep as impl-defined divergence; add/retain explicit fixture rows where families can resync. |
| `PARSER.batch.11` / `.12` delimiter-state checks | Add to more applicable JSON/XML/Pythonic fixture families, not just Llama semicolon grammar. |
| `PARSER.batch.13` unknown-tool behavior | Preserve per-engine expectations; SGLang default drop vs Dynamo/vLLM forwarding is a real divergence. |
| Rust `PARSER.xml.1` entity decoding | Add XML-family fixture rows for `&lt;`, `&amp;`, numeric entities, and mixed HTML+JSON values. |
| Prior universal gaps covered by LightSeek | Convert empty/Unicode/long function name, huge payload, duplicate-key, and boundary-buffer cases into explicit follow-up fixtures or update the gap list. |
| Streaming-heavy rows | Defer to a stream fixture harness; record rows now so they are not lost. |

## Model / Parser Sections

### SGLang Python - TestHermesDetector

| Test | Bucket(s) | Link | Notes |
| -- | -- | -- | -- |
| `TestHermesDetector.test_has_tool_call_true` | `PARSER.fmt.3` | [source](https://github.com/sgl-project/sglang/blob/612785ffdcaf35552f1ed433a981d596ca9fe900/test/registered/unit/function_call/test_hermes_detector.py#L56) | Format detection / marker-recognition helper coverage. |
| `TestHermesDetector.test_has_tool_call_false` | `PARSER.fmt.3` | [source](https://github.com/sgl-project/sglang/blob/612785ffdcaf35552f1ed433a981d596ca9fe900/test/registered/unit/function_call/test_hermes_detector.py#L60) | Format detection / marker-recognition helper coverage. |
| `TestHermesDetector.test_single_tool_call` | `PARSER.batch.1` | [source](https://github.com/sgl-project/sglang/blob/612785ffdcaf35552f1ed433a981d596ca9fe900/test/registered/unit/function_call/test_hermes_detector.py#L66) | Single tool call. |
| `TestHermesDetector.test_multiple_tool_calls` | `PARSER.batch.2.a` | [source](https://github.com/sgl-project/sglang/blob/612785ffdcaf35552f1ed433a981d596ca9fe900/test/registered/unit/function_call/test_hermes_detector.py#L75) | Multiple tool calls. |
| `TestHermesDetector.test_tool_call_with_leading_text` | `PARSER.batch.8.a` | [source](https://github.com/sgl-project/sglang/blob/612785ffdcaf35552f1ed433a981d596ca9fe900/test/registered/unit/function_call/test_hermes_detector.py#L85) | Tool call with leading text. |
| `TestHermesDetector.test_no_tool_call` | `PARSER.batch.3` | [source](https://github.com/sgl-project/sglang/blob/612785ffdcaf35552f1ed433a981d596ca9fe900/test/registered/unit/function_call/test_hermes_detector.py#L92) | No tool call. |
| `TestHermesDetector.test_tool_call_with_multiple_arguments` | `PARSER.batch.2.a`, `PARSER.batch.7.a` | [source](https://github.com/sgl-project/sglang/blob/612785ffdcaf35552f1ed433a981d596ca9fe900/test/registered/unit/function_call/test_hermes_detector.py#L98) | Tool call with multiple arguments. |
| `TestHermesDetector.test_malformed_json_returns_original_text` | `PARSER.batch.4.b` | [source](https://github.com/sgl-project/sglang/blob/612785ffdcaf35552f1ed433a981d596ca9fe900/test/registered/unit/function_call/test_hermes_detector.py#L106) | Malformed json returns original text. |
| `TestHermesDetector.test_structure_info` | `// helper` | [source](https://github.com/sgl-project/sglang/blob/612785ffdcaf35552f1ed433a981d596ca9fe900/test/registered/unit/function_call/test_hermes_detector.py#L114) | Registry, construction, serialization, or helper-only coverage. |
| `TestHermesDetector.test_streaming_single_tool_call` | `PARSER.batch.1`, `PARSER.stream.1` | [source](https://github.com/sgl-project/sglang/blob/612785ffdcaf35552f1ed433a981d596ca9fe900/test/registered/unit/function_call/test_hermes_detector.py#L123) | Streaming single tool call. |
| `TestHermesDetector.test_streaming_normal_text_before_tool` | `PARSER.batch.8.a`, `PARSER.stream.1` | [source](https://github.com/sgl-project/sglang/blob/612785ffdcaf35552f1ed433a981d596ca9fe900/test/registered/unit/function_call/test_hermes_detector.py#L146) | Streaming normal text before tool. |
| `TestHermesDetector.test_streaming_text_then_tool_call` | `PARSER.batch.8.a`, `PARSER.stream.1` | [source](https://github.com/sgl-project/sglang/blob/612785ffdcaf35552f1ed433a981d596ca9fe900/test/registered/unit/function_call/test_hermes_detector.py#L152) | Streaming text then tool call. |

### SGLang Python - TestLlama32Detector

| Test | Bucket(s) | Link | Notes |
| -- | -- | -- | -- |
| `TestLlama32Detector.test_has_tool_call_with_python_tag` | `PARSER.fmt.3` | [source](https://github.com/sgl-project/sglang/blob/612785ffdcaf35552f1ed433a981d596ca9fe900/test/registered/unit/function_call/test_llama32_detector.py#L56) | Format detection / marker-recognition helper coverage. |
| `TestLlama32Detector.test_has_tool_call_with_json_start` | `PARSER.fmt.3` | [source](https://github.com/sgl-project/sglang/blob/612785ffdcaf35552f1ed433a981d596ca9fe900/test/registered/unit/function_call/test_llama32_detector.py#L60) | Format detection / marker-recognition helper coverage. |
| `TestLlama32Detector.test_has_tool_call_false` | `PARSER.fmt.3` | [source](https://github.com/sgl-project/sglang/blob/612785ffdcaf35552f1ed433a981d596ca9fe900/test/registered/unit/function_call/test_llama32_detector.py#L64) | Format detection / marker-recognition helper coverage. |
| `TestLlama32Detector.test_single_tool_call_with_python_tag` | `PARSER.batch.1`, `PARSER.fmt.3` | [source](https://github.com/sgl-project/sglang/blob/612785ffdcaf35552f1ed433a981d596ca9fe900/test/registered/unit/function_call/test_llama32_detector.py#L70) | Single tool call with python tag. |
| `TestLlama32Detector.test_single_tool_call_without_python_tag` | `PARSER.batch.1`, `PARSER.fmt.3` | [source](https://github.com/sgl-project/sglang/blob/612785ffdcaf35552f1ed433a981d596ca9fe900/test/registered/unit/function_call/test_llama32_detector.py#L78) | Single tool call without python tag. |
| `TestLlama32Detector.test_normal_text_before_python_tag` | `PARSER.batch.8.a`, `PARSER.fmt.3` | [source](https://github.com/sgl-project/sglang/blob/612785ffdcaf35552f1ed433a981d596ca9fe900/test/registered/unit/function_call/test_llama32_detector.py#L84) | Normal text before python tag. |
| `TestLlama32Detector.test_no_tool_call` | `PARSER.batch.3` | [source](https://github.com/sgl-project/sglang/blob/612785ffdcaf35552f1ed433a981d596ca9fe900/test/registered/unit/function_call/test_llama32_detector.py#L90) | No tool call. |
| `TestLlama32Detector.test_multiple_json_objects` | `PARSER.batch.2.a` | [source](https://github.com/sgl-project/sglang/blob/612785ffdcaf35552f1ed433a981d596ca9fe900/test/registered/unit/function_call/test_llama32_detector.py#L96) | Multiple json objects. |
| `TestLlama32Detector.test_tool_call_with_multiple_arguments` | `PARSER.batch.2.a`, `PARSER.batch.7.a` | [source](https://github.com/sgl-project/sglang/blob/612785ffdcaf35552f1ed433a981d596ca9fe900/test/registered/unit/function_call/test_llama32_detector.py#L103) | Tool call with multiple arguments. |
| `TestLlama32Detector.test_convert_python_dict_to_json` | `PARSER.batch.7.d` | [source](https://github.com/sgl-project/sglang/blob/612785ffdcaf35552f1ed433a981d596ca9fe900/test/registered/unit/function_call/test_llama32_detector.py#L113) | Convert python dict to json. |
| `TestLlama32Detector.test_convert_invalid_string_returns_original` | `PARSER.batch.4.b` | [source](https://github.com/sgl-project/sglang/blob/612785ffdcaf35552f1ed433a981d596ca9fe900/test/registered/unit/function_call/test_llama32_detector.py#L119) | Convert invalid string returns original. |
| `TestLlama32Detector.test_structure_info` | `// helper` | [source](https://github.com/sgl-project/sglang/blob/612785ffdcaf35552f1ed433a981d596ca9fe900/test/registered/unit/function_call/test_llama32_detector.py#L126) | Registry, construction, serialization, or helper-only coverage. |
| `TestLlama32Detector.test_streaming_single_tool_call` | `PARSER.batch.1`, `PARSER.stream.1` | [source](https://github.com/sgl-project/sglang/blob/612785ffdcaf35552f1ed433a981d596ca9fe900/test/registered/unit/function_call/test_llama32_detector.py#L135) | Streaming single tool call. |
| `TestLlama32Detector.test_streaming_normal_text_before_tool` | `PARSER.batch.8.a`, `PARSER.stream.1` | [source](https://github.com/sgl-project/sglang/blob/612785ffdcaf35552f1ed433a981d596ca9fe900/test/registered/unit/function_call/test_llama32_detector.py#L156) | Streaming normal text before tool. |
| `TestLlama32Detector.test_streaming_text_then_tool_call` | `PARSER.batch.8.a`, `PARSER.stream.1` | [source](https://github.com/sgl-project/sglang/blob/612785ffdcaf35552f1ed433a981d596ca9fe900/test/registered/unit/function_call/test_llama32_detector.py#L164) | Streaming text then tool call. |
| `TestLlama32Detector.test_single_json` | `PARSER.batch.1` | [source](https://github.com/sgl-project/sglang/blob/612785ffdcaf35552f1ed433a981d596ca9fe900/test/registered/unit/function_call/test_function_call_parser.py#L906) | Single json. |
| `TestLlama32Detector.test_multiple_json_with_separator` | `PARSER.batch.2.a`, `PARSER.batch.12` | [source](https://github.com/sgl-project/sglang/blob/612785ffdcaf35552f1ed433a981d596ca9fe900/test/registered/unit/function_call/test_function_call_parser.py#L913) | Multiple json with separator. |
| `TestLlama32Detector.test_multiple_json_with_separator_customized` | `PARSER.batch.2.a`, `PARSER.batch.12` | [source](https://github.com/sgl-project/sglang/blob/612785ffdcaf35552f1ed433a981d596ca9fe900/test/registered/unit/function_call/test_function_call_parser.py#L923) | Multiple json with separator customized. |
| `TestLlama32Detector.test_json_with_trailing_text` | `PARSER.batch.8.b` | [source](https://github.com/sgl-project/sglang/blob/612785ffdcaf35552f1ed433a981d596ca9fe900/test/registered/unit/function_call/test_function_call_parser.py#L933) | Json with trailing text. |
| `TestLlama32Detector.test_invalid_then_valid_json` | `PARSER.batch.4.b`, `PARSER.batch.4.e` | [source](https://github.com/sgl-project/sglang/blob/612785ffdcaf35552f1ed433a981d596ca9fe900/test/registered/unit/function_call/test_function_call_parser.py#L939) | Invalid then valid json. |
| `TestLlama32Detector.test_plain_text_only` | `PARSER.batch.3` | [source](https://github.com/sgl-project/sglang/blob/612785ffdcaf35552f1ed433a981d596ca9fe900/test/registered/unit/function_call/test_function_call_parser.py#L948) | Plain text only. |
| `TestLlama32Detector.test_with_python_tag_prefix` | `PARSER.fmt.3` | [source](https://github.com/sgl-project/sglang/blob/612785ffdcaf35552f1ed433a981d596ca9fe900/test/registered/unit/function_call/test_function_call_parser.py#L954) | With python tag prefix. |

### SGLang Python - TestMistralDetector

| Test | Bucket(s) | Link | Notes |
| -- | -- | -- | -- |
| `TestMistralDetector.test_has_tool_call_json_array_format` | `PARSER.fmt.3` | [source](https://github.com/sgl-project/sglang/blob/612785ffdcaf35552f1ed433a981d596ca9fe900/test/registered/unit/function_call/test_mistral_detector.py#L56) | Format detection / marker-recognition helper coverage. |
| `TestMistralDetector.test_has_tool_call_compact_format` | `PARSER.fmt.3` | [source](https://github.com/sgl-project/sglang/blob/612785ffdcaf35552f1ed433a981d596ca9fe900/test/registered/unit/function_call/test_mistral_detector.py#L62) | Format detection / marker-recognition helper coverage. |
| `TestMistralDetector.test_has_tool_call_false` | `PARSER.fmt.3` | [source](https://github.com/sgl-project/sglang/blob/612785ffdcaf35552f1ed433a981d596ca9fe900/test/registered/unit/function_call/test_mistral_detector.py#L66) | Format detection / marker-recognition helper coverage. |
| `TestMistralDetector.test_json_array_single_tool_call` | `PARSER.batch.1`, `PARSER.batch.7.d` | [source](https://github.com/sgl-project/sglang/blob/612785ffdcaf35552f1ed433a981d596ca9fe900/test/registered/unit/function_call/test_mistral_detector.py#L72) | Json array single tool call. |
| `TestMistralDetector.test_json_array_multiple_tool_calls` | `PARSER.batch.2.a`, `PARSER.batch.7.d` | [source](https://github.com/sgl-project/sglang/blob/612785ffdcaf35552f1ed433a981d596ca9fe900/test/registered/unit/function_call/test_mistral_detector.py#L83) | Json array multiple tool calls. |
| `TestMistralDetector.test_json_array_with_leading_text` | `PARSER.batch.7.d`, `PARSER.batch.8.a` | [source](https://github.com/sgl-project/sglang/blob/612785ffdcaf35552f1ed433a981d596ca9fe900/test/registered/unit/function_call/test_mistral_detector.py#L90) | Json array with leading text. |
| `TestMistralDetector.test_compact_format_single_tool_call` | `PARSER.batch.1`, `PARSER.fmt.3` | [source](https://github.com/sgl-project/sglang/blob/612785ffdcaf35552f1ed433a981d596ca9fe900/test/registered/unit/function_call/test_mistral_detector.py#L98) | Compact format single tool call. |
| `TestMistralDetector.test_compact_format_with_leading_text` | `PARSER.batch.8.a`, `PARSER.fmt.3` | [source](https://github.com/sgl-project/sglang/blob/612785ffdcaf35552f1ed433a981d596ca9fe900/test/registered/unit/function_call/test_mistral_detector.py#L106) | Compact format with leading text. |
| `TestMistralDetector.test_no_tool_call` | `PARSER.batch.3` | [source](https://github.com/sgl-project/sglang/blob/612785ffdcaf35552f1ed433a981d596ca9fe900/test/registered/unit/function_call/test_mistral_detector.py#L114) | No tool call. |
| `TestMistralDetector.test_tool_call_with_nested_json` | `PARSER.batch.7.d` | [source](https://github.com/sgl-project/sglang/blob/612785ffdcaf35552f1ed433a981d596ca9fe900/test/registered/unit/function_call/test_mistral_detector.py#L122) | Tool call with nested json. |
| `TestMistralDetector.test_json_array_with_invalid_json` | `PARSER.batch.4.b`, `PARSER.batch.7.d` | [source](https://github.com/sgl-project/sglang/blob/612785ffdcaf35552f1ed433a981d596ca9fe900/test/registered/unit/function_call/test_mistral_detector.py#L129) | Json array with invalid json. |
| `TestMistralDetector.test_extract_json_array` | `PARSER.batch.7.d` | [source](https://github.com/sgl-project/sglang/blob/612785ffdcaf35552f1ed433a981d596ca9fe900/test/registered/unit/function_call/test_mistral_detector.py#L137) | Extract json array. |
| `TestMistralDetector.test_extract_json_array_nested_brackets` | `PARSER.batch.7.d` | [source](https://github.com/sgl-project/sglang/blob/612785ffdcaf35552f1ed433a981d596ca9fe900/test/registered/unit/function_call/test_mistral_detector.py#L147) | Extract json array nested brackets. |
| `TestMistralDetector.test_extract_json_array_no_marker` | `PARSER.batch.7.d`, `PARSER.fmt.3` | [source](https://github.com/sgl-project/sglang/blob/612785ffdcaf35552f1ed433a981d596ca9fe900/test/registered/unit/function_call/test_mistral_detector.py#L156) | Extract json array no marker. |
| `TestMistralDetector.test_structure_info` | `// helper` | [source](https://github.com/sgl-project/sglang/blob/612785ffdcaf35552f1ed433a981d596ca9fe900/test/registered/unit/function_call/test_mistral_detector.py#L163) | Registry, construction, serialization, or helper-only coverage. |
| `TestMistralDetector.test_streaming_compact_format` | `PARSER.fmt.3`, `PARSER.stream.1` | [source](https://github.com/sgl-project/sglang/blob/612785ffdcaf35552f1ed433a981d596ca9fe900/test/registered/unit/function_call/test_mistral_detector.py#L172) | Streaming compact format. |
| `TestMistralDetector.test_streaming_normal_text_before_tool` | `PARSER.batch.8.a`, `PARSER.stream.1` | [source](https://github.com/sgl-project/sglang/blob/612785ffdcaf35552f1ed433a981d596ca9fe900/test/registered/unit/function_call/test_mistral_detector.py#L192) | Streaming normal text before tool. |
| `TestMistralDetector.test_streaming_text_then_tool_call` | `PARSER.batch.8.a`, `PARSER.stream.1` | [source](https://github.com/sgl-project/sglang/blob/612785ffdcaf35552f1ed433a981d596ca9fe900/test/registered/unit/function_call/test_mistral_detector.py#L198) | Streaming text then tool call. |
| `TestMistralDetector.test_detect_and_parse_with_nested_brackets_in_content` | `PARSER.batch.1`, `PARSER.batch.7.d` | [source](https://github.com/sgl-project/sglang/blob/612785ffdcaf35552f1ed433a981d596ca9fe900/test/registered/unit/function_call/test_function_call_parser.py#L385) | Detect and parse with nested brackets in content. |
| `TestMistralDetector.test_detect_and_parse_simple_case` | `PARSER.batch.1` | [source](https://github.com/sgl-project/sglang/blob/612785ffdcaf35552f1ed433a981d596ca9fe900/test/registered/unit/function_call/test_function_call_parser.py#L425) | Detect and parse simple case. |
| `TestMistralDetector.test_detect_and_parse_no_tool_calls` | `PARSER.batch.3`, `PARSER.batch.1` | [source](https://github.com/sgl-project/sglang/blob/612785ffdcaf35552f1ed433a981d596ca9fe900/test/registered/unit/function_call/test_function_call_parser.py#L439) | Detect and parse no tool calls. |
| `TestMistralDetector.test_detect_and_parse_with_text_before_tool_call` | `PARSER.batch.1`, `PARSER.batch.8.a` | [source](https://github.com/sgl-project/sglang/blob/612785ffdcaf35552f1ed433a981d596ca9fe900/test/registered/unit/function_call/test_function_call_parser.py#L452) | Detect and parse with text before tool call. |
| `TestMistralDetector.test_detect_and_parse_compact_args_format` | `PARSER.batch.1`, `PARSER.fmt.3` | [source](https://github.com/sgl-project/sglang/blob/612785ffdcaf35552f1ed433a981d596ca9fe900/test/registered/unit/function_call/test_function_call_parser.py#L468) | Detect and parse compact args format. |
| `TestMistralDetector.test_streaming_compact_args_format_emits_tool_calls` | `PARSER.fmt.3`, `PARSER.stream.1` | [source](https://github.com/sgl-project/sglang/blob/612785ffdcaf35552f1ed433a981d596ca9fe900/test/registered/unit/function_call/test_function_call_parser.py#L479) | Streaming compact args format emits tool calls. |

### SGLang Python - TestParallelToolCalls

| Test | Bucket(s) | Link | Notes |
| -- | -- | -- | -- |
| `TestParallelToolCalls.test_parallel_tool_calls_with_array_parameters` | `PARSER.batch.2.a`, `PARSER.batch.7.d` | [source](https://github.com/sgl-project/sglang/blob/612785ffdcaf35552f1ed433a981d596ca9fe900/test/registered/unit/function_call/test_parallel_tool_calls.py#L70) | Parallel tool calls with array parameters. |
| `TestParallelToolCalls.test_simple_parallel_tool_calls` | `PARSER.batch.2.a` | [source](https://github.com/sgl-project/sglang/blob/612785ffdcaf35552f1ed433a981d596ca9fe900/test/registered/unit/function_call/test_parallel_tool_calls.py#L136) | Simple parallel tool calls. |

### SGLang Python - test_unknown_tool_name

| Test | Bucket(s) | Link | Notes |
| -- | -- | -- | -- |
| `test_unknown_tool_name_dropped_default` | `PARSER.batch.13` | [source](https://github.com/sgl-project/sglang/blob/612785ffdcaf35552f1ed433a981d596ca9fe900/test/registered/unit/function_call/test_unknown_tool_name.py#L29) | Unknown tool name dropped default. |
| `test_unknown_tool_name_forwarded` | `PARSER.batch.13` | [source](https://github.com/sgl-project/sglang/blob/612785ffdcaf35552f1ed433a981d596ca9fe900/test/registered/unit/function_call/test_unknown_tool_name.py#L53) | Unknown tool name forwarded. |

### SGLang Python - TestHunyuanDetectorHasToolCall

| Test | Bucket(s) | Link | Notes |
| -- | -- | -- | -- |
| `TestHunyuanDetectorHasToolCall.test_has_tool_call_true` | `PARSER.fmt.3` | [source](https://github.com/sgl-project/sglang/blob/612785ffdcaf35552f1ed433a981d596ca9fe900/test/registered/unit/function_call/test_hunyuan_detector.py#L82) | Format detection / marker-recognition helper coverage. |
| `TestHunyuanDetectorHasToolCall.test_has_tool_call_false` | `PARSER.fmt.3` | [source](https://github.com/sgl-project/sglang/blob/612785ffdcaf35552f1ed433a981d596ca9fe900/test/registered/unit/function_call/test_hunyuan_detector.py#L88) | Format detection / marker-recognition helper coverage. |
| `TestHunyuanDetectorHasToolCall.test_has_tool_call_partial_tag` | `PARSER.fmt.3` | [source](https://github.com/sgl-project/sglang/blob/612785ffdcaf35552f1ed433a981d596ca9fe900/test/registered/unit/function_call/test_hunyuan_detector.py#L93) | Format detection / marker-recognition helper coverage. |
| `TestHunyuanDetectorHasToolCall.test_has_tool_call_with_surrounding_text` | `PARSER.fmt.3` | [source](https://github.com/sgl-project/sglang/blob/612785ffdcaf35552f1ed433a981d596ca9fe900/test/registered/unit/function_call/test_hunyuan_detector.py#L97) | Format detection / marker-recognition helper coverage. |

### SGLang Python - TestHunyuanDetectorDetectAndParse

| Test | Bucket(s) | Link | Notes |
| -- | -- | -- | -- |
| `TestHunyuanDetectorDetectAndParse.test_no_tool_call` | `PARSER.batch.3` | [source](https://github.com/sgl-project/sglang/blob/612785ffdcaf35552f1ed433a981d596ca9fe900/test/registered/unit/function_call/test_hunyuan_detector.py#L108) | No tool call. |
| `TestHunyuanDetectorDetectAndParse.test_zero_arg_inline` | `PARSER.batch.6.a` | [source](https://github.com/sgl-project/sglang/blob/612785ffdcaf35552f1ed433a981d596ca9fe900/test/registered/unit/function_call/test_hunyuan_detector.py#L114) | Zero arg inline. |
| `TestHunyuanDetectorDetectAndParse.test_zero_arg_newline` | `PARSER.batch.6.b` | [source](https://github.com/sgl-project/sglang/blob/612785ffdcaf35552f1ed433a981d596ca9fe900/test/registered/unit/function_call/test_hunyuan_detector.py#L123) | Zero arg newline. |
| `TestHunyuanDetectorDetectAndParse.test_single_string_arg` | `// helper` | [source](https://github.com/sgl-project/sglang/blob/612785ffdcaf35552f1ed433a981d596ca9fe900/test/registered/unit/function_call/test_hunyuan_detector.py#L134) | Unclassified helper or parser-adjacent smoke test. |
| `TestHunyuanDetectorDetectAndParse.test_multiple_args_same_line` | `PARSER.batch.2.a` | [source](https://github.com/sgl-project/sglang/blob/612785ffdcaf35552f1ed433a981d596ca9fe900/test/registered/unit/function_call/test_hunyuan_detector.py#L145) | Multiple args same line. |
| `TestHunyuanDetectorDetectAndParse.test_args_with_newlines` | `PARSER.fmt.2` | [source](https://github.com/sgl-project/sglang/blob/612785ffdcaf35552f1ed433a981d596ca9fe900/test/registered/unit/function_call/test_hunyuan_detector.py#L157) | Args with newlines. |
| `TestHunyuanDetectorDetectAndParse.test_content_before_tool_call` | `PARSER.batch.8.a` | [source](https://github.com/sgl-project/sglang/blob/612785ffdcaf35552f1ed433a981d596ca9fe900/test/registered/unit/function_call/test_hunyuan_detector.py#L173) | Content before tool call. |
| `TestHunyuanDetectorDetectAndParse.test_multiple_tool_calls` | `PARSER.batch.2.a` | [source](https://github.com/sgl-project/sglang/blob/612785ffdcaf35552f1ed433a981d596ca9fe900/test/registered/unit/function_call/test_hunyuan_detector.py#L185) | Multiple tool calls. |
| `TestHunyuanDetectorDetectAndParse.test_empty_content_returns_empty_normal_text` | `PARSER.batch.9` | [source](https://github.com/sgl-project/sglang/blob/612785ffdcaf35552f1ed433a981d596ca9fe900/test/registered/unit/function_call/test_hunyuan_detector.py#L201) | Empty content returns empty normal text. |
| `TestHunyuanDetectorDetectAndParse.test_unknown_tool_skipped` | `PARSER.batch.13` | [source](https://github.com/sgl-project/sglang/blob/612785ffdcaf35552f1ed433a981d596ca9fe900/test/registered/unit/function_call/test_hunyuan_detector.py#L206) | Unknown tool skipped. |
| `TestHunyuanDetectorDetectAndParse.test_mixed_known_and_unknown_tools` | `PARSER.batch.13` | [source](https://github.com/sgl-project/sglang/blob/612785ffdcaf35552f1ed433a981d596ca9fe900/test/registered/unit/function_call/test_hunyuan_detector.py#L215) | Mixed known and unknown tools. |
| `TestHunyuanDetectorDetectAndParse.test_three_parallel_tool_calls` | `PARSER.batch.2.a` | [source](https://github.com/sgl-project/sglang/blob/612785ffdcaf35552f1ed433a981d596ca9fe900/test/registered/unit/function_call/test_hunyuan_detector.py#L231) | Three parallel tool calls. |

### SGLang Python - TestHunyuanDetectorArgDeserialization

| Test | Bucket(s) | Link | Notes |
| -- | -- | -- | -- |
| `TestHunyuanDetectorArgDeserialization.test_integer_arg` | `// helper` | [source](https://github.com/sgl-project/sglang/blob/612785ffdcaf35552f1ed433a981d596ca9fe900/test/registered/unit/function_call/test_hunyuan_detector.py#L260) | Registry, construction, serialization, or helper-only coverage. |
| `TestHunyuanDetectorArgDeserialization.test_float_arg` | `// helper` | [source](https://github.com/sgl-project/sglang/blob/612785ffdcaf35552f1ed433a981d596ca9fe900/test/registered/unit/function_call/test_hunyuan_detector.py#L273) | Registry, construction, serialization, or helper-only coverage. |
| `TestHunyuanDetectorArgDeserialization.test_boolean_arg` | `// helper` | [source](https://github.com/sgl-project/sglang/blob/612785ffdcaf35552f1ed433a981d596ca9fe900/test/registered/unit/function_call/test_hunyuan_detector.py#L285) | Registry, construction, serialization, or helper-only coverage. |
| `TestHunyuanDetectorArgDeserialization.test_string_arg_not_deserialized` | `// helper` | [source](https://github.com/sgl-project/sglang/blob/612785ffdcaf35552f1ed433a981d596ca9fe900/test/registered/unit/function_call/test_hunyuan_detector.py#L296) | Registry, construction, serialization, or helper-only coverage. |
| `TestHunyuanDetectorArgDeserialization.test_non_json_value_stays_string` | `// helper` | [source](https://github.com/sgl-project/sglang/blob/612785ffdcaf35552f1ed433a981d596ca9fe900/test/registered/unit/function_call/test_hunyuan_detector.py#L308) | Registry, construction, serialization, or helper-only coverage. |

### SGLang Python - TestHunyuanDetectorStreaming

| Test | Bucket(s) | Link | Notes |
| -- | -- | -- | -- |
| `TestHunyuanDetectorStreaming.test_normal_text_only` | `PARSER.batch.3`, `PARSER.stream.1` | [source](https://github.com/sgl-project/sglang/blob/612785ffdcaf35552f1ed433a981d596ca9fe900/test/registered/unit/function_call/test_hunyuan_detector.py#L343) | Normal text only. |
| `TestHunyuanDetectorStreaming.test_complete_tool_call_single_chunk` | `PARSER.stream.1`, `PARSER.stream.3` | [source](https://github.com/sgl-project/sglang/blob/612785ffdcaf35552f1ed433a981d596ca9fe900/test/registered/unit/function_call/test_hunyuan_detector.py#L351) | Complete tool call single chunk. |
| `TestHunyuanDetectorStreaming.test_chunked_tool_call` | `PARSER.stream.1`, `PARSER.stream.3` | [source](https://github.com/sgl-project/sglang/blob/612785ffdcaf35552f1ed433a981d596ca9fe900/test/registered/unit/function_call/test_hunyuan_detector.py#L364) | Chunked tool call. |
| `TestHunyuanDetectorStreaming.test_normal_text_before_tool` | `PARSER.batch.8.a`, `PARSER.stream.1` | [source](https://github.com/sgl-project/sglang/blob/612785ffdcaf35552f1ed433a981d596ca9fe900/test/registered/unit/function_call/test_hunyuan_detector.py#L385) | Normal text before tool. |
| `TestHunyuanDetectorStreaming.test_multiple_tool_calls_chunked` | `PARSER.batch.2.b`, `PARSER.stream.2`, `PARSER.stream.3` | [source](https://github.com/sgl-project/sglang/blob/612785ffdcaf35552f1ed433a981d596ca9fe900/test/registered/unit/function_call/test_hunyuan_detector.py#L397) | Multiple tool calls chunked. |
| `TestHunyuanDetectorStreaming.test_partial_bot_token_buffered` | `PARSER.fmt.3`, `PARSER.stream.1`, `PARSER.stream.3` | [source](https://github.com/sgl-project/sglang/blob/612785ffdcaf35552f1ed433a981d596ca9fe900/test/registered/unit/function_call/test_hunyuan_detector.py#L419) | Partial bot token buffered. |
| `TestHunyuanDetectorStreaming.test_char_by_char_streaming` | `PARSER.stream.1`, `PARSER.stream.3` | [source](https://github.com/sgl-project/sglang/blob/612785ffdcaf35552f1ed433a981d596ca9fe900/test/registered/unit/function_call/test_hunyuan_detector.py#L427) | Char by char streaming. |
| `TestHunyuanDetectorStreaming.test_streaming_with_args_char_by_char` | `PARSER.stream.1`, `PARSER.stream.3` | [source](https://github.com/sgl-project/sglang/blob/612785ffdcaf35552f1ed433a981d596ca9fe900/test/registered/unit/function_call/test_hunyuan_detector.py#L443) | Streaming with args char by char. |
| `TestHunyuanDetectorStreaming.test_streaming_three_tools_sequential` | `PARSER.batch.2.b`, `PARSER.stream.2` | [source](https://github.com/sgl-project/sglang/blob/612785ffdcaf35552f1ed433a981d596ca9fe900/test/registered/unit/function_call/test_hunyuan_detector.py#L460) | Streaming three tools sequential. |
| `TestHunyuanDetectorStreaming.test_streaming_normal_text_not_lost` | `PARSER.stream.1` | [source](https://github.com/sgl-project/sglang/blob/612785ffdcaf35552f1ed433a981d596ca9fe900/test/registered/unit/function_call/test_hunyuan_detector.py#L483) | Streaming normal text not lost. |
| `TestHunyuanDetectorStreaming.test_streaming_name_comes_before_args` | `PARSER.stream.1` | [source](https://github.com/sgl-project/sglang/blob/612785ffdcaf35552f1ed433a981d596ca9fe900/test/registered/unit/function_call/test_hunyuan_detector.py#L498) | Streaming name comes before args. |
| `TestHunyuanDetectorStreaming.test_streaming_typed_args_coerced` | `PARSER.batch.7.c`, `PARSER.xml.2`, `PARSER.stream.1` | [source](https://github.com/sgl-project/sglang/blob/612785ffdcaf35552f1ed433a981d596ca9fe900/test/registered/unit/function_call/test_hunyuan_detector.py#L516) | Streaming typed args coerced. |
| `TestHunyuanDetectorStreaming.test_streaming_string_arg_holds_back_partial_end_tag` | `PARSER.stream.1`, `PARSER.stream.3` | [source](https://github.com/sgl-project/sglang/blob/612785ffdcaf35552f1ed433a981d596ca9fe900/test/registered/unit/function_call/test_hunyuan_detector.py#L537) | Streaming string arg holds back partial end tag. |
| `TestHunyuanDetectorStreaming.test_streaming_all_in_one_delta` | `PARSER.stream.1` | [source](https://github.com/sgl-project/sglang/blob/612785ffdcaf35552f1ed433a981d596ca9fe900/test/registered/unit/function_call/test_hunyuan_detector.py#L553) | Streaming all in one delta. |
| `TestHunyuanDetectorStreaming.test_streaming_content_before` | `PARSER.batch.8.a`, `PARSER.stream.1` | [source](https://github.com/sgl-project/sglang/blob/612785ffdcaf35552f1ed433a981d596ca9fe900/test/registered/unit/function_call/test_hunyuan_detector.py#L566) | Streaming content before. |

### SGLang Python - TestHunyuanDetectorStructureInfo

| Test | Bucket(s) | Link | Notes |
| -- | -- | -- | -- |
| `TestHunyuanDetectorStructureInfo.test_structure_info_content` | `// helper` | [source](https://github.com/sgl-project/sglang/blob/612785ffdcaf35552f1ed433a981d596ca9fe900/test/registered/unit/function_call/test_hunyuan_detector.py#L594) | White-box helper/state test; no direct cross-impl fixture analogue. |
| `TestHunyuanDetectorStructureInfo.test_supports_structural_tag` | `// helper` | [source](https://github.com/sgl-project/sglang/blob/612785ffdcaf35552f1ed433a981d596ca9fe900/test/registered/unit/function_call/test_hunyuan_detector.py#L603) | White-box helper/state test; no direct cross-impl fixture analogue. |

### SGLang Python - TestHunyuanDetectorAccuracy

| Test | Bucket(s) | Link | Notes |
| -- | -- | -- | -- |
| `TestHunyuanDetectorAccuracy.test_reference_zero_arg_inline` | `PARSER.batch.6.a` | [source](https://github.com/sgl-project/sglang/blob/612785ffdcaf35552f1ed433a981d596ca9fe900/test/registered/unit/function_call/test_hunyuan_detector.py#L614) | Reference zero arg inline. |
| `TestHunyuanDetectorAccuracy.test_reference_zero_arg_newline` | `PARSER.batch.6.b` | [source](https://github.com/sgl-project/sglang/blob/612785ffdcaf35552f1ed433a981d596ca9fe900/test/registered/unit/function_call/test_hunyuan_detector.py#L624) | Reference zero arg newline. |
| `TestHunyuanDetectorAccuracy.test_reference_args_same_line` | `// helper` | [source](https://github.com/sgl-project/sglang/blob/612785ffdcaf35552f1ed433a981d596ca9fe900/test/registered/unit/function_call/test_hunyuan_detector.py#L630) | Unclassified helper or parser-adjacent smoke test. |
| `TestHunyuanDetectorAccuracy.test_reference_args_with_newlines` | `PARSER.fmt.2` | [source](https://github.com/sgl-project/sglang/blob/612785ffdcaf35552f1ed433a981d596ca9fe900/test/registered/unit/function_call/test_hunyuan_detector.py#L640) | Reference args with newlines. |
| `TestHunyuanDetectorAccuracy.test_reference_content_before` | `PARSER.batch.8.a` | [source](https://github.com/sgl-project/sglang/blob/612785ffdcaf35552f1ed433a981d596ca9fe900/test/registered/unit/function_call/test_hunyuan_detector.py#L650) | Reference content before. |
| `TestHunyuanDetectorAccuracy.test_reference_multiple` | `PARSER.batch.2.a` | [source](https://github.com/sgl-project/sglang/blob/612785ffdcaf35552f1ed433a981d596ca9fe900/test/registered/unit/function_call/test_hunyuan_detector.py#L656) | Reference multiple. |
| `TestHunyuanDetectorAccuracy.test_reference_empty_content_none` | `PARSER.batch.9` | [source](https://github.com/sgl-project/sglang/blob/612785ffdcaf35552f1ed433a981d596ca9fe900/test/registered/unit/function_call/test_hunyuan_detector.py#L666) | Reference empty content none. |
| `TestHunyuanDetectorAccuracy.test_reference_no_tool_call` | `PARSER.batch.3` | [source](https://github.com/sgl-project/sglang/blob/612785ffdcaf35552f1ed433a981d596ca9fe900/test/registered/unit/function_call/test_hunyuan_detector.py#L671) | Reference no tool call. |

### SGLang Python - TestHunyuanDetectorFunctionCallParser

| Test | Bucket(s) | Link | Notes |
| -- | -- | -- | -- |
| `TestHunyuanDetectorFunctionCallParser.test_parser_registry` | `// helper` | [source](https://github.com/sgl-project/sglang/blob/612785ffdcaf35552f1ed433a981d596ca9fe900/test/registered/unit/function_call/test_hunyuan_detector.py#L684) | Registry, construction, serialization, or helper-only coverage. |
| `TestHunyuanDetectorFunctionCallParser.test_parse_non_stream` | `PARSER.batch.1` | [source](https://github.com/sgl-project/sglang/blob/612785ffdcaf35552f1ed433a981d596ca9fe900/test/registered/unit/function_call/test_hunyuan_detector.py#L690) | Parse non stream. |
| `TestHunyuanDetectorFunctionCallParser.test_parse_stream_chunks` | `PARSER.stream.1`, `PARSER.stream.3` | [source](https://github.com/sgl-project/sglang/blob/612785ffdcaf35552f1ed433a981d596ca9fe900/test/registered/unit/function_call/test_hunyuan_detector.py#L705) | Parse stream chunks. |
| `TestHunyuanDetectorFunctionCallParser.test_has_tool_call_through_parser` | `PARSER.fmt.3` | [source](https://github.com/sgl-project/sglang/blob/612785ffdcaf35552f1ed433a981d596ca9fe900/test/registered/unit/function_call/test_hunyuan_detector.py#L724) | Format detection / marker-recognition helper coverage. |

### SGLang Python - TestKimiK2DetectorBasic

| Test | Bucket(s) | Link | Notes |
| -- | -- | -- | -- |
| `TestKimiK2DetectorBasic.test_single_tool_call` | `PARSER.batch.1` | [source](https://github.com/sgl-project/sglang/blob/612785ffdcaf35552f1ed433a981d596ca9fe900/test/registered/function_call/test_kimik2_detector.py#L80) | Single tool call. |
| `TestKimiK2DetectorBasic.test_multiple_tool_calls` | `PARSER.batch.2.a` | [source](https://github.com/sgl-project/sglang/blob/612785ffdcaf35552f1ed433a981d596ca9fe900/test/registered/function_call/test_kimik2_detector.py#L95) | Multiple tool calls. |
| `TestKimiK2DetectorBasic.test_normal_text_before_tool_call` | `PARSER.batch.8.a` | [source](https://github.com/sgl-project/sglang/blob/612785ffdcaf35552f1ed433a981d596ca9fe900/test/registered/function_call/test_kimik2_detector.py#L113) | Normal text before tool call. |
| `TestKimiK2DetectorBasic.test_no_tool_call` | `PARSER.batch.3` | [source](https://github.com/sgl-project/sglang/blob/612785ffdcaf35552f1ed433a981d596ca9fe900/test/registered/function_call/test_kimik2_detector.py#L127) | No tool call. |
| `TestKimiK2DetectorBasic.test_has_tool_call` | `PARSER.fmt.3` | [source](https://github.com/sgl-project/sglang/blob/612785ffdcaf35552f1ed433a981d596ca9fe900/test/registered/function_call/test_kimik2_detector.py#L134) | Format detection / marker-recognition helper coverage. |

### SGLang Python - TestKimiK2DetectorHyphenatedNames

| Test | Bucket(s) | Link | Notes |
| -- | -- | -- | -- |
| `TestKimiK2DetectorHyphenatedNames.test_hyphenated_name_non_streaming` | `PARSER.fmt.1`, `PARSER.stream.1` | [source](https://github.com/sgl-project/sglang/blob/612785ffdcaf35552f1ed433a981d596ca9fe900/test/registered/function_call/test_kimik2_detector.py#L152) | Hyphenated name non streaming. |
| `TestKimiK2DetectorHyphenatedNames.test_hyphenated_name_streaming` | `PARSER.fmt.1`, `PARSER.stream.1` | [source](https://github.com/sgl-project/sglang/blob/612785ffdcaf35552f1ed433a981d596ca9fe900/test/registered/function_call/test_kimik2_detector.py#L165) | Hyphenated name streaming. |

### SGLang Python - TestKimiK2DetectorStreaming

| Test | Bucket(s) | Link | Notes |
| -- | -- | -- | -- |
| `TestKimiK2DetectorStreaming.test_streaming_single_tool_call` | `PARSER.batch.1`, `PARSER.stream.1` | [source](https://github.com/sgl-project/sglang/blob/612785ffdcaf35552f1ed433a981d596ca9fe900/test/registered/function_call/test_kimik2_detector.py#L198) | Streaming single tool call. |
| `TestKimiK2DetectorStreaming.test_streaming_multiple_tool_calls` | `PARSER.batch.2.b`, `PARSER.stream.2` | [source](https://github.com/sgl-project/sglang/blob/612785ffdcaf35552f1ed433a981d596ca9fe900/test/registered/function_call/test_kimik2_detector.py#L214) | Streaming multiple tool calls. |
| `TestKimiK2DetectorStreaming.test_streaming_state_reset_after_completion` | `PARSER.stream.1`, `PARSER.stream.4` | [source](https://github.com/sgl-project/sglang/blob/612785ffdcaf35552f1ed433a981d596ca9fe900/test/registered/function_call/test_kimik2_detector.py#L233) | Streaming state reset after completion. |

### SGLang Python - TestKimiK2DetectorSpecialTokenLeakage

| Test | Bucket(s) | Link | Notes |
| -- | -- | -- | -- |
| `TestKimiK2DetectorSpecialTokenLeakage.test_no_leak_in_non_tool_text` | `PARSER.fmt.3` | [source](https://github.com/sgl-project/sglang/blob/612785ffdcaf35552f1ed433a981d596ca9fe900/test/registered/function_call/test_kimik2_detector.py#L256) | No leak in non tool text. |
| `TestKimiK2DetectorSpecialTokenLeakage.test_no_leak_of_argument_begin_token` | `PARSER.fmt.3` | [source](https://github.com/sgl-project/sglang/blob/612785ffdcaf35552f1ed433a981d596ca9fe900/test/registered/function_call/test_kimik2_detector.py#L265) | No leak of argument begin token. |
| `TestKimiK2DetectorSpecialTokenLeakage.test_no_leak_on_error_fallback` | `PARSER.fmt.3` | [source](https://github.com/sgl-project/sglang/blob/612785ffdcaf35552f1ed433a981d596ca9fe900/test/registered/function_call/test_kimik2_detector.py#L273) | No leak on error fallback. |
| `TestKimiK2DetectorSpecialTokenLeakage.test_strip_special_tokens_all_tokens` | `PARSER.fmt.3` | [source](https://github.com/sgl-project/sglang/blob/612785ffdcaf35552f1ed433a981d596ca9fe900/test/registered/function_call/test_kimik2_detector.py#L280) | Strip special tokens all tokens. |
| `TestKimiK2DetectorSpecialTokenLeakage.test_strip_preserves_normal_text` | `PARSER.fmt.3` | [source](https://github.com/sgl-project/sglang/blob/612785ffdcaf35552f1ed433a981d596ca9fe900/test/registered/function_call/test_kimik2_detector.py#L291) | Strip preserves normal text. |

### SGLang Python - TestKimiK2ReasoningDetectorNonStreaming

| Test | Bucket(s) | Link | Notes |
| -- | -- | -- | -- |
| `TestKimiK2ReasoningDetectorNonStreaming.test_normal_reasoning_with_think_end` | `REASONING.batch.1` | [source](https://github.com/sgl-project/sglang/blob/612785ffdcaf35552f1ed433a981d596ca9fe900/test/registered/function_call/test_kimik2_detector.py#L305) | Reasoning-to-tool transition coverage; parser tags included when a tool call is extracted. |
| `TestKimiK2ReasoningDetectorNonStreaming.test_tool_call_inside_think_without_close_tag` | `REASONING.batch.2`, `PARSER.batch.1`, `PARSER.stream.1`, `PARSER.stream.3` | [source](https://github.com/sgl-project/sglang/blob/612785ffdcaf35552f1ed433a981d596ca9fe900/test/registered/function_call/test_kimik2_detector.py#L320) | Reasoning-to-tool transition coverage; parser tags included when a tool call is extracted. |
| `TestKimiK2ReasoningDetectorNonStreaming.test_no_reasoning_just_tool_call` | `REASONING.batch.2`, `PARSER.batch.1`, `PARSER.stream.1` | [source](https://github.com/sgl-project/sglang/blob/612785ffdcaf35552f1ed433a981d596ca9fe900/test/registered/function_call/test_kimik2_detector.py#L349) | Reasoning-to-tool transition coverage; parser tags included when a tool call is extracted. |
| `TestKimiK2ReasoningDetectorNonStreaming.test_normal_text_without_reasoning` | `REASONING.batch.1` | [source](https://github.com/sgl-project/sglang/blob/612785ffdcaf35552f1ed433a981d596ca9fe900/test/registered/function_call/test_kimik2_detector.py#L363) | Reasoning-to-tool transition coverage; parser tags included when a tool call is extracted. |

### SGLang Python - TestKimiK2ReasoningDetectorStreaming

| Test | Bucket(s) | Link | Notes |
| -- | -- | -- | -- |
| `TestKimiK2ReasoningDetectorStreaming.test_streaming_normal_think_then_tool_call` | `REASONING.batch.2`, `PARSER.batch.1`, `PARSER.stream.1` | [source](https://github.com/sgl-project/sglang/blob/612785ffdcaf35552f1ed433a981d596ca9fe900/test/registered/function_call/test_kimik2_detector.py#L385) | Reasoning-to-tool transition coverage; parser tags included when a tool call is extracted. |
| `TestKimiK2ReasoningDetectorStreaming.test_streaming_tool_call_inside_think` | `REASONING.batch.2`, `PARSER.batch.1`, `PARSER.stream.1` | [source](https://github.com/sgl-project/sglang/blob/612785ffdcaf35552f1ed433a981d596ca9fe900/test/registered/function_call/test_kimik2_detector.py#L400) | Reasoning-to-tool transition coverage; parser tags included when a tool call is extracted. |
| `TestKimiK2ReasoningDetectorStreaming.test_streaming_tool_call_marker_in_single_chunk` | `REASONING.batch.2`, `PARSER.batch.1`, `PARSER.stream.1` | [source](https://github.com/sgl-project/sglang/blob/612785ffdcaf35552f1ed433a981d596ca9fe900/test/registered/function_call/test_kimik2_detector.py#L431) | Reasoning-to-tool transition coverage; parser tags included when a tool call is extracted. |
| `TestKimiK2ReasoningDetectorStreaming.test_streaming_partial_marker_buffering` | `REASONING.batch.1`, `PARSER.stream.3` | [source](https://github.com/sgl-project/sglang/blob/612785ffdcaf35552f1ed433a981d596ca9fe900/test/registered/function_call/test_kimik2_detector.py#L442) | Reasoning-to-tool transition coverage; parser tags included when a tool call is extracted. |
| `TestKimiK2ReasoningDetectorStreaming.test_streaming_no_reasoning_mode` | `REASONING.batch.1` | [source](https://github.com/sgl-project/sglang/blob/612785ffdcaf35552f1ed433a981d596ca9fe900/test/registered/function_call/test_kimik2_detector.py#L466) | Reasoning-to-tool transition coverage; parser tags included when a tool call is extracted. |
| `TestKimiK2ReasoningDetectorStreaming.test_streaming_force_reasoning` | `REASONING.batch.1` | [source](https://github.com/sgl-project/sglang/blob/612785ffdcaf35552f1ed433a981d596ca9fe900/test/registered/function_call/test_kimik2_detector.py#L478) | Reasoning-to-tool transition coverage; parser tags included when a tool call is extracted. |

### SGLang Python - TestKimiK2EndToEnd

| Test | Bucket(s) | Link | Notes |
| -- | -- | -- | -- |
| `TestKimiK2EndToEnd.test_e2e_streaming_reasoning_to_tool_call` | `REASONING.batch.2`, `PARSER.batch.1`, `PARSER.stream.1` | [source](https://github.com/sgl-project/sglang/blob/612785ffdcaf35552f1ed433a981d596ca9fe900/test/registered/function_call/test_kimik2_detector.py#L520) | Reasoning-to-tool transition coverage; parser tags included when a tool call is extracted. |
| `TestKimiK2EndToEnd.test_e2e_non_streaming_reasoning_to_tool_call` | `REASONING.batch.2`, `PARSER.batch.1`, `PARSER.stream.1` | [source](https://github.com/sgl-project/sglang/blob/612785ffdcaf35552f1ed433a981d596ca9fe900/test/registered/function_call/test_kimik2_detector.py#L570) | Reasoning-to-tool transition coverage; parser tags included when a tool call is extracted. |
| `TestKimiK2EndToEnd.test_e2e_normal_think_close_then_tool_call` | `REASONING.batch.2`, `PARSER.batch.1`, `PARSER.stream.1` | [source](https://github.com/sgl-project/sglang/blob/612785ffdcaf35552f1ed433a981d596ca9fe900/test/registered/function_call/test_kimik2_detector.py#L598) | Reasoning-to-tool transition coverage; parser tags included when a tool call is extracted. |
| `TestKimiK2EndToEnd.test_e2e_multiple_tool_calls_without_think_close` | `REASONING.batch.2`, `PARSER.batch.1`, `PARSER.stream.1` | [source](https://github.com/sgl-project/sglang/blob/612785ffdcaf35552f1ed433a981d596ca9fe900/test/registered/function_call/test_kimik2_detector.py#L629) | Reasoning-to-tool transition coverage; parser tags included when a tool call is extracted. |

### SGLang Python - Harmony - TestEvent

| Test | Bucket(s) | Link | Notes |
| -- | -- | -- | -- |
| `TestEvent.test_init` | `// helper` | [source](https://github.com/sgl-project/sglang/blob/612785ffdcaf35552f1ed433a981d596ca9fe900/test/registered/unit/parser/test_harmony_parser.py#L21) | White-box helper/state test; no direct cross-impl fixture analogue. |

### SGLang Python - Harmony - TestToken

| Test | Bucket(s) | Link | Notes |
| -- | -- | -- | -- |
| `TestToken.test_init` | `// helper` | [source](https://github.com/sgl-project/sglang/blob/612785ffdcaf35552f1ed433a981d596ca9fe900/test/registered/unit/parser/test_harmony_parser.py#L29) | White-box helper/state test; no direct cross-impl fixture analogue. |

### SGLang Python - Harmony - TestPrefixHold

| Test | Bucket(s) | Link | Notes |
| -- | -- | -- | -- |
| `TestPrefixHold.test_empty_text` | `// helper` | [source](https://github.com/sgl-project/sglang/blob/612785ffdcaf35552f1ed433a981d596ca9fe900/test/registered/unit/parser/test_harmony_parser.py#L38) | White-box helper/state test; no direct cross-impl fixture analogue. |
| `TestPrefixHold.test_no_matching_prefixes` | `// helper` | [source](https://github.com/sgl-project/sglang/blob/612785ffdcaf35552f1ed433a981d596ca9fe900/test/registered/unit/parser/test_harmony_parser.py#L44) | White-box helper/state test; no direct cross-impl fixture analogue. |
| `TestPrefixHold.test_partial_token_suffix` | `// helper` | [source](https://github.com/sgl-project/sglang/blob/612785ffdcaf35552f1ed433a981d596ca9fe900/test/registered/unit/parser/test_harmony_parser.py#L50) | White-box helper/state test; no direct cross-impl fixture analogue. |
| `TestPrefixHold.test_multiple_potential_matches` | `// helper` | [source](https://github.com/sgl-project/sglang/blob/612785ffdcaf35552f1ed433a981d596ca9fe900/test/registered/unit/parser/test_harmony_parser.py#L56) | White-box helper/state test; no direct cross-impl fixture analogue. |
| `TestPrefixHold.test_exact_token_match` | `// helper` | [source](https://github.com/sgl-project/sglang/blob/612785ffdcaf35552f1ed433a981d596ca9fe900/test/registered/unit/parser/test_harmony_parser.py#L62) | White-box helper/state test; no direct cross-impl fixture analogue. |

### SGLang Python - Harmony - TestIterTokens

| Test | Bucket(s) | Link | Notes |
| -- | -- | -- | -- |
| `TestIterTokens.test_empty_text` | `// helper` | [source](https://github.com/sgl-project/sglang/blob/612785ffdcaf35552f1ed433a981d596ca9fe900/test/registered/unit/parser/test_harmony_parser.py#L70) | White-box helper/state test; no direct cross-impl fixture analogue. |
| `TestIterTokens.test_plain_text` | `// helper` | [source](https://github.com/sgl-project/sglang/blob/612785ffdcaf35552f1ed433a981d596ca9fe900/test/registered/unit/parser/test_harmony_parser.py#L75) | White-box helper/state test; no direct cross-impl fixture analogue. |
| `TestIterTokens.test_single_token` | `// helper` | [source](https://github.com/sgl-project/sglang/blob/612785ffdcaf35552f1ed433a981d596ca9fe900/test/registered/unit/parser/test_harmony_parser.py#L83) | White-box helper/state test; no direct cross-impl fixture analogue. |
| `TestIterTokens.test_mixed_content` | `// helper` | [source](https://github.com/sgl-project/sglang/blob/612785ffdcaf35552f1ed433a981d596ca9fe900/test/registered/unit/parser/test_harmony_parser.py#L91) | White-box helper/state test; no direct cross-impl fixture analogue. |
| `TestIterTokens.test_unknown_token_partial_suffix` | `// helper` | [source](https://github.com/sgl-project/sglang/blob/612785ffdcaf35552f1ed433a981d596ca9fe900/test/registered/unit/parser/test_harmony_parser.py#L108) | White-box helper/state test; no direct cross-impl fixture analogue. |
| `TestIterTokens.test_unknown_token_middle` | `// helper` | [source](https://github.com/sgl-project/sglang/blob/612785ffdcaf35552f1ed433a981d596ca9fe900/test/registered/unit/parser/test_harmony_parser.py#L121) | White-box helper/state test; no direct cross-impl fixture analogue. |
| `TestIterTokens.test_all_structural_tokens` | `// helper` | [source](https://github.com/sgl-project/sglang/blob/612785ffdcaf35552f1ed433a981d596ca9fe900/test/registered/unit/parser/test_harmony_parser.py#L132) | White-box helper/state test; no direct cross-impl fixture analogue. |

### SGLang Python - Harmony - TestCanonicalStrategy

| Test | Bucket(s) | Link | Notes |
| -- | -- | -- | -- |
| `TestCanonicalStrategy.test_init` | `// helper` | [source](https://github.com/sgl-project/sglang/blob/612785ffdcaf35552f1ed433a981d596ca9fe900/test/registered/unit/parser/test_harmony_parser.py#L156) | Registry, construction, serialization, or helper-only coverage. |
| `TestCanonicalStrategy.test_extract_channel_type` | `PARSER.harmony.1`, `PARSER.harmony.2` | [source](https://github.com/sgl-project/sglang/blob/612785ffdcaf35552f1ed433a981d596ca9fe900/test/registered/unit/parser/test_harmony_parser.py#L161) | Harmony channel/envelope parser coverage. |
| `TestCanonicalStrategy.test_parse_single_analysis_block` | `PARSER.harmony.1`, `PARSER.harmony.2` | [source](https://github.com/sgl-project/sglang/blob/612785ffdcaf35552f1ed433a981d596ca9fe900/test/registered/unit/parser/test_harmony_parser.py#L172) | Harmony channel/envelope parser coverage. |
| `TestCanonicalStrategy.test_parse_single_commentary_block` | `PARSER.harmony.1`, `PARSER.harmony.2` | [source](https://github.com/sgl-project/sglang/blob/612785ffdcaf35552f1ed433a981d596ca9fe900/test/registered/unit/parser/test_harmony_parser.py#L182) | Harmony channel/envelope parser coverage. |
| `TestCanonicalStrategy.test_parse_single_final_block` | `PARSER.harmony.1`, `PARSER.harmony.2` | [source](https://github.com/sgl-project/sglang/blob/612785ffdcaf35552f1ed433a981d596ca9fe900/test/registered/unit/parser/test_harmony_parser.py#L192) | Harmony channel/envelope parser coverage. |
| `TestCanonicalStrategy.test_parse_tool_call_commentary` | `PARSER.harmony.1`, `PARSER.harmony.2`, `PARSER.batch.1` | [source](https://github.com/sgl-project/sglang/blob/612785ffdcaf35552f1ed433a981d596ca9fe900/test/registered/unit/parser/test_harmony_parser.py#L202) | Harmony channel/envelope parser coverage. |
| `TestCanonicalStrategy.test_parse_tool_call_analysis` | `PARSER.harmony.1`, `PARSER.harmony.2`, `PARSER.batch.1` | [source](https://github.com/sgl-project/sglang/blob/612785ffdcaf35552f1ed433a981d596ca9fe900/test/registered/unit/parser/test_harmony_parser.py#L212) | Harmony channel/envelope parser coverage. |
| `TestCanonicalStrategy.test_parse_complex_sequence` | `PARSER.harmony.1`, `PARSER.harmony.2` | [source](https://github.com/sgl-project/sglang/blob/612785ffdcaf35552f1ed433a981d596ca9fe900/test/registered/unit/parser/test_harmony_parser.py#L222) | Harmony channel/envelope parser coverage. |
| `TestCanonicalStrategy.test_parse_with_interspersed_text` | `PARSER.harmony.1`, `PARSER.harmony.2`, `PARSER.batch.8.d` | [source](https://github.com/sgl-project/sglang/blob/612785ffdcaf35552f1ed433a981d596ca9fe900/test/registered/unit/parser/test_harmony_parser.py#L238) | Harmony channel/envelope parser coverage. |
| `TestCanonicalStrategy.test_parse_incomplete_block` | `PARSER.harmony.1`, `PARSER.harmony.2`, `PARSER.batch.5.c` | [source](https://github.com/sgl-project/sglang/blob/612785ffdcaf35552f1ed433a981d596ca9fe900/test/registered/unit/parser/test_harmony_parser.py#L260) | Harmony channel/envelope parser coverage. |
| `TestCanonicalStrategy.test_parse_partial_token_suffix` | `PARSER.harmony.1`, `PARSER.harmony.2`, `PARSER.batch.5.c`, `PARSER.stream.1`, `PARSER.stream.3` | [source](https://github.com/sgl-project/sglang/blob/612785ffdcaf35552f1ed433a981d596ca9fe900/test/registered/unit/parser/test_harmony_parser.py#L270) | Harmony channel/envelope parser coverage. |
| `TestCanonicalStrategy.test_parse_tool_response_message` | `PARSER.harmony.1`, `PARSER.harmony.2` | [source](https://github.com/sgl-project/sglang/blob/612785ffdcaf35552f1ed433a981d596ca9fe900/test/registered/unit/parser/test_harmony_parser.py#L280) | Harmony channel/envelope parser coverage. |
| `TestCanonicalStrategy.test_parse_empty_content_blocks` | `PARSER.harmony.1`, `PARSER.harmony.2` | [source](https://github.com/sgl-project/sglang/blob/612785ffdcaf35552f1ed433a981d596ca9fe900/test/registered/unit/parser/test_harmony_parser.py#L290) | Harmony channel/envelope parser coverage. |
| `TestCanonicalStrategy.test_parse_commentary_filler_between_blocks` | `PARSER.harmony.1`, `PARSER.harmony.2`, `PARSER.batch.8.d` | [source](https://github.com/sgl-project/sglang/blob/612785ffdcaf35552f1ed433a981d596ca9fe900/test/registered/unit/parser/test_harmony_parser.py#L300) | Harmony channel/envelope parser coverage. |

### SGLang Python - Harmony - TestTextStrategy

| Test | Bucket(s) | Link | Notes |
| -- | -- | -- | -- |
| `TestTextStrategy.test_init` | `// helper` | [source](https://github.com/sgl-project/sglang/blob/612785ffdcaf35552f1ed433a981d596ca9fe900/test/registered/unit/parser/test_harmony_parser.py#L332) | Registry, construction, serialization, or helper-only coverage. |
| `TestTextStrategy.test_parse_analysis_then_final` | `PARSER.harmony.1` | [source](https://github.com/sgl-project/sglang/blob/612785ffdcaf35552f1ed433a981d596ca9fe900/test/registered/unit/parser/test_harmony_parser.py#L336) | Harmony channel/envelope parser coverage. |
| `TestTextStrategy.test_parse_commentary_then_final` | `PARSER.harmony.1` | [source](https://github.com/sgl-project/sglang/blob/612785ffdcaf35552f1ed433a981d596ca9fe900/test/registered/unit/parser/test_harmony_parser.py#L348) | Harmony channel/envelope parser coverage. |
| `TestTextStrategy.test_parse_final_only` | `PARSER.harmony.1`, `PARSER.batch.3` | [source](https://github.com/sgl-project/sglang/blob/612785ffdcaf35552f1ed433a981d596ca9fe900/test/registered/unit/parser/test_harmony_parser.py#L360) | Harmony channel/envelope parser coverage. |
| `TestTextStrategy.test_parse_analysis_only` | `PARSER.harmony.1`, `PARSER.batch.3` | [source](https://github.com/sgl-project/sglang/blob/612785ffdcaf35552f1ed433a981d596ca9fe900/test/registered/unit/parser/test_harmony_parser.py#L370) | Harmony channel/envelope parser coverage. |
| `TestTextStrategy.test_parse_incomplete_assistantfinal` | `PARSER.harmony.1`, `PARSER.batch.5.c` | [source](https://github.com/sgl-project/sglang/blob/612785ffdcaf35552f1ed433a981d596ca9fe900/test/registered/unit/parser/test_harmony_parser.py#L381) | Harmony channel/envelope parser coverage. |
| `TestTextStrategy.test_parse_partial_analysis_streaming` | `PARSER.harmony.1`, `PARSER.batch.5.c`, `PARSER.stream.1`, `PARSER.stream.3` | [source](https://github.com/sgl-project/sglang/blob/612785ffdcaf35552f1ed433a981d596ca9fe900/test/registered/unit/parser/test_harmony_parser.py#L389) | Harmony channel/envelope parser coverage. |
| `TestTextStrategy.test_parse_case_insensitive` | `PARSER.harmony.1` | [source](https://github.com/sgl-project/sglang/blob/612785ffdcaf35552f1ed433a981d596ca9fe900/test/registered/unit/parser/test_harmony_parser.py#L399) | Harmony channel/envelope parser coverage. |
| `TestTextStrategy.test_parse_plain_text_fallback` | `PARSER.harmony.1`, `PARSER.batch.3` | [source](https://github.com/sgl-project/sglang/blob/612785ffdcaf35552f1ed433a981d596ca9fe900/test/registered/unit/parser/test_harmony_parser.py#L408) | Harmony channel/envelope parser coverage. |
| `TestTextStrategy.test_parse_analysis_no_space_after_header` | `PARSER.harmony.1` | [source](https://github.com/sgl-project/sglang/blob/612785ffdcaf35552f1ed433a981d596ca9fe900/test/registered/unit/parser/test_harmony_parser.py#L418) | Harmony channel/envelope parser coverage. |

### SGLang Python - Harmony - TestHarmonyParser

| Test | Bucket(s) | Link | Notes |
| -- | -- | -- | -- |
| `TestHarmonyParser.test_init` | `// helper` | [source](https://github.com/sgl-project/sglang/blob/612785ffdcaf35552f1ed433a981d596ca9fe900/test/registered/unit/parser/test_harmony_parser.py#L439) | Registry, construction, serialization, or helper-only coverage. |
| `TestHarmonyParser.test_strategy_selection_canonical` | `// helper` | [source](https://github.com/sgl-project/sglang/blob/612785ffdcaf35552f1ed433a981d596ca9fe900/test/registered/unit/parser/test_harmony_parser.py#L444) | Registry, construction, serialization, or helper-only coverage. |
| `TestHarmonyParser.test_strategy_selection_text` | `// helper` | [source](https://github.com/sgl-project/sglang/blob/612785ffdcaf35552f1ed433a981d596ca9fe900/test/registered/unit/parser/test_harmony_parser.py#L452) | Registry, construction, serialization, or helper-only coverage. |
| `TestHarmonyParser.test_strategy_selection_delayed` | `// helper` | [source](https://github.com/sgl-project/sglang/blob/612785ffdcaf35552f1ed433a981d596ca9fe900/test/registered/unit/parser/test_harmony_parser.py#L460) | Registry, construction, serialization, or helper-only coverage. |
| `TestHarmonyParser.test_streaming_canonical_format` | `PARSER.harmony.1`, `PARSER.harmony.2`, `PARSER.stream.1` | [source](https://github.com/sgl-project/sglang/blob/612785ffdcaf35552f1ed433a981d596ca9fe900/test/registered/unit/parser/test_harmony_parser.py#L472) | Harmony channel/envelope parser coverage. |
| `TestHarmonyParser.test_streaming_text_format` | `PARSER.harmony.1`, `PARSER.stream.1` | [source](https://github.com/sgl-project/sglang/blob/612785ffdcaf35552f1ed433a981d596ca9fe900/test/registered/unit/parser/test_harmony_parser.py#L506) | Harmony channel/envelope parser coverage. |
| `TestHarmonyParser.test_streaming_commentary_filler` | `PARSER.harmony.1`, `PARSER.batch.8.d`, `PARSER.stream.1` | [source](https://github.com/sgl-project/sglang/blob/612785ffdcaf35552f1ed433a981d596ca9fe900/test/registered/unit/parser/test_harmony_parser.py#L522) | Harmony channel/envelope parser coverage. |
| `TestHarmonyParser.test_repetitive_tool_calls_with_commentary_filler` | `PARSER.harmony.1`, `PARSER.harmony.2`, `PARSER.batch.1`, `PARSER.batch.2`, `PARSER.batch.8.d` | [source](https://github.com/sgl-project/sglang/blob/612785ffdcaf35552f1ed433a981d596ca9fe900/test/registered/unit/parser/test_harmony_parser.py#L570) | Harmony channel/envelope parser coverage. |

### SGLang Python - Harmony - TestIntegrationScenarios

| Test | Bucket(s) | Link | Notes |
| -- | -- | -- | -- |
| `TestIntegrationScenarios.test_complete_reasoning_flow` | `REASONING.batch.1` | [source](https://github.com/sgl-project/sglang/blob/612785ffdcaf35552f1ed433a981d596ca9fe900/test/registered/unit/parser/test_harmony_parser.py#L615) | Reasoning-to-tool transition coverage; parser tags included when a tool call is extracted. |
| `TestIntegrationScenarios.test_tool_call_sequence` | `PARSER.harmony.1`, `PARSER.harmony.2`, `PARSER.batch.1` | [source](https://github.com/sgl-project/sglang/blob/612785ffdcaf35552f1ed433a981d596ca9fe900/test/registered/unit/parser/test_harmony_parser.py#L632) | Harmony channel/envelope parser coverage. |
| `TestIntegrationScenarios.test_preamble_sequence` | `PARSER.harmony.1` | [source](https://github.com/sgl-project/sglang/blob/612785ffdcaf35552f1ed433a981d596ca9fe900/test/registered/unit/parser/test_harmony_parser.py#L650) | Harmony channel/envelope parser coverage. |
| `TestIntegrationScenarios.test_built_in_tool_call` | `PARSER.harmony.1`, `PARSER.harmony.2`, `PARSER.batch.1` | [source](https://github.com/sgl-project/sglang/blob/612785ffdcaf35552f1ed433a981d596ca9fe900/test/registered/unit/parser/test_harmony_parser.py#L669) | Harmony channel/envelope parser coverage. |
| `TestIntegrationScenarios.test_tool_response_handling` | `PARSER.harmony.1` | [source](https://github.com/sgl-project/sglang/blob/612785ffdcaf35552f1ed433a981d596ca9fe900/test/registered/unit/parser/test_harmony_parser.py#L681) | Harmony channel/envelope parser coverage. |
| `TestIntegrationScenarios.test_text_fallback_formats` | `PARSER.harmony.1`, `PARSER.batch.3` | [source](https://github.com/sgl-project/sglang/blob/612785ffdcaf35552f1ed433a981d596ca9fe900/test/registered/unit/parser/test_harmony_parser.py#L693) | Harmony channel/envelope parser coverage. |
| `TestIntegrationScenarios.test_streaming_property_canonical` | `PARSER.harmony.1`, `PARSER.harmony.2`, `PARSER.stream.1`, `PARSER.stream.3` | [source](https://github.com/sgl-project/sglang/blob/612785ffdcaf35552f1ed433a981d596ca9fe900/test/registered/unit/parser/test_harmony_parser.py#L710) | Harmony channel/envelope parser coverage. |
| `TestIntegrationScenarios.test_streaming_property_text` | `PARSER.harmony.1`, `PARSER.stream.1`, `PARSER.stream.3` | [source](https://github.com/sgl-project/sglang/blob/612785ffdcaf35552f1ed433a981d596ca9fe900/test/registered/unit/parser/test_harmony_parser.py#L758) | Harmony channel/envelope parser coverage. |

### SGLang Python - Harmony - TestEdgeCases

| Test | Bucket(s) | Link | Notes |
| -- | -- | -- | -- |
| `TestEdgeCases.test_malformed_channel_headers` | `PARSER.harmony.1`, `PARSER.harmony.2`, `PARSER.batch.4.d` | [source](https://github.com/sgl-project/sglang/blob/612785ffdcaf35552f1ed433a981d596ca9fe900/test/registered/unit/parser/test_harmony_parser.py#L796) | Harmony channel/envelope parser coverage. |
| `TestEdgeCases.test_mixed_unknown_tokens` | `PARSER.harmony.1`, `PARSER.harmony.2`, `PARSER.batch.4.d` | [source](https://github.com/sgl-project/sglang/blob/612785ffdcaf35552f1ed433a981d596ca9fe900/test/registered/unit/parser/test_harmony_parser.py#L807) | Harmony channel/envelope parser coverage. |
| `TestEdgeCases.test_empty_input` | `PARSER.harmony.1`, `PARSER.batch.9` | [source](https://github.com/sgl-project/sglang/blob/612785ffdcaf35552f1ed433a981d596ca9fe900/test/registered/unit/parser/test_harmony_parser.py#L821) | Harmony channel/envelope parser coverage. |
| `TestEdgeCases.test_whitespace_preservation` | `PARSER.harmony.1` | [source](https://github.com/sgl-project/sglang/blob/612785ffdcaf35552f1ed433a981d596ca9fe900/test/registered/unit/parser/test_harmony_parser.py#L827) | Harmony channel/envelope parser coverage. |
| `TestEdgeCases.test_streaming_whitespace_preservation` | `PARSER.harmony.1`, `PARSER.stream.1` | [source](https://github.com/sgl-project/sglang/blob/612785ffdcaf35552f1ed433a981d596ca9fe900/test/registered/unit/parser/test_harmony_parser.py#L837) | Harmony channel/envelope parser coverage. |
| `TestEdgeCases.test_consecutive_blocks_same_type` | `PARSER.harmony.1`, `PARSER.batch.2` | [source](https://github.com/sgl-project/sglang/blob/612785ffdcaf35552f1ed433a981d596ca9fe900/test/registered/unit/parser/test_harmony_parser.py#L860) | Harmony channel/envelope parser coverage. |

### SGLang Python - Harmony - TestAdditionalEdgeCases

| Test | Bucket(s) | Link | Notes |
| -- | -- | -- | -- |
| `TestAdditionalEdgeCases.test_prefix_hold_with_empty_token_in_list` | `PARSER.harmony.1`, `PARSER.harmony.2`, `PARSER.stream.1`, `PARSER.stream.3` | [source](https://github.com/sgl-project/sglang/blob/612785ffdcaf35552f1ed433a981d596ca9fe900/test/registered/unit/parser/test_harmony_parser.py#L880) | Harmony channel/envelope parser coverage. |
| `TestAdditionalEdgeCases.test_iter_tokens_unknown_token_no_closing` | `PARSER.harmony.1`, `PARSER.harmony.2`, `PARSER.batch.4.d` | [source](https://github.com/sgl-project/sglang/blob/612785ffdcaf35552f1ed433a981d596ca9fe900/test/registered/unit/parser/test_harmony_parser.py#L888) | Harmony channel/envelope parser coverage. |
| `TestAdditionalEdgeCases.test_canonical_commentary_filler_after_call` | `PARSER.harmony.1`, `PARSER.harmony.2`, `PARSER.batch.8.d` | [source](https://github.com/sgl-project/sglang/blob/612785ffdcaf35552f1ed433a981d596ca9fe900/test/registered/unit/parser/test_harmony_parser.py#L896) | Harmony channel/envelope parser coverage. |
| `TestAdditionalEdgeCases.test_canonical_standalone_structural_token_filtered` | `PARSER.harmony.1`, `PARSER.harmony.2` | [source](https://github.com/sgl-project/sglang/blob/612785ffdcaf35552f1ed433a981d596ca9fe900/test/registered/unit/parser/test_harmony_parser.py#L907) | Harmony channel/envelope parser coverage. |
| `TestAdditionalEdgeCases.test_canonical_incomplete_block_returns_partial` | `PARSER.harmony.1`, `PARSER.harmony.2`, `PARSER.batch.5.c` | [source](https://github.com/sgl-project/sglang/blob/612785ffdcaf35552f1ed433a981d596ca9fe900/test/registered/unit/parser/test_harmony_parser.py#L918) | Harmony channel/envelope parser coverage. |
| `TestAdditionalEdgeCases.test_text_strategy_commentary_channel` | `PARSER.harmony.1`, `PARSER.harmony.2` | [source](https://github.com/sgl-project/sglang/blob/612785ffdcaf35552f1ed433a981d596ca9fe900/test/registered/unit/parser/test_harmony_parser.py#L931) | Harmony channel/envelope parser coverage. |
| `TestAdditionalEdgeCases.test_canonical_call_with_text_commentary_after` | `PARSER.harmony.1`, `PARSER.harmony.2`, `PARSER.batch.8.d` | [source](https://github.com/sgl-project/sglang/blob/612785ffdcaf35552f1ed433a981d596ca9fe900/test/registered/unit/parser/test_harmony_parser.py#L941) | Harmony channel/envelope parser coverage. |
| `TestAdditionalEdgeCases.test_canonical_return_without_final` | `PARSER.harmony.1`, `PARSER.harmony.2` | [source](https://github.com/sgl-project/sglang/blob/612785ffdcaf35552f1ed433a981d596ca9fe900/test/registered/unit/parser/test_harmony_parser.py#L951) | Harmony channel/envelope parser coverage. |
| `TestAdditionalEdgeCases.test_iter_tokens_unknown_at_end_no_next_marker` | `PARSER.harmony.1`, `PARSER.harmony.2` | [source](https://github.com/sgl-project/sglang/blob/612785ffdcaf35552f1ed433a981d596ca9fe900/test/registered/unit/parser/test_harmony_parser.py#L962) | Harmony channel/envelope parser coverage. |
| `TestAdditionalEdgeCases.test_canonical_standalone_end_token_filtered` | `PARSER.harmony.1`, `PARSER.harmony.2` | [source](https://github.com/sgl-project/sglang/blob/612785ffdcaf35552f1ed433a981d596ca9fe900/test/registered/unit/parser/test_harmony_parser.py#L974) | Harmony channel/envelope parser coverage. |
| `TestAdditionalEdgeCases.test_canonical_incomplete_parse_block_no_end` | `PARSER.harmony.1`, `PARSER.harmony.2`, `PARSER.batch.5.c` | [source](https://github.com/sgl-project/sglang/blob/612785ffdcaf35552f1ed433a981d596ca9fe900/test/registered/unit/parser/test_harmony_parser.py#L986) | Harmony channel/envelope parser coverage. |
| `TestAdditionalEdgeCases.test_text_strategy_commentary_only` | `PARSER.harmony.1`, `PARSER.batch.3` | [source](https://github.com/sgl-project/sglang/blob/612785ffdcaf35552f1ed433a981d596ca9fe900/test/registered/unit/parser/test_harmony_parser.py#L998) | Harmony channel/envelope parser coverage. |
| `TestAdditionalEdgeCases.test_text_strategy_commentary_with_hold` | `PARSER.harmony.1`, `PARSER.stream.1`, `PARSER.stream.3` | [source](https://github.com/sgl-project/sglang/blob/612785ffdcaf35552f1ed433a981d596ca9fe900/test/registered/unit/parser/test_harmony_parser.py#L1010) | Harmony channel/envelope parser coverage. |

### SGLang Python - TestPythonicDetector

| Test | Bucket(s) | Link | Notes |
| -- | -- | -- | -- |
| `TestPythonicDetector.test_parse_streaming_no_brackets` | `PARSER.batch.3`, `PARSER.stream.1` | [source](https://github.com/sgl-project/sglang/blob/612785ffdcaf35552f1ed433a981d596ca9fe900/test/registered/unit/function_call/test_function_call_parser.py#L74) | Parse streaming no brackets. |
| `TestPythonicDetector.test_parse_streaming_complete_tool_call` | `PARSER.stream.1` | [source](https://github.com/sgl-project/sglang/blob/612785ffdcaf35552f1ed433a981d596ca9fe900/test/registered/unit/function_call/test_function_call_parser.py#L83) | Parse streaming complete tool call. |
| `TestPythonicDetector.test_parse_streaming_text_before_tool_call` | `PARSER.batch.8.a`, `PARSER.stream.1` | [source](https://github.com/sgl-project/sglang/blob/612785ffdcaf35552f1ed433a981d596ca9fe900/test/registered/unit/function_call/test_function_call_parser.py#L100) | Parse streaming text before tool call. |
| `TestPythonicDetector.test_parse_streaming_partial_tool_call` | `PARSER.batch.5.c`, `PARSER.stream.1`, `PARSER.stream.3` | [source](https://github.com/sgl-project/sglang/blob/612785ffdcaf35552f1ed433a981d596ca9fe900/test/registered/unit/function_call/test_function_call_parser.py#L113) | Parse streaming partial tool call. |
| `TestPythonicDetector.test_parse_streaming_bracket_without_text_before` | `PARSER.batch.5.a`, `PARSER.batch.8.a`, `PARSER.stream.1` | [source](https://github.com/sgl-project/sglang/blob/612785ffdcaf35552f1ed433a981d596ca9fe900/test/registered/unit/function_call/test_function_call_parser.py#L140) | Parse streaming bracket without text before. |
| `TestPythonicDetector.test_parse_streaming_text_after_tool_call` | `PARSER.batch.8.b`, `PARSER.stream.1` | [source](https://github.com/sgl-project/sglang/blob/612785ffdcaf35552f1ed433a981d596ca9fe900/test/registered/unit/function_call/test_function_call_parser.py#L153) | Parse streaming text after tool call. |
| `TestPythonicDetector.test_parse_streaming_multiple_tool_calls` | `PARSER.batch.2.b`, `PARSER.stream.2` | [source](https://github.com/sgl-project/sglang/blob/612785ffdcaf35552f1ed433a981d596ca9fe900/test/registered/unit/function_call/test_function_call_parser.py#L172) | Parse streaming multiple tool calls. |
| `TestPythonicDetector.test_parse_streaming_opening_bracket_only` | `PARSER.batch.5.a`, `PARSER.stream.1` | [source](https://github.com/sgl-project/sglang/blob/612785ffdcaf35552f1ed433a981d596ca9fe900/test/registered/unit/function_call/test_function_call_parser.py#L189) | Parse streaming opening bracket only. |
| `TestPythonicDetector.test_parse_streaming_nested_brackets` | `PARSER.batch.7.d`, `PARSER.stream.1` | [source](https://github.com/sgl-project/sglang/blob/612785ffdcaf35552f1ed433a981d596ca9fe900/test/registered/unit/function_call/test_function_call_parser.py#L200) | Parse streaming nested brackets. |
| `TestPythonicDetector.test_parse_streaming_nested_brackets_dict` | `PARSER.batch.7.d`, `PARSER.stream.1` | [source](https://github.com/sgl-project/sglang/blob/612785ffdcaf35552f1ed433a981d596ca9fe900/test/registered/unit/function_call/test_function_call_parser.py#L217) | Parse streaming nested brackets dict. |
| `TestPythonicDetector.test_parse_streaming_multiple_tools_with_nested_brackets` | `PARSER.batch.2.b`, `PARSER.batch.7.d`, `PARSER.stream.2` | [source](https://github.com/sgl-project/sglang/blob/612785ffdcaf35552f1ed433a981d596ca9fe900/test/registered/unit/function_call/test_function_call_parser.py#L234) | Parse streaming multiple tools with nested brackets. |
| `TestPythonicDetector.test_parse_streaming_partial_nested_brackets` | `PARSER.batch.5.c`, `PARSER.batch.7.d`, `PARSER.stream.1`, `PARSER.stream.3` | [source](https://github.com/sgl-project/sglang/blob/612785ffdcaf35552f1ed433a981d596ca9fe900/test/registered/unit/function_call/test_function_call_parser.py#L255) | Parse streaming partial nested brackets. |
| `TestPythonicDetector.test_parse_streaming_with_python_start_and_end_token` | `PARSER.fmt.3`, `PARSER.stream.1`, `PARSER.stream.4` | [source](https://github.com/sgl-project/sglang/blob/612785ffdcaf35552f1ed433a981d596ca9fe900/test/registered/unit/function_call/test_function_call_parser.py#L281) | Parse streaming with python start and end token. |
| `TestPythonicDetector.test_detect_and_parse_with_python_start_and_end_token` | `PARSER.batch.1`, `PARSER.fmt.3` | [source](https://github.com/sgl-project/sglang/blob/612785ffdcaf35552f1ed433a981d596ca9fe900/test/registered/unit/function_call/test_function_call_parser.py#L338) | Detect and parse with python start and end token. |

### SGLang Python - TestBaseFormatDetector

| Test | Bucket(s) | Link | Notes |
| -- | -- | -- | -- |
| `TestBaseFormatDetector.test_sequential_tool_index_assignment` | `// helper` | [source](https://github.com/sgl-project/sglang/blob/612785ffdcaf35552f1ed433a981d596ca9fe900/test/registered/unit/function_call/test_function_call_parser.py#L565) | White-box helper/state test; no direct cross-impl fixture analogue. |
| `TestBaseFormatDetector.test_buffer_content_preservation` | `// helper` | [source](https://github.com/sgl-project/sglang/blob/612785ffdcaf35552f1ed433a981d596ca9fe900/test/registered/unit/function_call/test_function_call_parser.py#L596) | White-box helper/state test; no direct cross-impl fixture analogue. |
| `TestBaseFormatDetector.test_current_tool_id_increment_on_completion` | `// helper` | [source](https://github.com/sgl-project/sglang/blob/612785ffdcaf35552f1ed433a981d596ca9fe900/test/registered/unit/function_call/test_function_call_parser.py#L625) | White-box helper/state test; no direct cross-impl fixture analogue. |
| `TestBaseFormatDetector.test_tool_name_streaming_with_correct_index` | `// helper` | [source](https://github.com/sgl-project/sglang/blob/612785ffdcaf35552f1ed433a981d596ca9fe900/test/registered/unit/function_call/test_function_call_parser.py#L675) | White-box helper/state test; no direct cross-impl fixture analogue. |
| `TestBaseFormatDetector.test_buffer_reset_on_invalid_tool` | `// helper` | [source](https://github.com/sgl-project/sglang/blob/612785ffdcaf35552f1ed433a981d596ca9fe900/test/registered/unit/function_call/test_function_call_parser.py#L712) | White-box helper/state test; no direct cross-impl fixture analogue. |
| `TestBaseFormatDetector.test_chinese_characters_not_double_escaped` | `// helper` | [source](https://github.com/sgl-project/sglang/blob/612785ffdcaf35552f1ed433a981d596ca9fe900/test/registered/unit/function_call/test_function_call_parser.py#L730) | White-box helper/state test; no direct cross-impl fixture analogue. |
| `TestBaseFormatDetector.test_chinese_characters_incremental_streaming` | `// helper` | [source](https://github.com/sgl-project/sglang/blob/612785ffdcaf35552f1ed433a981d596ca9fe900/test/registered/unit/function_call/test_function_call_parser.py#L774) | White-box helper/state test; no direct cross-impl fixture analogue. |
| `TestBaseFormatDetector.test_multiple_chinese_parameters` | `// helper` | [source](https://github.com/sgl-project/sglang/blob/612785ffdcaf35552f1ed433a981d596ca9fe900/test/registered/unit/function_call/test_function_call_parser.py#L814) | White-box helper/state test; no direct cross-impl fixture analogue. |

### SGLang Python - TestKimiK2Detector

| Test | Bucket(s) | Link | Notes |
| -- | -- | -- | -- |
| `TestKimiK2Detector.test_single_tool_call` | `PARSER.batch.1` | [source](https://github.com/sgl-project/sglang/blob/612785ffdcaf35552f1ed433a981d596ca9fe900/test/registered/unit/function_call/test_function_call_parser.py#L1003) | Single tool call. |
| `TestKimiK2Detector.test_multiple_tool_calls` | `PARSER.batch.2.a` | [source](https://github.com/sgl-project/sglang/blob/612785ffdcaf35552f1ed433a981d596ca9fe900/test/registered/unit/function_call/test_function_call_parser.py#L1012) | Multiple tool calls. |
| `TestKimiK2Detector.test_streaming_tool_call` | `PARSER.stream.1` | [source](https://github.com/sgl-project/sglang/blob/612785ffdcaf35552f1ed433a981d596ca9fe900/test/registered/unit/function_call/test_function_call_parser.py#L1023) | Streaming tool call. |
| `TestKimiK2Detector.test_streaming_multiple_tool_calls` | `PARSER.batch.2.b`, `PARSER.stream.2` | [source](https://github.com/sgl-project/sglang/blob/612785ffdcaf35552f1ed433a981d596ca9fe900/test/registered/unit/function_call/test_function_call_parser.py#L1052) | Streaming multiple tool calls. |
| `TestKimiK2Detector.test_tool_call_completion` | `// helper` | [source](https://github.com/sgl-project/sglang/blob/612785ffdcaf35552f1ed433a981d596ca9fe900/test/registered/unit/function_call/test_function_call_parser.py#L1086) | Unclassified helper or parser-adjacent smoke test. |
| `TestKimiK2Detector.test_tool_name_streaming` | `PARSER.stream.1` | [source](https://github.com/sgl-project/sglang/blob/612785ffdcaf35552f1ed433a981d596ca9fe900/test/registered/unit/function_call/test_function_call_parser.py#L1103) | Tool name streaming. |
| `TestKimiK2Detector.test_invalid_tool_call` | `PARSER.batch.4.b` | [source](https://github.com/sgl-project/sglang/blob/612785ffdcaf35552f1ed433a981d596ca9fe900/test/registered/unit/function_call/test_function_call_parser.py#L1134) | Invalid tool call. |
| `TestKimiK2Detector.test_partial_tool_call` | `PARSER.batch.5.c` | [source](https://github.com/sgl-project/sglang/blob/612785ffdcaf35552f1ed433a981d596ca9fe900/test/registered/unit/function_call/test_function_call_parser.py#L1141) | Partial tool call. |

### SGLang Python - TestDeepSeekV3Detector

| Test | Bucket(s) | Link | Notes |
| -- | -- | -- | -- |
| `TestDeepSeekV3Detector.test_parse_streaming_multiple_tool_calls_with_multi_token_chunk` | `PARSER.batch.2.b`, `PARSER.fmt.3`, `PARSER.stream.2`, `PARSER.stream.3` | [source](https://github.com/sgl-project/sglang/blob/612785ffdcaf35552f1ed433a981d596ca9fe900/test/registered/unit/function_call/test_function_call_parser.py#L1210) | Parse streaming multiple tool calls with multi token chunk. |

### SGLang Python - TestDeepSeekV32Detector

| Test | Bucket(s) | Link | Notes |
| -- | -- | -- | -- |
| `TestDeepSeekV32Detector.test_detect_and_parse_xml_format` | `PARSER.batch.1`, `PARSER.fmt.3` | [source](https://github.com/sgl-project/sglang/blob/612785ffdcaf35552f1ed433a981d596ca9fe900/test/registered/unit/function_call/test_function_call_parser.py#L1309) | Detect and parse xml format. |
| `TestDeepSeekV32Detector.test_detect_and_parse_json_format` | `PARSER.batch.1`, `PARSER.fmt.3` | [source](https://github.com/sgl-project/sglang/blob/612785ffdcaf35552f1ed433a981d596ca9fe900/test/registered/unit/function_call/test_function_call_parser.py#L1342) | Detect and parse json format. |
| `TestDeepSeekV32Detector.test_streaming_xml_format` | `PARSER.fmt.3`, `PARSER.stream.1` | [source](https://github.com/sgl-project/sglang/blob/612785ffdcaf35552f1ed433a981d596ca9fe900/test/registered/unit/function_call/test_function_call_parser.py#L1380) | Streaming xml format. |
| `TestDeepSeekV32Detector.test_streaming_json_format` | `PARSER.fmt.3`, `PARSER.stream.1` | [source](https://github.com/sgl-project/sglang/blob/612785ffdcaf35552f1ed433a981d596ca9fe900/test/registered/unit/function_call/test_function_call_parser.py#L1430) | Streaming json format. |
| `TestDeepSeekV32Detector.test_detect_and_parse_no_parameters` | `PARSER.batch.1`, `PARSER.batch.6.c` | [source](https://github.com/sgl-project/sglang/blob/612785ffdcaf35552f1ed433a981d596ca9fe900/test/registered/unit/function_call/test_function_call_parser.py#L1483) | Detect and parse no parameters. |
| `TestDeepSeekV32Detector.test_streaming_no_parameters` | `PARSER.batch.6.c`, `PARSER.stream.1` | [source](https://github.com/sgl-project/sglang/blob/612785ffdcaf35552f1ed433a981d596ca9fe900/test/registered/unit/function_call/test_function_call_parser.py#L1514) | Streaming no parameters. |
| `TestDeepSeekV32Detector.test_streaming_no_parameters_with_whitespace` | `PARSER.batch.6.c`, `PARSER.fmt.2`, `PARSER.stream.1` | [source](https://github.com/sgl-project/sglang/blob/612785ffdcaf35552f1ed433a981d596ca9fe900/test/registered/unit/function_call/test_function_call_parser.py#L1578) | Streaming no parameters with whitespace. |

### SGLang Python - TestQwen3CoderDetector

| Test | Bucket(s) | Link | Notes |
| -- | -- | -- | -- |
| `TestQwen3CoderDetector.test_plain_text_only` | `PARSER.batch.3` | [source](https://github.com/sgl-project/sglang/blob/612785ffdcaf35552f1ed433a981d596ca9fe900/test/registered/unit/function_call/test_function_call_parser.py#L1700) | Plain text only. |
| `TestQwen3CoderDetector.test_single_tool_call` | `PARSER.batch.1` | [source](https://github.com/sgl-project/sglang/blob/612785ffdcaf35552f1ed433a981d596ca9fe900/test/registered/unit/function_call/test_function_call_parser.py#L1713) | Single tool call. |
| `TestQwen3CoderDetector.test_single_tool_call_with_text_prefix` | `PARSER.batch.1` | [source](https://github.com/sgl-project/sglang/blob/612785ffdcaf35552f1ed433a981d596ca9fe900/test/registered/unit/function_call/test_function_call_parser.py#L1737) | Single tool call with text prefix. |
| `TestQwen3CoderDetector.test_multiple_tool_calls` | `PARSER.batch.2.a` | [source](https://github.com/sgl-project/sglang/blob/612785ffdcaf35552f1ed433a981d596ca9fe900/test/registered/unit/function_call/test_function_call_parser.py#L1757) | Multiple tool calls. |
| `TestQwen3CoderDetector.test_streaming_single_tool_call` | `PARSER.batch.1`, `PARSER.stream.1` | [source](https://github.com/sgl-project/sglang/blob/612785ffdcaf35552f1ed433a981d596ca9fe900/test/registered/unit/function_call/test_function_call_parser.py#L1790) | Streaming single tool call. |
| `TestQwen3CoderDetector.test_streaming_with_text_and_tool` | `PARSER.stream.1` | [source](https://github.com/sgl-project/sglang/blob/612785ffdcaf35552f1ed433a981d596ca9fe900/test/registered/unit/function_call/test_function_call_parser.py#L1828) | Streaming with text and tool. |
| `TestQwen3CoderDetector.test_integer_parameter_conversion` | `PARSER.batch.7.c` | [source](https://github.com/sgl-project/sglang/blob/612785ffdcaf35552f1ed433a981d596ca9fe900/test/registered/unit/function_call/test_function_call_parser.py#L1860) | Integer parameter conversion. |
| `TestQwen3CoderDetector.test_boolean_parameter_conversion` | `PARSER.batch.7.c` | [source](https://github.com/sgl-project/sglang/blob/612785ffdcaf35552f1ed433a981d596ca9fe900/test/registered/unit/function_call/test_function_call_parser.py#L1879) | Boolean parameter conversion. |
| `TestQwen3CoderDetector.test_complex_array_parameter` | `PARSER.batch.7.d` | [source](https://github.com/sgl-project/sglang/blob/612785ffdcaf35552f1ed433a981d596ca9fe900/test/registered/unit/function_call/test_function_call_parser.py#L1898) | Complex array parameter. |
| `TestQwen3CoderDetector.test_empty_parameter_value` | `PARSER.batch.6.b` | [source](https://github.com/sgl-project/sglang/blob/612785ffdcaf35552f1ed433a981d596ca9fe900/test/registered/unit/function_call/test_function_call_parser.py#L1925) | Empty parameter value. |
| `TestQwen3CoderDetector.test_parameter_with_special_characters` | `PARSER.batch.7.b` | [source](https://github.com/sgl-project/sglang/blob/612785ffdcaf35552f1ed433a981d596ca9fe900/test/registered/unit/function_call/test_function_call_parser.py#L1943) | Parameter with special characters. |
| `TestQwen3CoderDetector.test_incomplete_tool_call` | `PARSER.batch.5.c` | [source](https://github.com/sgl-project/sglang/blob/612785ffdcaf35552f1ed433a981d596ca9fe900/test/registered/unit/function_call/test_function_call_parser.py#L1961) | Incomplete tool call. |
| `TestQwen3CoderDetector.test_has_tool_call_detection` | `PARSER.fmt.3` | [source](https://github.com/sgl-project/sglang/blob/612785ffdcaf35552f1ed433a981d596ca9fe900/test/registered/unit/function_call/test_function_call_parser.py#L1976) | Format detection / marker-recognition helper coverage. |

### SGLang Python - TestGlm4MoeDetector

| Test | Bucket(s) | Link | Notes |
| -- | -- | -- | -- |
| `TestGlm4MoeDetector.test_single_tool_call` | `PARSER.batch.1` | [source](https://github.com/sgl-project/sglang/blob/612785ffdcaf35552f1ed433a981d596ca9fe900/test/registered/unit/function_call/test_function_call_parser.py#L2010) | Single tool call. |
| `TestGlm4MoeDetector.test_multiple_tool_calls` | `PARSER.batch.2.a` | [source](https://github.com/sgl-project/sglang/blob/612785ffdcaf35552f1ed433a981d596ca9fe900/test/registered/unit/function_call/test_function_call_parser.py#L2025) | Multiple tool calls. |
| `TestGlm4MoeDetector.test_streaming_tool_call` | `PARSER.stream.1` | [source](https://github.com/sgl-project/sglang/blob/612785ffdcaf35552f1ed433a981d596ca9fe900/test/registered/unit/function_call/test_function_call_parser.py#L2048) | Streaming tool call. |
| `TestGlm4MoeDetector.test_streaming_multiple_tool_calls` | `PARSER.batch.2.b`, `PARSER.stream.2` | [source](https://github.com/sgl-project/sglang/blob/612785ffdcaf35552f1ed433a981d596ca9fe900/test/registered/unit/function_call/test_function_call_parser.py#L2077) | Streaming multiple tool calls. |
| `TestGlm4MoeDetector.test_tool_call_id` | `PARSER.fmt.5` | [source](https://github.com/sgl-project/sglang/blob/612785ffdcaf35552f1ed433a981d596ca9fe900/test/registered/unit/function_call/test_function_call_parser.py#L2113) | Tool call id. |
| `TestGlm4MoeDetector.test_invalid_tool_call` | `PARSER.batch.4.b` | [source](https://github.com/sgl-project/sglang/blob/612785ffdcaf35552f1ed433a981d596ca9fe900/test/registered/unit/function_call/test_function_call_parser.py#L2125) | Invalid tool call. |
| `TestGlm4MoeDetector.test_partial_tool_call` | `PARSER.batch.5.c` | [source](https://github.com/sgl-project/sglang/blob/612785ffdcaf35552f1ed433a981d596ca9fe900/test/registered/unit/function_call/test_function_call_parser.py#L2131) | Partial tool call. |
| `TestGlm4MoeDetector.test_array_argument_with_escaped_json` | `PARSER.batch.7.d`, `PARSER.batch.7.b` | [source](https://github.com/sgl-project/sglang/blob/612785ffdcaf35552f1ed433a981d596ca9fe900/test/registered/unit/function_call/test_function_call_parser.py#L2161) | Array argument with escaped json. |
| `TestGlm4MoeDetector.test_empty_function_name_handling` | `PARSER.fmt.1` | [source](https://github.com/sgl-project/sglang/blob/612785ffdcaf35552f1ed433a981d596ca9fe900/test/registered/unit/function_call/test_function_call_parser.py#L2264) | Empty function name handling. |
| `TestGlm4MoeDetector.test_whitespace_preserved_in_arg_values` | `PARSER.fmt.2` | [source](https://github.com/sgl-project/sglang/blob/612785ffdcaf35552f1ed433a981d596ca9fe900/test/registered/unit/function_call/test_function_call_parser.py#L2279) | Whitespace preserved in arg values. |

### SGLang Python - TestGlm47MoeDetector

| Test | Bucket(s) | Link | Notes |
| -- | -- | -- | -- |
| `TestGlm47MoeDetector.test_single_tool_call` | `PARSER.batch.1` | [source](https://github.com/sgl-project/sglang/blob/612785ffdcaf35552f1ed433a981d596ca9fe900/test/registered/unit/function_call/test_function_call_parser.py#L2334) | Single tool call. |
| `TestGlm47MoeDetector.test_multiple_tool_calls` | `PARSER.batch.2.a` | [source](https://github.com/sgl-project/sglang/blob/612785ffdcaf35552f1ed433a981d596ca9fe900/test/registered/unit/function_call/test_function_call_parser.py#L2349) | Multiple tool calls. |
| `TestGlm47MoeDetector.test_streaming_tool_call` | `PARSER.stream.1` | [source](https://github.com/sgl-project/sglang/blob/612785ffdcaf35552f1ed433a981d596ca9fe900/test/registered/unit/function_call/test_function_call_parser.py#L2372) | Streaming tool call. |
| `TestGlm47MoeDetector.test_streaming_multiple_tool_calls` | `PARSER.batch.2.b`, `PARSER.stream.2` | [source](https://github.com/sgl-project/sglang/blob/612785ffdcaf35552f1ed433a981d596ca9fe900/test/registered/unit/function_call/test_function_call_parser.py#L2401) | Streaming multiple tool calls. |
| `TestGlm47MoeDetector.test_tool_call_id` | `PARSER.fmt.5` | [source](https://github.com/sgl-project/sglang/blob/612785ffdcaf35552f1ed433a981d596ca9fe900/test/registered/unit/function_call/test_function_call_parser.py#L2437) | Tool call id. |
| `TestGlm47MoeDetector.test_invalid_tool_call` | `PARSER.batch.4.b` | [source](https://github.com/sgl-project/sglang/blob/612785ffdcaf35552f1ed433a981d596ca9fe900/test/registered/unit/function_call/test_function_call_parser.py#L2449) | Invalid tool call. |
| `TestGlm47MoeDetector.test_partial_tool_call` | `PARSER.batch.5.c` | [source](https://github.com/sgl-project/sglang/blob/612785ffdcaf35552f1ed433a981d596ca9fe900/test/registered/unit/function_call/test_function_call_parser.py#L2455) | Partial tool call. |
| `TestGlm47MoeDetector.test_array_argument_with_escaped_json` | `PARSER.batch.7.d`, `PARSER.batch.7.b` | [source](https://github.com/sgl-project/sglang/blob/612785ffdcaf35552f1ed433a981d596ca9fe900/test/registered/unit/function_call/test_function_call_parser.py#L2485) | Array argument with escaped json. |
| `TestGlm47MoeDetector.test_whitespace_preserved_in_arg_values` | `PARSER.fmt.2` | [source](https://github.com/sgl-project/sglang/blob/612785ffdcaf35552f1ed433a981d596ca9fe900/test/registered/unit/function_call/test_function_call_parser.py#L2588) | Whitespace preserved in arg values. |

### SGLang Python - TestJsonArrayParser

| Test | Bucket(s) | Link | Notes |
| -- | -- | -- | -- |
| `TestJsonArrayParser.test_json_detector_has_no_ebnf` | `// helper` | [source](https://github.com/sgl-project/sglang/blob/612785ffdcaf35552f1ed433a981d596ca9fe900/test/registered/unit/function_call/test_function_call_parser.py#L2666) | Registry, construction, serialization, or helper-only coverage. |
| `TestJsonArrayParser.test_parse_streaming_increment_malformed_json` | `PARSER.batch.4.b`, `PARSER.batch.7.d`, `PARSER.stream.1` | [source](https://github.com/sgl-project/sglang/blob/612785ffdcaf35552f1ed433a981d596ca9fe900/test/registered/unit/function_call/test_function_call_parser.py#L2673) | Parse streaming increment malformed json. |
| `TestJsonArrayParser.test_parse_streaming_increment_empty_input` | `PARSER.batch.9`, `PARSER.batch.7.d`, `PARSER.stream.1` | [source](https://github.com/sgl-project/sglang/blob/612785ffdcaf35552f1ed433a981d596ca9fe900/test/registered/unit/function_call/test_function_call_parser.py#L2687) | Parse streaming increment empty input. |
| `TestJsonArrayParser.test_parse_streaming_increment_whitespace_handling` | `PARSER.batch.7.d`, `PARSER.fmt.2`, `PARSER.stream.1` | [source](https://github.com/sgl-project/sglang/blob/612785ffdcaf35552f1ed433a981d596ca9fe900/test/registered/unit/function_call/test_function_call_parser.py#L2693) | Parse streaming increment whitespace handling. |
| `TestJsonArrayParser.test_parse_streaming_increment_nested_objects` | `PARSER.batch.7.d`, `PARSER.stream.1` | [source](https://github.com/sgl-project/sglang/blob/612785ffdcaf35552f1ed433a981d596ca9fe900/test/registered/unit/function_call/test_function_call_parser.py#L2705) | Parse streaming increment nested objects. |
| `TestJsonArrayParser.test_json_parsing_with_commas` | `PARSER.batch.7.d`, `PARSER.batch.12` | [source](https://github.com/sgl-project/sglang/blob/612785ffdcaf35552f1ed433a981d596ca9fe900/test/registered/unit/function_call/test_function_call_parser.py#L2716) | Json parsing with commas. |
| `TestJsonArrayParser.test_braces_in_strings` | `PARSER.batch.7.d`, `PARSER.batch.12` | [source](https://github.com/sgl-project/sglang/blob/612785ffdcaf35552f1ed433a981d596ca9fe900/test/registered/unit/function_call/test_function_call_parser.py#L2736) | Braces in strings. |
| `TestJsonArrayParser.test_separator_in_same_chunk` | `PARSER.batch.7.d`, `PARSER.batch.12`, `PARSER.stream.1`, `PARSER.stream.3` | [source](https://github.com/sgl-project/sglang/blob/612785ffdcaf35552f1ed433a981d596ca9fe900/test/registered/unit/function_call/test_function_call_parser.py#L2765) | Separator in same chunk. |
| `TestJsonArrayParser.test_separator_in_separate_chunk` | `PARSER.batch.7.d`, `PARSER.batch.12`, `PARSER.stream.1`, `PARSER.stream.3` | [source](https://github.com/sgl-project/sglang/blob/612785ffdcaf35552f1ed433a981d596ca9fe900/test/registered/unit/function_call/test_function_call_parser.py#L2780) | Separator in separate chunk. |
| `TestJsonArrayParser.test_incomplete_json_across_chunks` | `PARSER.batch.5.c`, `PARSER.batch.7.d`, `PARSER.stream.1`, `PARSER.stream.3` | [source](https://github.com/sgl-project/sglang/blob/612785ffdcaf35552f1ed433a981d596ca9fe900/test/registered/unit/function_call/test_function_call_parser.py#L2799) | Incomplete json across chunks. |
| `TestJsonArrayParser.test_malformed_json_recovery` | `PARSER.batch.4.b`, `PARSER.batch.4.e`, `PARSER.batch.7.d` | [source](https://github.com/sgl-project/sglang/blob/612785ffdcaf35552f1ed433a981d596ca9fe900/test/registered/unit/function_call/test_function_call_parser.py#L2813) | Malformed json recovery. |
| `TestJsonArrayParser.test_nested_objects_with_commas` | `PARSER.batch.7.d`, `PARSER.batch.12` | [source](https://github.com/sgl-project/sglang/blob/612785ffdcaf35552f1ed433a981d596ca9fe900/test/registered/unit/function_call/test_function_call_parser.py#L2831) | Nested objects with commas. |
| `TestJsonArrayParser.test_empty_objects` | `PARSER.batch.7.d` | [source](https://github.com/sgl-project/sglang/blob/612785ffdcaf35552f1ed433a981d596ca9fe900/test/registered/unit/function_call/test_function_call_parser.py#L2844) | Empty objects. |
| `TestJsonArrayParser.test_whitespace_handling` | `PARSER.batch.7.d`, `PARSER.fmt.2` | [source](https://github.com/sgl-project/sglang/blob/612785ffdcaf35552f1ed433a981d596ca9fe900/test/registered/unit/function_call/test_function_call_parser.py#L2854) | Whitespace handling. |
| `TestJsonArrayParser.test_multiple_commas_in_chunk` | `PARSER.batch.2.b`, `PARSER.batch.7.d`, `PARSER.batch.12`, `PARSER.stream.2`, `PARSER.stream.3` | [source](https://github.com/sgl-project/sglang/blob/612785ffdcaf35552f1ed433a981d596ca9fe900/test/registered/unit/function_call/test_function_call_parser.py#L2864) | Multiple commas in chunk. |
| `TestJsonArrayParser.test_complete_tool_call_with_trailing_comma` | `PARSER.batch.5.c`, `PARSER.batch.7.d` | [source](https://github.com/sgl-project/sglang/blob/612785ffdcaf35552f1ed433a981d596ca9fe900/test/registered/unit/function_call/test_function_call_parser.py#L2888) | Complete tool call with trailing comma. |
| `TestJsonArrayParser.test_three_tool_calls_separate_chunks_with_commas` | `PARSER.batch.2.b`, `PARSER.batch.7.d`, `PARSER.batch.12`, `PARSER.stream.2`, `PARSER.stream.3` | [source](https://github.com/sgl-project/sglang/blob/612785ffdcaf35552f1ed433a981d596ca9fe900/test/registered/unit/function_call/test_function_call_parser.py#L2907) | Three tool calls separate chunks with commas. |

### SGLang Python - TestLfm2Detector

| Test | Bucket(s) | Link | Notes |
| -- | -- | -- | -- |
| `TestLfm2Detector.test_has_tool_call_true` | `PARSER.fmt.3` | [source](https://github.com/sgl-project/sglang/blob/612785ffdcaf35552f1ed433a981d596ca9fe900/test/registered/unit/function_call/test_function_call_parser.py#L3004) | Format detection / marker-recognition helper coverage. |
| `TestLfm2Detector.test_has_tool_call_false` | `PARSER.fmt.3` | [source](https://github.com/sgl-project/sglang/blob/612785ffdcaf35552f1ed433a981d596ca9fe900/test/registered/unit/function_call/test_function_call_parser.py#L3009) | Format detection / marker-recognition helper coverage. |
| `TestLfm2Detector.test_has_tool_call_partial_marker` | `PARSER.fmt.3` | [source](https://github.com/sgl-project/sglang/blob/612785ffdcaf35552f1ed433a981d596ca9fe900/test/registered/unit/function_call/test_function_call_parser.py#L3014) | Format detection / marker-recognition helper coverage. |
| `TestLfm2Detector.test_detect_and_parse_pythonic_simple` | `PARSER.batch.1` | [source](https://github.com/sgl-project/sglang/blob/612785ffdcaf35552f1ed433a981d596ca9fe900/test/registered/unit/function_call/test_function_call_parser.py#L3021) | Detect and parse pythonic simple. |
| `TestLfm2Detector.test_detect_and_parse_pythonic_multiple_args` | `PARSER.batch.1`, `PARSER.batch.2.a` | [source](https://github.com/sgl-project/sglang/blob/612785ffdcaf35552f1ed433a981d596ca9fe900/test/registered/unit/function_call/test_function_call_parser.py#L3033) | Detect and parse pythonic multiple args. |
| `TestLfm2Detector.test_detect_and_parse_pythonic_no_args` | `PARSER.batch.1` | [source](https://github.com/sgl-project/sglang/blob/612785ffdcaf35552f1ed433a981d596ca9fe900/test/registered/unit/function_call/test_function_call_parser.py#L3045) | Detect and parse pythonic no args. |
| `TestLfm2Detector.test_detect_and_parse_pythonic_multiple_calls` | `PARSER.batch.1`, `PARSER.batch.2.a` | [source](https://github.com/sgl-project/sglang/blob/612785ffdcaf35552f1ed433a981d596ca9fe900/test/registered/unit/function_call/test_function_call_parser.py#L3064) | Detect and parse pythonic multiple calls. |
| `TestLfm2Detector.test_detect_and_parse_with_normal_text_before` | `PARSER.batch.1`, `PARSER.batch.8.a` | [source](https://github.com/sgl-project/sglang/blob/612785ffdcaf35552f1ed433a981d596ca9fe900/test/registered/unit/function_call/test_function_call_parser.py#L3078) | Detect and parse with normal text before. |
| `TestLfm2Detector.test_detect_and_parse_special_characters_in_value` | `PARSER.batch.1`, `PARSER.batch.7.b` | [source](https://github.com/sgl-project/sglang/blob/612785ffdcaf35552f1ed433a981d596ca9fe900/test/registered/unit/function_call/test_function_call_parser.py#L3087) | Detect and parse special characters in value. |
| `TestLfm2Detector.test_detect_and_parse_numeric_values` | `PARSER.batch.1`, `PARSER.batch.7.c` | [source](https://github.com/sgl-project/sglang/blob/612785ffdcaf35552f1ed433a981d596ca9fe900/test/registered/unit/function_call/test_function_call_parser.py#L3098) | Detect and parse numeric values. |
| `TestLfm2Detector.test_detect_and_parse_json_simple` | `PARSER.batch.1` | [source](https://github.com/sgl-project/sglang/blob/612785ffdcaf35552f1ed433a981d596ca9fe900/test/registered/unit/function_call/test_function_call_parser.py#L3108) | Detect and parse json simple. |
| `TestLfm2Detector.test_detect_and_parse_json_multiple_calls` | `PARSER.batch.1`, `PARSER.batch.2.a` | [source](https://github.com/sgl-project/sglang/blob/612785ffdcaf35552f1ed433a981d596ca9fe900/test/registered/unit/function_call/test_function_call_parser.py#L3119) | Detect and parse json multiple calls. |
| `TestLfm2Detector.test_detect_and_parse_json_with_parameters_key` | `PARSER.batch.1`, `PARSER.batch.7.a`, `PARSER.fmt.5` | [source](https://github.com/sgl-project/sglang/blob/612785ffdcaf35552f1ed433a981d596ca9fe900/test/registered/unit/function_call/test_function_call_parser.py#L3128) | Detect and parse json with parameters key. |
| `TestLfm2Detector.test_detect_and_parse_no_tool_call` | `PARSER.batch.3`, `PARSER.batch.1` | [source](https://github.com/sgl-project/sglang/blob/612785ffdcaf35552f1ed433a981d596ca9fe900/test/registered/unit/function_call/test_function_call_parser.py#L3139) | Detect and parse no tool call. |
| `TestLfm2Detector.test_detect_and_parse_unknown_function` | `PARSER.batch.1`, `PARSER.batch.13` | [source](https://github.com/sgl-project/sglang/blob/612785ffdcaf35552f1ed433a981d596ca9fe900/test/registered/unit/function_call/test_function_call_parser.py#L3147) | Detect and parse unknown function. |
| `TestLfm2Detector.test_detect_and_parse_empty_content` | `PARSER.batch.9`, `PARSER.batch.1` | [source](https://github.com/sgl-project/sglang/blob/612785ffdcaf35552f1ed433a981d596ca9fe900/test/registered/unit/function_call/test_function_call_parser.py#L3155) | Detect and parse empty content. |
| `TestLfm2Detector.test_detect_and_parse_multiple_blocks` | `PARSER.batch.1`, `PARSER.batch.2.a` | [source](https://github.com/sgl-project/sglang/blob/612785ffdcaf35552f1ed433a981d596ca9fe900/test/registered/unit/function_call/test_function_call_parser.py#L3162) | Detect and parse multiple blocks. |
| `TestLfm2Detector.test_streaming_json_complete_in_one_chunk` | `PARSER.stream.1`, `PARSER.stream.3` | [source](https://github.com/sgl-project/sglang/blob/612785ffdcaf35552f1ed433a981d596ca9fe900/test/registered/unit/function_call/test_function_call_parser.py#L3176) | Streaming json complete in one chunk. |
| `TestLfm2Detector.test_streaming_json_split_across_chunks` | `PARSER.stream.1`, `PARSER.stream.3` | [source](https://github.com/sgl-project/sglang/blob/612785ffdcaf35552f1ed433a981d596ca9fe900/test/registered/unit/function_call/test_function_call_parser.py#L3184) | Streaming json split across chunks. |
| `TestLfm2Detector.test_streaming_json_normal_text_before_tool_call` | `PARSER.batch.8.a`, `PARSER.stream.1` | [source](https://github.com/sgl-project/sglang/blob/612785ffdcaf35552f1ed433a981d596ca9fe900/test/registered/unit/function_call/test_function_call_parser.py#L3205) | Streaming json normal text before tool call. |
| `TestLfm2Detector.test_streaming_eot_token_filtering` | `PARSER.fmt.3`, `PARSER.stream.1`, `PARSER.stream.4`, `PIPELINE.finish_reason` | [source](https://github.com/sgl-project/sglang/blob/612785ffdcaf35552f1ed433a981d596ca9fe900/test/registered/unit/function_call/test_function_call_parser.py#L3221) | Streaming eot token filtering. |
| `TestLfm2Detector.test_streaming_pythonic_complete_in_one_chunk` | `PARSER.stream.1`, `PARSER.stream.3` | [source](https://github.com/sgl-project/sglang/blob/612785ffdcaf35552f1ed433a981d596ca9fe900/test/registered/unit/function_call/test_function_call_parser.py#L3235) | Streaming pythonic complete in one chunk. |
| `TestLfm2Detector.test_streaming_pythonic_split_across_chunks` | `PARSER.stream.1`, `PARSER.stream.3` | [source](https://github.com/sgl-project/sglang/blob/612785ffdcaf35552f1ed433a981d596ca9fe900/test/registered/unit/function_call/test_function_call_parser.py#L3245) | Streaming pythonic split across chunks. |
| `TestLfm2Detector.test_streaming_pythonic_multiple_calls` | `PARSER.batch.2.b`, `PARSER.stream.2` | [source](https://github.com/sgl-project/sglang/blob/612785ffdcaf35552f1ed433a981d596ca9fe900/test/registered/unit/function_call/test_function_call_parser.py#L3265) | Streaming pythonic multiple calls. |
| `TestLfm2Detector.test_supports_structural_tag` | `// helper` | [source](https://github.com/sgl-project/sglang/blob/612785ffdcaf35552f1ed433a981d596ca9fe900/test/registered/unit/function_call/test_function_call_parser.py#L3278) | Registry, construction, serialization, or helper-only coverage. |
| `TestLfm2Detector.test_structure_info` | `// helper` | [source](https://github.com/sgl-project/sglang/blob/612785ffdcaf35552f1ed433a981d596ca9fe900/test/registered/unit/function_call/test_function_call_parser.py#L3284) | Registry, construction, serialization, or helper-only coverage. |

### SGLang Python - TestGigaChat3Detector

| Test | Bucket(s) | Link | Notes |
| -- | -- | -- | -- |
| `TestGigaChat3Detector.test_has_tool_call` | `PARSER.fmt.3` | [source](https://github.com/sgl-project/sglang/blob/612785ffdcaf35552f1ed433a981d596ca9fe900/test/registered/unit/function_call/test_function_call_parser.py#L3346) | Format detection / marker-recognition helper coverage. |
| `TestGigaChat3Detector.test_detect_and_parse_no_tool_call` | `PARSER.batch.3`, `PARSER.batch.1` | [source](https://github.com/sgl-project/sglang/blob/612785ffdcaf35552f1ed433a981d596ca9fe900/test/registered/unit/function_call/test_function_call_parser.py#L3352) | Detect and parse no tool call. |
| `TestGigaChat3Detector.test_detect_and_parse_simple_tool_call` | `PARSER.batch.1` | [source](https://github.com/sgl-project/sglang/blob/612785ffdcaf35552f1ed433a981d596ca9fe900/test/registered/unit/function_call/test_function_call_parser.py#L3360) | Detect and parse simple tool call. |
| `TestGigaChat3Detector.test_detect_and_parse_parameterless_tool_call` | `PARSER.batch.1`, `PARSER.batch.6.c` | [source](https://github.com/sgl-project/sglang/blob/612785ffdcaf35552f1ed433a981d596ca9fe900/test/registered/unit/function_call/test_function_call_parser.py#L3374) | Detect and parse parameterless tool call. |
| `TestGigaChat3Detector.test_detect_and_parse_complex_tool_call` | `PARSER.batch.1`, `PARSER.batch.7.d` | [source](https://github.com/sgl-project/sglang/blob/612785ffdcaf35552f1ed433a981d596ca9fe900/test/registered/unit/function_call/test_function_call_parser.py#L3386) | Detect and parse complex tool call. |
| `TestGigaChat3Detector.test_detect_and_parse_with_content_before` | `PARSER.batch.1`, `PARSER.batch.8.a` | [source](https://github.com/sgl-project/sglang/blob/612785ffdcaf35552f1ed433a981d596ca9fe900/test/registered/unit/function_call/test_function_call_parser.py#L3406) | Detect and parse with content before. |
| `TestGigaChat3Detector.test_detect_and_parse_with_eos_token` | `PARSER.batch.1`, `PARSER.fmt.3`, `PIPELINE.finish_reason` | [source](https://github.com/sgl-project/sglang/blob/612785ffdcaf35552f1ed433a981d596ca9fe900/test/registered/unit/function_call/test_function_call_parser.py#L3415) | Detect and parse with eos token. |
| `TestGigaChat3Detector.test_detect_and_parse_with_content_and_eos` | `PARSER.batch.1`, `PARSER.batch.8.b`, `PIPELINE.finish_reason` | [source](https://github.com/sgl-project/sglang/blob/612785ffdcaf35552f1ed433a981d596ca9fe900/test/registered/unit/function_call/test_function_call_parser.py#L3428) | Detect and parse with content and eos. |
| `TestGigaChat3Detector.test_detect_and_parse_invalid_json` | `PARSER.batch.1`, `PARSER.batch.4.b` | [source](https://github.com/sgl-project/sglang/blob/612785ffdcaf35552f1ed433a981d596ca9fe900/test/registered/unit/function_call/test_function_call_parser.py#L3437) | Detect and parse invalid json. |
| `TestGigaChat3Detector.test_detect_and_parse_missing_name` | `PARSER.batch.1`, `PARSER.batch.4.c` | [source](https://github.com/sgl-project/sglang/blob/612785ffdcaf35552f1ed433a981d596ca9fe900/test/registered/unit/function_call/test_function_call_parser.py#L3446) | Detect and parse missing name. |
| `TestGigaChat3Detector.test_detect_and_parse_missing_arguments` | `PARSER.batch.1`, `PARSER.batch.4.c`, `PARSER.batch.7.a` | [source](https://github.com/sgl-project/sglang/blob/612785ffdcaf35552f1ed433a981d596ca9fe900/test/registered/unit/function_call/test_function_call_parser.py#L3454) | Detect and parse missing arguments. |
| `TestGigaChat3Detector.test_detect_and_parse_arguments_not_dict` | `PARSER.batch.1`, `PARSER.batch.4.b`, `PARSER.batch.7.d` | [source](https://github.com/sgl-project/sglang/blob/612785ffdcaf35552f1ed433a981d596ca9fe900/test/registered/unit/function_call/test_function_call_parser.py#L3462) | Detect and parse arguments not dict. |
| `TestGigaChat3Detector.test_streaming_no_tool_call` | `PARSER.batch.3`, `PARSER.stream.1` | [source](https://github.com/sgl-project/sglang/blob/612785ffdcaf35552f1ed433a981d596ca9fe900/test/registered/unit/function_call/test_function_call_parser.py#L3470) | Streaming no tool call. |
| `TestGigaChat3Detector.test_streaming_simple_tool_call` | `PARSER.batch.1`, `PARSER.stream.1` | [source](https://github.com/sgl-project/sglang/blob/612785ffdcaf35552f1ed433a981d596ca9fe900/test/registered/unit/function_call/test_function_call_parser.py#L3482) | Streaming simple tool call. |
| `TestGigaChat3Detector.test_streaming_with_content_before` | `PARSER.batch.8.a`, `PARSER.stream.1` | [source](https://github.com/sgl-project/sglang/blob/612785ffdcaf35552f1ed433a981d596ca9fe900/test/registered/unit/function_call/test_function_call_parser.py#L3519) | Streaming with content before. |
| `TestGigaChat3Detector.test_streaming_complex_arguments` | `PARSER.batch.7.d`, `PARSER.stream.1` | [source](https://github.com/sgl-project/sglang/blob/612785ffdcaf35552f1ed433a981d596ca9fe900/test/registered/unit/function_call/test_function_call_parser.py#L3561) | Streaming complex arguments. |
| `TestGigaChat3Detector.test_streaming_with_eos_token` | `PARSER.fmt.3`, `PARSER.stream.1`, `PARSER.stream.4`, `PIPELINE.finish_reason` | [source](https://github.com/sgl-project/sglang/blob/612785ffdcaf35552f1ed433a981d596ca9fe900/test/registered/unit/function_call/test_function_call_parser.py#L3602) | Streaming with eos token. |
| `TestGigaChat3Detector.test_streaming_incomplete_json` | `PARSER.batch.5.c`, `PARSER.stream.1` | [source](https://github.com/sgl-project/sglang/blob/612785ffdcaf35552f1ed433a981d596ca9fe900/test/registered/unit/function_call/test_function_call_parser.py#L3639) | Streaming incomplete json. |
| `TestGigaChat3Detector.test_streaming_large_steps` | `PARSER.stream.1` | [source](https://github.com/sgl-project/sglang/blob/612785ffdcaf35552f1ed433a981d596ca9fe900/test/registered/unit/function_call/test_function_call_parser.py#L3675) | Streaming large steps. |
| `TestGigaChat3Detector.test_streaming_very_small_chunks` | `PARSER.stream.1`, `PARSER.stream.3` | [source](https://github.com/sgl-project/sglang/blob/612785ffdcaf35552f1ed433a981d596ca9fe900/test/registered/unit/function_call/test_function_call_parser.py#L3717) | Streaming very small chunks. |
| `TestGigaChat3Detector.test_streaming_json_split_at_quotes` | `PARSER.batch.7.b`, `PARSER.stream.1`, `PARSER.stream.3` | [source](https://github.com/sgl-project/sglang/blob/612785ffdcaf35552f1ed433a981d596ca9fe900/test/registered/unit/function_call/test_function_call_parser.py#L3759) | Streaming json split at quotes. |
| `TestGigaChat3Detector.test_detect_and_parse_function_call_marker_simple_tool_call` | `PARSER.batch.1`, `PARSER.fmt.3` | [source](https://github.com/sgl-project/sglang/blob/612785ffdcaf35552f1ed433a981d596ca9fe900/test/registered/unit/function_call/test_function_call_parser.py#L3799) | Detect and parse function call marker simple tool call. |
| `TestGigaChat3Detector.test_detect_and_parse_function_call_marker_with_content_before` | `PARSER.batch.1`, `PARSER.batch.8.a`, `PARSER.fmt.3` | [source](https://github.com/sgl-project/sglang/blob/612785ffdcaf35552f1ed433a981d596ca9fe900/test/registered/unit/function_call/test_function_call_parser.py#L3812) | Detect and parse function call marker with content before. |
| `TestGigaChat3Detector.test_detect_and_parse_function_call_marker_with_eos_token` | `PARSER.batch.1`, `PARSER.fmt.3`, `PIPELINE.finish_reason` | [source](https://github.com/sgl-project/sglang/blob/612785ffdcaf35552f1ed433a981d596ca9fe900/test/registered/unit/function_call/test_function_call_parser.py#L3827) | Detect and parse function call marker with eos token. |
| `TestGigaChat3Detector.test_detect_and_parse_function_call_marker_invalid_json` | `PARSER.batch.1`, `PARSER.batch.4.b`, `PARSER.fmt.3` | [source](https://github.com/sgl-project/sglang/blob/612785ffdcaf35552f1ed433a981d596ca9fe900/test/registered/unit/function_call/test_function_call_parser.py#L3840) | Detect and parse function call marker invalid json. |
| `TestGigaChat3Detector.test_streaming_function_call_marker_simple_tool_call` | `PARSER.batch.1`, `PARSER.fmt.3`, `PARSER.stream.1` | [source](https://github.com/sgl-project/sglang/blob/612785ffdcaf35552f1ed433a981d596ca9fe900/test/registered/unit/function_call/test_function_call_parser.py#L3848) | Streaming function call marker simple tool call. |
| `TestGigaChat3Detector.test_streaming_function_call_marker_json_split_at_quotes` | `PARSER.batch.7.b`, `PARSER.fmt.3`, `PARSER.stream.1`, `PARSER.stream.3` | [source](https://github.com/sgl-project/sglang/blob/612785ffdcaf35552f1ed433a981d596ca9fe900/test/registered/unit/function_call/test_function_call_parser.py#L3887) | Streaming function call marker json split at quotes. |

### SGLang Python - TestGetStructureConstraint

| Test | Bucket(s) | Link | Notes |
| -- | -- | -- | -- |
| `TestGetStructureConstraint.test_kimi_required_strict_returns_structural_tag` | `FRONTEND.3/.6`, `FRONTEND.3` | [source](https://github.com/sgl-project/sglang/blob/612785ffdcaf35552f1ed433a981d596ca9fe900/test/registered/unit/function_call/test_function_call_parser.py#L3959) | Request-time structured-output / tool-choice constraint selection; not parser extraction. |
| `TestGetStructureConstraint.test_kimi_required_no_strict_returns_structural_tag` | `FRONTEND.3/.6`, `FRONTEND.3` | [source](https://github.com/sgl-project/sglang/blob/612785ffdcaf35552f1ed433a981d596ca9fe900/test/registered/unit/function_call/test_function_call_parser.py#L3966) | Request-time structured-output / tool-choice constraint selection; not parser extraction. |
| `TestGetStructureConstraint.test_kimi_auto_strict_returns_structural_tag` | `FRONTEND.3/.6` | [source](https://github.com/sgl-project/sglang/blob/612785ffdcaf35552f1ed433a981d596ca9fe900/test/registered/unit/function_call/test_function_call_parser.py#L3974) | Request-time structured-output / tool-choice constraint selection; not parser extraction. |
| `TestGetStructureConstraint.test_kimi_auto_no_strict_returns_none` | `FRONTEND.3/.6` | [source](https://github.com/sgl-project/sglang/blob/612785ffdcaf35552f1ed433a981d596ca9fe900/test/registered/unit/function_call/test_function_call_parser.py#L3981) | Request-time structured-output / tool-choice constraint selection; not parser extraction. |
| `TestGetStructureConstraint.test_kimi_named_tool_choice_returns_structural_tag` | `FRONTEND.3/.6`, `FRONTEND.3` | [source](https://github.com/sgl-project/sglang/blob/612785ffdcaf35552f1ed433a981d596ca9fe900/test/registered/unit/function_call/test_function_call_parser.py#L3987) | Request-time structured-output / tool-choice constraint selection; not parser extraction. |
| `TestGetStructureConstraint.test_deepseekv3_required_no_strict_returns_structural_tag` | `FRONTEND.3/.6`, `FRONTEND.3` | [source](https://github.com/sgl-project/sglang/blob/612785ffdcaf35552f1ed433a981d596ca9fe900/test/registered/unit/function_call/test_function_call_parser.py#L3999) | Request-time structured-output / tool-choice constraint selection; not parser extraction. |
| `TestGetStructureConstraint.test_qwen25_required_no_strict_returns_structural_tag` | `FRONTEND.3/.6`, `FRONTEND.3` | [source](https://github.com/sgl-project/sglang/blob/612785ffdcaf35552f1ed433a981d596ca9fe900/test/registered/unit/function_call/test_function_call_parser.py#L4005) | Request-time structured-output / tool-choice constraint selection; not parser extraction. |
| `TestGetStructureConstraint.test_kimi_structural_tag_has_kimi_tokens` | `FRONTEND.3/.6` | [source](https://github.com/sgl-project/sglang/blob/612785ffdcaf35552f1ed433a981d596ca9fe900/test/registered/unit/function_call/test_function_call_parser.py#L4013) | Request-time structured-output / tool-choice constraint selection; not parser extraction. |
| `TestGetStructureConstraint.test_kimi_required_no_strict_uses_empty_schema` | `FRONTEND.3/.6`, `FRONTEND.3` | [source](https://github.com/sgl-project/sglang/blob/612785ffdcaf35552f1ed433a981d596ca9fe900/test/registered/unit/function_call/test_function_call_parser.py#L4023) | Request-time structured-output / tool-choice constraint selection; not parser extraction. |
| `TestGetStructureConstraint.test_kimi_required_strict_uses_tool_schema` | `FRONTEND.3/.6`, `FRONTEND.3` | [source](https://github.com/sgl-project/sglang/blob/612785ffdcaf35552f1ed433a981d596ca9fe900/test/registered/unit/function_call/test_function_call_parser.py#L4031) | Request-time structured-output / tool-choice constraint selection; not parser extraction. |

### SGLang Python - TestQwen25Detector

| Test | Bucket(s) | Link | Notes |
| -- | -- | -- | -- |
| `TestQwen25Detector.test_detect_and_parse_single_tool_call` | `PARSER.batch.1` | [source](https://github.com/sgl-project/sglang/blob/612785ffdcaf35552f1ed433a981d596ca9fe900/test/registered/unit/function_call/test_function_call_parser.py#L4078) | Detect and parse single tool call. |
| `TestQwen25Detector.test_detect_and_parse_multiple_tool_calls` | `PARSER.batch.1`, `PARSER.batch.2.a` | [source](https://github.com/sgl-project/sglang/blob/612785ffdcaf35552f1ed433a981d596ca9fe900/test/registered/unit/function_call/test_function_call_parser.py#L4086) | Detect and parse multiple tool calls. |
| `TestQwen25Detector.test_detect_and_parse_with_normal_text_prefix` | `PARSER.batch.1` | [source](https://github.com/sgl-project/sglang/blob/612785ffdcaf35552f1ed433a981d596ca9fe900/test/registered/unit/function_call/test_function_call_parser.py#L4098) | Detect and parse with normal text prefix. |
| `TestQwen25Detector.test_streaming_single_tool_call` | `PARSER.batch.1`, `PARSER.stream.1` | [source](https://github.com/sgl-project/sglang/blob/612785ffdcaf35552f1ed433a981d596ca9fe900/test/registered/unit/function_call/test_function_call_parser.py#L4129) | Streaming single tool call. |
| `TestQwen25Detector.test_streaming_multiple_tool_calls` | `PARSER.batch.2.b`, `PARSER.stream.2` | [source](https://github.com/sgl-project/sglang/blob/612785ffdcaf35552f1ed433a981d596ca9fe900/test/registered/unit/function_call/test_function_call_parser.py#L4144) | Streaming multiple tool calls. |
| `TestQwen25Detector.test_streaming_multiple_tool_calls_fused_chunks` | `PARSER.batch.2.b`, `PARSER.stream.2`, `PARSER.stream.3` | [source](https://github.com/sgl-project/sglang/blob/612785ffdcaf35552f1ed433a981d596ca9fe900/test/registered/unit/function_call/test_function_call_parser.py#L4165) | Streaming multiple tool calls fused chunks. |
| `TestQwen25Detector.test_streaming_multiple_tool_calls_char_by_char_separator` | `PARSER.batch.2.b`, `PARSER.batch.12`, `PARSER.stream.2`, `PARSER.stream.3` | [source](https://github.com/sgl-project/sglang/blob/612785ffdcaf35552f1ed433a981d596ca9fe900/test/registered/unit/function_call/test_function_call_parser.py#L4178) | Streaming multiple tool calls char by char separator. |

### SGLang Python - TestGemma4Detector

| Test | Bucket(s) | Link | Notes |
| -- | -- | -- | -- |
| `TestGemma4Detector.test_detect_and_parse` | `PARSER.batch.1` | [source](https://github.com/sgl-project/sglang/blob/612785ffdcaf35552f1ed433a981d596ca9fe900/test/registered/unit/function_call/test_function_call_parser.py#L4220) | Detect and parse. |
| `TestGemma4Detector.test_parse_streaming_increment` | `PARSER.stream.1` | [source](https://github.com/sgl-project/sglang/blob/612785ffdcaf35552f1ed433a981d596ca9fe900/test/registered/unit/function_call/test_function_call_parser.py#L4231) | Parse streaming increment. |
| `TestGemma4Detector.test_nested_array_streaming` | `PARSER.batch.7.d`, `PARSER.stream.1` | [source](https://github.com/sgl-project/sglang/blob/612785ffdcaf35552f1ed433a981d596ca9fe900/test/registered/unit/function_call/test_function_call_parser.py#L4263) | Nested array streaming. |
| `TestGemma4Detector.test_has_tool_call` | `PARSER.fmt.3` | [source](https://github.com/sgl-project/sglang/blob/612785ffdcaf35552f1ed433a981d596ca9fe900/test/registered/unit/function_call/test_function_call_parser.py#L4291) | Format detection / marker-recognition helper coverage. |
| `TestGemma4Detector.test_detect_and_parse_no_tool_call` | `PARSER.batch.3`, `PARSER.batch.1` | [source](https://github.com/sgl-project/sglang/blob/612785ffdcaf35552f1ed433a981d596ca9fe900/test/registered/unit/function_call/test_function_call_parser.py#L4299) | Detect and parse no tool call. |
| `TestGemma4Detector.test_detect_and_parse_tool_index` | `PARSER.batch.1`, `PARSER.fmt.5` | [source](https://github.com/sgl-project/sglang/blob/612785ffdcaf35552f1ed433a981d596ca9fe900/test/registered/unit/function_call/test_function_call_parser.py#L4305) | Detect and parse tool index. |
| `TestGemma4Detector.test_detect_and_parse_unknown_tool_index` | `PARSER.batch.1`, `PARSER.batch.13`, `PARSER.fmt.5` | [source](https://github.com/sgl-project/sglang/blob/612785ffdcaf35552f1ed433a981d596ca9fe900/test/registered/unit/function_call/test_function_call_parser.py#L4312) | Detect and parse unknown tool index. |
| `TestGemma4Detector.test_detect_and_parse_nested_object` | `PARSER.batch.1`, `PARSER.batch.7.d` | [source](https://github.com/sgl-project/sglang/blob/612785ffdcaf35552f1ed433a981d596ca9fe900/test/registered/unit/function_call/test_function_call_parser.py#L4318) | Detect and parse nested object. |
| `TestGemma4Detector.test_detect_and_parse_multiple_calls` | `PARSER.batch.1`, `PARSER.batch.2.a` | [source](https://github.com/sgl-project/sglang/blob/612785ffdcaf35552f1ed433a981d596ca9fe900/test/registered/unit/function_call/test_function_call_parser.py#L4328) | Detect and parse multiple calls. |
| `TestGemma4Detector.test_parse_gemma4_args_empty` | `// helper` | [source](https://github.com/sgl-project/sglang/blob/612785ffdcaf35552f1ed433a981d596ca9fe900/test/registered/unit/function_call/test_function_call_parser.py#L4352) | Unclassified helper or parser-adjacent smoke test. |
| `TestGemma4Detector.test_parse_gemma4_args_booleans` | `PARSER.batch.7.c` | [source](https://github.com/sgl-project/sglang/blob/612785ffdcaf35552f1ed433a981d596ca9fe900/test/registered/unit/function_call/test_function_call_parser.py#L4356) | Parse gemma4 args booleans. |
| `TestGemma4Detector.test_parse_gemma4_args_numbers` | `// helper` | [source](https://github.com/sgl-project/sglang/blob/612785ffdcaf35552f1ed433a981d596ca9fe900/test/registered/unit/function_call/test_function_call_parser.py#L4361) | Unclassified helper or parser-adjacent smoke test. |
| `TestGemma4Detector.test_parse_gemma4_args_string_with_colon` | `// helper` | [source](https://github.com/sgl-project/sglang/blob/612785ffdcaf35552f1ed433a981d596ca9fe900/test/registered/unit/function_call/test_function_call_parser.py#L4366) | Unclassified helper or parser-adjacent smoke test. |
| `TestGemma4Detector.test_parse_gemma4_args_nested_object` | `PARSER.batch.7.d` | [source](https://github.com/sgl-project/sglang/blob/612785ffdcaf35552f1ed433a981d596ca9fe900/test/registered/unit/function_call/test_function_call_parser.py#L4370) | Parse gemma4 args nested object. |
| `TestGemma4Detector.test_parse_gemma4_array_mixed_types` | `PARSER.batch.7.d` | [source](https://github.com/sgl-project/sglang/blob/612785ffdcaf35552f1ed433a981d596ca9fe900/test/registered/unit/function_call/test_function_call_parser.py#L4376) | Parse gemma4 array mixed types. |
| `TestGemma4Detector.test_parse_gemma4_value_types` | `// helper` | [source](https://github.com/sgl-project/sglang/blob/612785ffdcaf35552f1ed433a981d596ca9fe900/test/registered/unit/function_call/test_function_call_parser.py#L4384) | Unclassified helper or parser-adjacent smoke test. |
| `TestGemma4Detector.test_streaming_multiple_tool_calls` | `PARSER.batch.2.b`, `PARSER.stream.2` | [source](https://github.com/sgl-project/sglang/blob/612785ffdcaf35552f1ed433a981d596ca9fe900/test/registered/unit/function_call/test_function_call_parser.py#L4414) | Streaming multiple tool calls. |
| `TestGemma4Detector.test_streaming_very_small_chunks` | `PARSER.stream.1`, `PARSER.stream.3` | [source](https://github.com/sgl-project/sglang/blob/612785ffdcaf35552f1ed433a981d596ca9fe900/test/registered/unit/function_call/test_function_call_parser.py#L4462) | Streaming very small chunks. |
| `TestGemma4Detector.test_streaming_empty_args` | `PARSER.batch.6.a`, `PARSER.stream.1` | [source](https://github.com/sgl-project/sglang/blob/612785ffdcaf35552f1ed433a981d596ca9fe900/test/registered/unit/function_call/test_function_call_parser.py#L4474) | Streaming empty args. |
| `TestGemma4Detector.test_streaming_text_between_tool_calls` | `PARSER.batch.8.d`, `PARSER.stream.1` | [source](https://github.com/sgl-project/sglang/blob/612785ffdcaf35552f1ed433a981d596ca9fe900/test/registered/unit/function_call/test_function_call_parser.py#L4481) | Streaming text between tool calls. |

### LightSeek Rust tool-parser - Minimax M2

| Test | Bucket(s) | Link | Notes |
| -- | -- | -- | -- |
| `test_minimax_complete_parsing` | `PARSER.batch.1` | [source](https://github.com/lightseekorg/smg/blob/e3eccacf96bc6e041a7ec6623e2251dba1129f28/crates/tool_parser/tests/tool_parser_minimax_m2.rs#L8) | Minimax complete parsing. |
| `test_minimax_multiple_tools` | `PARSER.batch.2.a` | [source](https://github.com/lightseekorg/smg/blob/e3eccacf96bc6e041a7ec6623e2251dba1129f28/crates/tool_parser/tests/tool_parser_minimax_m2.rs#L31) | Minimax multiple tools. |
| `test_minimax_type_conversion` | `PARSER.batch.7.c` | [source](https://github.com/lightseekorg/smg/blob/e3eccacf96bc6e041a7ec6623e2251dba1129f28/crates/tool_parser/tests/tool_parser_minimax_m2.rs#L54) | Minimax type conversion. |
| `test_minimax_streaming_basic` | `PARSER.stream.1` | [source](https://github.com/lightseekorg/smg/blob/e3eccacf96bc6e041a7ec6623e2251dba1129f28/crates/tool_parser/tests/tool_parser_minimax_m2.rs#L80) | Minimax streaming basic. |
| `test_minimax_format_detection` | `PARSER.fmt.3` | [source](https://github.com/lightseekorg/smg/blob/e3eccacf96bc6e041a7ec6623e2251dba1129f28/crates/tool_parser/tests/tool_parser_minimax_m2.rs#L117) | Format detection / marker-recognition helper coverage. |
| `test_minimax_python_literals` | `PARSER.batch.7.c` | [source](https://github.com/lightseekorg/smg/blob/e3eccacf96bc6e041a7ec6623e2251dba1129f28/crates/tool_parser/tests/tool_parser_minimax_m2.rs#L132) | Minimax python literals. |
| `test_minimax_nested_json_in_parameters` | `PARSER.batch.7.d` | [source](https://github.com/lightseekorg/smg/blob/e3eccacf96bc6e041a7ec6623e2251dba1129f28/crates/tool_parser/tests/tool_parser_minimax_m2.rs#L154) | Minimax nested json in parameters. |
| `test_minimax_xml_entities` | `PARSER.xml.1` | [source](https://github.com/lightseekorg/smg/blob/e3eccacf96bc6e041a7ec6623e2251dba1129f28/crates/tool_parser/tests/tool_parser_minimax_m2.rs#L177) | Minimax xml entities. |
| `test_minimax_streaming_partial_tags` | `PARSER.batch.5.a`, `PARSER.stream.1`, `PARSER.stream.3` | [source](https://github.com/lightseekorg/smg/blob/e3eccacf96bc6e041a7ec6623e2251dba1129f28/crates/tool_parser/tests/tool_parser_minimax_m2.rs#L198) | Minimax streaming partial tags. |
| `test_minimax_streaming_incremental_json` | `PARSER.stream.1` | [source](https://github.com/lightseekorg/smg/blob/e3eccacf96bc6e041a7ec6623e2251dba1129f28/crates/tool_parser/tests/tool_parser_minimax_m2.rs#L236) | Minimax streaming incremental json. |
| `test_minimax_multiple_tools_boundary` | `PARSER.batch.2.a`, `PARSER.stream.2`, `PARSER.stream.3` | [source](https://github.com/lightseekorg/smg/blob/e3eccacf96bc6e041a7ec6623e2251dba1129f28/crates/tool_parser/tests/tool_parser_minimax_m2.rs#L287) | Minimax multiple tools boundary. |
| `test_minimax_invalid_function_name` | `PARSER.batch.4.b`, `PARSER.fmt.1` | [source](https://github.com/lightseekorg/smg/blob/e3eccacf96bc6e041a7ec6623e2251dba1129f28/crates/tool_parser/tests/tool_parser_minimax_m2.rs#L315) | Minimax invalid function name. |
| `test_minimax_empty_parameters` | `PARSER.batch.6.a` | [source](https://github.com/lightseekorg/smg/blob/e3eccacf96bc6e041a7ec6623e2251dba1129f28/crates/tool_parser/tests/tool_parser_minimax_m2.rs#L345) | Minimax empty parameters. |
| `test_minimax_multiline_parameter_values` | `PARSER.batch.7.a`, `PARSER.fmt.2` | [source](https://github.com/lightseekorg/smg/blob/e3eccacf96bc6e041a7ec6623e2251dba1129f28/crates/tool_parser/tests/tool_parser_minimax_m2.rs#L362) | Minimax multiline parameter values. |
| `test_minimax_nested_xml_like_content` | `PARSER.batch.7.d`, `PARSER.batch.7.b` | [source](https://github.com/lightseekorg/smg/blob/e3eccacf96bc6e041a7ec6623e2251dba1129f28/crates/tool_parser/tests/tool_parser_minimax_m2.rs#L383) | Minimax nested xml like content. |
| `test_minimax_streaming_state_reset` | `PARSER.stream.1`, `PARSER.stream.4` | [source](https://github.com/lightseekorg/smg/blob/e3eccacf96bc6e041a7ec6623e2251dba1129f28/crates/tool_parser/tests/tool_parser_minimax_m2.rs#L406) | Minimax streaming state reset. |
| `test_minimax_many_parameters` | `PARSER.batch.2.a`, `PARSER.batch.7.d` | [source](https://github.com/lightseekorg/smg/blob/e3eccacf96bc6e041a7ec6623e2251dba1129f28/crates/tool_parser/tests/tool_parser_minimax_m2.rs#L442) | Minimax many parameters. |
| `test_minimax_character_by_character_streaming` | `PARSER.stream.1` | [source](https://github.com/lightseekorg/smg/blob/e3eccacf96bc6e041a7ec6623e2251dba1129f28/crates/tool_parser/tests/tool_parser_minimax_m2.rs#L476) | Minimax character by character streaming. |
| `test_minimax_content_before_and_after_tool_calls` | `PARSER.batch.8.a`, `PARSER.batch.8.c`, `PARSER.batch.8.d` | [source](https://github.com/lightseekorg/smg/blob/e3eccacf96bc6e041a7ec6623e2251dba1129f28/crates/tool_parser/tests/tool_parser_minimax_m2.rs#L524) | Minimax content before and after tool calls. |
| `test_minimax_incomplete_tool_call` | `PARSER.batch.5.c` | [source](https://github.com/lightseekorg/smg/blob/e3eccacf96bc6e041a7ec6623e2251dba1129f28/crates/tool_parser/tests/tool_parser_minimax_m2.rs#L553) | Minimax incomplete tool call. |
| `test_minimax_malformed_invoke_tag` | `PARSER.batch.4.d` | [source](https://github.com/lightseekorg/smg/blob/e3eccacf96bc6e041a7ec6623e2251dba1129f28/crates/tool_parser/tests/tool_parser_minimax_m2.rs#L569) | Minimax malformed invoke tag. |
| `test_minimax_streaming_with_invalid_function_progressive` | `PARSER.batch.4.b`, `PARSER.stream.1` | [source](https://github.com/lightseekorg/smg/blob/e3eccacf96bc6e041a7ec6623e2251dba1129f28/crates/tool_parser/tests/tool_parser_minimax_m2.rs#L587) | Minimax streaming with invalid function progressive. |
| `test_minimax_rapid_streaming_bursts` | `PARSER.stream.1`, `PARSER.stream.3` | [source](https://github.com/lightseekorg/smg/blob/e3eccacf96bc6e041a7ec6623e2251dba1129f28/crates/tool_parser/tests/tool_parser_minimax_m2.rs#L626) | Minimax rapid streaming bursts. |
| `test_minimax_special_characters_in_values` | `PARSER.batch.7.b` | [source](https://github.com/lightseekorg/smg/blob/e3eccacf96bc6e041a7ec6623e2251dba1129f28/crates/tool_parser/tests/tool_parser_minimax_m2.rs#L662) | Minimax special characters in values. |
| `test_minimax_whitespace_handling` | `PARSER.fmt.2` | [source](https://github.com/lightseekorg/smg/blob/e3eccacf96bc6e041a7ec6623e2251dba1129f28/crates/tool_parser/tests/tool_parser_minimax_m2.rs#L683) | Minimax whitespace handling. |
| `test_minimax_no_tools` | `PARSER.batch.3` | [source](https://github.com/lightseekorg/smg/blob/e3eccacf96bc6e041a7ec6623e2251dba1129f28/crates/tool_parser/tests/tool_parser_minimax_m2.rs#L709) | Minimax no tools. |
| `test_minimax_invalid_json_in_parameters` | `PARSER.batch.4.b` | [source](https://github.com/lightseekorg/smg/blob/e3eccacf96bc6e041a7ec6623e2251dba1129f28/crates/tool_parser/tests/tool_parser_minimax_m2.rs#L735) | Minimax invalid json in parameters. |

### LightSeek Rust tool-parser - Mistral

| Test | Bucket(s) | Link | Notes |
| -- | -- | -- | -- |
| `test_mistral_single_tool` | `PARSER.batch.1` | [source](https://github.com/lightseekorg/smg/blob/e3eccacf96bc6e041a7ec6623e2251dba1129f28/crates/tool_parser/tests/tool_parser_mistral.rs#L10) | Mistral single tool. |
| `test_mistral_multiple_tools` | `PARSER.batch.2.a` | [source](https://github.com/lightseekorg/smg/blob/e3eccacf96bc6e041a7ec6623e2251dba1129f28/crates/tool_parser/tests/tool_parser_mistral.rs#L26) | Mistral multiple tools. |
| `test_mistral_nested_json` | `PARSER.batch.7.d` | [source](https://github.com/lightseekorg/smg/blob/e3eccacf96bc6e041a7ec6623e2251dba1129f28/crates/tool_parser/tests/tool_parser_mistral.rs#L48) | Mistral nested json. |
| `test_mistral_with_text_after` | `PARSER.batch.8.b` | [source](https://github.com/lightseekorg/smg/blob/e3eccacf96bc6e041a7ec6623e2251dba1129f28/crates/tool_parser/tests/tool_parser_mistral.rs#L63) | Mistral with text after. |
| `test_mistral_empty_arguments` | `PARSER.batch.6.a`, `PARSER.batch.7.a` | [source](https://github.com/lightseekorg/smg/blob/e3eccacf96bc6e041a7ec6623e2251dba1129f28/crates/tool_parser/tests/tool_parser_mistral.rs#L75) | Mistral empty arguments. |
| `test_mistral_with_brackets_in_strings` | `PARSER.batch.11` | [source](https://github.com/lightseekorg/smg/blob/e3eccacf96bc6e041a7ec6623e2251dba1129f28/crates/tool_parser/tests/tool_parser_mistral.rs#L85) | Mistral with brackets in strings. |
| `test_mistral_format_detection` | `PARSER.fmt.3` | [source](https://github.com/lightseekorg/smg/blob/e3eccacf96bc6e041a7ec6623e2251dba1129f28/crates/tool_parser/tests/tool_parser_mistral.rs#L97) | Format detection / marker-recognition helper coverage. |
| `test_mistral_malformed_json` | `PARSER.batch.4.b` | [source](https://github.com/lightseekorg/smg/blob/e3eccacf96bc6e041a7ec6623e2251dba1129f28/crates/tool_parser/tests/tool_parser_mistral.rs#L107) | Mistral malformed json. |
| `test_mistral_real_world_output` | `PARSER.batch.1`, `PARSER.batch.7.d` | [source](https://github.com/lightseekorg/smg/blob/e3eccacf96bc6e041a7ec6623e2251dba1129f28/crates/tool_parser/tests/tool_parser_mistral.rs#L126) | Mistral real world output. |
| `test_mistral_streaming_closing_bracket` | `PARSER.stream.1` | [source](https://github.com/lightseekorg/smg/blob/e3eccacf96bc6e041a7ec6623e2251dba1129f28/crates/tool_parser/tests/tool_parser_mistral.rs#L161) | Mistral streaming closing bracket. |
| `test_mistral_streaming_bracket_in_text_after_tools` | `PARSER.batch.8.b`, `PARSER.batch.8.d`, `PARSER.stream.1` | [source](https://github.com/lightseekorg/smg/blob/e3eccacf96bc6e041a7ec6623e2251dba1129f28/crates/tool_parser/tests/tool_parser_mistral.rs#L220) | Mistral streaming bracket in text after tools. |

### LightSeek Rust tool-parser - Cohere

| Test | Bucket(s) | Link | Notes |
| -- | -- | -- | -- |
| `test_cohere_single_tool` | `PARSER.batch.1` | [source](https://github.com/lightseekorg/smg/blob/e3eccacf96bc6e041a7ec6623e2251dba1129f28/crates/tool_parser/tests/tool_parser_cohere.rs#L11) | Cohere single tool. |
| `test_cohere_multiple_tools_array` | `PARSER.batch.2.a`, `PARSER.batch.7.d` | [source](https://github.com/lightseekorg/smg/blob/e3eccacf96bc6e041a7ec6623e2251dba1129f28/crates/tool_parser/tests/tool_parser_cohere.rs#L29) | Cohere multiple tools array. |
| `test_cohere_nested_json` | `PARSER.batch.7.d` | [source](https://github.com/lightseekorg/smg/blob/e3eccacf96bc6e041a7ec6623e2251dba1129f28/crates/tool_parser/tests/tool_parser_cohere.rs#L51) | Cohere nested json. |
| `test_cohere_with_text_before_and_after` | `PARSER.batch.8.a`, `PARSER.batch.8.c` | [source](https://github.com/lightseekorg/smg/blob/e3eccacf96bc6e041a7ec6623e2251dba1129f28/crates/tool_parser/tests/tool_parser_cohere.rs#L66) | Cohere with text before and after. |
| `test_cohere_empty_parameters` | `PARSER.batch.6.a` | [source](https://github.com/lightseekorg/smg/blob/e3eccacf96bc6e041a7ec6623e2251dba1129f28/crates/tool_parser/tests/tool_parser_cohere.rs#L80) | Cohere empty parameters. |
| `test_cohere_with_special_chars_in_strings` | `PARSER.batch.7.b` | [source](https://github.com/lightseekorg/smg/blob/e3eccacf96bc6e041a7ec6623e2251dba1129f28/crates/tool_parser/tests/tool_parser_cohere.rs#L91) | Cohere with special chars in strings. |
| `test_cohere_format_detection` | `PARSER.fmt.3` | [source](https://github.com/lightseekorg/smg/blob/e3eccacf96bc6e041a7ec6623e2251dba1129f28/crates/tool_parser/tests/tool_parser_cohere.rs#L103) | Format detection / marker-recognition helper coverage. |
| `test_cohere_malformed_json` | `PARSER.batch.4.b` | [source](https://github.com/lightseekorg/smg/blob/e3eccacf96bc6e041a7ec6623e2251dba1129f28/crates/tool_parser/tests/tool_parser_cohere.rs#L115) | Cohere malformed json. |
| `test_cohere_no_tool_calls` | `PARSER.batch.3` | [source](https://github.com/lightseekorg/smg/blob/e3eccacf96bc6e041a7ec6623e2251dba1129f28/crates/tool_parser/tests/tool_parser_cohere.rs#L133) | Cohere no tool calls. |
| `test_cohere_real_world_output` | `PARSER.batch.1`, `PARSER.batch.7.d` | [source](https://github.com/lightseekorg/smg/blob/e3eccacf96bc6e041a7ec6623e2251dba1129f28/crates/tool_parser/tests/tool_parser_cohere.rs#L143) | Cohere real world output. |
| `test_cohere_alternative_field_names` | `PARSER.fmt.5` | [source](https://github.com/lightseekorg/smg/blob/e3eccacf96bc6e041a7ec6623e2251dba1129f28/crates/tool_parser/tests/tool_parser_cohere.rs#L179) | Cohere alternative field names. |
| `test_cohere_streaming_basic` | `PARSER.stream.1` | [source](https://github.com/lightseekorg/smg/blob/e3eccacf96bc6e041a7ec6623e2251dba1129f28/crates/tool_parser/tests/tool_parser_cohere.rs#L194) | Cohere streaming basic. |
| `test_cohere_reset` | `// helper` | [source](https://github.com/lightseekorg/smg/blob/e3eccacf96bc6e041a7ec6623e2251dba1129f28/crates/tool_parser/tests/tool_parser_cohere.rs#L254) | Unclassified helper or parser-adjacent smoke test. |
| `test_cohere_multiple_action_blocks` | `PARSER.batch.2.a` | [source](https://github.com/lightseekorg/smg/blob/e3eccacf96bc6e041a7ec6623e2251dba1129f28/crates/tool_parser/tests/tool_parser_cohere.rs#L271) | Cohere multiple action blocks. |
| `test_cohere_escaped_quotes_in_parameters` | `PARSER.batch.7.b` | [source](https://github.com/lightseekorg/smg/blob/e3eccacf96bc6e041a7ec6623e2251dba1129f28/crates/tool_parser/tests/tool_parser_cohere.rs#L289) | Cohere escaped quotes in parameters. |
| `test_cohere_unicode_in_parameters` | `PARSER.batch.7.b` | [source](https://github.com/lightseekorg/smg/blob/e3eccacf96bc6e041a7ec6623e2251dba1129f28/crates/tool_parser/tests/tool_parser_cohere.rs#L301) | Cohere unicode in parameters. |
| `test_cohere_whitespace_handling` | `PARSER.fmt.2` | [source](https://github.com/lightseekorg/smg/blob/e3eccacf96bc6e041a7ec6623e2251dba1129f28/crates/tool_parser/tests/tool_parser_cohere.rs#L314) | Cohere whitespace handling. |
| `test_cohere_tool_call_id_field` | `PARSER.fmt.5` | [source](https://github.com/lightseekorg/smg/blob/e3eccacf96bc6e041a7ec6623e2251dba1129f28/crates/tool_parser/tests/tool_parser_cohere.rs#L329) | Cohere tool call id field. |

### LightSeek Rust tool-parser - Core Crate Tests

| Test | Bucket(s) | Link | Notes |
| -- | -- | -- | -- |
| `test_tool_parser_factory` | `// helper` | [source](https://github.com/lightseekorg/smg/blob/e3eccacf96bc6e041a7ec6623e2251dba1129f28/crates/tool_parser/src/tests.rs#L11) | Registry, construction, serialization, or helper-only coverage. |
| `test_tool_parser_factory_model_mapping` | `// helper` | [source](https://github.com/lightseekorg/smg/blob/e3eccacf96bc6e041a7ec6623e2251dba1129f28/crates/tool_parser/src/tests.rs#L21) | Registry, construction, serialization, or helper-only coverage. |
| `test_tool_call_serialization` | `// helper` | [source](https://github.com/lightseekorg/smg/blob/e3eccacf96bc6e041a7ec6623e2251dba1129f28/crates/tool_parser/src/tests.rs#L34) | Registry, construction, serialization, or helper-only coverage. |
| `test_partial_json_parser` | `// helper` | [source](https://github.com/lightseekorg/smg/blob/e3eccacf96bc6e041a7ec6623e2251dba1129f28/crates/tool_parser/src/tests.rs#L55) | Unclassified helper or parser-adjacent smoke test. |
| `test_partial_json_depth_limit` | `// helper` | [source](https://github.com/lightseekorg/smg/blob/e3eccacf96bc6e041a7ec6623e2251dba1129f28/crates/tool_parser/src/tests.rs#L81) | Unclassified helper or parser-adjacent smoke test. |
| `test_partial_tool_call` | `PARSER.batch.5.c` | [source](https://github.com/lightseekorg/smg/blob/e3eccacf96bc6e041a7ec6623e2251dba1129f28/crates/tool_parser/src/tests.rs#L105) | Partial tool call. |
| `test_json_parser_complete_single` | `PARSER.batch.1` | [source](https://github.com/lightseekorg/smg/blob/e3eccacf96bc6e041a7ec6623e2251dba1129f28/crates/tool_parser/src/tests.rs#L130) | Json parser complete single. |
| `test_json_parser_complete_array` | `PARSER.batch.7.d` | [source](https://github.com/lightseekorg/smg/blob/e3eccacf96bc6e041a7ec6623e2251dba1129f28/crates/tool_parser/src/tests.rs#L143) | Json parser complete array. |
| `test_json_parser_with_parameters` | `PARSER.batch.7.a` | [source](https://github.com/lightseekorg/smg/blob/e3eccacf96bc6e041a7ec6623e2251dba1129f28/crates/tool_parser/src/tests.rs#L159) | Json parser with parameters. |
| `test_multiline_json_array` | `PARSER.batch.7.d`, `PARSER.fmt.2` | [source](https://github.com/lightseekorg/smg/blob/e3eccacf96bc6e041a7ec6623e2251dba1129f28/crates/tool_parser/src/tests.rs#L175) | Multiline json array. |
| `test_json_parser_format_detection` | `PARSER.fmt.3` | [source](https://github.com/lightseekorg/smg/blob/e3eccacf96bc6e041a7ec6623e2251dba1129f28/crates/tool_parser/src/tests.rs#L204) | Format detection / marker-recognition helper coverage. |
| `test_factory_with_json_parser` | `// helper` | [source](https://github.com/lightseekorg/smg/blob/e3eccacf96bc6e041a7ec6623e2251dba1129f28/crates/tool_parser/src/tests.rs#L217) | Registry, construction, serialization, or helper-only coverage. |
| `test_json_parser_invalid_input` | `PARSER.batch.4.b` | [source](https://github.com/lightseekorg/smg/blob/e3eccacf96bc6e041a7ec6623e2251dba1129f28/crates/tool_parser/src/tests.rs#L231) | Json parser invalid input. |
| `test_json_parser_empty_arguments` | `PARSER.batch.6.a`, `PARSER.batch.7.a` | [source](https://github.com/lightseekorg/smg/blob/e3eccacf96bc6e041a7ec6623e2251dba1129f28/crates/tool_parser/src/tests.rs#L241) | Json parser empty arguments. |
| `test_malformed_tool_missing_name` | `PARSER.batch.4.c` | [source](https://github.com/lightseekorg/smg/blob/e3eccacf96bc6e041a7ec6623e2251dba1129f28/crates/tool_parser/src/tests.rs#L258) | Malformed tool missing name. |
| `test_invalid_arguments_json` | `PARSER.batch.4.b`, `PARSER.batch.7.a` | [source](https://github.com/lightseekorg/smg/blob/e3eccacf96bc6e041a7ec6623e2251dba1129f28/crates/tool_parser/src/tests.rs#L274) | Invalid arguments json. |
| `test_invalid_json_structures` | `PARSER.batch.4.b` | [source](https://github.com/lightseekorg/smg/blob/e3eccacf96bc6e041a7ec6623e2251dba1129f28/crates/tool_parser/src/tests.rs#L300) | Invalid json structures. |
| `test_unicode_in_names_and_arguments` | `PARSER.batch.7.b` | [source](https://github.com/lightseekorg/smg/blob/e3eccacf96bc6e041a7ec6623e2251dba1129f28/crates/tool_parser/src/tests.rs#L325) | Unicode in names and arguments. |
| `test_escaped_characters` | `PARSER.batch.7.b` | [source](https://github.com/lightseekorg/smg/blob/e3eccacf96bc6e041a7ec6623e2251dba1129f28/crates/tool_parser/src/tests.rs#L344) | Escaped characters. |
| `test_very_large_payloads` | `PARSER.batch.7.d` | [source](https://github.com/lightseekorg/smg/blob/e3eccacf96bc6e041a7ec6623e2251dba1129f28/crates/tool_parser/src/tests.rs#L368) | Very large payloads. |
| `test_mixed_array_tools_and_non_tools` | `PARSER.batch.7.d` | [source](https://github.com/lightseekorg/smg/blob/e3eccacf96bc6e041a7ec6623e2251dba1129f28/crates/tool_parser/src/tests.rs#L399) | Mixed array tools and non tools. |
| `test_duplicate_keys_in_json` | `PARSER.batch.7.d` | [source](https://github.com/lightseekorg/smg/blob/e3eccacf96bc6e041a7ec6623e2251dba1129f28/crates/tool_parser/src/tests.rs#L417) | Duplicate keys in json. |
| `test_null_values_in_arguments` | `PARSER.batch.7.c` | [source](https://github.com/lightseekorg/smg/blob/e3eccacf96bc6e041a7ec6623e2251dba1129f28/crates/tool_parser/src/tests.rs#L435) | Null values in arguments. |
| `test_special_json_values` | `PARSER.batch.7.c` | [source](https://github.com/lightseekorg/smg/blob/e3eccacf96bc6e041a7ec6623e2251dba1129f28/crates/tool_parser/src/tests.rs#L452) | Special json values. |
| `test_function_field_alternative` | `PARSER.fmt.5` | [source](https://github.com/lightseekorg/smg/blob/e3eccacf96bc6e041a7ec6623e2251dba1129f28/crates/tool_parser/src/tests.rs#L479) | Function field alternative. |
| `test_whitespace_handling` | `PARSER.fmt.2` | [source](https://github.com/lightseekorg/smg/blob/e3eccacf96bc6e041a7ec6623e2251dba1129f28/crates/tool_parser/src/tests.rs#L496) | Whitespace handling. |
| `test_deeply_nested_arguments` | `PARSER.batch.7.d` | [source](https://github.com/lightseekorg/smg/blob/e3eccacf96bc6e041a7ec6623e2251dba1129f28/crates/tool_parser/src/tests.rs#L523) | Deeply nested arguments. |
| `test_concurrent_parser_usage` | `// helper` | [source](https://github.com/lightseekorg/smg/blob/e3eccacf96bc6e041a7ec6623e2251dba1129f28/crates/tool_parser/src/tests.rs#L554) | Registry, construction, serialization, or helper-only coverage. |
| `test_qwen_xml_incremental_parameter_streaming` | `PARSER.stream.1` | [source](https://github.com/lightseekorg/smg/blob/e3eccacf96bc6e041a7ec6623e2251dba1129f28/crates/tool_parser/src/tests.rs#L599) | Qwen xml incremental parameter streaming. |
| `test_qwen_xml_incremental_parameter_streaming_with_partial_values` | `PARSER.stream.1`, `PARSER.stream.3` | [source](https://github.com/lightseekorg/smg/blob/e3eccacf96bc6e041a7ec6623e2251dba1129f28/crates/tool_parser/src/tests.rs#L654) | Qwen xml incremental parameter streaming with partial values. |
| `test_qwen_xml_nested_json_parameter` | `PARSER.batch.7.d` | [source](https://github.com/lightseekorg/smg/blob/e3eccacf96bc6e041a7ec6623e2251dba1129f28/crates/tool_parser/src/tests.rs#L694) | Qwen xml nested json parameter. |
| `test_qwen_xml_model_mappings` | `// helper` | [source](https://github.com/lightseekorg/smg/blob/e3eccacf96bc6e041a7ec6623e2251dba1129f28/crates/tool_parser/src/tests.rs#L744) | Registry, construction, serialization, or helper-only coverage. |
| `test_qwen_json_model_mappings` | `// helper` | [source](https://github.com/lightseekorg/smg/blob/e3eccacf96bc6e041a7ec6623e2251dba1129f28/crates/tool_parser/src/tests.rs#L767) | Registry, construction, serialization, or helper-only coverage. |

### LightSeek Rust tool-parser - Mixed Edge Cases

| Test | Bucket(s) | Link | Notes |
| -- | -- | -- | -- |
| `test_mixed_formats_in_text` | `PARSER.fmt.3` | [source](https://github.com/lightseekorg/smg/blob/e3eccacf96bc6e041a7ec6623e2251dba1129f28/crates/tool_parser/tests/tool_parser_mixed_edge_cases.rs#L11) | Mixed formats in text. |
| `test_format_markers_in_string_content` | `PARSER.batch.7.b`, `PARSER.batch.11`, `PARSER.fmt.3` | [source](https://github.com/lightseekorg/smg/blob/e3eccacf96bc6e041a7ec6623e2251dba1129f28/crates/tool_parser/tests/tool_parser_mixed_edge_cases.rs#L36) | Format markers in string content. |
| `test_deeply_nested_json_structures` | `PARSER.batch.7.d` | [source](https://github.com/lightseekorg/smg/blob/e3eccacf96bc6e041a7ec6623e2251dba1129f28/crates/tool_parser/tests/tool_parser_mixed_edge_cases.rs#L57) | Deeply nested json structures. |
| `test_multiple_sequential_calls_different_formats` | `PARSER.batch.2.a`, `PARSER.fmt.3` | [source](https://github.com/lightseekorg/smg/blob/e3eccacf96bc6e041a7ec6623e2251dba1129f28/crates/tool_parser/tests/tool_parser_mixed_edge_cases.rs#L86) | Multiple sequential calls different formats. |
| `test_empty_and_whitespace_variations` | `PARSER.fmt.2` | [source](https://github.com/lightseekorg/smg/blob/e3eccacf96bc6e041a7ec6623e2251dba1129f28/crates/tool_parser/tests/tool_parser_mixed_edge_cases.rs#L106) | Empty and whitespace variations. |
| `test_special_json_values` | `PARSER.batch.7.c` | [source](https://github.com/lightseekorg/smg/blob/e3eccacf96bc6e041a7ec6623e2251dba1129f28/crates/tool_parser/tests/tool_parser_mixed_edge_cases.rs#L127) | Special json values. |
| `test_parser_recovery_after_invalid_input` | `PARSER.batch.4.b`, `PARSER.batch.4.e` | [source](https://github.com/lightseekorg/smg/blob/e3eccacf96bc6e041a7ec6623e2251dba1129f28/crates/tool_parser/tests/tool_parser_mixed_edge_cases.rs#L153) | Parser recovery after invalid input. |
| `test_boundary_cases_for_extraction` | `PARSER.stream.1`, `PARSER.stream.3` | [source](https://github.com/lightseekorg/smg/blob/e3eccacf96bc6e041a7ec6623e2251dba1129f28/crates/tool_parser/tests/tool_parser_mixed_edge_cases.rs#L175) | Boundary cases for extraction. |
| `test_pythonic_edge_cases` | `// helper` | [source](https://github.com/lightseekorg/smg/blob/e3eccacf96bc6e041a7ec6623e2251dba1129f28/crates/tool_parser/tests/tool_parser_mixed_edge_cases.rs#L199) | Unclassified helper or parser-adjacent smoke test. |
| `test_mistral_with_pretty_json` | `PARSER.fmt.2` | [source](https://github.com/lightseekorg/smg/blob/e3eccacf96bc6e041a7ec6623e2251dba1129f28/crates/tool_parser/tests/tool_parser_mixed_edge_cases.rs#L217) | Mistral with pretty json. |
| `test_qwen_with_cdata_like_content` | `PARSER.batch.7.b`, `PARSER.fmt.3` | [source](https://github.com/lightseekorg/smg/blob/e3eccacf96bc6e041a7ec6623e2251dba1129f28/crates/tool_parser/tests/tool_parser_mixed_edge_cases.rs#L247) | Qwen with cdata like content. |
| `test_extremely_long_function_names` | `PARSER.fmt.1` | [source](https://github.com/lightseekorg/smg/blob/e3eccacf96bc6e041a7ec6623e2251dba1129f28/crates/tool_parser/tests/tool_parser_mixed_edge_cases.rs#L264) | Extremely long function names. |
| `test_json_with_duplicate_keys` | `PARSER.batch.7.d` | [source](https://github.com/lightseekorg/smg/blob/e3eccacf96bc6e041a7ec6623e2251dba1129f28/crates/tool_parser/tests/tool_parser_mixed_edge_cases.rs#L276) | Json with duplicate keys. |

### LightSeek Rust tool-parser - Deepseek31

| Test | Bucket(s) | Link | Notes |
| -- | -- | -- | -- |
| `test_deepseek31_complete_single_tool` | `PARSER.batch.1` | [source](https://github.com/lightseekorg/smg/blob/e3eccacf96bc6e041a7ec6623e2251dba1129f28/crates/tool_parser/tests/tool_parser_deepseek31.rs#L8) | Deepseek31 complete single tool. |
| `test_deepseek31_complete_multiple_tools` | `PARSER.batch.2.a` | [source](https://github.com/lightseekorg/smg/blob/e3eccacf96bc6e041a7ec6623e2251dba1129f28/crates/tool_parser/tests/tool_parser_deepseek31.rs#L31) | Deepseek31 complete multiple tools. |
| `test_deepseek31_complete_nested_json` | `PARSER.batch.7.d` | [source](https://github.com/lightseekorg/smg/blob/e3eccacf96bc6e041a7ec6623e2251dba1129f28/crates/tool_parser/tests/tool_parser_deepseek31.rs#L52) | Deepseek31 complete nested json. |
| `test_deepseek31_complete_malformed_json` | `PARSER.batch.4.b` | [source](https://github.com/lightseekorg/smg/blob/e3eccacf96bc6e041a7ec6623e2251dba1129f28/crates/tool_parser/tests/tool_parser_deepseek31.rs#L72) | Deepseek31 complete malformed json. |
| `test_deepseek31_format_detection` | `PARSER.fmt.3` | [source](https://github.com/lightseekorg/smg/blob/e3eccacf96bc6e041a7ec6623e2251dba1129f28/crates/tool_parser/tests/tool_parser_deepseek31.rs#L92) | Format detection / marker-recognition helper coverage. |
| `test_deepseek31_no_tool_calls` | `PARSER.batch.3` | [source](https://github.com/lightseekorg/smg/blob/e3eccacf96bc6e041a7ec6623e2251dba1129f28/crates/tool_parser/tests/tool_parser_deepseek31.rs#L104) | Deepseek31 no tool calls. |
| `test_deepseek31_streaming_single_tool` | `PARSER.batch.1`, `PARSER.stream.1` | [source](https://github.com/lightseekorg/smg/blob/e3eccacf96bc6e041a7ec6623e2251dba1129f28/crates/tool_parser/tests/tool_parser_deepseek31.rs#L114) | Deepseek31 streaming single tool. |
| `test_deepseek31_streaming_multiple_tools` | `PARSER.batch.2.b`, `PARSER.stream.2` | [source](https://github.com/lightseekorg/smg/blob/e3eccacf96bc6e041a7ec6623e2251dba1129f28/crates/tool_parser/tests/tool_parser_deepseek31.rs#L152) | Deepseek31 streaming multiple tools. |
| `test_deepseek31_streaming_text_before_tools` | `PARSER.batch.8.a`, `PARSER.stream.1` | [source](https://github.com/lightseekorg/smg/blob/e3eccacf96bc6e041a7ec6623e2251dba1129f28/crates/tool_parser/tests/tool_parser_deepseek31.rs#L182) | Deepseek31 streaming text before tools. |
| `test_deepseek31_streaming_end_tokens_stripped` | `PARSER.fmt.3`, `PARSER.stream.1`, `PARSER.stream.4`, `PIPELINE.finish_reason` | [source](https://github.com/lightseekorg/smg/blob/e3eccacf96bc6e041a7ec6623e2251dba1129f28/crates/tool_parser/tests/tool_parser_deepseek31.rs#L212) | Deepseek31 streaming end tokens stripped. |
| `test_deepseek31_streaming_end_marker_not_leaked_into_args` | `PARSER.fmt.3`, `PARSER.stream.1` | [source](https://github.com/lightseekorg/smg/blob/e3eccacf96bc6e041a7ec6623e2251dba1129f28/crates/tool_parser/tests/tool_parser_deepseek31.rs#L228) | Deepseek31 streaming end marker not leaked into args. |
| `test_deepseek31_factory_registration` | `// helper` | [source](https://github.com/lightseekorg/smg/blob/e3eccacf96bc6e041a7ec6623e2251dba1129f28/crates/tool_parser/tests/tool_parser_deepseek31.rs#L270) | Registry, construction, serialization, or helper-only coverage. |

### LightSeek Rust tool-parser - Deepseek Dsml

| Test | Bucket(s) | Link | Notes |
| -- | -- | -- | -- |
| `test_deepseek32_complete_single_tool` | `PARSER.batch.1`, `PARSER.fmt.3` | [source](https://github.com/lightseekorg/smg/blob/e3eccacf96bc6e041a7ec6623e2251dba1129f28/crates/tool_parser/tests/tool_parser_deepseek_dsml.rs#L8) | Deepseek32 complete single tool. |
| `test_deepseek32_complete_multiple_tools` | `PARSER.batch.2.a`, `PARSER.fmt.3` | [source](https://github.com/lightseekorg/smg/blob/e3eccacf96bc6e041a7ec6623e2251dba1129f28/crates/tool_parser/tests/tool_parser_deepseek_dsml.rs#L32) | Deepseek32 complete multiple tools. |
| `test_deepseek32_complete_direct_json` | `PARSER.fmt.3` | [source](https://github.com/lightseekorg/smg/blob/e3eccacf96bc6e041a7ec6623e2251dba1129f28/crates/tool_parser/tests/tool_parser_deepseek_dsml.rs#L54) | Deepseek32 complete direct json. |
| `test_deepseek32_complete_mixed_types` | `PARSER.batch.7.d`, `PARSER.fmt.3` | [source](https://github.com/lightseekorg/smg/blob/e3eccacf96bc6e041a7ec6623e2251dba1129f28/crates/tool_parser/tests/tool_parser_deepseek_dsml.rs#L75) | Deepseek32 complete mixed types. |
| `test_deepseek32_complete_nested_json_param` | `PARSER.batch.7.d`, `PARSER.fmt.3` | [source](https://github.com/lightseekorg/smg/blob/e3eccacf96bc6e041a7ec6623e2251dba1129f28/crates/tool_parser/tests/tool_parser_deepseek_dsml.rs#L98) | Deepseek32 complete nested json param. |
| `test_deepseek32_complete_malformed_skips` | `PARSER.batch.4.b`, `PARSER.fmt.3` | [source](https://github.com/lightseekorg/smg/blob/e3eccacf96bc6e041a7ec6623e2251dba1129f28/crates/tool_parser/tests/tool_parser_deepseek_dsml.rs#L117) | Deepseek32 complete malformed skips. |
| `test_deepseek32_format_detection` | `PARSER.fmt.3` | [source](https://github.com/lightseekorg/smg/blob/e3eccacf96bc6e041a7ec6623e2251dba1129f28/crates/tool_parser/tests/tool_parser_deepseek_dsml.rs#L138) | Format detection / marker-recognition helper coverage. |
| `test_deepseek32_no_tool_calls` | `PARSER.batch.3`, `PARSER.fmt.3` | [source](https://github.com/lightseekorg/smg/blob/e3eccacf96bc6e041a7ec6623e2251dba1129f28/crates/tool_parser/tests/tool_parser_deepseek_dsml.rs#L150) | Deepseek32 no tool calls. |
| `test_deepseek32_streaming_single_tool` | `PARSER.batch.1`, `PARSER.fmt.3`, `PARSER.stream.1` | [source](https://github.com/lightseekorg/smg/blob/e3eccacf96bc6e041a7ec6623e2251dba1129f28/crates/tool_parser/tests/tool_parser_deepseek_dsml.rs#L160) | Deepseek32 streaming single tool. |
| `test_deepseek32_streaming_multiple_tools` | `PARSER.batch.2.b`, `PARSER.fmt.3`, `PARSER.stream.2` | [source](https://github.com/lightseekorg/smg/blob/e3eccacf96bc6e041a7ec6623e2251dba1129f28/crates/tool_parser/tests/tool_parser_deepseek_dsml.rs#L198) | Deepseek32 streaming multiple tools. |
| `test_deepseek32_streaming_text_before_tools` | `PARSER.batch.8.a`, `PARSER.fmt.3`, `PARSER.stream.1` | [source](https://github.com/lightseekorg/smg/blob/e3eccacf96bc6e041a7ec6623e2251dba1129f28/crates/tool_parser/tests/tool_parser_deepseek_dsml.rs#L228) | Deepseek32 streaming text before tools. |
| `test_deepseek32_streaming_end_tokens_stripped` | `PARSER.fmt.3`, `PARSER.stream.1`, `PARSER.stream.4`, `PIPELINE.finish_reason` | [source](https://github.com/lightseekorg/smg/blob/e3eccacf96bc6e041a7ec6623e2251dba1129f28/crates/tool_parser/tests/tool_parser_deepseek_dsml.rs#L260) | Deepseek32 streaming end tokens stripped. |
| `test_deepseek32_factory_registration` | `// helper` | [source](https://github.com/lightseekorg/smg/blob/e3eccacf96bc6e041a7ec6623e2251dba1129f28/crates/tool_parser/tests/tool_parser_deepseek_dsml.rs#L274) | Registry, construction, serialization, or helper-only coverage. |
| `test_deepseek_v4_complete_single_tool` | `PARSER.batch.1`, `PARSER.fmt.3` | [source](https://github.com/lightseekorg/smg/blob/e3eccacf96bc6e041a7ec6623e2251dba1129f28/crates/tool_parser/tests/tool_parser_deepseek_dsml.rs#L330) | Deepseek v4 complete single tool. |
| `test_deepseek_v4_complete_mixed_types` | `PARSER.batch.7.d`, `PARSER.fmt.3` | [source](https://github.com/lightseekorg/smg/blob/e3eccacf96bc6e041a7ec6623e2251dba1129f28/crates/tool_parser/tests/tool_parser_deepseek_dsml.rs#L352) | Deepseek v4 complete mixed types. |
| `test_deepseek_v4_format_detection` | `PARSER.fmt.3` | [source](https://github.com/lightseekorg/smg/blob/e3eccacf96bc6e041a7ec6623e2251dba1129f28/crates/tool_parser/tests/tool_parser_deepseek_dsml.rs#L375) | Format detection / marker-recognition helper coverage. |
| `test_deepseek_v32_does_not_match_v4_block` | `PARSER.fmt.3` | [source](https://github.com/lightseekorg/smg/blob/e3eccacf96bc6e041a7ec6623e2251dba1129f28/crates/tool_parser/tests/tool_parser_deepseek_dsml.rs#L387) | Deepseek v32 does not match v4 block. |
| `test_deepseek_v4_cross_variant_payload_passthrough` | `PARSER.fmt.3` | [source](https://github.com/lightseekorg/smg/blob/e3eccacf96bc6e041a7ec6623e2251dba1129f28/crates/tool_parser/tests/tool_parser_deepseek_dsml.rs#L394) | Deepseek v4 cross variant payload passthrough. |
| `test_deepseek_v4_streaming_single_tool` | `PARSER.batch.1`, `PARSER.fmt.3`, `PARSER.stream.1` | [source](https://github.com/lightseekorg/smg/blob/e3eccacf96bc6e041a7ec6623e2251dba1129f28/crates/tool_parser/tests/tool_parser_deepseek_dsml.rs#L411) | Deepseek v4 streaming single tool. |
| `test_deepseek_v4_factory_registration` | `// helper` | [source](https://github.com/lightseekorg/smg/blob/e3eccacf96bc6e041a7ec6623e2251dba1129f28/crates/tool_parser/tests/tool_parser_deepseek_dsml.rs#L446) | Registry, construction, serialization, or helper-only coverage. |
| `test_deepseek_dsml_streaming_strips_eos_from_partial_parameter` | `PARSER.fmt.3`, `PARSER.stream.1`, `PARSER.stream.3`, `PARSER.stream.4`, `PIPELINE.finish_reason` | [source](https://github.com/lightseekorg/smg/blob/e3eccacf96bc6e041a7ec6623e2251dba1129f28/crates/tool_parser/tests/tool_parser_deepseek_dsml.rs#L478) | Deepseek dsml streaming strips eos from partial parameter. |
| `test_deepseek_dsml_v4_streaming_strips_eos_from_partial_parameter` | `PARSER.fmt.3`, `PARSER.stream.1`, `PARSER.stream.3`, `PARSER.stream.4`, `PIPELINE.finish_reason` | [source](https://github.com/lightseekorg/smg/blob/e3eccacf96bc6e041a7ec6623e2251dba1129f28/crates/tool_parser/tests/tool_parser_deepseek_dsml.rs#L509) | Deepseek dsml v4 streaming strips eos from partial parameter. |
| `test_deepseek_dsml_streaming_malformed_empty_name_does_not_trap_buffer` | `PARSER.batch.4.b`, `PARSER.fmt.3`, `PARSER.stream.1` | [source](https://github.com/lightseekorg/smg/blob/e3eccacf96bc6e041a7ec6623e2251dba1129f28/crates/tool_parser/tests/tool_parser_deepseek_dsml.rs#L544) | Deepseek dsml streaming malformed empty name does not trap buffer. |
| `test_deepseek_dsml_v4_streaming_bpe_chunked_opener` | `PARSER.fmt.3`, `PARSER.stream.1`, `PARSER.stream.3` | [source](https://github.com/lightseekorg/smg/blob/e3eccacf96bc6e041a7ec6623e2251dba1129f28/crates/tool_parser/tests/tool_parser_deepseek_dsml.rs#L600) | Deepseek dsml v4 streaming bpe chunked opener. |
| `test_deepseek_dsml_v32_streaming_bpe_chunked_opener` | `PARSER.fmt.3`, `PARSER.stream.1`, `PARSER.stream.3` | [source](https://github.com/lightseekorg/smg/blob/e3eccacf96bc6e041a7ec6623e2251dba1129f28/crates/tool_parser/tests/tool_parser_deepseek_dsml.rs#L686) | Deepseek dsml v32 streaming bpe chunked opener. |
| `test_deepseek_dsml_v4_streaming_malformed_empty_name_does_not_trap_buffer` | `PARSER.batch.4.b`, `PARSER.fmt.3`, `PARSER.stream.1` | [source](https://github.com/lightseekorg/smg/blob/e3eccacf96bc6e041a7ec6623e2251dba1129f28/crates/tool_parser/tests/tool_parser_deepseek_dsml.rs#L734) | Deepseek dsml v4 streaming malformed empty name does not trap buffer. |

### LightSeek Rust tool-parser - Partial Json

| Test | Bucket(s) | Link | Notes |
| -- | -- | -- | -- |
| `test_partial_string_flag_disallows_incomplete_strings` | `PARSER.batch.5.c` | [source](https://github.com/lightseekorg/smg/blob/e3eccacf96bc6e041a7ec6623e2251dba1129f28/crates/tool_parser/tests/tool_parser_partial_json.rs#L9) | Partial string flag disallows incomplete strings. |
| `test_partial_string_flag_allows_incomplete_strings` | `PARSER.batch.5.c` | [source](https://github.com/lightseekorg/smg/blob/e3eccacf96bc6e041a7ec6623e2251dba1129f28/crates/tool_parser/tests/tool_parser_partial_json.rs#L35) | Partial string flag allows incomplete strings. |
| `test_partial_string_flag_complete_json` | `PARSER.batch.5.c` | [source](https://github.com/lightseekorg/smg/blob/e3eccacf96bc6e041a7ec6623e2251dba1129f28/crates/tool_parser/tests/tool_parser_partial_json.rs#L60) | Partial string flag complete json. |
| `test_backward_compatibility_default` | `// helper` | [source](https://github.com/lightseekorg/smg/blob/e3eccacf96bc6e041a7ec6623e2251dba1129f28/crates/tool_parser/tests/tool_parser_partial_json.rs#L86) | Unclassified helper or parser-adjacent smoke test. |
| `test_partial_string_in_nested_object` | `PARSER.batch.5.c`, `PARSER.batch.7.d` | [source](https://github.com/lightseekorg/smg/blob/e3eccacf96bc6e041a7ec6623e2251dba1129f28/crates/tool_parser/tests/tool_parser_partial_json.rs#L106) | Partial string in nested object. |
| `test_bug_fix_exact_scenario` | `// helper` | [source](https://github.com/lightseekorg/smg/blob/e3eccacf96bc6e041a7ec6623e2251dba1129f28/crates/tool_parser/tests/tool_parser_partial_json.rs#L131) | Unclassified helper or parser-adjacent smoke test. |

### LightSeek Rust tool-parser - Deepseek

| Test | Bucket(s) | Link | Notes |
| -- | -- | -- | -- |
| `test_deepseek_complete_parsing` | `PARSER.batch.1` | [source](https://github.com/lightseekorg/smg/blob/e3eccacf96bc6e041a7ec6623e2251dba1129f28/crates/tool_parser/tests/tool_parser_deepseek.rs#L8) | Deepseek complete parsing. |
| `test_deepseek_multiple_tools` | `PARSER.batch.2.a` | [source](https://github.com/lightseekorg/smg/blob/e3eccacf96bc6e041a7ec6623e2251dba1129f28/crates/tool_parser/tests/tool_parser_deepseek.rs#L29) | Deepseek multiple tools. |
| `test_deepseek_streaming` | `PARSER.stream.1` | [source](https://github.com/lightseekorg/smg/blob/e3eccacf96bc6e041a7ec6623e2251dba1129f28/crates/tool_parser/tests/tool_parser_deepseek.rs#L50) | Deepseek streaming. |
| `test_deepseek_nested_json` | `PARSER.batch.7.d` | [source](https://github.com/lightseekorg/smg/blob/e3eccacf96bc6e041a7ec6623e2251dba1129f28/crates/tool_parser/tests/tool_parser_deepseek.rs#L83) | Deepseek nested json. |
| `test_deepseek_format_detection` | `PARSER.fmt.3` | [source](https://github.com/lightseekorg/smg/blob/e3eccacf96bc6e041a7ec6623e2251dba1129f28/crates/tool_parser/tests/tool_parser_deepseek.rs#L106) | Format detection / marker-recognition helper coverage. |
| `test_deepseek_malformed_json_handling` | `PARSER.batch.4.b` | [source](https://github.com/lightseekorg/smg/blob/e3eccacf96bc6e041a7ec6623e2251dba1129f28/crates/tool_parser/tests/tool_parser_deepseek.rs#L120) | Deepseek malformed json handling. |
| `test_multiple_tool_calls` | `PARSER.batch.2.a` | [source](https://github.com/lightseekorg/smg/blob/e3eccacf96bc6e041a7ec6623e2251dba1129f28/crates/tool_parser/tests/tool_parser_deepseek.rs#L142) | Multiple tool calls. |

### LightSeek Rust tool-parser - Pythonic

| Test | Bucket(s) | Link | Notes |
| -- | -- | -- | -- |
| `test_pythonic_single_function` | `// helper` | [source](https://github.com/lightseekorg/smg/blob/e3eccacf96bc6e041a7ec6623e2251dba1129f28/crates/tool_parser/tests/tool_parser_pythonic.rs#L11) | Unclassified helper or parser-adjacent smoke test. |
| `test_pythonic_multiple_functions` | `PARSER.batch.2.a` | [source](https://github.com/lightseekorg/smg/blob/e3eccacf96bc6e041a7ec6623e2251dba1129f28/crates/tool_parser/tests/tool_parser_pythonic.rs#L25) | Pythonic multiple functions. |
| `test_pythonic_with_python_literals` | `PARSER.batch.7.c` | [source](https://github.com/lightseekorg/smg/blob/e3eccacf96bc6e041a7ec6623e2251dba1129f28/crates/tool_parser/tests/tool_parser_pythonic.rs#L41) | Pythonic with python literals. |
| `test_pythonic_with_lists_and_dicts` | `PARSER.batch.7.d` | [source](https://github.com/lightseekorg/smg/blob/e3eccacf96bc6e041a7ec6623e2251dba1129f28/crates/tool_parser/tests/tool_parser_pythonic.rs#L55) | Pythonic with lists and dicts. |
| `test_pythonic_with_special_tokens` | `PARSER.fmt.3` | [source](https://github.com/lightseekorg/smg/blob/e3eccacf96bc6e041a7ec6623e2251dba1129f28/crates/tool_parser/tests/tool_parser_pythonic.rs#L70) | Pythonic with special tokens. |
| `test_pythonic_with_nested_parentheses` | `PARSER.batch.7.d` | [source](https://github.com/lightseekorg/smg/blob/e3eccacf96bc6e041a7ec6623e2251dba1129f28/crates/tool_parser/tests/tool_parser_pythonic.rs#L86) | Pythonic with nested parentheses. |
| `test_pythonic_with_escaped_quotes` | `PARSER.batch.7.b` | [source](https://github.com/lightseekorg/smg/blob/e3eccacf96bc6e041a7ec6623e2251dba1129f28/crates/tool_parser/tests/tool_parser_pythonic.rs#L99) | Pythonic with escaped quotes. |
| `test_pythonic_empty_arguments` | `PARSER.batch.6.a`, `PARSER.batch.7.a` | [source](https://github.com/lightseekorg/smg/blob/e3eccacf96bc6e041a7ec6623e2251dba1129f28/crates/tool_parser/tests/tool_parser_pythonic.rs#L111) | Pythonic empty arguments. |
| `test_pythonic_format_detection` | `PARSER.fmt.3` | [source](https://github.com/lightseekorg/smg/blob/e3eccacf96bc6e041a7ec6623e2251dba1129f28/crates/tool_parser/tests/tool_parser_pythonic.rs#L124) | Format detection / marker-recognition helper coverage. |
| `test_pythonic_invalid_syntax` | `PARSER.batch.4.b` | [source](https://github.com/lightseekorg/smg/blob/e3eccacf96bc6e041a7ec6623e2251dba1129f28/crates/tool_parser/tests/tool_parser_pythonic.rs#L134) | Pythonic invalid syntax. |
| `test_pythonic_real_world_llama4` | `PARSER.batch.1`, `PARSER.batch.7.d` | [source](https://github.com/lightseekorg/smg/blob/e3eccacf96bc6e041a7ec6623e2251dba1129f28/crates/tool_parser/tests/tool_parser_pythonic.rs#L157) | Pythonic real world llama4. |
| `test_pythonic_nested_brackets_in_lists` | `PARSER.batch.7.d` | [source](https://github.com/lightseekorg/smg/blob/e3eccacf96bc6e041a7ec6623e2251dba1129f28/crates/tool_parser/tests/tool_parser_pythonic.rs#L182) | Pythonic nested brackets in lists. |
| `test_pythonic_nested_brackets_in_dicts` | `PARSER.batch.7.d` | [source](https://github.com/lightseekorg/smg/blob/e3eccacf96bc6e041a7ec6623e2251dba1129f28/crates/tool_parser/tests/tool_parser_pythonic.rs#L197) | Pythonic nested brackets in dicts. |
| `test_pythonic_mixed_quotes` | `PARSER.batch.7.b` | [source](https://github.com/lightseekorg/smg/blob/e3eccacf96bc6e041a7ec6623e2251dba1129f28/crates/tool_parser/tests/tool_parser_pythonic.rs#L213) | Pythonic mixed quotes. |
| `test_pythonic_complex_nesting` | `PARSER.batch.7.d` | [source](https://github.com/lightseekorg/smg/blob/e3eccacf96bc6e041a7ec6623e2251dba1129f28/crates/tool_parser/tests/tool_parser_pythonic.rs#L229) | Pythonic complex nesting. |
| `test_parse_streaming_no_brackets` | `PARSER.batch.3`, `PARSER.stream.1` | [source](https://github.com/lightseekorg/smg/blob/e3eccacf96bc6e041a7ec6623e2251dba1129f28/crates/tool_parser/tests/tool_parser_pythonic.rs#L250) | Parse streaming no brackets. |
| `test_parse_streaming_complete_tool_call` | `PARSER.stream.1` | [source](https://github.com/lightseekorg/smg/blob/e3eccacf96bc6e041a7ec6623e2251dba1129f28/crates/tool_parser/tests/tool_parser_pythonic.rs#L263) | Parse streaming complete tool call. |
| `test_parse_streaming_text_before_tool_call` | `PARSER.batch.8.a`, `PARSER.stream.1` | [source](https://github.com/lightseekorg/smg/blob/e3eccacf96bc6e041a7ec6623e2251dba1129f28/crates/tool_parser/tests/tool_parser_pythonic.rs#L279) | Parse streaming text before tool call. |
| `test_parse_streaming_partial_tool_call` | `PARSER.batch.5.c`, `PARSER.stream.1`, `PARSER.stream.3` | [source](https://github.com/lightseekorg/smg/blob/e3eccacf96bc6e041a7ec6623e2251dba1129f28/crates/tool_parser/tests/tool_parser_pythonic.rs#L294) | Parse streaming partial tool call. |
| `test_parse_streaming_bracket_without_text_before` | `PARSER.batch.5.a`, `PARSER.batch.8.a`, `PARSER.stream.1` | [source](https://github.com/lightseekorg/smg/blob/e3eccacf96bc6e041a7ec6623e2251dba1129f28/crates/tool_parser/tests/tool_parser_pythonic.rs#L323) | Parse streaming bracket without text before. |
| `test_parse_streaming_text_after_tool_call` | `PARSER.batch.8.b`, `PARSER.stream.1` | [source](https://github.com/lightseekorg/smg/blob/e3eccacf96bc6e041a7ec6623e2251dba1129f28/crates/tool_parser/tests/tool_parser_pythonic.rs#L338) | Parse streaming text after tool call. |
| `test_parse_streaming_multiple_tool_calls` | `PARSER.batch.2.b`, `PARSER.stream.2` | [source](https://github.com/lightseekorg/smg/blob/e3eccacf96bc6e041a7ec6623e2251dba1129f28/crates/tool_parser/tests/tool_parser_pythonic.rs#L353) | Parse streaming multiple tool calls. |
| `test_parse_streaming_opening_bracket_only` | `PARSER.batch.5.a`, `PARSER.stream.1` | [source](https://github.com/lightseekorg/smg/blob/e3eccacf96bc6e041a7ec6623e2251dba1129f28/crates/tool_parser/tests/tool_parser_pythonic.rs#L373) | Parse streaming opening bracket only. |
| `test_parse_streaming_nested_brackets` | `PARSER.batch.7.d`, `PARSER.stream.1` | [source](https://github.com/lightseekorg/smg/blob/e3eccacf96bc6e041a7ec6623e2251dba1129f28/crates/tool_parser/tests/tool_parser_pythonic.rs#L389) | Parse streaming nested brackets. |
| `test_parse_streaming_nested_brackets_dict` | `PARSER.batch.7.d`, `PARSER.stream.1` | [source](https://github.com/lightseekorg/smg/blob/e3eccacf96bc6e041a7ec6623e2251dba1129f28/crates/tool_parser/tests/tool_parser_pythonic.rs#L409) | Parse streaming nested brackets dict. |
| `test_parse_streaming_multiple_tools_with_nested_brackets` | `PARSER.batch.2.b`, `PARSER.batch.7.d`, `PARSER.stream.2` | [source](https://github.com/lightseekorg/smg/blob/e3eccacf96bc6e041a7ec6623e2251dba1129f28/crates/tool_parser/tests/tool_parser_pythonic.rs#L428) | Parse streaming multiple tools with nested brackets. |
| `test_parse_streaming_partial_nested_brackets` | `PARSER.batch.5.c`, `PARSER.batch.7.d`, `PARSER.stream.1`, `PARSER.stream.3` | [source](https://github.com/lightseekorg/smg/blob/e3eccacf96bc6e041a7ec6623e2251dba1129f28/crates/tool_parser/tests/tool_parser_pythonic.rs#L445) | Parse streaming partial nested brackets. |
| `test_parse_streaming_with_python_start_and_end_token` | `PARSER.fmt.3`, `PARSER.stream.1`, `PARSER.stream.4` | [source](https://github.com/lightseekorg/smg/blob/e3eccacf96bc6e041a7ec6623e2251dba1129f28/crates/tool_parser/tests/tool_parser_pythonic.rs#L472) | Parse streaming with python start and end token. |
| `test_detect_and_parse_with_python_start_and_end_token` | `PARSER.batch.1`, `PARSER.fmt.3` | [source](https://github.com/lightseekorg/smg/blob/e3eccacf96bc6e041a7ec6623e2251dba1129f28/crates/tool_parser/tests/tool_parser_pythonic.rs#L505) | Detect and parse with python start and end token. |

### LightSeek Rust tool-parser - Fallback

| Test | Bucket(s) | Link | Notes |
| -- | -- | -- | -- |
| `test_json_parser_invalid_json_returns_as_normal_text` | `PARSER.batch.4.b` | [source](https://github.com/lightseekorg/smg/blob/e3eccacf96bc6e041a7ec6623e2251dba1129f28/crates/tool_parser/tests/tool_parser_fallback.rs#L10) | Json parser invalid json returns as normal text. |
| `test_qwen_parser_invalid_format_returns_as_normal_text` | `PARSER.batch.4.d`, `PARSER.fmt.3` | [source](https://github.com/lightseekorg/smg/blob/e3eccacf96bc6e041a7ec6623e2251dba1129f28/crates/tool_parser/tests/tool_parser_fallback.rs#L36) | Qwen parser invalid format returns as normal text. |
| `test_llama_parser_invalid_format_returns_as_normal_text` | `PARSER.batch.4.d`, `PARSER.fmt.3` | [source](https://github.com/lightseekorg/smg/blob/e3eccacf96bc6e041a7ec6623e2251dba1129f28/crates/tool_parser/tests/tool_parser_fallback.rs#L64) | Llama parser invalid format returns as normal text. |
| `test_mistral_parser_invalid_format_returns_as_normal_text` | `PARSER.batch.4.d`, `PARSER.fmt.3` | [source](https://github.com/lightseekorg/smg/blob/e3eccacf96bc6e041a7ec6623e2251dba1129f28/crates/tool_parser/tests/tool_parser_fallback.rs#L87) | Mistral parser invalid format returns as normal text. |
| `test_deepseek_parser_invalid_format_returns_as_normal_text` | `PARSER.batch.4.d`, `PARSER.fmt.3` | [source](https://github.com/lightseekorg/smg/blob/e3eccacf96bc6e041a7ec6623e2251dba1129f28/crates/tool_parser/tests/tool_parser_fallback.rs#L110) | Deepseek parser invalid format returns as normal text. |
| `test_mixed_valid_and_invalid_content` | `PARSER.batch.4.b`, `PARSER.batch.4.e` | [source](https://github.com/lightseekorg/smg/blob/e3eccacf96bc6e041a7ec6623e2251dba1129f28/crates/tool_parser/tests/tool_parser_fallback.rs#L139) | Mixed valid and invalid content. |
| `test_partial_tool_markers` | `PARSER.batch.5.c`, `PARSER.fmt.3` | [source](https://github.com/lightseekorg/smg/blob/e3eccacf96bc6e041a7ec6623e2251dba1129f28/crates/tool_parser/tests/tool_parser_fallback.rs#L161) | Partial tool markers. |
| `test_escaped_json_like_content` | `PARSER.batch.7.b` | [source](https://github.com/lightseekorg/smg/blob/e3eccacf96bc6e041a7ec6623e2251dba1129f28/crates/tool_parser/tests/tool_parser_fallback.rs#L184) | Escaped json like content. |
| `test_unicode_and_special_chars_in_failed_parsing` | `PARSER.batch.7.b` | [source](https://github.com/lightseekorg/smg/blob/e3eccacf96bc6e041a7ec6623e2251dba1129f28/crates/tool_parser/tests/tool_parser_fallback.rs#L206) | Unicode and special chars in failed parsing. |
| `test_very_long_invalid_input` | `PARSER.batch.4.b` | [source](https://github.com/lightseekorg/smg/blob/e3eccacf96bc6e041a7ec6623e2251dba1129f28/crates/tool_parser/tests/tool_parser_fallback.rs#L228) | Very long invalid input. |
| `test_almost_valid_tool_calls` | `PARSER.batch.4.b` | [source](https://github.com/lightseekorg/smg/blob/e3eccacf96bc6e041a7ec6623e2251dba1129f28/crates/tool_parser/tests/tool_parser_fallback.rs#L244) | Almost valid tool calls. |

### LightSeek Rust tool-parser - Qwen

| Test | Bucket(s) | Link | Notes |
| -- | -- | -- | -- |
| `test_qwen_single_tool` | `PARSER.batch.1` | [source](https://github.com/lightseekorg/smg/blob/e3eccacf96bc6e041a7ec6623e2251dba1129f28/crates/tool_parser/tests/tool_parser_qwen.rs#L11) | Qwen single tool. |
| `test_qwen_multiple_sequential_tools` | `PARSER.batch.2.a` | [source](https://github.com/lightseekorg/smg/blob/e3eccacf96bc6e041a7ec6623e2251dba1129f28/crates/tool_parser/tests/tool_parser_qwen.rs#L27) | Qwen multiple sequential tools. |
| `test_qwen_pretty_printed_json` | `PARSER.fmt.2` | [source](https://github.com/lightseekorg/smg/blob/e3eccacf96bc6e041a7ec6623e2251dba1129f28/crates/tool_parser/tests/tool_parser_qwen.rs#L45) | Qwen pretty printed json. |
| `test_qwen_with_text_between` | `PARSER.batch.8.d` | [source](https://github.com/lightseekorg/smg/blob/e3eccacf96bc6e041a7ec6623e2251dba1129f28/crates/tool_parser/tests/tool_parser_qwen.rs#L71) | Qwen with text between. |
| `test_qwen_empty_arguments` | `PARSER.batch.6.a`, `PARSER.batch.7.a` | [source](https://github.com/lightseekorg/smg/blob/e3eccacf96bc6e041a7ec6623e2251dba1129f28/crates/tool_parser/tests/tool_parser_qwen.rs#L93) | Qwen empty arguments. |
| `test_qwen_with_newlines_in_strings` | `PARSER.batch.7.b`, `PARSER.fmt.2` | [source](https://github.com/lightseekorg/smg/blob/e3eccacf96bc6e041a7ec6623e2251dba1129f28/crates/tool_parser/tests/tool_parser_qwen.rs#L105) | Qwen with newlines in strings. |
| `test_qwen_format_detection` | `PARSER.fmt.3` | [source](https://github.com/lightseekorg/smg/blob/e3eccacf96bc6e041a7ec6623e2251dba1129f28/crates/tool_parser/tests/tool_parser_qwen.rs#L119) | Format detection / marker-recognition helper coverage. |
| `test_qwen_incomplete_tags` | `PARSER.batch.5.c` | [source](https://github.com/lightseekorg/smg/blob/e3eccacf96bc6e041a7ec6623e2251dba1129f28/crates/tool_parser/tests/tool_parser_qwen.rs#L129) | Qwen incomplete tags. |
| `test_qwen_real_world_output` | `PARSER.batch.1`, `PARSER.batch.7.d` | [source](https://github.com/lightseekorg/smg/blob/e3eccacf96bc6e041a7ec6623e2251dba1129f28/crates/tool_parser/tests/tool_parser_qwen.rs#L146) | Qwen real world output. |
| `test_buffer_drain_optimization` | `// helper` | [source](https://github.com/lightseekorg/smg/blob/e3eccacf96bc6e041a7ec6623e2251dba1129f28/crates/tool_parser/tests/tool_parser_qwen.rs#L193) | Registry, construction, serialization, or helper-only coverage. |
| `test_buffer_efficiency_with_multiple_tools` | `// helper` | [source](https://github.com/lightseekorg/smg/blob/e3eccacf96bc6e041a7ec6623e2251dba1129f28/crates/tool_parser/tests/tool_parser_qwen.rs#L227) | Registry, construction, serialization, or helper-only coverage. |
| `test_qwen_realistic_chunks_with_xml_tags` | `PARSER.batch.1`, `PARSER.batch.7.d`, `PARSER.stream.1`, `PARSER.stream.3` | [source](https://github.com/lightseekorg/smg/blob/e3eccacf96bc6e041a7ec6623e2251dba1129f28/crates/tool_parser/tests/tool_parser_qwen.rs#L258) | Qwen realistic chunks with xml tags. |
| `test_qwen_xml_tag_arrives_in_parts` | `PARSER.batch.5.a`, `PARSER.stream.1`, `PARSER.stream.3` | [source](https://github.com/lightseekorg/smg/blob/e3eccacf96bc6e041a7ec6623e2251dba1129f28/crates/tool_parser/tests/tool_parser_qwen.rs#L283) | Qwen xml tag arrives in parts. |

### LightSeek Rust tool-parser - Edge Cases

| Test | Bucket(s) | Link | Notes |
| -- | -- | -- | -- |
| `test_empty_input` | `PARSER.batch.9` | [source](https://github.com/lightseekorg/smg/blob/e3eccacf96bc6e041a7ec6623e2251dba1129f28/crates/tool_parser/tests/tool_parser_edge_cases.rs#L10) | Empty input. |
| `test_plain_text_no_tools` | `PARSER.batch.3` | [source](https://github.com/lightseekorg/smg/blob/e3eccacf96bc6e041a7ec6623e2251dba1129f28/crates/tool_parser/tests/tool_parser_edge_cases.rs#L46) | Plain text no tools. |
| `test_incomplete_json` | `PARSER.batch.5.c` | [source](https://github.com/lightseekorg/smg/blob/e3eccacf96bc6e041a7ec6623e2251dba1129f28/crates/tool_parser/tests/tool_parser_edge_cases.rs#L95) | Incomplete json. |
| `test_malformed_mistral` | `PARSER.batch.4.b` | [source](https://github.com/lightseekorg/smg/blob/e3eccacf96bc6e041a7ec6623e2251dba1129f28/crates/tool_parser/tests/tool_parser_edge_cases.rs#L119) | Malformed mistral. |
| `test_missing_required_fields` | `PARSER.batch.4.c` | [source](https://github.com/lightseekorg/smg/blob/e3eccacf96bc6e041a7ec6623e2251dba1129f28/crates/tool_parser/tests/tool_parser_edge_cases.rs#L144) | Missing required fields. |
| `test_very_long_strings` | `// helper` | [source](https://github.com/lightseekorg/smg/blob/e3eccacf96bc6e041a7ec6623e2251dba1129f28/crates/tool_parser/tests/tool_parser_edge_cases.rs#L159) | Unclassified helper or parser-adjacent smoke test. |
| `test_unicode_edge_cases` | `PARSER.batch.7.b` | [source](https://github.com/lightseekorg/smg/blob/e3eccacf96bc6e041a7ec6623e2251dba1129f28/crates/tool_parser/tests/tool_parser_edge_cases.rs#L174) | Unicode edge cases. |
| `test_nested_brackets_in_strings` | `PARSER.batch.7.d`, `PARSER.batch.11` | [source](https://github.com/lightseekorg/smg/blob/e3eccacf96bc6e041a7ec6623e2251dba1129f28/crates/tool_parser/tests/tool_parser_edge_cases.rs#L188) | Nested brackets in strings. |
| `test_multiple_formats_in_text` | `PARSER.batch.2.a`, `PARSER.fmt.3` | [source](https://github.com/lightseekorg/smg/blob/e3eccacf96bc6e041a7ec6623e2251dba1129f28/crates/tool_parser/tests/tool_parser_edge_cases.rs#L205) | Multiple formats in text. |
| `test_escaped_characters` | `PARSER.batch.7.b` | [source](https://github.com/lightseekorg/smg/blob/e3eccacf96bc6e041a7ec6623e2251dba1129f28/crates/tool_parser/tests/tool_parser_edge_cases.rs#L219) | Escaped characters. |
| `test_numeric_edge_cases` | `PARSER.batch.7.c` | [source](https://github.com/lightseekorg/smg/blob/e3eccacf96bc6e041a7ec6623e2251dba1129f28/crates/tool_parser/tests/tool_parser_edge_cases.rs#L236) | Numeric edge cases. |
| `test_null_and_boolean_values` | `PARSER.batch.7.c` | [source](https://github.com/lightseekorg/smg/blob/e3eccacf96bc6e041a7ec6623e2251dba1129f28/crates/tool_parser/tests/tool_parser_edge_cases.rs#L264) | Null and boolean values. |
| `test_partial_token_at_buffer_boundary` | `PARSER.batch.5.a`, `PARSER.fmt.3`, `PARSER.stream.1`, `PARSER.stream.3` | [source](https://github.com/lightseekorg/smg/blob/e3eccacf96bc6e041a7ec6623e2251dba1129f28/crates/tool_parser/tests/tool_parser_edge_cases.rs#L286) | Partial token at buffer boundary. |
| `test_exact_prefix_lengths` | `PARSER.batch.5.a` | [source](https://github.com/lightseekorg/smg/blob/e3eccacf96bc6e041a7ec6623e2251dba1129f28/crates/tool_parser/tests/tool_parser_edge_cases.rs#L316) | Exact prefix lengths. |

### LightSeek Rust tool-parser - Step3

| Test | Bucket(s) | Link | Notes |
| -- | -- | -- | -- |
| `test_step3_complete_parsing` | `PARSER.batch.1` | [source](https://github.com/lightseekorg/smg/blob/e3eccacf96bc6e041a7ec6623e2251dba1129f28/crates/tool_parser/tests/tool_parser_step3.rs#L8) | Step3 complete parsing. |
| `test_step3_multiple_tools` | `PARSER.batch.2.a` | [source](https://github.com/lightseekorg/smg/blob/e3eccacf96bc6e041a7ec6623e2251dba1129f28/crates/tool_parser/tests/tool_parser_step3.rs#L31) | Step3 multiple tools. |
| `test_step3_type_conversion` | `PARSER.batch.7.c` | [source](https://github.com/lightseekorg/smg/blob/e3eccacf96bc6e041a7ec6623e2251dba1129f28/crates/tool_parser/tests/tool_parser_step3.rs#L51) | Step3 type conversion. |
| `test_step3_streaming` | `PARSER.stream.1` | [source](https://github.com/lightseekorg/smg/blob/e3eccacf96bc6e041a7ec6623e2251dba1129f28/crates/tool_parser/tests/tool_parser_step3.rs#L76) | Step3 streaming. |
| `test_step3_format_detection` | `PARSER.fmt.3` | [source](https://github.com/lightseekorg/smg/blob/e3eccacf96bc6e041a7ec6623e2251dba1129f28/crates/tool_parser/tests/tool_parser_step3.rs#L109) | Format detection / marker-recognition helper coverage. |
| `test_step3_nested_steptml` | `PARSER.batch.7.d`, `PARSER.fmt.3` | [source](https://github.com/lightseekorg/smg/blob/e3eccacf96bc6e041a7ec6623e2251dba1129f28/crates/tool_parser/tests/tool_parser_step3.rs#L123) | Step3 nested steptml. |
| `test_step3_python_literals` | `PARSER.batch.7.c` | [source](https://github.com/lightseekorg/smg/blob/e3eccacf96bc6e041a7ec6623e2251dba1129f28/crates/tool_parser/tests/tool_parser_step3.rs#L143) | Step3 python literals. |
| `test_steptml_format` | `PARSER.fmt.3` | [source](https://github.com/lightseekorg/smg/blob/e3eccacf96bc6e041a7ec6623e2251dba1129f28/crates/tool_parser/tests/tool_parser_step3.rs#L164) | Steptml format. |
| `test_json_parameter_values` | `PARSER.batch.7.d` | [source](https://github.com/lightseekorg/smg/blob/e3eccacf96bc6e041a7ec6623e2251dba1129f28/crates/tool_parser/tests/tool_parser_step3.rs#L187) | Json parameter values. |
| `test_step3_parameter_with_angle_brackets` | `// helper` | [source](https://github.com/lightseekorg/smg/blob/e3eccacf96bc6e041a7ec6623e2251dba1129f28/crates/tool_parser/tests/tool_parser_step3.rs#L206) | Unclassified helper or parser-adjacent smoke test. |
| `test_step3_empty_function_name` | `PARSER.fmt.1` | [source](https://github.com/lightseekorg/smg/blob/e3eccacf96bc6e041a7ec6623e2251dba1129f28/crates/tool_parser/tests/tool_parser_step3.rs#L226) | Step3 empty function name. |

### LightSeek Rust tool-parser - Glm47 Moe

| Test | Bucket(s) | Link | Notes |
| -- | -- | -- | -- |
| `test_glm47_complete_parsing` | `PARSER.batch.1` | [source](https://github.com/lightseekorg/smg/blob/e3eccacf96bc6e041a7ec6623e2251dba1129f28/crates/tool_parser/tests/tool_parser_glm47_moe.rs#L8) | Glm47 complete parsing. |
| `test_glm47_multiple_tools` | `PARSER.batch.2.a` | [source](https://github.com/lightseekorg/smg/blob/e3eccacf96bc6e041a7ec6623e2251dba1129f28/crates/tool_parser/tests/tool_parser_glm47_moe.rs#L26) | Glm47 multiple tools. |
| `test_glm47_type_conversion` | `PARSER.batch.7.c` | [source](https://github.com/lightseekorg/smg/blob/e3eccacf96bc6e041a7ec6623e2251dba1129f28/crates/tool_parser/tests/tool_parser_glm47_moe.rs#L39) | Glm47 type conversion. |
| `test_glm47_streaming` | `PARSER.stream.1` | [source](https://github.com/lightseekorg/smg/blob/e3eccacf96bc6e041a7ec6623e2251dba1129f28/crates/tool_parser/tests/tool_parser_glm47_moe.rs#L57) | Glm47 streaming. |
| `test_glm47_format_detection` | `PARSER.fmt.3` | [source](https://github.com/lightseekorg/smg/blob/e3eccacf96bc6e041a7ec6623e2251dba1129f28/crates/tool_parser/tests/tool_parser_glm47_moe.rs#L90) | Format detection / marker-recognition helper coverage. |
| `test_python_literals` | `PARSER.batch.7.c` | [source](https://github.com/lightseekorg/smg/blob/e3eccacf96bc6e041a7ec6623e2251dba1129f28/crates/tool_parser/tests/tool_parser_glm47_moe.rs#L104) | Python literals. |
| `test_glm47_nested_json_in_arg_values` | `PARSER.batch.7.d` | [source](https://github.com/lightseekorg/smg/blob/e3eccacf96bc6e041a7ec6623e2251dba1129f28/crates/tool_parser/tests/tool_parser_glm47_moe.rs#L120) | Glm47 nested json in arg values. |

### LightSeek Rust tool-parser - Qwen Xml

| Test | Bucket(s) | Link | Notes |
| -- | -- | -- | -- |
| `test_qwen_xml_single_tool` | `PARSER.batch.1` | [source](https://github.com/lightseekorg/smg/blob/e3eccacf96bc6e041a7ec6623e2251dba1129f28/crates/tool_parser/tests/tool_parser_qwen_xml.rs#L12) | Qwen xml single tool. |
| `test_qwen_xml_multiple_sequential_tools` | `PARSER.batch.2.a` | [source](https://github.com/lightseekorg/smg/blob/e3eccacf96bc6e041a7ec6623e2251dba1129f28/crates/tool_parser/tests/tool_parser_qwen_xml.rs#L31) | Qwen xml multiple sequential tools. |
| `test_qwen_xml_nested_json_in_parameters` | `PARSER.batch.7.d` | [source](https://github.com/lightseekorg/smg/blob/e3eccacf96bc6e041a7ec6623e2251dba1129f28/crates/tool_parser/tests/tool_parser_qwen_xml.rs#L54) | Qwen xml nested json in parameters. |
| `test_qwen_xml_string_parameters` | `// helper` | [source](https://github.com/lightseekorg/smg/blob/e3eccacf96bc6e041a7ec6623e2251dba1129f28/crates/tool_parser/tests/tool_parser_qwen_xml.rs#L74) | Unclassified helper or parser-adjacent smoke test. |
| `test_qwen_xml_empty_arguments` | `PARSER.batch.6.a`, `PARSER.batch.7.a` | [source](https://github.com/lightseekorg/smg/blob/e3eccacf96bc6e041a7ec6623e2251dba1129f28/crates/tool_parser/tests/tool_parser_qwen_xml.rs#L93) | Qwen xml empty arguments. |
| `test_qwen_xml_multiline_parameter_values` | `PARSER.batch.7.a`, `PARSER.fmt.2` | [source](https://github.com/lightseekorg/smg/blob/e3eccacf96bc6e041a7ec6623e2251dba1129f28/crates/tool_parser/tests/tool_parser_qwen_xml.rs#L109) | Qwen xml multiline parameter values. |
| `test_qwen_xml_format_detection` | `PARSER.fmt.3` | [source](https://github.com/lightseekorg/smg/blob/e3eccacf96bc6e041a7ec6623e2251dba1129f28/crates/tool_parser/tests/tool_parser_qwen_xml.rs#L129) | Format detection / marker-recognition helper coverage. |
| `test_qwen_xml_incomplete_tags` | `PARSER.batch.5.c` | [source](https://github.com/lightseekorg/smg/blob/e3eccacf96bc6e041a7ec6623e2251dba1129f28/crates/tool_parser/tests/tool_parser_qwen_xml.rs#L139) | Qwen xml incomplete tags. |
| `test_qwen_xml_streaming_basic` | `PARSER.stream.1` | [source](https://github.com/lightseekorg/smg/blob/e3eccacf96bc6e041a7ec6623e2251dba1129f28/crates/tool_parser/tests/tool_parser_qwen_xml.rs#L158) | Qwen xml streaming basic. |
| `test_qwen_xml_streaming_incremental_json` | `PARSER.stream.1` | [source](https://github.com/lightseekorg/smg/blob/e3eccacf96bc6e041a7ec6623e2251dba1129f28/crates/tool_parser/tests/tool_parser_qwen_xml.rs#L194) | Qwen xml streaming incremental json. |
| `test_qwen_xml_streaming_partial_tags` | `PARSER.batch.5.a`, `PARSER.stream.1`, `PARSER.stream.3` | [source](https://github.com/lightseekorg/smg/blob/e3eccacf96bc6e041a7ec6623e2251dba1129f28/crates/tool_parser/tests/tool_parser_qwen_xml.rs#L237) | Qwen xml streaming partial tags. |
| `test_qwen_xml_multiple_tools_boundary` | `PARSER.batch.2.a`, `PARSER.stream.2`, `PARSER.stream.3` | [source](https://github.com/lightseekorg/smg/blob/e3eccacf96bc6e041a7ec6623e2251dba1129f28/crates/tool_parser/tests/tool_parser_qwen_xml.rs#L274) | Qwen xml multiple tools boundary. |
| `test_qwen_xml_invalid_function_name` | `PARSER.batch.4.d`, `PARSER.fmt.1` | [source](https://github.com/lightseekorg/smg/blob/e3eccacf96bc6e041a7ec6623e2251dba1129f28/crates/tool_parser/tests/tool_parser_qwen_xml.rs#L302) | Qwen xml invalid function name. |
| `test_qwen_xml_type_conversion` | `PARSER.batch.7.c`, `PARSER.xml.2` | [source](https://github.com/lightseekorg/smg/blob/e3eccacf96bc6e041a7ec6623e2251dba1129f28/crates/tool_parser/tests/tool_parser_qwen_xml.rs#L332) | Qwen xml type conversion. |
| `test_qwen_xml_special_characters_in_values` | `PARSER.batch.7.b` | [source](https://github.com/lightseekorg/smg/blob/e3eccacf96bc6e041a7ec6623e2251dba1129f28/crates/tool_parser/tests/tool_parser_qwen_xml.rs#L358) | Qwen xml special characters in values. |
| `test_qwen_xml_whitespace_handling` | `PARSER.fmt.2` | [source](https://github.com/lightseekorg/smg/blob/e3eccacf96bc6e041a7ec6623e2251dba1129f28/crates/tool_parser/tests/tool_parser_qwen_xml.rs#L379) | Qwen xml whitespace handling. |
| `test_qwen_xml_no_tools` | `PARSER.batch.3` | [source](https://github.com/lightseekorg/smg/blob/e3eccacf96bc6e041a7ec6623e2251dba1129f28/crates/tool_parser/tests/tool_parser_qwen_xml.rs#L404) | Qwen xml no tools. |
| `test_qwen_xml_streaming_state_reset` | `PARSER.stream.1`, `PARSER.stream.4` | [source](https://github.com/lightseekorg/smg/blob/e3eccacf96bc6e041a7ec6623e2251dba1129f28/crates/tool_parser/tests/tool_parser_qwen_xml.rs#L430) | Qwen xml streaming state reset. |
| `test_qwen_xml_realistic_chunks` | `PARSER.batch.1`, `PARSER.batch.7.d`, `PARSER.stream.1`, `PARSER.stream.3` | [source](https://github.com/lightseekorg/smg/blob/e3eccacf96bc6e041a7ec6623e2251dba1129f28/crates/tool_parser/tests/tool_parser_qwen_xml.rs#L466) | Qwen xml realistic chunks. |
| `test_qwen_xml_xml_tag_arrives_in_parts` | `PARSER.batch.5.a`, `PARSER.stream.1`, `PARSER.stream.3` | [source](https://github.com/lightseekorg/smg/blob/e3eccacf96bc6e041a7ec6623e2251dba1129f28/crates/tool_parser/tests/tool_parser_qwen_xml.rs#L496) | Qwen xml xml tag arrives in parts. |
| `test_qwen_xml_content_before_and_after_tool_calls` | `PARSER.batch.8.a`, `PARSER.batch.8.c`, `PARSER.batch.8.d` | [source](https://github.com/lightseekorg/smg/blob/e3eccacf96bc6e041a7ec6623e2251dba1129f28/crates/tool_parser/tests/tool_parser_qwen_xml.rs#L522) | Qwen xml content before and after tool calls. |
| `test_qwen_xml_incomplete_tool_call` | `PARSER.batch.5.c` | [source](https://github.com/lightseekorg/smg/blob/e3eccacf96bc6e041a7ec6623e2251dba1129f28/crates/tool_parser/tests/tool_parser_qwen_xml.rs#L551) | Qwen xml incomplete tool call. |
| `test_qwen_xml_malformed_function_tag` | `PARSER.batch.4.d` | [source](https://github.com/lightseekorg/smg/blob/e3eccacf96bc6e041a7ec6623e2251dba1129f28/crates/tool_parser/tests/tool_parser_qwen_xml.rs#L567) | Qwen xml malformed function tag. |
| `test_qwen_xml_many_parameters` | `PARSER.batch.2.a`, `PARSER.batch.7.d` | [source](https://github.com/lightseekorg/smg/blob/e3eccacf96bc6e041a7ec6623e2251dba1129f28/crates/tool_parser/tests/tool_parser_qwen_xml.rs#L585) | Qwen xml many parameters. |
| `test_qwen_xml_malformed_xml_missing_parameter_close` | `PARSER.batch.4.d`, `PARSER.batch.5.c` | [source](https://github.com/lightseekorg/smg/blob/e3eccacf96bc6e041a7ec6623e2251dba1129f28/crates/tool_parser/tests/tool_parser_qwen_xml.rs#L623) | Qwen xml malformed xml missing parameter close. |
| `test_qwen_xml_malformed_xml_unclosed_function` | `PARSER.batch.4.d`, `PARSER.batch.5.c` | [source](https://github.com/lightseekorg/smg/blob/e3eccacf96bc6e041a7ec6623e2251dba1129f28/crates/tool_parser/tests/tool_parser_qwen_xml.rs#L648) | Qwen xml malformed xml unclosed function. |
| `test_qwen_xml_malformed_xml_nested_tool_calls` | `PARSER.batch.4.d`, `PARSER.batch.7.d` | [source](https://github.com/lightseekorg/smg/blob/e3eccacf96bc6e041a7ec6623e2251dba1129f28/crates/tool_parser/tests/tool_parser_qwen_xml.rs#L666) | Qwen xml malformed xml nested tool calls. |
| `test_qwen_xml_unicode_parameter_names` | `PARSER.batch.7.b` | [source](https://github.com/lightseekorg/smg/blob/e3eccacf96bc6e041a7ec6623e2251dba1129f28/crates/tool_parser/tests/tool_parser_qwen_xml.rs#L687) | Qwen xml unicode parameter names. |
| `test_qwen_xml_unicode_function_name` | `PARSER.batch.7.b`, `PARSER.fmt.1` | [source](https://github.com/lightseekorg/smg/blob/e3eccacf96bc6e041a7ec6623e2251dba1129f28/crates/tool_parser/tests/tool_parser_qwen_xml.rs#L710) | Qwen xml unicode function name. |
| `test_qwen_xml_very_large_parameter_value` | `PARSER.batch.7.d` | [source](https://github.com/lightseekorg/smg/blob/e3eccacf96bc6e041a7ec6623e2251dba1129f28/crates/tool_parser/tests/tool_parser_qwen_xml.rs#L729) | Qwen xml very large parameter value. |
| `test_qwen_xml_very_large_nested_json_parameter` | `PARSER.batch.7.d` | [source](https://github.com/lightseekorg/smg/blob/e3eccacf96bc6e041a7ec6623e2251dba1129f28/crates/tool_parser/tests/tool_parser_qwen_xml.rs#L752) | Qwen xml very large nested json parameter. |
| `test_qwen_xml_streaming_malformed_recovery` | `PARSER.batch.4.d`, `PARSER.stream.1` | [source](https://github.com/lightseekorg/smg/blob/e3eccacf96bc6e041a7ec6623e2251dba1129f28/crates/tool_parser/tests/tool_parser_qwen_xml.rs#L780) | Qwen xml streaming malformed recovery. |
| `test_qwen_xml_parameter_with_xml_like_content` | `PARSER.batch.7.b` | [source](https://github.com/lightseekorg/smg/blob/e3eccacf96bc6e041a7ec6623e2251dba1129f28/crates/tool_parser/tests/tool_parser_qwen_xml.rs#L811) | Qwen xml parameter with xml like content. |
| `test_qwen_xml_empty_parameter_value` | `PARSER.batch.6.b` | [source](https://github.com/lightseekorg/smg/blob/e3eccacf96bc6e041a7ec6623e2251dba1129f28/crates/tool_parser/tests/tool_parser_qwen_xml.rs#L837) | Qwen xml empty parameter value. |
| `test_qwen_xml_html_entity_decoding` | `PARSER.xml.1` | [source](https://github.com/lightseekorg/smg/blob/e3eccacf96bc6e041a7ec6623e2251dba1129f28/crates/tool_parser/tests/tool_parser_qwen_xml.rs#L862) | Qwen xml html entity decoding. |
| `test_qwen_xml_html_numeric_entities` | `PARSER.batch.7.c`, `PARSER.xml.1` | [source](https://github.com/lightseekorg/smg/blob/e3eccacf96bc6e041a7ec6623e2251dba1129f28/crates/tool_parser/tests/tool_parser_qwen_xml.rs#L884) | Qwen xml html numeric entities. |
| `test_qwen_xml_python_literals` | `PARSER.batch.7.c` | [source](https://github.com/lightseekorg/smg/blob/e3eccacf96bc6e041a7ec6623e2251dba1129f28/crates/tool_parser/tests/tool_parser_qwen_xml.rs#L904) | Qwen xml python literals. |
| `test_qwen_xml_mixed_html_and_json` | `PARSER.xml.1` | [source](https://github.com/lightseekorg/smg/blob/e3eccacf96bc6e041a7ec6623e2251dba1129f28/crates/tool_parser/tests/tool_parser_qwen_xml.rs#L934) | Qwen xml mixed html and json. |

### LightSeek Rust tool-parser - Kimik2

| Test | Bucket(s) | Link | Notes |
| -- | -- | -- | -- |
| `test_kimik2_complete_parsing` | `PARSER.batch.1` | [source](https://github.com/lightseekorg/smg/blob/e3eccacf96bc6e041a7ec6623e2251dba1129f28/crates/tool_parser/tests/tool_parser_kimik2.rs#L8) | Kimik2 complete parsing. |
| `test_kimik2_multiple_tools` | `PARSER.batch.2.a` | [source](https://github.com/lightseekorg/smg/blob/e3eccacf96bc6e041a7ec6623e2251dba1129f28/crates/tool_parser/tests/tool_parser_kimik2.rs#L28) | Kimik2 multiple tools. |
| `test_kimik2_with_whitespace` | `PARSER.fmt.2` | [source](https://github.com/lightseekorg/smg/blob/e3eccacf96bc6e041a7ec6623e2251dba1129f28/crates/tool_parser/tests/tool_parser_kimik2.rs#L44) | Kimik2 with whitespace. |
| `test_kimik2_streaming` | `PARSER.stream.1` | [source](https://github.com/lightseekorg/smg/blob/e3eccacf96bc6e041a7ec6623e2251dba1129f28/crates/tool_parser/tests/tool_parser_kimik2.rs#L62) | Kimik2 streaming. |
| `test_kimik2_format_detection` | `PARSER.fmt.3` | [source](https://github.com/lightseekorg/smg/blob/e3eccacf96bc6e041a7ec6623e2251dba1129f28/crates/tool_parser/tests/tool_parser_kimik2.rs#L96) | Format detection / marker-recognition helper coverage. |
| `test_kimik2_sequential_indices` | `PARSER.batch.2.a`, `PARSER.fmt.5`, `PARSER.batch.2.d` | [source](https://github.com/lightseekorg/smg/blob/e3eccacf96bc6e041a7ec6623e2251dba1129f28/crates/tool_parser/tests/tool_parser_kimik2.rs#L110) | Kimik2 sequential indices. |
| `test_function_index_extraction` | `PARSER.fmt.5` | [source](https://github.com/lightseekorg/smg/blob/e3eccacf96bc6e041a7ec6623e2251dba1129f28/crates/tool_parser/tests/tool_parser_kimik2.rs#L128) | Function index extraction. |
| `test_namespace_extraction` | `PARSER.fmt.1` | [source](https://github.com/lightseekorg/smg/blob/e3eccacf96bc6e041a7ec6623e2251dba1129f28/crates/tool_parser/tests/tool_parser_kimik2.rs#L146) | Namespace extraction. |

### LightSeek Rust tool-parser - Llama

| Test | Bucket(s) | Link | Notes |
| -- | -- | -- | -- |
| `test_llama_python_tag_format` | `PARSER.batch.1`, `PARSER.fmt.3` | [source](https://github.com/lightseekorg/smg/blob/e3eccacf96bc6e041a7ec6623e2251dba1129f28/crates/tool_parser/tests/tool_parser_llama.rs#L10) | Llama python tag format. |
| `test_llama_with_semicolon_separation` | `PARSER.batch.2.a`, `PARSER.batch.12` | [source](https://github.com/lightseekorg/smg/blob/e3eccacf96bc6e041a7ec6623e2251dba1129f28/crates/tool_parser/tests/tool_parser_llama.rs#L24) | Llama with semicolon separation. |
| `test_llama_no_tool_calls` | `PARSER.batch.3` | [source](https://github.com/lightseekorg/smg/blob/e3eccacf96bc6e041a7ec6623e2251dba1129f28/crates/tool_parser/tests/tool_parser_llama.rs#L37) | Llama no tool calls. |
| `test_llama_plain_json_fallback` | `PARSER.batch.1` | [source](https://github.com/lightseekorg/smg/blob/e3eccacf96bc6e041a7ec6623e2251dba1129f28/crates/tool_parser/tests/tool_parser_llama.rs#L47) | Llama plain json fallback. |
| `test_llama_with_text_before` | `PARSER.batch.8.a` | [source](https://github.com/lightseekorg/smg/blob/e3eccacf96bc6e041a7ec6623e2251dba1129f28/crates/tool_parser/tests/tool_parser_llama.rs#L61) | Llama with text before. |
| `test_llama_with_nested_json` | `PARSER.batch.7.d` | [source](https://github.com/lightseekorg/smg/blob/e3eccacf96bc6e041a7ec6623e2251dba1129f28/crates/tool_parser/tests/tool_parser_llama.rs#L75) | Llama with nested json. |
| `test_llama_empty_arguments` | `PARSER.batch.6.a`, `PARSER.batch.7.a` | [source](https://github.com/lightseekorg/smg/blob/e3eccacf96bc6e041a7ec6623e2251dba1129f28/crates/tool_parser/tests/tool_parser_llama.rs#L98) | Llama empty arguments. |
| `test_llama_format_detection` | `PARSER.fmt.3` | [source](https://github.com/lightseekorg/smg/blob/e3eccacf96bc6e041a7ec6623e2251dba1129f28/crates/tool_parser/tests/tool_parser_llama.rs#L115) | Format detection / marker-recognition helper coverage. |
| `test_llama_invalid_json_after_tag` | `PARSER.batch.4.d` | [source](https://github.com/lightseekorg/smg/blob/e3eccacf96bc6e041a7ec6623e2251dba1129f28/crates/tool_parser/tests/tool_parser_llama.rs#L124) | Llama invalid json after tag. |
| `test_llama_real_world_output` | `PARSER.batch.1`, `PARSER.batch.7.d` | [source](https://github.com/lightseekorg/smg/blob/e3eccacf96bc6e041a7ec6623e2251dba1129f28/crates/tool_parser/tests/tool_parser_llama.rs#L134) | Llama real world output. |
| `test_single_json` | `PARSER.batch.1` | [source](https://github.com/lightseekorg/smg/blob/e3eccacf96bc6e041a7ec6623e2251dba1129f28/crates/tool_parser/tests/tool_parser_llama.rs#L160) | Single json. |
| `test_multiple_json_with_separator` | `PARSER.batch.2.a`, `PARSER.batch.12` | [source](https://github.com/lightseekorg/smg/blob/e3eccacf96bc6e041a7ec6623e2251dba1129f28/crates/tool_parser/tests/tool_parser_llama.rs#L173) | Multiple json with separator. |
| `test_json_with_trailing_text` | `PARSER.batch.8.b` | [source](https://github.com/lightseekorg/smg/blob/e3eccacf96bc6e041a7ec6623e2251dba1129f28/crates/tool_parser/tests/tool_parser_llama.rs#L184) | Json with trailing text. |
| `test_invalid_then_valid_json` | `PARSER.batch.4.b`, `PARSER.batch.4.e` | [source](https://github.com/lightseekorg/smg/blob/e3eccacf96bc6e041a7ec6623e2251dba1129f28/crates/tool_parser/tests/tool_parser_llama.rs#L197) | Invalid then valid json. |
| `test_plain_text_only` | `PARSER.batch.3` | [source](https://github.com/lightseekorg/smg/blob/e3eccacf96bc6e041a7ec6623e2251dba1129f28/crates/tool_parser/tests/tool_parser_llama.rs#L210) | Plain text only. |
| `test_with_python_tag_prefix` | `PARSER.fmt.3` | [source](https://github.com/lightseekorg/smg/blob/e3eccacf96bc6e041a7ec6623e2251dba1129f28/crates/tool_parser/tests/tool_parser_llama.rs#L219) | With python tag prefix. |
| `test_llama_streaming_simple` | `PARSER.stream.1` | [source](https://github.com/lightseekorg/smg/blob/e3eccacf96bc6e041a7ec6623e2251dba1129f28/crates/tool_parser/tests/tool_parser_llama.rs#L231) | Llama streaming simple. |
| `test_llama_streaming_partial` | `PARSER.stream.1`, `PARSER.stream.3` | [source](https://github.com/lightseekorg/smg/blob/e3eccacf96bc6e041a7ec6623e2251dba1129f28/crates/tool_parser/tests/tool_parser_llama.rs#L249) | Llama streaming partial. |
| `test_llama_streaming_plain_json` | `PARSER.stream.1` | [source](https://github.com/lightseekorg/smg/blob/e3eccacf96bc6e041a7ec6623e2251dba1129f28/crates/tool_parser/tests/tool_parser_llama.rs#L279) | Llama streaming plain json. |
| `test_llama_streaming_with_text_before` | `PARSER.batch.8.a`, `PARSER.stream.1` | [source](https://github.com/lightseekorg/smg/blob/e3eccacf96bc6e041a7ec6623e2251dba1129f28/crates/tool_parser/tests/tool_parser_llama.rs#L309) | Llama streaming with text before. |
| `test_llama_streaming_multiple_tools` | `PARSER.batch.2.b`, `PARSER.stream.2` | [source](https://github.com/lightseekorg/smg/blob/e3eccacf96bc6e041a7ec6623e2251dba1129f28/crates/tool_parser/tests/tool_parser_llama.rs#L338) | Llama streaming multiple tools. |
| `test_llama_streaming_multiple_tools_chunked` | `PARSER.batch.2.b`, `PARSER.stream.2`, `PARSER.stream.3` | [source](https://github.com/lightseekorg/smg/blob/e3eccacf96bc6e041a7ec6623e2251dba1129f28/crates/tool_parser/tests/tool_parser_llama.rs#L367) | Llama streaming multiple tools chunked. |
| `test_llama_realistic_chunks_with_python_tag` | `PARSER.batch.1`, `PARSER.batch.7.d`, `PARSER.fmt.3`, `PARSER.stream.1`, `PARSER.stream.3` | [source](https://github.com/lightseekorg/smg/blob/e3eccacf96bc6e041a7ec6623e2251dba1129f28/crates/tool_parser/tests/tool_parser_llama.rs#L405) | Llama realistic chunks with python tag. |
| `test_llama_python_tag_arrives_in_parts` | `PARSER.batch.5.a`, `PARSER.fmt.3`, `PARSER.stream.1`, `PARSER.stream.3` | [source](https://github.com/lightseekorg/smg/blob/e3eccacf96bc6e041a7ec6623e2251dba1129f28/crates/tool_parser/tests/tool_parser_llama.rs#L430) | Llama python tag arrives in parts. |

### LightSeek Rust tool-parser - Glm4 Moe

| Test | Bucket(s) | Link | Notes |
| -- | -- | -- | -- |
| `test_glm4_complete_parsing` | `PARSER.batch.1` | [source](https://github.com/lightseekorg/smg/blob/e3eccacf96bc6e041a7ec6623e2251dba1129f28/crates/tool_parser/tests/tool_parser_glm4_moe.rs#L8) | Glm4 complete parsing. |
| `test_glm4_multiple_tools` | `PARSER.batch.2.a` | [source](https://github.com/lightseekorg/smg/blob/e3eccacf96bc6e041a7ec6623e2251dba1129f28/crates/tool_parser/tests/tool_parser_glm4_moe.rs#L31) | Glm4 multiple tools. |
| `test_glm4_type_conversion` | `PARSER.batch.7.c` | [source](https://github.com/lightseekorg/smg/blob/e3eccacf96bc6e041a7ec6623e2251dba1129f28/crates/tool_parser/tests/tool_parser_glm4_moe.rs#L53) | Glm4 type conversion. |
| `test_glm4_streaming` | `PARSER.stream.1` | [source](https://github.com/lightseekorg/smg/blob/e3eccacf96bc6e041a7ec6623e2251dba1129f28/crates/tool_parser/tests/tool_parser_glm4_moe.rs#L82) | Glm4 streaming. |
| `test_glm4_format_detection` | `PARSER.fmt.3` | [source](https://github.com/lightseekorg/smg/blob/e3eccacf96bc6e041a7ec6623e2251dba1129f28/crates/tool_parser/tests/tool_parser_glm4_moe.rs#L115) | Format detection / marker-recognition helper coverage. |
| `test_python_literals` | `PARSER.batch.7.c` | [source](https://github.com/lightseekorg/smg/blob/e3eccacf96bc6e041a7ec6623e2251dba1129f28/crates/tool_parser/tests/tool_parser_glm4_moe.rs#L129) | Python literals. |
| `test_glm4_nested_json_in_arg_values` | `PARSER.batch.7.d` | [source](https://github.com/lightseekorg/smg/blob/e3eccacf96bc6e041a7ec6623e2251dba1129f28/crates/tool_parser/tests/tool_parser_glm4_moe.rs#L152) | Glm4 nested json in arg values. |

### LightSeek Rust tool-parser - Json

| Test | Bucket(s) | Link | Notes |
| -- | -- | -- | -- |
| `test_simple_json_tool_call` | `PARSER.batch.1` | [source](https://github.com/lightseekorg/smg/blob/e3eccacf96bc6e041a7ec6623e2251dba1129f28/crates/tool_parser/tests/tool_parser_json.rs#L11) | Simple json tool call. |
| `test_json_array_of_tools` | `PARSER.batch.2.a`, `PARSER.batch.7.d` | [source](https://github.com/lightseekorg/smg/blob/e3eccacf96bc6e041a7ec6623e2251dba1129f28/crates/tool_parser/tests/tool_parser_json.rs#L25) | Json array of tools. |
| `test_json_with_parameters_key` | `PARSER.batch.7.a`, `PARSER.fmt.5` | [source](https://github.com/lightseekorg/smg/blob/e3eccacf96bc6e041a7ec6623e2251dba1129f28/crates/tool_parser/tests/tool_parser_json.rs#L40) | Json with parameters key. |
| `test_json_extraction_from_text` | `PARSER.batch.8.a` | [source](https://github.com/lightseekorg/smg/blob/e3eccacf96bc6e041a7ec6623e2251dba1129f28/crates/tool_parser/tests/tool_parser_json.rs#L55) | Json extraction from text. |
| `test_json_with_nested_objects` | `PARSER.batch.7.d` | [source](https://github.com/lightseekorg/smg/blob/e3eccacf96bc6e041a7ec6623e2251dba1129f28/crates/tool_parser/tests/tool_parser_json.rs#L69) | Json with nested objects. |
| `test_json_with_special_characters` | `PARSER.batch.7.b` | [source](https://github.com/lightseekorg/smg/blob/e3eccacf96bc6e041a7ec6623e2251dba1129f28/crates/tool_parser/tests/tool_parser_json.rs#L96) | Json with special characters. |
| `test_json_with_unicode` | `PARSER.batch.7.b` | [source](https://github.com/lightseekorg/smg/blob/e3eccacf96bc6e041a7ec6623e2251dba1129f28/crates/tool_parser/tests/tool_parser_json.rs#L110) | Json with unicode. |
| `test_json_empty_arguments` | `PARSER.batch.6.a`, `PARSER.batch.7.a` | [source](https://github.com/lightseekorg/smg/blob/e3eccacf96bc6e041a7ec6623e2251dba1129f28/crates/tool_parser/tests/tool_parser_json.rs#L124) | Json empty arguments. |
| `test_json_invalid_format` | `PARSER.batch.4.d`, `PARSER.fmt.3` | [source](https://github.com/lightseekorg/smg/blob/e3eccacf96bc6e041a7ec6623e2251dba1129f28/crates/tool_parser/tests/tool_parser_json.rs#L138) | Json invalid format. |
| `test_json_format_detection` | `PARSER.fmt.3` | [source](https://github.com/lightseekorg/smg/blob/e3eccacf96bc6e041a7ec6623e2251dba1129f28/crates/tool_parser/tests/tool_parser_json.rs#L157) | Format detection / marker-recognition helper coverage. |
| `test_json_array_streaming_required_mode` | `PARSER.batch.7.d`, `PARSER.stream.1` | [source](https://github.com/lightseekorg/smg/blob/e3eccacf96bc6e041a7ec6623e2251dba1129f28/crates/tool_parser/tests/tool_parser_json.rs#L167) | Json array streaming required mode. |
| `test_json_array_multiple_tools_streaming` | `PARSER.batch.2.b`, `PARSER.batch.7.d`, `PARSER.stream.2` | [source](https://github.com/lightseekorg/smg/blob/e3eccacf96bc6e041a7ec6623e2251dba1129f28/crates/tool_parser/tests/tool_parser_json.rs#L241) | Json array multiple tools streaming. |
| `test_json_array_closing_bracket_separate_chunk` | `PARSER.batch.7.d`, `PARSER.stream.1`, `PARSER.stream.3` | [source](https://github.com/lightseekorg/smg/blob/e3eccacf96bc6e041a7ec6623e2251dba1129f28/crates/tool_parser/tests/tool_parser_json.rs#L306) | Json array closing bracket separate chunk. |
| `test_json_single_object_with_trailing_text` | `PARSER.batch.8.b` | [source](https://github.com/lightseekorg/smg/blob/e3eccacf96bc6e041a7ec6623e2251dba1129f28/crates/tool_parser/tests/tool_parser_json.rs#L366) | Json single object with trailing text. |
| `test_json_single_object_with_bracket_in_text` | `PARSER.batch.8.b` | [source](https://github.com/lightseekorg/smg/blob/e3eccacf96bc6e041a7ec6623e2251dba1129f28/crates/tool_parser/tests/tool_parser_json.rs#L419) | Json single object with bracket in text. |
| `test_json_array_bracket_in_text_after_tools` | `PARSER.batch.7.d`, `PARSER.batch.8.b`, `PARSER.batch.8.d` | [source](https://github.com/lightseekorg/smg/blob/e3eccacf96bc6e041a7ec6623e2251dba1129f28/crates/tool_parser/tests/tool_parser_json.rs#L470) | Json array bracket in text after tools. |
| `test_json_bug_incomplete_tool_name_string` | `PARSER.batch.5.c` | [source](https://github.com/lightseekorg/smg/blob/e3eccacf96bc6e041a7ec6623e2251dba1129f28/crates/tool_parser/tests/tool_parser_json.rs#L528) | Json bug incomplete tool name string. |
| `test_json_realistic_chunks_simple_tool` | `PARSER.batch.1`, `PARSER.batch.7.d`, `PARSER.stream.1`, `PARSER.stream.3` | [source](https://github.com/lightseekorg/smg/blob/e3eccacf96bc6e041a7ec6623e2251dba1129f28/crates/tool_parser/tests/tool_parser_json.rs#L587) | Json realistic chunks simple tool. |
| `test_json_strategic_chunks_with_quotes` | `PARSER.batch.7.b`, `PARSER.stream.1`, `PARSER.stream.3` | [source](https://github.com/lightseekorg/smg/blob/e3eccacf96bc6e041a7ec6623e2251dba1129f28/crates/tool_parser/tests/tool_parser_json.rs#L612) | Json strategic chunks with quotes. |
| `test_json_incremental_arguments_streaming` | `PARSER.batch.7.a`, `PARSER.stream.1` | [source](https://github.com/lightseekorg/smg/blob/e3eccacf96bc6e041a7ec6623e2251dba1129f28/crates/tool_parser/tests/tool_parser_json.rs#L637) | Json incremental arguments streaming. |
| `test_json_very_long_url_in_arguments` | `PARSER.batch.7.b` | [source](https://github.com/lightseekorg/smg/blob/e3eccacf96bc6e041a7ec6623e2251dba1129f28/crates/tool_parser/tests/tool_parser_json.rs#L664) | Json very long url in arguments. |
| `test_json_unicode` | `PARSER.batch.7.b` | [source](https://github.com/lightseekorg/smg/blob/e3eccacf96bc6e041a7ec6623e2251dba1129f28/crates/tool_parser/tests/tool_parser_json.rs#L690) | Json unicode. |
