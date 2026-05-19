# SGLang Tool Parser Test Audit

Source: local SGLang checkout at `612785ffdcaf35552f1ed433a981d596ca9fe900`
under `/sgl-workspace/sglang`.

Scope: in-tree Python parser and detector tests:

- `test/registered/unit/function_call/test_function_call_parser.py`
- `test/registered/unit/function_call/test_{hermes,llama32,mistral}_detector.py`
- `test/registered/unit/function_call/test_{parallel_tool_calls,unknown_tool_name}.py`
- `test/registered/function_call/test_kimik2_detector.py`
- `test/registered/unit/parser/test_harmony_parser.py`

Excluded from parser-taxonomy changes:

- `test_json_schema_constraint.py` and `openai_server/function_call/*` test request
  shaping, guided schema constraints, `tool_choice`, and API wiring. Those are
  `FRONTEND.*` / guided-decoding concerns, not parser extraction cases.
- Hunyuan, LFM2, GigaChat, GLM4-MoE, and similar SGLang-only detectors do not
  have Dynamo parser families today. Their rows were used as gap signal only.

## Bucket Summary

| SGLang test shape | Bucket |
| -- | -- |
| Single complete call | `PARSER.batch.1` |
| Multiple / parallel calls | `PARSER.batch.2`, `PARSER.batch.2.{a,b,c,d}` |
| Plain text / no tool call | `PARSER.batch.3` |
| Malformed body / missing key / malformed wrapper | `PARSER.batch.4.{a,b,c,d}` |
| Malformed prefix followed by a later valid call | `PARSER.batch.4.e` |
| Missing end token / partial call | `PARSER.batch.5.{a,c}` |
| Empty args / no args / no arguments key | `PARSER.batch.6.{a,b,c}` |
| Nested args, arrays, unicode, escaped chars, typed values | `PARSER.batch.7.{a,b,c,d}` |
| Text before / after / around tool calls | `PARSER.batch.8.{a,b,c,d}` |
| Empty input | `PARSER.batch.9` |
| Duplicate tool calls | `PARSER.batch.10` |
| Separator character inside a string argument | `PARSER.batch.11` |
| Multi-call response with separator character inside one argument | `PARSER.batch.12` |
| Tool name absent from supplied `tools` | `PARSER.batch.13` |
| Streaming assembly, multi-call streaming, token-boundary splits, termination | `PARSER.stream.{1,2,3,4}` |
| Function-name surface, token variants, arguments-vs-parameters aliases | `PARSER.fmt.{1,3,5}` |
| Harmony channel / recipient / envelope grammar | `PARSER.harmony.{1,2}` |
| White-box helper-only state tests | `// helper` |

## Existing-Test Overlap Decision

- `PARSER.batch.11` is close to `PARSER.batch.7.b` because both involve
  special characters inside argument strings. The difference is parser state:
  `7.b` proves JSON/string decoding, while `11` proves a delimiter-aware
  parser does not split on a grammar separator while it is inside a string.
- `PARSER.batch.12` is the same delimiter-state check on the parallel-call
  path. It overlaps with `PARSER.batch.2.a`, but `2.a` only proves real
  separators between calls; it does not prove the parser ignores an earlier
  separator-looking character inside a string value.
- `PARSER.batch.13` does not fit `PARSER.batch.4.c`: the call is structurally
  valid and parseable. The tested behavior is the schema/tool-registry
  lookup after syntax parsing, which SGLang explicitly tests with both drop
  and forwarding modes.

## SGLang-Only Taxonomy Additions

### `PARSER.batch.4.e`

SGLang's `TestLlama32Detector.test_invalid_then_valid_json` verifies that a
malformed JSON object followed by a valid JSON object can still recover the
later call. Dynamo and vLLM currently treat the whole string as normal text.
This is still an impl-defined malformed-input contract, but it needs its own
subcase because it tests resynchronization, not just "invalid JSON does not
crash."

Fixture added:

- `tests/parity/parser/fixtures/llama3_json/PARSER.batch.4.yaml`

### `PARSER.batch.11` / `PARSER.batch.12`

SGLang's Llama tests cover semicolon-separated JSON objects, and its generic
JSON-array parser tests cover separator-state handling for commas and braces in
string values. Dynamo already had `llama3_json` fixture rows for the exact
semicolon-in-string shape (`PARSER.batch.11` and `.12`); this audit expanded
the same delimiter-state check to other applicable grammars.

Fixtures documented or added:

- `tests/parity/parser/fixtures/llama3_json/PARSER.batch.yaml`
- `tests/parity/parser/fixtures/gemma4/PARSER.batch.yaml`
- `tests/parity/parser/fixtures/jamba/PARSER.batch.yaml`
- `tests/parity/parser/fixtures/mistral/PARSER.batch.yaml`
- `tests/parity/parser/fixtures/phi4/PARSER.batch.yaml`
- `tests/parity/parser/fixtures/pythonic/PARSER.batch.yaml`

### `PARSER.batch.13`

SGLang has explicit tests for unknown tool names:

- default: drop unknown tool calls and log a warning
- opt-in: forward unknown tool calls with `tool_index = -1`

Dynamo and vLLM generally forward unknown tool names. SGLang behavior depends
on detector family: base-JSON-derived detectors drop by default, while some
detectors that do not route through `parse_base_json` forward.

Fixtures added:

- `tests/parity/parser/fixtures/gemma4/PARSER.batch.yaml` — all three forward
- `tests/parity/parser/fixtures/hermes/PARSER.batch.yaml` — SGLang default drop
- `tests/parity/parser/fixtures/jamba/PARSER.batch.yaml` — no SGLang detector
- `tests/parity/parser/fixtures/llama3_json/PARSER.batch.yaml` — SGLang default drop
- `tests/parity/parser/fixtures/mistral/PARSER.batch.yaml` — SGLang Mistral format divergence masks unknown-tool lookup
- `tests/parity/parser/fixtures/phi4/PARSER.batch.yaml` — no SGLang detector
- `tests/parity/parser/fixtures/pythonic/PARSER.batch.yaml` — SGLang default drop

## Existing-Bucket Mapping

| SGLang file / class | Representative tests | Bucket(s) |
| -- | -- | -- |
| `test_hermes_detector.py::TestHermesDetector` | `test_single_tool_call`, `test_multiple_tool_calls`, `test_tool_call_with_leading_text`, `test_malformed_json_returns_original_text`, streaming variants | `batch.{1,2,3,4,7,8}`, `stream.{1,3}` |
| `test_llama32_detector.py::TestLlama32Detector` | with/without `<|python_tag|>`, semicolon-separated calls, normal text before call, streaming split tag | `batch.{1,2,3,7,8}`, `fmt.3`, `stream.{1,3}` |
| `test_function_call_parser.py::TestLlama32Detector` | trailing text, invalid-then-valid JSON, repeated `<|python_tag|>` separator | `batch.4.e`, `batch.8.b`, `fmt.3` |
| `test_mistral_detector.py::TestMistralDetector` | JSON-array and compact formats, nested JSON, invalid JSON, streaming compact format | `batch.{1,2,3,4,7,8}`, `fmt.3`, `stream.{1,3}` |
| `test_function_call_parser.py::TestPythonicDetector` | no brackets, complete call, partial call, nested brackets, text after tool call | `batch.{1,2,3,4,7,8}`, `stream.{1,2,3}` |
| `test_function_call_parser.py::TestQwen3CoderDetector` | XML-style plain text, single/multiple calls, typed conversion, special chars, incomplete call | `batch.{1,2,3,4,7,8}`, `xml.2`, `stream.{1,2}` |
| `test_function_call_parser.py::TestGlm47MoeDetector` | single/multiple calls, tool IDs, invalid/partial calls, escaped JSON arrays, whitespace preservation | `batch.{1,2,4,5,7}`, `fmt.{1,2}`, `stream.{1,2}` |
| `test_function_call_parser.py::TestDeepSeekV32Detector` | XML and JSON formats, no parameters, whitespace in no-parameter calls, streaming XML/JSON | `batch.{1,6}`, `fmt.3`, `stream.{1,3}` |
| `test_function_call_parser.py::TestGemma4Detector` | unknown tool index, nested args, multiple calls, empty args, char-by-char streaming | `batch.{1,2,3,6,7,13}`, `stream.{1,2,3}` |
| `test_kimik2_detector.py::TestKimiK2Detector*` | hyphenated names, multiple calls, state reset, special-token leakage, reasoning-to-tool transitions | `batch.{1,2,3,8}`, `fmt.1`, `stream.{1,2,3,4}`, `REASONING.*` |
| `test_parallel_tool_calls.py` | JSON-array parallel calls with array parameters and separators | `batch.2`, `batch.7`, `stream.2` |
| `test_unknown_tool_name.py` | default drop and opt-in forwarding of unknown tools | `PARSER.batch.13` |
| `test_harmony_parser.py` | canonical channels, commentary tool calls, built-in analysis tool calls, malformed/partial tokens, commentary filler | `PARSER.harmony.{1,2}`, `batch.{1,2,3,4,6,8,9}`, `stream.{1,3}` |

## Follow-Up Gaps

- SGLang has extensive streaming tests; Dynamo's shared YAML harness is still
  batch-only. The stream rows above should feed future stream fixture work.
- SGLang's `JsonArrayParser` has many separator-state tests around commas,
  trailing commas, braces in strings, and three-call streams. These mostly fit
  `PARSER.stream.{2,3}` plus `PARSER.batch.{7,11,12}` and do not require more
  taxonomy labels yet.
- SGLang's unknown-tool forwarding mode is environment-controlled. The parity
  fixture added here pins the default mode; testing opt-in forwarding would
  require a fixture-level environment override in the harness.
