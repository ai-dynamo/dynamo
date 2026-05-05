# Parser-parity fixtures

Static JSON fixtures that drive the cross-impl parity harness
(`tests/parity/parser/test_parity_*.py`). Each file holds a set of
inputs + Dynamo's expected output; the harness feeds every input
through Dynamo, vLLM, and SGLang and diffs results.

The `regenerate_fixtures.py` script (one level up) recomputes these
files using Dynamo as the reference oracle. Re-run it after Dynamo
parser-side changes to refresh the expected outputs.

## File layout

```
fixtures/
├── README.md                       (this file)
└── <parser_family>/
    ├── PARSER.batch.json           ← non-streaming parser cases (today)
    ├── PARSER.stream.json          ← streaming parser cases (future)
    ├── PARSER.fmt.json             ← format-conditional cases (future)
    ├── PARSER.xml.json             ← XML-family-only cases (future)
    └── PARSER.harmony.json         ← Harmony-only cases (future)
```

`<parser_family>` is one of the parser names recognized by Dynamo's
Rust parser registry: `kimi_k2`, `qwen3_coder`, `glm47`,
`deepseek_v3_1`, `harmony`, `minimax_m2`, `nemotron_deci`. Each
family directory holds its own copy of every applicable mode file —
the parity matrix is `family × mode × case`.

## What goes where, by mode

| file | scope | invocation surface |
|---|---|---|
| `PARSER.batch.json` | full model output as one string → `(calls, normal_text)` | `parser.detect_and_parse(text, tools)` (Method 2), `/v1/chat/completions` (Method 3) |
| `PARSER.stream.json` | incremental delta_text → per-chunk state | `parser.parse_streaming_increment(delta)` |
| `PARSER.fmt.json` | format-variant inputs (whitespace, name conventions, alt tokens) | same as `batch` |
| `PARSER.xml.json` | XML-family-only behaviors (entity decoding, schema-aware coercion) | same as `batch` |
| `PARSER.harmony.json` | Harmony-only envelope grammar | same as `batch` |

Today only `PARSER.batch.json` is populated; the other modes land in
follow-up PRs (see `lib/parsers/PARSER_CASES.md` for the case
taxonomy).

## File schema (`PARSER.batch.json`)

```json
{
  "family": "kimi_k2",
  "mode": "batch",
  "cases": {
    "1": {
      "description": "Single tool call (happy path)",
      "model_text": "<|tool_calls_section_begin|>...",
      "tools": [{"name": "...", "parameters": {...}}],
      "expected": {
        "calls": [{"name": "...", "arguments": {...}}],
        "normal_text": ""
      }
    },
    "2": { ... },
    ...
  }
}
```

Case keys are `"1"`–`"10"` (string-typed because JSON object keys
are strings); the harness reconstructs the full case ID
`PARSER.batch.<n>` for test IDs and the `KNOWN_DIVERGENCES` registry.

UTF-8 encoding with `ensure_ascii=False`, so DeepSeek special tokens
(`｜` U+FF5C, `▁` U+2581) appear as literal characters rather than
`\uXXXX` escapes.

## Out of scope for this directory

The directory is named `parity/parser` because today every fixture
exercises a **parser** (text → calls). If we add fixtures for
adjacent stages — `PREPROCESS.*` (request preprocessing,
chat-template materialization) or `POSTPROCESS.*` (parser output →
OpenAI wire response) — they'll either land alongside these as
siblings or move to a renamed parent directory. See
`lib/parsers/PARSER_CASES.md`,
`components/src/dynamo/frontend/tests/FRONTEND_CASES.md`, and
`lib/parsers/PIPELINE_CASES.md` for the surrounding taxonomy.

## Adding a new fixture

1. Edit `INPUTS` in `regenerate_fixtures.py` with `(family, "PARSER.<mode>.<n>")` → `{description, text, tools}`.
2. Run `python3 tests/parity/parser/regenerate_fixtures.py` from a container with `dynamo._core` built.
3. Verify the harness with `pytest tests/parity/parser/`.
4. If the new case introduces a new cross-impl divergence, classify it and add an entry to `KNOWN_DIVERGENCES` in the relevant `test_parity_*.py`.
