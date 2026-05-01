# dynamo-parsers

Rust crate for parsing **tool calls** and **reasoning content** out of raw LLM
output. Wire-format-aware, streaming-first, model-family-aware.

This is the post-model side of Dynamo's chat-completions pipeline: given a
token stream from vLLM or SGLang, extract structured `Vec<ToolCall>` +
`reasoning_content` for the client. The pre-model side (prompt formatting)
lives in `lib/llm/src/preprocessor/prompt/`.

## What's in the crate

Two top-level modules, each with its own parser registry:

```
lib/parsers/
├── src/
│   ├── tool_calling/        ← tool-call extraction (18 registered parsers)
│   │   ├── parsers.rs         — registry + dispatch (detect_and_parse_tool_call)
│   │   ├── config.rs          — per-parser ToolCallConfig
│   │   ├── response.rs        — ToolCallResponse shape (wire type)
│   │   ├── dsml/              — DeepSeek V3.2 / V4 DSML grammar
│   │   ├── gemma4/            — Google Gemma 4 custom non-JSON grammar (`<|"|>`-delimited strings)
│   │   ├── xml/               — hermes, glm47, kimi_k2, minimax_m2, qwen3_coder
│   │   ├── json/              — deepseek_v3, deepseek_v3_1, nemotron_deci/nano, jamba, mistral, phi4, llama3_json
│   │   ├── harmony/           — OpenAI gpt-oss (Harmony token stream, uses openai_harmony crate)
│   │   └── pythonic/          — Python function-call syntax (some Llama variants)
│   └── reasoning/           ← reasoning-content extraction (15 registered parsers)
│       ├── mod.rs             — registry + dispatch
│       ├── base_parser.rs     — BasicReasoningParser (<think> ... </think>)
│       ├── gemma4_parser.rs   — Gemma 4 (`<|channel>thought\n...<channel|>`)
│       ├── gpt_oss_parser.rs  — Harmony channel parsing
│       ├── granite_parser.rs  — Granite-style
│       └── minimax_append_think_parser.rs  — MiniMax inline-reasoning
```

## How a request flows through the crate

```
  token stream from engine
            │
            ▼
  ┌─────────────────────────────────┐
  │ reasoning parser                │  — registered by name via
  │   (basic / gpt_oss / ...)       │    reasoning::mod.rs get_reasoning_parser_map()
  │                                 │    returns: (reasoning_content, non_reasoning_tail)
  └─────────────────────────────────┘
            │
            ▼ (non-reasoning tail)
  ┌─────────────────────────────────┐
  │ tool-call parser                │  — registered by name via
  │   dispatched on parser name     │    tool_calling::parsers::get_tool_parser_map()
  │   which picks a ParserConfig:   │
  │     - Dsml(DsmlParserConfig)    │  → try_tool_call_parse_dsml
  │     - Json(JsonParserConfig)    │  → try_tool_call_parse_json
  │     - Xml(XmlParserConfig)      │  → try_tool_call_parse_xml
  │     - KimiK2(KimiK2ParserConfig)│  → try_tool_call_parse_kimi_k2
  │     - Pythonic / Harmony        │
  └─────────────────────────────────┘
            │
            ▼
  Vec<ToolCallResponse> + normal_text
```

Main public entry points in `tool_calling/parsers.rs`:

- `detect_and_parse_tool_call(input, parser_name, schema) -> (calls, normal_text)`
- `try_tool_call_parse(input, config) -> (calls, normal_text)` (lower-level, bypasses the registry)
- `detect_tool_call_start(chunk, parser_name)` — streaming: "is this chunk starting a tool-call block?"
- `find_tool_call_end_position(chunk, parser_name)` — streaming: "where does the block end in this chunk?"

## Parser-family cheat sheet

When adding a new model, the right parser family is usually one of:

| Family | Grammar | Shared engine | Examples |
| -- | -- | -- | -- |
| **DSML** | `<｜DSML｜tool_calls>...` with typed `string="true|false"` parameters | `dsml/parser.rs` | DeepSeek V3.2, V4 |
| **XML** | `<tool_call>...</tool_call>` with nested `<parameter>` or `<function>` | `xml/parser.rs` (generic) or own file for variants | hermes, qwen3_coder, minimax_m2, glm47 (own), kimi_k2 (own, special-token XML) |
| **JSON** | Start sentinel + bare JSON array of `{name, arguments}` | `json/base_json_parser.rs` | deepseek_v3, deepseek_v3_1, nemotron_deci/nano |
| **Harmony** | OpenAI Harmony token stream with `<\|channel\|>`, `<\|message\|>`, `<\|call\|>` | `harmony/harmony_parser.rs` (wraps external `openai_harmony` crate) | gpt-oss-20B / 120B |
| **Pythonic** | `[func_name(arg=value, ...)]` Python function-call syntax | `pythonic/pythonic_parser.rs` | some Llama variants |
| **Gemma 4** | Custom: `<\|tool_call>call:name{key:<\|"\|>val<\|"\|>}<tool_call\|>`, bare keys, custom string delimiter | `gemma4/parser.rs` (recursive-descent into `serde_json::Value`) | Google Gemma 4 thinking models |

Reasoning parsers:

| Family | Grammar | Shared engine | Examples |
| -- | -- | -- | -- |
| **Basic (think-tag)** | `<think>...</think>` | `reasoning/base_parser.rs` (BasicReasoningParser) | Qwen3, Nemotron, Kimi K2.5, DeepSeek R1 / V4, GLM-4.5+ |
| **Append-think** | `<think>...</think>` left inline as text, with `<think>` prefix on first chunk | `reasoning/minimax_append_think_parser.rs` | MiniMax M2 |
| **Harmony channel** | Hidden `analysis` channel | `reasoning/gpt_oss_parser.rs` (wraps external `openai_harmony`) | gpt-oss-20B / 120B |
| **Granite** | Custom start/end tokens | `reasoning/granite_parser.rs` | IBM Granite |
| **Gemma 4 channel** | `<\|channel>thought\n...<channel\|>` with role-label prefix stripped | `reasoning/gemma4_parser.rs` | Google Gemma 4 thinking models |

## Adding a new parser

1. **Pick the family** from the cheat sheet above. If an existing config-driven
   family fits, add a `ToolCallConfig::<your_model>()` constructor in
   `tool_calling/config.rs`, register it in `tool_calling/parsers.rs`. Done —
   you inherit all the shared parser and tests.

2. **If the grammar is genuinely new**, add a module under `tool_calling/` and
   add a `ParserConfig` variant in `config.rs`. Follow the existing parser
   modules for layout.

3. **For reasoning**, prefer aliasing to `BasicReasoningParser` unless the
   grammar truly diverges (append-think, Harmony channels). Most new models
   use plain `<think>...</think>` and can share.

4. **Write tests.** Minimum viable set is in [`TEST_CASES.md`](./TEST_CASES.md) (T1–T20
   taxonomy). At minimum: T1/T2/T3 for correctness, T5 for truncation
   behavior, T8/T9 for streaming, T14 for interleaved text. `N/A` categories
   should be explicitly called out in a comment rather than silently skipped.

## Related docs

- [`TEST_CASES.md`](./TEST_CASES.md) — corner-case taxonomy (T1–T20). What every
  parser should be tested against, what's N/A per family, what's a universal
  gap today.
- `lib/llm/tests/data/` — captured streaming fixtures per (engine × model)
  that feed `test_streaming_tool_parsers.rs`. The replay side of the testing
  story.

## Integration with the rest of Dynamo

- `lib/llm/src/preprocessor/prompt/` — pre-model side. Writes the prompts
  that (eventually) come back and get parsed here.
- `lib/llm/src/preprocessor.rs` — top-level request/response pipeline.
  Decides whether to run the reasoning parser based on
  `is_reasoning_disabled_by_request`, then hands the reasoning-stripped
  tail to the tool-call parser.
- `components/src/dynamo/frontend/` — Python frontend that surfaces parsed
  output as OpenAI-compatible SSE chunks to the client.
