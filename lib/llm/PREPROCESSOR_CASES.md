# Frontend Preprocessor — Per-Parser Decisions

Reference taxonomy for the per-request decisions the OpenAI preprocessor
makes based on the configured tool-call / reasoning parser. Each case
below names one decision the preprocessor takes; the **per-parser truth
table** at the bottom records what each parser expects.

This is the preprocessor-layer counterpart to
[`lib/parsers/TEST_CASES.md`](../parsers/TEST_CASES.md). Parsers are
unit-tested for output correctness on input shapes (CASE.\*); the
preprocessor is unit-tested for *whether the right config knob fires for
the right (parser, request) pair* (PRE.\*). Most preprocessor bugs are
"forgot to add parser X to the truth table" — a parser ships, the
preprocessor doesn't know about it, the new path silently runs with a
wrong default and the parser sees malformed input.

When adding a new parser, walk every PRE.\* row in the truth table below
and decide the value explicitly (including `N/A`). Don't silently inherit
the default — write the row.

---

## PRE.1 — `skip_special_tokens` default

**Function:** `OpenAIPreprocessor::parser_requires_special_tokens`

vLLM defaults to `skip_special_tokens=true` when decoding tokens to text.
For parsers whose grammar markers (`<|channel|>`, `<|message|>`,
`<|tool_calls_section_begin|>`, `<|think|>`, etc.) are *single special
tokens* in the model's vocabulary, this strips the markers from the
decoded text and the parser sees plain prose with no structure to match.

The preprocessor flips the default to `false` for parsers whose markers
are special tokens. Caller can still override explicitly via
`output_options.skip_special_tokens`.

**Wrong value silently produces:** empty `reasoning_content` /
`tool_calls` even when the model emitted them correctly. No error path.
This is exactly the failure mode that broke `test_reasoning_effort` for
gpt-oss before the harmony/gpt_oss whitelist landed.

## PRE.2 — Per-request reasoning gate

**Function:** `OpenAIPreprocessor::is_reasoning_disabled_by_request`

Some parsers should be turned **off** based on `chat_template_args` in
the request. The exact arg name and value varies per family:

- `kimi_k25` — `thinking: false`
- `nemotron_nano` / `nemotron3` — `enable_thinking: false` OR
  `force_nonempty_content: true`
- `deepseek_r1` / `deepseek_v4` — `thinking: false` OR
  `thinking_mode: "chat"` (matches V4 formatter's `resolve_thinking_mode`
  convention; keeps parser and prompt synchronized)
- `gemma4` — `enable_thinking: false`

When the gate fires, no reasoning parser runs; the model output is
treated as plain content.

**Wrong value silently produces:** mislabeled `reasoning_content`
(content emitted as reasoning when reasoning is actually off, or vice
versa). Per-parser correctness — depends on each model's chat template
behavior.

## PRE.3 — Force-reasoning + tool-continuation interaction

**Function:** inline gate in
`OpenAIPreprocessor::postprocessor_parsing_stream`

When the chat template injects a reasoning start token (e.g. `<think>`)
into the prompt, the preprocessor sets `prompt_injected_reasoning=true`
so the parser starts in reasoning mode immediately. Combined with the
`last_is_tool` gate:

- `last_is_tool == true` → force-reasoning **off** (current behavior).
  Rationale: tool-continuation turns produce the final user-facing
  answer directly from the tool result; force-reasoning would mislabel
  that final answer as `reasoning_content`. Matches SGLang's observed
  Kimi K2.5 behavior.
- `last_is_tool == false` → force-reasoning honored.

**Tension:** DSv4 disagrees with this gate — the V4 formatter *seeds*
`<think>` into the prompt after a merged tool result, so DSv4 needs
force-reasoning **on** even when `last_is_tool`. Tracked in
[#8901](https://github.com/ai-dynamo/dynamo/pull/8901). Resolution
likely requires per-parser handling of this gate (mirror PRE.2's
match-on-parser shape) rather than a global behavior.

**Wrong value silently produces:** `</think>` literal leaking into
`content` (DSv4) or final answer mislabeled as reasoning (Kimi K2.5).

## PRE.4 — `tool_choice` forcing guided JSON

**Function:** inline gate in
`OpenAIPreprocessor::postprocessor_parsing_stream`

`tool_choice = required | named` forces the backend into guided
decoding, which constrains output to bare JSON with no reasoning
wrapper. The preprocessor turns the reasoning parser off for these cases.

**Wrong value silently produces:** parsers that inject `<think>`
unconditionally (e.g. `minimax_append_think`) contaminate the tool-call
JSON fed into the jail.

## PRE.5 — `ignore_eos` / EOS token ids

**Function:** `stop_conditions.apply_ignore_eos`

When `ignore_eos = true` the preprocessor does not propagate the model's
EOS token ids; otherwise it does. Universal — not parser-specific —
included here as a baseline reminder that the preprocessor owns this.

---

## Per-parser truth table

Walk this table when adding a new parser. **`?`** means unverified —
if you know the answer, fill it in.

| Parser | PRE.1 needs special tokens | PRE.2 reasoning gate | PRE.3 force-reasoning override on tool-continuation | Notes |
|---|---|---|---|---|
| `harmony` (tool) / `gpt_oss` (reasoning) | **YES** | — | — | Channels: `<\|channel\|>analysis<\|message\|>...<\|end\|>`. gpt-oss-20B/120B. |
| `gemma4` (tool + reasoning) | **YES** | `enable_thinking=false` | — | `<\|think\|>...<\|/think\|>`. |
| `kimi_k25` (reasoning) | ? — markers are `<\|tool_calls_section_*\|>`, likely YES | `thinking=false` | OFF when last_is_tool (currently global) | Special-token markers in K2/K2.5/K2.6. |
| `deepseek_v3` (tool) | ? — Unicode markers (`<｜tool_calls_section_begin｜>`); likely YES | — | — | DSv3 grammar. |
| `deepseek_v3_2` / `deepseek_v4` (DSML) | ? — DSML markers (`<｜DSML｜tool_calls>`); likely YES | `thinking=false` / `thinking_mode=chat` | **NEEDS ON** even when last_is_tool (V4 formatter seeds `<think>`); see #8901 | DSv3.2 / DSv4 grammar. |
| `deepseek_r1` (reasoning) | NO (uses plain `<think>`) | `thinking=false` | — | DeepSeek-R1. |
| `nemotron_deci` (tool) / `nemotron_nano` / `nemotron3` (reasoning) | ? | `enable_thinking=false` / `force_nonempty_content=true` (nano/n3 only) | — | Nemotron family. |
| `llama3_json` (tool) | ? — `<\|python_tag\|>` is a special token, likely YES | — | — | Llama 3.x. |
| `hermes` (tool) | NO | — | — | Plain XML `<tool_call>...</tool_call>`. |
| `qwen3_coder` (tool) | NO | — | — | Plain XML `<tool_call><function=...>`. |
| `pythonic` (tool) | NO | — | — | Python list literal. |
| `mistral` (tool) | NO | — | — | `[TOOL_CALLS]` plain text. |
| `phi4` (tool) | NO | — | — | `functools[...]` plain text. |
| `minimax_m2` (tool) / `minimax_append_think` (reasoning) | NO | — | OFF on `tool_choice=required/named` (universal, PRE.4) | XML markers, plain text. |
| `glm47` (tool) | NO | — | — | Plain XML. |
| `jamba` (tool) | NO | — | — | `<tool_calls>` plain text wrapper. |
| `qwen` (reasoning, basic `<think>`) | NO | — | — | Plain `<think>...</think>`. |

---

## Adding a new parser — checklist

1. PRE.1: does the parser's grammar use markers that the model's
   tokenizer treats as special tokens? Run a quick sanity check: encode
   the marker string with the model's tokenizer; if it returns a single
   token id from the special-token range, **add the parser to
   `parser_requires_special_tokens`**.
2. PRE.2: does the parser need to be silenced based on
   `chat_template_args`? Look at the model's chat template: does it
   gate emission of the parser's markers on a flag like `enable_thinking`
   / `thinking` / `thinking_mode`? If yes, add the case to
   `is_reasoning_disabled_by_request`.
3. PRE.3: when the previous turn is a tool call, does the model
   re-enter reasoning (DSv4) or skip straight to answer (Kimi K2.5)?
   Document explicitly in this table; current code uses a global gate
   that may need to become parser-specific (see #8901).
4. PRE.4: confirm `tool_choice = required/named` doesn't conflict with
   the parser's behavior. Universal default is to disable reasoning
   parsing in this case.
5. Add a row to the truth table above with explicit values. `N/A` is
   acceptable but must be stated, not omitted.
6. Add a unit test in `lib/llm/src/preprocessor.rs`'s `#[cfg(test)] mod`
   that asserts `parser_requires_special_tokens(...)` returns the
   expected value for the new parser. Table-driven, one row per parser
   in this doc.
