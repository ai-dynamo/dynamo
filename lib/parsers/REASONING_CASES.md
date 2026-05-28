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

### Reasoning, batch mode

The numbering mirrors `PARSER.batch.{N}` so that "PARSER.batch.X tests
the same shape on the tool-call side that REASONING.batch.X tests on
the reasoning side." Some numbers are N/A on the reasoning side (no
analog of "empty args" for a reasoning block) and are tagged
explicitly as `N/A` rather than silently absent.

- **`REASONING.batch.1`** Single reasoning block, happy path — `<think>...</think>` (or analog) present, no tool call. Parser populates `reasoning_text`, leaves `tool_calls` empty.
- **`REASONING.batch.2`** Paired reasoning + downstream tool call — model emits `<think>...</think>` followed by tool-call tokens. Reasoning parser must extract the think content **and leave the tool-call markers intact in `normal_text`** for the downstream tool-call parser to consume. Watch for the "unclosed think-tag swallows tool call" bug.
- **`REASONING.batch.3`** Reasoning + normal text (no tool call) — `<think>...</think>` interleaved with user-visible narration. Reasoning content goes to `reasoning_text`; narration goes to `normal_text`.
- **`REASONING.batch.4`** Malformed reasoning content — dangling end marker (close without open), open without close (truncation), or invalid syntax inside the reasoning block. Behavior is impl-defined: graceful fallback to `normal_text`, partial extraction, or explicit error are all valid.
- **`REASONING.batch.5`** Missing end-marker recovery — analog of `PARSER.batch.5` for reasoning. Engine hit `max_tokens` mid-think; pin behavior (recover the partial reasoning, surface as truncated, or treat all as `normal_text`).
- **`REASONING.batch.6`** Empty reasoning content — `<think></think>` (or analog with zero bytes between markers). Must still register as a reasoning span, not be silently dropped.
- **`REASONING.batch.7`** Complex reasoning content — large blocks, multi-paragraph, special characters, Unicode, newlines. Verify content makes it through without truncation or escape bugs.
- **`REASONING.batch.8`** Empty/null content variants — empty input, whitespace-only input, null in upstream chunk. Parser must not crash and must produce coherent output.
- **`REASONING.batch.9`** Multiple reasoning spans in one response — e.g., two `<think>...</think>` blocks back-to-back. Behavior is impl-defined: concatenate, surface only the first, or surface as a list.

### Reasoning, stream mode

- **`REASONING.stream.1`** Single reasoning block across N chunks — incremental assembly of `<think>...</think>` content over multiple chunks of any sizing.
- **`REASONING.stream.2`** Start-marker chunk-boundary split — opening `<think>` (or analog) straddles a chunk boundary; partial-token matching must keep buffering instead of flushing as plain text.
- **`REASONING.stream.3`** End-marker chunk-boundary split — closing `</think>` straddles a chunk boundary; same buffering contract.
- **`REASONING.stream.4`** Diverged accumulated text — accumulated text never matches the start marker; parser must give up cleanly and treat all input as `normal_text`.

### Format-conditional tags (cross-stage)

The `PARSER.fmt|xml|harmony.*` tags from `PARSER_CASES.md` also apply
to reasoning parsers when they consume the corresponding format. For
example, the GPT-OSS reasoning parser exercises Harmony's
`<|channel|>analysis<|message|>...<|end|>` envelope, so its tests
carry both `REASONING.batch.1` and `PARSER.harmony.1`.

### Categories that are N/A for reasoning parsers

- `PARSER.batch.2` — "multiple tool calls" maps to `REASONING.batch.9`
  (multiple reasoning spans), not directly. The tool-call-shaped tag
  doesn't apply.
- `PARSER.batch.8` — interleaved tool-call markers and normal text;
  reasoning's analog is `REASONING.batch.3` (reasoning + normal text).
- `FRONTEND.tool_choice` — request-time gating, see
  `FRONTEND_CASES.md`.
- `PIPELINE.finish_reason` — pipeline-boundary contract, see
  `PIPELINE_CASES.md`.

---

## `REASONING.batch.1` — Reasoning content only

`<think>...</think>` (or analog: `<seed:think>`, harmony's
`<|channel|>analysis`) present, no tool call.

- Applies to every reasoning parser.
- Parser populates `reasoning_text` with the content between the
  markers; `tool_calls` empty; `normal_text` either empty or carries
  any text outside the reasoning block (see `REASONING.batch.3`).

## `REASONING.batch.2` — Paired reasoning + downstream tool call

Model emits reasoning content followed by tool-call tokens:
`<think>...</think><|tool_call_begin|>...`. Both must be extracted —
`reasoning_text` populated AND the tool-call markers preserved in
`normal_text` so the downstream tool-call parser can consume them.

- Applies to every (reasoning, tool-call) parser pair.
- Failure mode: greedy reasoning parser eats the tool-call content
  that follows. Pin the boundary explicitly.

## `REASONING.batch.3` — Reasoning + normal text

`<think>...</think>` followed (or preceded) by user-visible narration,
no tool call. Reasoning content → `reasoning_text`; narration →
`normal_text`.

- Applies to every reasoning parser.

---

## `REASONING.stream.1` — Single reasoning block across N chunks

One complete reasoning block delivered across multiple SSE chunks of
any sizing. Parser incrementally reconstructs `reasoning_text`.

- Applies to every reasoning parser. Dominant production path.

## `REASONING.stream.2` — Start-marker chunk-boundary split

The opening reasoning marker (`<think>`, `<|channel|>analysis`, etc.)
straddles a chunk boundary. Partial-token matching must return "keep
buffering" rather than flushing the partial bytes as `normal_text` and
completing the match when the next chunk lands.

- Applies to every reasoning parser.

## `REASONING.stream.3` — End-marker chunk-boundary split

The closing reasoning marker (`</think>`, `<|end|>`, etc.) straddles a
chunk boundary. Same buffering contract as `REASONING.stream.2`.

- Applies to every reasoning parser.

## `REASONING.stream.4` — Diverged accumulated text

The accumulated text never matches the start marker (the model emitted
plain text from the start, no reasoning block). Parser must give up
cleanly and treat the entire stream as `normal_text` rather than
buffering forever.

- Applies to every reasoning parser.

---

## Customer-incident regression tests

Same convention as `PARSER_CASES.md`: include the originating
reference inline in the `#[test]` comment.

```rust
#[test] // REASONING.batch.2 (PR #1234)
fn test_unclosed_think_tag_no_longer_swallows_tool_call() { ... }
```

---

## Adding a new reasoning parser: what you must include

Minimum viable set:

1. `REASONING.batch.{1, 3}` — baseline reasoning extraction +
   reasoning-with-narration split.
2. `REASONING.batch.2` — boundary contract with downstream tool-call
   parser. Non-negotiable for any reasoning parser used alongside
   tool calling.
3. `REASONING.batch.{4, 5}` — malformed input + missing-end-marker
   recovery. Pin behavior; silent drop is the failure mode.
4. `REASONING.batch.{6, 7, 8}` — empty / complex / null content.
5. `REASONING.batch.9` — multiple reasoning spans. Document the
   contract.
6. `REASONING.stream.{1, 2, 3, 4}` — streaming. Essentially
   non-negotiable for any parser behind a streaming frontend.
7. Format tags where applicable: `PARSER.harmony.1` for parsers
   consuming Harmony-format input, etc.
