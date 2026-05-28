# Pipeline Boundary Cases

Reference taxonomy for tests that pin contracts at the **boundary
between the parser and the rest of the pipeline** — output formats
that aren't parser-internal but aren't request-time gating either.

Sibling files:

- **Tool-call parsers**: `PARSER_CASES.md`
- **Reasoning parsers**: `REASONING_CASES.md`
- **Frontend gating**:
  `components/src/dynamo/frontend/tests/FRONTEND_CASES.md`

## Quick reference

- **`PIPELINE.finish_reason`** — Parser output is independent of the
  upstream `finish_reason` (`stop` / `tool_calls` / `length`). Same
  input text must produce the same parsed result regardless of which
  stream-end reason the engine reported.

## `PIPELINE.finish_reason` — output independence from upstream stream-end

When the engine reports `finish_reason=tool_calls` because a tool call
landed, parsers must not "trust" that signal — they must extract calls
based purely on the text. Conversely, when the engine reports
`finish_reason=length` (truncation), parsers must still recover any
complete calls that fit before the truncation point (see also
`PARSER.batch.5`).

The parser's job is to map `text → (calls, normal_text)`; the mapping
from `(calls, finish_reason)` to the response wire format
(`finish_reason: tool_calls` etc.) is the **frontend's** job, not the
parser's.

- Applies to every tool-call parser.
- Test convention: feed the same text twice with different upstream
  `finish_reason` values; assert the parser output is byte-identical
  both times.
- Companion frontend assertion (the half about *propagating*
  `finish_reason=tool_calls` when calls land) lives in
  `FRONTEND_CASES.md`.
