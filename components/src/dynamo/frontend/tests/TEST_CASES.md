# Frontend Chat-Processor Test Cases

Reference taxonomy for unit testing the **frontend chat-processor layer** —
the code under `components/src/dynamo/frontend/` that sits between the
OpenAI-shaped HTTP request and the backend engine. Tests for this layer
live under `components/src/dynamo/frontend/tests/`.

This is the **frontend** companion to `lib/parsers/TEST_CASES.md`. The
two taxonomies cover different surfaces:

| File | Scope | Prefix |
|---|---|---|
| `lib/parsers/TEST_CASES.md` | Tool-call & reasoning parser behavior on **model output** | `CASE.X` |
| `components/src/dynamo/frontend/tests/TEST_CASES.md` | Chat-processor layer: request preprocessing, output assembly, error surface, worker plumbing | `FE.X` |

Backends covered by this taxonomy: **vllm** (`prepost.py` + `vllm_processor.py`)
and **sglang** (`sglang_prepost.py` + `sglang_processor.py`). trtllm has its
own architecture under `components/src/dynamo/trtllm/` and is out of scope here.

## Quick reference

- **`FE.1`** Chat-template input preprocessing — multi-turn assistant `tool_calls` with JSON-string `arguments`, message materialization, role handling, tool messages, system-merging. (Where Richard's qwen3.5 fix in #8792 lives.)
- **`FE.2`** Parser construction & dispatch — instantiate the right tool-call / reasoning parser from a request's `chat_template_kwargs`, model name, runtime config; handle "no parser" gracefully.
- **`FE.3`** Request shaping & sampling-param projection — OpenAI fields → backend kwargs, tool stripping when `tool_choice="none"`, guided-decoding setup.
- **`FE.4`** Tool-call output assembly — model output stream → OpenAI-shaped `tool_calls` deltas. Single, multiple, content-mixed, fallback paths.
- **`FE.5`** Finish-reason mapping — frontend-layer remap (`stop`/`length`/`tool_calls`). Distinct from parser-layer `CASE.12` which covers the parser's view of the raw signal.
- **`FE.6`** Incremental detokenization — token-id stream → text, prompt-token-id normalization, fast plain-text path.
- **`FE.7`** Worker subprocess boundary — preprocessing runs in a subprocess; result picklability, init, error propagation across the boundary.
- **`FE.8`** Error surface — `BackendError` / `InternalError` / engine-error handling, malformed responses, stream errors, deprecation warnings.
- **`FE.9`** Reasoning ↔ tool-call orchestration — both parsers active on the same response; distinct from parser-layer `CASE.9` which is purely on output text.

## Annotation convention

Tests carry a one-line trailing comment naming the FE.X(s) they cover:

```python
class TestMapFinishReason:  # FE.5
    def test_stop_to_tool_calls_when_emitted(self): ...
```

Or per-test when a class spans multiple categories:

```python
class TestUtilities:
    def test_make_backend_error(self): ...  # FE.8
    def test_normalize_prompt_token_ids(self): ...  # FE.6
```

`grep -r 'FE.1' components/src/dynamo/frontend/tests/` returns every
chat-template-preprocessing test across vllm + sglang in one shot.

---

## `FE.1` — Chat-template input preprocessing

The frontend rebuilds the prompt from a multi-turn message history before
handing it to the backend. Several quirks live here:

- Some chat templates expect assistant `tool_calls.function.arguments` as
  a **dict** (because the template does `arguments | items`), but the
  OpenAI wire format ships them as **JSON strings**. Frontend has to
  normalize per backend / per model.
- Materialization: pydantic models / dataclasses / mapping types must all
  end up as plain dicts before the template runs.
- Mutations to the materialized dicts must NOT leak back to the
  caller-owned request object.

Examples: `TestPreprocessChatRequest` (sglang),
`TestPrepareRequestToolStripping` (vllm).

## `FE.2` — Parser construction & dispatch

For a given request, frontend must pick the right tool-call parser and the
right reasoning parser — based on the model name, `chat_template_kwargs`,
and runtime config. Tests pin the dispatch matrix.

Examples: `TestCreateParsers`, `TestRuntimeConfigParserName`,
`TestNoReasoningParser`.

## `FE.3` — Request shaping & sampling-param projection

OpenAI request fields → backend `SamplingParams` / equivalent. Tool
stripping when `tool_choice="none"`. Guided-decoding configuration when
`tool_choice` requires it.

Examples: `TestConvertTools`, `TestBuildToolCallGuidedDecoding`,
`TestPrepareRequestToolStripping`.

## `FE.4` — Tool-call output assembly

Backend output stream → OpenAI-shaped `tool_calls` deltas. Single, multiple,
parallel, content-then-tool, fallback when no parser fires.

Examples: `TestSingleToolCall`, `TestMultipleToolCalls`,
`TestContentWithToolCalls`, `TestSingleChunkFallback`,
`TestMalformedToolCalls`, `TestJsonArrayParserReparse`,
`TestKimiToolCallIds`.

## `FE.5` — Finish-reason mapping

Frontend layer maps the engine's raw finish_reason into OpenAI's enum,
remapping `stop` → `tool_calls` once any tool call has been emitted on a
choice (per-choice tracking). Distinct from `CASE.12` in the parser
taxonomy: that's about the parser's view; this is about the frontend's
post-parser remap.

Example: `TestMapFinishReason`.

## `FE.6` — Incremental detokenization

Token-id streams arriving from the engine → user-facing text deltas.
Includes prompt-token-id normalization, fast plain-text path (skip parser
when no tool/reasoning markers detected), and chunk-boundary handling.

Examples: `TestIncrementalDetokenization`, `TestFastPlainTextPath`,
`TestNormalizePromptTokenIds`, `TestParseJsonArrayBuffer`.

## `FE.7` — Worker subprocess boundary

Preprocessing runs in a worker subprocess (avoids the GIL on tokenizer
calls). Result objects must pickle; init must be robust; errors must
propagate cleanly across the boundary.

Examples: `TestBuildDynamoPreproc`, `TestWorkerResultPicklability`.

## `FE.8` — Error surface

How the frontend reports errors back to the client: `BackendError` for
backend issues, `InternalError` for our bugs, engine errors mapped to
HTTP-friendly shapes. Also covers deprecation warnings on legacy fields.

Examples: `TestMakeBackendError`, `TestMakeInternalError`,
`TestHandleEngineError`, `TestDeprecationWarning`.

## `FE.9` — Reasoning ↔ tool-call orchestration

When both a reasoning parser and a tool-call parser fire on the same
response, the frontend orchestrates routing (text → reasoning vs text →
tool-call markup vs text → user-visible content). Distinct from parser
`CASE.9`: that's the parser-internal view; this is the frontend assembly
view.

Examples: `TestReasoningParsing`.
