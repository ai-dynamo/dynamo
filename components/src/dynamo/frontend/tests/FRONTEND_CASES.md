# Frontend Chat-Processor Test Cases

## What is the frontend (FE)?

The **frontend (FE)** is the node between the OpenAI HTTP API and the inference engine. It owns request **preprocessing** (turn an OpenAI request into the token_ids the engine consumes) and response **assembly** (turn the engine's raw token/text stream back into OpenAI-shaped chunks). It does not run the model — it shapes the engine's input and re-shapes its output. Code lives under `components/src/dynamo/frontend/`; tests under `components/src/dynamo/frontend/tests/`.

Per-request pipeline (vllm path shown; sglang mirrors it with `sglang_prepost.py` / `sglang_processor.py`):

```text
USER REQUEST (OpenAI /chat/completions)
  -> preprocess_chat_request   validate, pick tool parser, run chat template, tokenize  -> prompt_token_ids
  -> routed_engine.generate    engine streams raw outputs (token_ids + text + finish_reason)
  -> StreamingPostProcessor.process_output (per chunk)   assemble OpenAI delta (content / reasoning / tool_calls), remap finish_reason
  -> yield OpenAI ChatCompletionChunk(s) back to the client
```

The bug-prone core is `process_output` — a stateful streaming state machine (it accumulates text/token_ids and tracks reasoning-done, in-progress tool calls, and per-choice emission across chunks). That is where reasoning/tool markup leaks, wrong finish_reason, and dropped/merged tool calls happen.

## Inputs to the FE (examples)

The FE has two input surfaces.

**1. The client request** — an OpenAI `/chat/completions` payload. Relevant fields: `messages[]` (multi-turn; roles system/user/assistant/tool; an assistant message may carry `tool_calls` whose `arguments` arrive as JSON **strings** on the wire), `tools[]` (function schemas), `tool_choice` (`auto` / `none` / `required` / named), sampling params (`temperature`, `top_p`, `max_tokens`, `stop`, `seed`, `n`), `stream`, `chat_template_kwargs` (e.g. `enable_thinking`, `reasoning_effort`), `add_generation_prompt`.

Example — tool-calling request:

```json
{
  "model": "Qwen/Qwen3-...",
  "messages": [
    {"role": "system", "content": "You are a weather bot."},
    {"role": "user", "content": "Weather in NYC?"}
  ],
  "tools": [
    {"type": "function", "function": {"name": "get_weather",
      "parameters": {"type": "object", "properties": {"location": {"type": "string"}}}}}
  ],
  "tool_choice": "auto",
  "stream": true
}
```

Example — multi-turn with a prior assistant tool call (the FE.preprocess.1 normalization case: `arguments` is a JSON string, not a dict):

```json
{
  "messages": [
    {"role": "user", "content": "Weather in NYC?"},
    {"role": "assistant", "tool_calls": [
      {"id": "call_1", "type": "function",
       "function": {"name": "get_weather", "arguments": "{\"location\": \"NYC\"}"}}]},
    {"role": "tool", "tool_call_id": "call_1", "content": "{\"temp\": 72}"},
    {"role": "user", "content": "And tomorrow?"}
  ],
  "tools": [ /* ... */ ]
}
```

**2. The engine output stream** — one output object per `output.index` (`n>1` interleaves choices). Each carries `token_ids` (delta), `text` (delta, already stop-trimmed by the engine), `finish_reason` (raw engine/router string), `logprobs`. Conceptually:

```text
output(index=0, text="<tool_call>{\"name\":\"get_weather\"",            token_ids=[...], finish_reason=None)
output(index=0, text=",\"arguments\":{\"location\":\"NYC\"}}</tool_call>", token_ids=[...], finish_reason="stop")
```

## Outputs from the FE (examples)

The FE emits OpenAI `ChatCompletionChunk` choices, each `{index, delta, finish_reason, logprobs}` where `delta` is `{role, content?, reasoning_content?, tool_calls?}`.

Plain text:

```json
{"index": 0, "delta": {"role": "assistant", "content": "It is 72F in NYC."}, "finish_reason": null}
{"index": 0, "delta": {}, "finish_reason": "stop"}
```

Tool call — note `finish_reason` remapped `stop` -> `tool_calls`:

```json
{"index": 0, "delta": {"role": "assistant", "tool_calls": [
  {"index": 0, "id": "call_abc", "type": "function",
   "function": {"name": "get_weather", "arguments": "{\"location\": \"NYC\"}"}}]},
 "finish_reason": "tool_calls"}
```

Reasoning + tool call — reasoning text routed to `reasoning_content`, never leaked into `content` or into the tool markup:

```json
{"index": 0, "delta": {"role": "assistant", "reasoning_content": "User wants weather; I'll call the tool."}, "finish_reason": null}
{"index": 0, "delta": {"role": "assistant", "tool_calls": [
  {"index": 0, "id": "call_abc", "type": "function",
   "function": {"name": "get_weather", "arguments": "{\"location\": \"NYC\"}"}}]},
 "finish_reason": "tool_calls"}
```

The contract these tests pin (the concrete failures): no empty `tool_calls` when the model called a tool; no tool-call markup or special tokens leaked into `content` / `reasoning_content`; no duplicated or merged calls; correct call IDs (Kimi uses sequential, not tool_index); `finish_reason == "tool_calls"` once a tool was called on that choice.

## How Dynamo works vs peer implementations

**Entry point (Dynamo).** `python -m dynamo.frontend` (`main.py`) starts the OpenAI HTTP server — the server itself is Rust in `lib/llm` (`main.py` calls `run_input(runtime, "http", engine)`). Per request the Rust service calls back into the Python **processor** wired through `EngineFactory.chat_engine_factory` (`main.py::setup_engine_factory`). The per-request Python entry point is `VllmProcessor.generator` (`vllm_processor.py`), or `SglangProcessor` for the sglang backend — it runs preprocess + engine + postprocess on this node.

**Peers (vLLM / SGLang).** The reference implementations are the engines' own OpenAI serving layers: vLLM's `OpenAIServingChat` (`vllm/entrypoints/openai/serving_chat.py`) and SGLang's `serving_chat.py` (`sglang/srt/entrypoints/openai/serving_chat.py`). Those own the full request -> response loop inside the engine process.

**The key difference.** Dynamo does **not** call the peer end-to-end serving layer. It reuses the peers' **primitives** and reimplements the **orchestration** itself, so the FE can plug into Dynamo's router / disagg engine and own the streaming contract. What Dynamo imports from each peer:

| Concern | Dynamo code | vLLM library | SGLang library |
|---|---|---|---|
| Request / delta models | `prepost.py` | `vllm.entrypoints.openai...protocol` (`ChatCompletionRequest`, `DeltaToolCall`, `DeltaMessage`) | `sglang.srt.entrypoints.openai.protocol` (`Tool`, `Function`, `ToolChoice`) |
| Tool parser | `prepost.py` / `sglang_prepost.py` | `vllm.tool_parsers.ToolParser` | `sglang.srt.function_call.FunctionCallParser`, `JsonArrayParser` |
| Reasoning parser | `prepost.py` / `sglang_prepost.py` | `vllm.reasoning.ReasoningParser` | `sglang.srt.parser.reasoning_parser.ReasoningParser` |
| Chat template / tokenizer | `prepost.py` / `sglang_processor.py` | `vllm.renderers.ChatParams`, `vllm.tokenizers` | `sglang.srt.utils.hf_transformers_utils.get_tokenizer` |
| Request shaping (orchestration) | **Dynamo-owned** `_prepare_request`, `preprocess_chat_request` | vLLM does this inside `OpenAIServingChat` | SGLang does this inside `serving_chat.py` |
| Streaming response assembly | **Dynamo-owned** `StreamingPostProcessor.process_output` | vLLM: `OpenAIServingChat` | SGLang: `serving_chat.py` |

Consequence: when a peer fixes a tool/reasoning quirk inside its own `serving_chat.py`, Dynamo does **not** inherit it for free — the orchestration is reimplemented here, so the fix has to be ported. That reimplemented orchestration is exactly what the FE.* cases test. (Dynamo deliberately mirrors specific peer helpers — e.g. `sglang_prepost.py` mirrors `serving_chat._get_reasoning_from_request` — but by copy, not by call.)

## Companion taxonomies

`FE.*` is one of four sibling test taxonomies; the others cover adjacent surfaces:

| File | Scope | Prefix |
|---|---|---|
| `lib/parsers/TOOLCALLING_CASES.md` | Tool-call parser behavior on **model output** | `TOOLCALLING.batch.*`, `TOOLCALLING.stream.*`, `TOOLCALLING.fmt.*`, `TOOLCALLING.xml.*`, `TOOLCALLING.harmony.*` |
| `lib/parsers/REASONING_CASES.md` | Reasoning parser behavior on **model output** | `REASONING.batch.*`, `REASONING.stream.*` |
| `lib/parsers/PIPELINE_CASES.md` | Pipeline-boundary contracts (parser output independence from upstream metadata) | `PIPELINE.*` |
| `components/src/dynamo/frontend/tests/FRONTEND_CASES.md` | Chat-processor layer: request preprocessing, output assembly, error surface, worker plumbing | `FE.process_output.*` (behavioral fixtures) + `FE.preprocess.*` / `FE.response_misc.*` (unit) |

Backends covered by this taxonomy: **vllm** (`prepost.py` + `vllm_processor.py`) and **sglang** (`sglang_prepost.py` + `sglang_processor.py`). trtllm has its own architecture under `components/src/dynamo/trtllm/` and is out of scope here.

## Cases: FE.process_output (fixtures) vs FE.preprocess / FE.response_misc (unit)

The 9 pipeline stages split into three groups, all prefixed `FE.*`. The number is the pipeline position; the prefix tells you the group and how the stage is tested.

- **`FE.process_output.{4,6,9}`** — stages that are a deterministic input → output transform both backends implement with the *same* `process_output` contract, so the same case replays on vllm **and** sglang from shared YAML fixtures (`fixtures/frontend_*.yaml`) and is rendered in the behavioral parity matrix (`tests/parity/frontend/PARITY.html`): **4** tool-call assembly, **6** incremental detok, **9** reasoning ↔ tool orchestration.
- **`FE.preprocess.{1,2,3,7}`** — request-side stages (request → prompt): **1** chat-template, **2** parser dispatch, **3** request shaping, **7** worker subprocess boundary. Per-backend annotated unit tests, not a shared replay.
- **`FE.response_misc.{5,8}`** — response-side stages *outside* `process_output`'s streaming assembly: **5** finish-reason mapping, **8** error surface. Per-backend annotated unit tests.

A test declares its stage(s) with a trailing `# FE.<group>.N` comment, so `grep -r 'FE.response_misc.5' components/src/dynamo/frontend/tests/` finds every test for that stage across both backends.

## Quick reference

- **`FE.preprocess.1`** Chat-template input preprocessing — multi-turn assistant `tool_calls` with JSON-string `arguments`, message materialization, role handling, tool messages, system-merging. (Where Richard's qwen3.5 fix in #8792 lives.)
- **`FE.preprocess.2`** Parser construction & dispatch — instantiate the right tool-call / reasoning parser from a request's `chat_template_kwargs`, model name, runtime config; handle "no parser" gracefully.
- **`FE.preprocess.3`** Request shaping & sampling-param projection — OpenAI fields → backend kwargs, tool stripping when `tool_choice="none"`, guided-decoding setup.
- **`FE.process_output.4`** Tool-call output assembly — model output stream → OpenAI-shaped `tool_calls` deltas. Single, multiple, content-mixed, fallback paths.
- **`FE.response_misc.5`** Finish-reason mapping — frontend-layer remap (`stop`/`length`/`tool_calls`). Distinct from parser-layer `PIPELINE.finish_reason` which covers the parser's view of the raw signal.
- **`FE.process_output.6`** Incremental detokenization — token-id stream → text, prompt-token-id normalization, fast plain-text path.
- **`FE.preprocess.7`** Worker subprocess boundary — preprocessing runs in a subprocess; result picklability, init, error propagation across the boundary.
- **`FE.response_misc.8`** Error surface — `BackendError` / `InternalError` / engine-error handling, malformed responses, stream errors, deprecation warnings.
- **`FE.process_output.9`** Reasoning ↔ tool-call orchestration — both parsers active on the same response; distinct from `REASONING.batch.2` which is purely on output text.

## Annotation convention

Tests carry a one-line trailing comment naming the stage(s) they cover — `# FE.process_output.N` for the behavioral-fixture stages (4/6/9), `# FE.preprocess.N` / `# FE.response_misc.N` for the unit stages:

```python
class TestMapFinishReason:  # FE.response_misc.5
    def test_stop_to_tool_calls_when_emitted(self): ...
```

Or per-test when a class spans multiple categories:

```python
class TestUtilities:
    def test_make_backend_error(self): ...  # FE.response_misc.8
    def test_normalize_prompt_token_ids(self): ...  # FE.process_output.6
```

`grep -r 'FE.preprocess.1' components/src/dynamo/frontend/tests/` returns every chat-template-preprocessing test across vllm + sglang in one shot.

---

## `FE.preprocess.1` — Chat-template input preprocessing

The frontend rebuilds the prompt from a multi-turn message history before handing it to the backend. Several quirks live here:

- Some chat templates expect assistant `tool_calls.function.arguments` as a **dict** (because the template does `arguments | items`), but the OpenAI wire format ships them as **JSON strings**. Frontend has to normalize per backend / per model.
- Materialization: pydantic models / dataclasses / mapping types must all end up as plain dicts before the template runs.
- Mutations to the materialized dicts must NOT leak back to the caller-owned request object.

Examples: `TestPreprocessChatRequest` (sglang), `TestPrepareRequestToolStripping` (vllm).

## `FE.preprocess.2` — Parser construction & dispatch

For a given request, frontend must pick the right tool-call parser and the right reasoning parser — based on the model name, `chat_template_kwargs`, and runtime config. Tests pin the dispatch matrix.

Examples: `TestCreateParsers`, `TestRuntimeConfigParserName`, `TestNoReasoningParser`.

## `FE.preprocess.3` — Request shaping & sampling-param projection

OpenAI request fields → backend `SamplingParams` / equivalent. Tool stripping when `tool_choice="none"`. Guided-decoding configuration when `tool_choice` requires it.

Examples: `TestConvertTools`, `TestBuildToolCallGuidedDecoding`, `TestPrepareRequestToolStripping`.

## `FE.process_output.4` — Tool-call output assembly

Backend output stream → OpenAI-shaped `tool_calls` deltas. Single, multiple, parallel, content-then-tool, fallback when no parser fires.

Examples: `TestSingleToolCall`, `TestMultipleToolCalls`, `TestContentWithToolCalls`, `TestSingleChunkFallback`, `TestMalformedToolCalls`, `TestJsonArrayParserReparse`, `TestKimiToolCallIds`.

## `FE.response_misc.5` — Finish-reason mapping

Frontend layer maps the engine's raw finish_reason into OpenAI's enum, remapping `stop` → `tool_calls` once any tool call has been emitted on a choice (per-choice tracking). Distinct from `PIPELINE.finish_reason` in the parser taxonomy: that's about the parser's view (output independence from upstream signal); this is about the frontend's post-parser remap.

Example: `TestMapFinishReason`.

## `FE.process_output.6` — Incremental detokenization

Token-id streams arriving from the engine → user-facing text deltas. Includes prompt-token-id normalization, fast plain-text path (skip parser when no tool/reasoning markers detected), and chunk-boundary handling.

Examples: `TestIncrementalDetokenization`, `TestFastPlainTextPath`, `TestNormalizePromptTokenIds`, `TestParseJsonArrayBuffer`.

## `FE.preprocess.7` — Worker subprocess boundary

Preprocessing runs in a worker subprocess (avoids the GIL on tokenizer calls). Result objects must pickle; init must be robust; errors must propagate cleanly across the boundary.

Examples: `TestBuildDynamoPreproc`, `TestWorkerResultPicklability`.

## `FE.response_misc.8` — Error surface

How the frontend reports errors back to the client: `BackendError` for backend issues, `InternalError` for our bugs, engine errors mapped to HTTP-friendly shapes. Also covers deprecation warnings on legacy fields.

Examples: `TestMakeBackendError`, `TestMakeInternalError`, `TestHandleEngineError`, `TestDeprecationWarning`.

## `FE.process_output.9` — Reasoning ↔ tool-call orchestration

When both a reasoning parser and a tool-call parser fire on the same response, the frontend orchestrates routing (text → reasoning vs text → tool-call markup vs text → user-visible content). Distinct from `REASONING.batch.2`: that's the parser-internal view (reasoning parser must leave tool-call markers intact for downstream); this is the frontend assembly view.

Examples: `TestReasoningParsing`.
