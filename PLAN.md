# PLAN: Independently toggleable Dynamo / SGLang pre- and post-processing

## Goal

Let users delegate Dynamo's Rust preprocessing and postprocessing paths to the
backend engine (SGLang first; vLLM later) **independently**, so that:

- Bugs and gaps in Dynamo's Rust tool-call / reasoning parsers / detokenizer
  can be bypassed by handing post-processing to the backend.
- Preprocessing can still be done by Dynamo (for KV-routing) **or** by the
  backend, on a per-deployment basis.
- The `dynamo.sglang` deployment can be made behaviourally identical to
  `sglang.launch_server` for `v1/chat/completions` whenever the user wants
  that, including tool calls and reasoning.
- The same shape will extend to vLLM later without surface-area churn.

The user-visible primitives:

- Worker flags: `--preprocessor {dynamo,sglang}` and
  `--postprocessor {dynamo,sglang}` (env `DYN_SGL_PREPROCESSOR`,
  `DYN_SGL_POSTPROCESSOR`).
- Rust model metadata: a new `ModelOutput` enum (`Text` | `Tokens`) that is
  the dual of the existing `ModelInput` enum.
- All four combinations valid: `Tokens/Tokens`, `Tokens/Text`, `Text/Text`,
  `Text/Tokens`.
- Deprecate `--use-sglang-tokenizer` (kept as a backwards-compatible alias
  for `--preprocessor sglang --postprocessor sglang`).
- Remove `--dyn-chat-processor sglang` from the frontend — SGLang processing
  is now configured on the worker, not on the frontend.

## Non-goals

- vLLM implementation. Design must accommodate it; implementation is deferred.
- Completions / embeddings / multimodal-only routes don't get the new
  `Text/Tokens` chat pipeline yet (warnings emitted where applicable).
- Direct/KV routing for `Text/Tokens` (rejected — KV routing needs the token
  ids that only the Dynamo preprocessor produces).

---

## Design

### Rust core

1. **`ModelOutput` enum** (`lib/llm/src/model_type.rs`)
   - Variants `Text` (default), `Tokens`.
   - `From<ModelInput>`: `Text→Text`, `Tokens→Tokens`, `Tensor→Text`. Used so
     existing call sites that only know `ModelInput` keep working unchanged.

2. **`ModelDeploymentCard`** (`lib/llm/src/model_card.rs`)
   - New `model_output: ModelOutput` (serde-defaulted for backwards compat
     with existing MDCs in etcd).
   - `requires_postprocessing()` helper.

3. **Pipeline construction by `(ModelInput, ModelOutput)`**
   (`lib/llm/src/discovery/watcher.rs`)

   | Input  | Output | Pipeline                                                                               |
   |--------|--------|----------------------------------------------------------------------------------------|
   | Tokens | Tokens | Existing `build_routed_pipeline` (Dynamo pre + Dynamo post). Completions still works.  |
   | Tokens | Text   | New `build_routed_pipeline_text_output` (Dynamo pre, backend post). KV-routable.       |
   | Text   | Text   | `PushRouter` direct passthrough. No KV routing (`round-robin`/`random` only).          |
   | Text   | Tokens | New `build_routed_pipeline_text_input_token_output`. No KV/Direct routing.             |

   Completions endpoint is registered only when `model_output == Tokens`
   today; a warning is logged otherwise (extension left to a follow-up).

4. **Two new `OpenAIPreprocessor` `Operator` impls**
   (`lib/llm/src/preprocessor.rs`)
   - `(NvChat → NvChatStream) over (PreprocessedRequest → NvChatStream)` —
     Tokens-in/Text-out. Dynamo preprocesses to `PreprocessedRequest`, stashes
     the original OpenAI request inside `extra_args["openai_request"]` so the
     SGLang worker can rebuild the `ChatCompletionRequest`, and forwards the
     parsed chat-completion stream untouched.
   - `(NvChat → NvChatStream) over (NvChat → LLMEngineOutput)` —
     Text-in/Tokens-out. Dynamo runs its preprocessor for state tracking,
     forwards the raw `NvChat` request to the backend, and reuses
     `Backend::process_token_stream` + the existing postprocessor parsing
     stream to detokenize, parse tools/reasoning, and emit chat chunks.

5. **`Backend` refactor** (`lib/llm/src/backend.rs`)
   - Extracted `Backend::process_token_stream` so the preprocessor can run
     the same detokenization / stop-sequence handling on any token stream,
     not just the one produced by `Backend`'s own `Operator` impl.
   - Promoted `DecoderParams::from_request` to `pub(crate)`.

6. **`KvPushRouter` and `DirectRoutingRouter` made generic**
   (`lib/llm/src/kv_router/push_router.rs`)
   - New `KvRouterResponse` trait: `generated_token_count`,
     `is_generation_progress`, `query_response`.
   - Implemented for `LLMEngineOutput` (existing behaviour) and
     `NvCreateChatCompletionStreamResponse` (chunks count as progress when
     they carry choices or a usage block).
   - This is what unlocks KV routing for the `Tokens/Text` pipeline — the
     router can carry text-format chunks end-to-end while still observing
     prefill completion etc.

7. **OpenAI request schema** (`lib/llm/src/protocols/openai/chat_completions.rs`)
   - Added `separate_reasoning: Option<bool>` and `stream_reasoning:
     Option<bool>` so SGLang's native reasoning-controls pass through the
     frontend instead of being rejected as unsupported.
   - All `NvCreateChatCompletionRequest` constructors in cross-protocol
     conversions and tests updated.

8. **Python bindings** (`lib/bindings/python/rust/lib.rs`,
   `_core.pyi`, `dynamo/llm/__init__.py`)
   - `ModelOutput` exported as a pyclass.
   - `register_model` and `LocalModel::attach` gain an optional
     `model_output` argument; falls back to `model_input.into()` to keep
     existing callers source-compatible.

### SGLang worker (`components/src/dynamo/sglang/`)

1. **CLI** (`backend_args.py`)
   - `--preprocessor`, `--postprocessor` choices `{dynamo, sglang}` with
     env-var backing.
   - `--use-sglang-tokenizer` marked deprecated but functional.

2. **Legacy shim** (`args.py::_apply_legacy_tokenizer_shim`)
   - When `--use-sglang-tokenizer` is set and neither `--preprocessor` nor
     `--postprocessor` is given (CLI **and** env), set both to `sglang`.
   - Explicit flags or env vars always win — supports mixed deployments
     during migration.
   - `--custom-jinja-template` + `--preprocessor sglang` rejected at startup
     (template requires Dynamo preprocessor).
   - `--skip-tokenizer-init` + any `sglang` pre/post rejected — SGLang needs
     its tokenizer for delegated processing.
   - `_merge_sglang_config_args` shim added so the SGLang `ConfigArgumentMerger`
     keeps working across upstream API versions.

3. **Model registration** (`register.py`)
   - `_model_input_from_args`, `_model_output_from_args` derive
     `ModelInput`/`ModelOutput` independently from the two flags.
   - Embedding workers continue to follow `--preprocessor` for
     `ModelInput` and `--postprocessor` for `ModelOutput`.
   - Prefill workers stay at `ModelOutput::Tokens`.
   - For chat with `ModelOutput::Text`, registration narrows the model type
     to `Chat` (`v1/completions` is not yet wired for the Text-output path).

4. **Decode handler** (`request_handlers/llm/decode_handler.py`)
   - Splits the old `use_sglang_tokenizer` flag into independent
     `use_sglang_preprocessor` / `use_sglang_postprocessor` toggles.
   - `_build_sglang_chat_request` reconstructs SGLang's
     `ChatCompletionRequest` either from the raw incoming request (when
     SGLang owns preprocessing) or from `extra_args["openai_request"]` (when
     Dynamo preprocessed and forwarded the original).
   - When SGLang owns preprocessing for chat, requests go through
     `OpenAIServingChat._process_messages` so chat templates, tools, and
     multimodal data take SGLang's native code path.
   - When SGLang owns postprocessing, replace the old
     `_process_text_stream` with one that drives SGLang's
     `_process_reasoning_stream`, `_process_tool_call_stream`,
     `_check_for_unstreamed_tool_args`, hidden-state emission, and
     `UsageProcessor.calculate_streaming_usage` — i.e. the same pipeline as
     `sglang.launch_server`.
   - Health-check payload now picks text vs token probes based on
     `--preprocessor`, not the deprecated flag.

5. **`_build_sampling_params`** updated to read split flags and to forward
   `skip_special_tokens` from Dynamo's `output_options` so the engine
   honours it under `Tokens/*` modes.

### Frontend

- Drop `sglang` from `--dyn-chat-processor` choices; SGLang pre/post is now
  worker-side. `vllm` remains.
- `--dyn-preprocessor-instrumentation` / `--dyn-preprocessor-pool-size` help
  text adjusted (sglang path no longer applicable).

### Docs

- `docs/backends/sglang/sglang-chat-processor.md` rewritten as "SGLang
  Processing Modes" with the 4-mode table and migration guide.
- `docs/agents/chat-processor-options.md` updated:
  - Old option C (SGLang chat-processor) is now the `Tokens/Text` SGLang
    postprocessor-delegation row.
  - New option E (full SGLang delegation) replaces the deprecated tokenizer
    flag entry.
- `docs/backends/sglang/sglang-reference-guide.md` updated with the new
  flags and the deprecated alias note.
- `docs/index.yml` nav entry renamed.

---

## Test plan

### Unit / integration tests

Rust:
- `lib/llm/src/http/service/openai.rs::test_chat_completions_accepts_sglang_reasoning_controls`
  — verifies `separate_reasoning` and `stream_reasoning` parse and are not
  rejected as unsupported.
- All `NvCreateChatCompletionRequest` fixtures (across
  `lib/llm/tests/*.rs`, `protocols/anthropic`, `protocols/openai/responses`,
  `protocols/unified`, `entrypoint/input/text.rs`) updated for the new
  fields.

Python (SGLang):
- `test_sglang_unit.py::test_legacy_use_sglang_tokenizer_enables_split_modes`
  — shim defaults both sides to sglang.
- `test_sglang_unit.py::test_legacy_use_sglang_tokenizer_respects_explicit_split_modes`
  — explicit CLI/env wins over shim.
- `test_sglang_unit.py::test_register_model_input_output_modes_are_independent`
  — `(pre,post)` pairs map to the right `(ModelInput, ModelOutput)`.
- `test_sglang_decode_handler.py` — new cases for SGLang reasoning and
  tool-call streaming delegation, finish-reason `matched_stop` instead of
  legacy `nvext.stop_reason`.

Frontend:
- `test_sglang_processor_unit.py::TestDeprecationWarning` — deprecation
  message points users at the new flag combo.

### End-to-end functional parity

Verify by running the same prompt against each of the five deployments and
diffing responses:

1. `sglang.launch_server` (reference)
2. `dynamo.frontend` + `dynamo.sglang --preprocessor sglang --postprocessor sglang` (`Text/Text`)
3. `dynamo.frontend --router-mode kv` + `dynamo.sglang --preprocessor dynamo --postprocessor sglang` (`Tokens/Text`)
4. `dynamo.frontend --router-mode kv` + `dynamo.sglang` (default, `Tokens/Tokens`)
5. `dynamo.frontend --router-mode round-robin` + `dynamo.sglang --preprocessor sglang --postprocessor dynamo` (`Text/Tokens`)

Each run exercises:
- Plain chat completion (streaming + non-streaming)
- Tool call (`tool_choice=auto`, with parser/tool spec)
- Reasoning model (`separate_reasoning=true`)
- Combined tools + reasoning

Model under test: `Qwen/Qwen3-VL-8B-Instruct-FP8`.

### Performance

aiperf, 1k/1k ISL/OSL, concurrency 64, 128 warmup requests, against the five
deployments above. Artifacts are written under
`artifacts/sglang-prepost/{direct-sglang,text-text,tokens-text,tokens-tokens,text-tokens,aiperf-smoke}/`
with `profile_export_aiperf.{json,csv}` and worker/frontend logs for diffing.

### Environment

- Fresh `uv venv` based on latest dynamo `main`.
- Install: `cd lib/bindings/python && maturin develop --uv && cd <root> && uv pip install -e .`
- Install matching SGLang release and latest `aiperf`.
- Single H100/H200 GPU per worker is enough for the 8B FP8 model.

---

## Rollout / compatibility

- Existing deployments using `--use-sglang-tokenizer` continue to work
  unchanged; a `FutureWarning` points at the new flags.
- Existing MDCs in etcd without `model_output` deserialize with default
  `Text`, which (via `From<ModelInput>`) collapses to the old behaviour for
  Tokens-in workers — no migration required.
- Python callers of `register_model` / `LocalModel::attach` that omit
  `model_output` keep the prior behaviour automatically.

## Future work (out of scope here)

- Mirror the worker-side pre/post flags in `dynamo.vllm` and wire them to
  the same `ModelInput`/`ModelOutput` plumbing.
- Re-enable `v1/completions` for `ModelOutput::Text` once the SGLang side
  has a delegated completions path equivalent to `OpenAIServingChat`.
- KV routing for `Text/Tokens` once a routing strategy that does not depend
  on Dynamo-side tokens is defined.
