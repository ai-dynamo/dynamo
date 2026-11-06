<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Agent protocol compatibility over TensorRT-LLM OpenEngine

Reference serving profile: JunyiXu-nv/TensorRT-LLM branch
`dev-junyix-feat-kimi-k3-lhr-serving` at `eb1f4d7bd0b75159b2cc384deb28ed212d531875`.
Dynamo baseline: `a908c1a14501b4bb03a7c80833edec0f0e571986`.

The path is the Dynamo HTTP frontend, its prompt/token/parser pipeline, the
TensorRT-LLM sidecar, and the existing `openengine.v1.Inference/Generate` RPC.
HTTP compatibility belongs to the frontend. The sidecar transports token stops,
sampling controls and results; it does not proxy an HTTP body into an engine.
No protobuf wire definitions change.

## Request and replay behavior

- Responses `additional_tools` input items are moved into `tools` before typed
  validation. Existing tool declarations are retained; malformed arrays fail
  before dispatch. Namespace and custom-tool validation continue to use the
  current Dynamo tool registry.
- Responses conversation routing uses explicit `conversation_params.conversation_id`,
  supported session headers, then `prompt_cache_key`, `client_metadata.thread_id`,
  and `client_metadata.session_id`. The fallback also supplies the gRPC engine's
  conversation identity. Existing parent/subagent metadata is retained. These
  values are routing hints, not authentication or stored conversation history.
  Sticky routing still requires the deployment's session-affinity configuration.
- Session aliases `x-claude-session-id`, `x-correlation-id`, `x-session-affinity`
  and `x-multi-turn-session-id` are accepted as fallbacks after existing explicit
  Dynamo/agent headers.
- Requested `reasoning.encrypted_content` contains `trtllm-reasoning-v1:` followed
  by standard base64 of the exact UTF-8 reasoning. This is **not encryption**.
  Readable content takes precedence on replay; owned encoded payloads are decoded
  when readable content is absent. Unknown provider encodings remain opaque.
  Malformed owned encodings are rejected rather than silently losing history.
- Kimi Responses history merges split results for one call within a consecutive
  result run. Unmatched results become user notes after that run, preserving their
  content and the binding of valid results. Other model families keep their
  existing history policy.
- Kimi namespaced calls retain an opaque `_bare` call-id marker when the model
  wrote the bare name. Native replay preserves that spelling. When an older
  client dropped the namespace of a qualified call, a unique declared namespace
  is restored; ambiguous declarations are not guessed.

## Generation boundary across gRPC

For explicit `parallel_tool_calls=false`, Kimi's complete call close
`<|close|>call<|sep|>` and GLM's `</tool_call>` are generation stops. This applies
when tools are effective, tool_choice is not none, and the worker advertises
`visible_stop_strings` in its runtime metadata. The constraint is shared by
Responses, Chat Completions and Anthropic's converted chat request.

The internal optional `StopConditions.stop_visible` field carries these parser
boundaries. The sidecar translates them into existing OpenEngine StopText
conditions and enables include_stop_in_output. The frontend decoder retains the
parser boundary while trimming ordinary user stop strings and hidden token
stops. This prevents the tool parser from seeing a cut-off call envelope.

Absent capability metadata preserves the previous guided-decoding/output-filter
fallback. Absent stop_visible remains decodable on mixed-version request wires.
Other parser families retain their existing guidance and filtering behavior.

## Response and error behavior

Dynamo retains its incremental function/custom-tool payload events, usage,
response ids, monotonic sequence numbers, early opening and disconnect handling.
It does not delay tool payloads until the whole generation finishes.

Streamed context overflow, whether encountered while obtaining the engine stream,
during the precommit peek, or after opening, receives an in-protocol failed
response with `error.code=context_length_exceeded`. A pre-stream overflow produces
HTTP 200 and created/in_progress/failed events sharing one response id. Ordinary
invalid arguments keep their existing HTTP rejection behavior. Non-streaming
overflow remains an HTTP error.

Other Responses backend failures close active output items and emit an `error`
event before the full `response.failed` snapshot. Structured context-length
error details are preferred. OpenEngine v1 does not expose structured overflow
fields, so the frontend also recognizes the current TensorRT-LLM bounded-input
error messages; unrelated invalid arguments are not classified as overflow.

## Deployment boundaries

The compatibility target is the stateless disaggregated serving profile.
Dynamo and Junyi's disaggregated service both reject previous_response_id.
This change does not introduce standalone response storage, Anthropic Message
Batches, a Slurm fleet gateway, model-specific NaN/pad-token detection, or new
KV-write verification guarantees. Those require their own storage, deployment
or backend contracts and are not supplied by the OpenEngine adapter.

Native namespace rendering, image input, existing guided constraints and
Dynamo's current worker/backend contracts are preserved.

## Validation

Run in a Docker development environment with the repository-pinned Rust 1.96.1:

```bash
cargo test --locked -p dynamo-trtllm-sidecar --lib
cargo test --locked -p dynamo-llm --no-default-features --features mm-routing --lib -- \
  protocols::openai::responses:: protocols::agents::tests:: \
  http::service::openai::tests::context_overflow_is_a_typed_failed_sse_response \
  http::service::openai::tests::ordinary_invalid_arguments_are_not_context_overflows \
  http::service::openai::tests::body_affinity_obeys_explicit_body_then_header_then_fallback_precedence \
  preprocessor::tests::single_call_boundary_requires_worker_support_and_explicit_false \
  backend::tests::parser_stops_are_visible_while_user_stops_stay_hidden
cargo fmt --all -- --check
```

The sidecar suite exercises actual local gRPC client/server transport with a fake
engine, including parser-stop delivery and hidden-token retention behavior.
The frontend suite exercises request conversion, history replay, decoder stops
and serialized SSE output. These checks do not prove GPU sampling, Kimi model
quality, or physical KV transfer safety. Live-engine tests remain separately
available under `src/tests/e2e.rs` and require running OpenEngine endpoints.

Validated in the task Docker container: 195 frontend tests passed; 82 sidecar
tests passed, with 4 live-engine tests ignored. A focused readable/encoded
reasoning precedence regression also passed after the final adjustment.
Clippy passed with `-D warnings` for the sidecar's all-targets check and the
frontend library; formatting passed. The test image used
`CUDARC_CUDA_VERSION=13010` because its cached toolkit is CUDA 13.4, newer than
the pinned cudarc build detector supports. No GPU execution was performed.
