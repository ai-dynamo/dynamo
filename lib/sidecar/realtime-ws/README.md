<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Experimental upstream WebSocket sidecar

This local prototype registers a real Dynamo worker and uses upstream vLLM or
vLLM-Omni WebSockets for inference. It requires no Dynamo code in the upstream
engine container. It is intentionally limited to one utterance per connection.

```text
ASR / Omni:
client WS -> Dynamo frontend -> Dynamo bidirectional runtime -> Rust sidecar
          -> upstream Python API WS -> upstream engine

TTS:
client HTTP speech -> Dynamo frontend -> Dynamo streaming runtime -> Rust sidecar
                   -> upstream speech WS -> upstream engine
```

The prototype uses generic JSON values on the worker boundary; the frontend
retains its existing typed public request/response validation. The sidecar owns
the documented translations below. Model preprocessing, scheduling, generation,
and model-specific output processing remain upstream-owned.

## Build and start

```bash
cargo build --release -p dynamo-realtime-ws-sidecar --no-default-features
cargo test -p dynamo-realtime-ws-sidecar --no-default-features
```

The build needs the repository's normal Rust system prerequisites, including
`protoc`. The executable is `target/release/dynamo-realtime-ws-sidecar` unless
`CARGO_TARGET_DIR` is set. Compile optimized code before measuring performance.

Use the same discovery/request/response settings as the Dynamo frontend. The
sidecar accepts the existing `--discovery-backend`, `--request-plane`,
`--response-plane`, and `--event-plane` flags and their `DYN_*` environments.
The upstream service must already be running and support the selected model.

```bash
dynamo-realtime-ws-sidecar --profile asr --model Qwen/Qwen3-ASR-0.6B \
  --upstream-url ws://127.0.0.1:8001/v1/realtime \
  --namespace rtws --component backend --endpoint generate

dynamo-realtime-ws-sidecar --profile omni --model Qwen/Qwen3-Omni-30B-A3B-Instruct \
  --upstream-url ws://127.0.0.1:8001/v1/realtime \
  --namespace rtws --component backend --endpoint generate

dynamo-realtime-ws-sidecar --profile tts --model Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice \
  --upstream-url ws://127.0.0.1:8001/v1/audio/speech/stream \
  --namespace rtws --component backend --endpoint generate
```

These examples select protocol profiles; they do not supply the model-specific
upstream launch configuration. In particular, Qwen3-ASR needs an upstream model
architecture with realtime support. Use the benchmark campaign's pinned engine
versions, configuration, and hardware for comparisons.

## Implemented paths

| Profile | Dynamo contract | Upstream contract | Translation |
| --- | --- | --- | --- |
| ASR | `/v1/realtime`, nested transcription session, 24 kHz PCM16 | `/v1/realtime`, top-level model selection, 16 kHz PCM16 | Start upstream generation on first append; translate final commit and transcription output events |
| Omni | `/v1/realtime`, legacy Dynamo 16 kHz audio input | `/v1/realtime`, modern 24 kHz turn-based audio session | Normalize audio session, resample input, commit then `response.create`, forward modern output events |
| TTS | `/v1/audio/speech`, PCM HTTP streaming | `/v1/audio/speech/stream`, `session.config`, `input.text`, `input.done` | Forward supported voice/utterance controls; convert binary PCM frames to worker audio chunks |

`--input-rate` can override ASR/Omni input to 16000 or 24000 for diagnostics;
clients must match it. `--max-output-tokens` sets Omni's upstream session limit.
The public frontend emits `session.created`; the sidecar suppresses upstream's
duplicate and acknowledges the original session update itself.
It also suppresses Omni v0.31.0rc1's obsolete `conversation.created` handshake
event, which the current Dynamo public schema does not expose.

Audio conversion uses a Kaiser-5 polyphase FIR and 100 ms input blocks. Its filter
construction follows SciPy's default `resample_poly`. ASR's current Python worker
also batches on 100 ms boundaries. Use 100 ms audio chunks for the initial paired
Omni comparison to avoid introducing a distinct batching delay. The native WS
wire adds a PCM16 quantization step versus the current Python engine's float
input; evaluate transcript/output quality along with latency. The prototype
does not promise sample-identical engine inputs across both paths.

TTS requires `response_format=pcm`, `speed=1`, and inline audio. It preserves
`voice`, `language`, `task_type`, `instructions`, `ref_audio`, `ref_text`, and
`max_new_tokens`. Frontend media passthrough supports `seed`,
`initial_codec_chunk_frames`, `non_streaming_mode`, `x_vector_only_mode`, and
`speaker_embedding`, which exist in the upstream session config. Unknown
passthrough settings fail explicitly. `stream` is a frontend delivery hint;
the upstream WS always requests progressive PCM audio.

## Deliberate gaps

| Feature | Prototype / upstream limitation |
| --- | --- |
| Multiple turns, queued turns, overlapping Omni responses | Prototype serves one utterance then closes. Additional Rust orchestration and potentially per-turn upstream sockets can restore current behavior. |
| Clear/reset, mid-generation session updates | Explicitly rejected by the prototype. Native ASR lacks clear/cancel events; use a new upstream connection and validate cleanup. |
| Text realtime and incremental text prefill | Not implemented. ASR WS cannot provide ordinary text generation; use another upstream API and implement the session state. Streaming prefill requires an upstream extension. |
| Omni incremental input into active inference | Current turn-based upstream WS buffers until commit/response creation. Sidecar changes alone cannot restore model computation during append. |
| Text-only Omni responses and custom output configuration | Prototype always requests audio plus its native transcript at 24 kHz. Extend normalization and test each supported upstream output configuration. |
| Conversation/history/tool behavior | No compatibility work here. The prototype forwards only update/append/commit; upstream-native richer sessions require deliberate public semantics. |
| Audio formats, speed, URL delivery, timestamps | Prototype only emits mono 24 kHz PCM for TTS. Some require codec/delivery adaptation; upstream streaming TTS specifically requires speed=1. |
| Production worker lifecycle | Registers through the real runtime and unregisters on signal, but does not implement full `Worker` admission, upstream readiness probes, telemetry, or management controls. |
| TLS and authentication | Prototype accepts `ws://` for a colocated upstream only. Add authenticated/TLS upstream configuration for other placements. |

Transport cancellation closes the upstream socket; stream drop also drops the
socket. Writes, reads, and normal close have bounds. This does not constitute an
engine abort acknowledgement; GPU cleanup must be verified against each upstream
release. Backpressure is pull-driven through the runtime response stream.

## Validation

Fake-upstream tests exercise ASR rate conversion and commit ordering, parse its
outputs using the frontend's real `RealtimeServerEvent` schema, exercise TTS binary
PCM and terminal completion, and verify cancellation closes an idle upstream
socket. These CPU tests do not establish model compatibility or performance;
real-model results belong to the benchmark campaign under `benchmarks/realtime_ws`.
