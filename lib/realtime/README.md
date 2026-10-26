<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Realtime Sidecar Adapters

**Experimental.** `dynamo-realtime` owns upstream connections, audio transport
conversion, and session/turn state. It has no dependency on the Dynamo frontend,
worker bootstrap, or inference engines. The vLLM sidecar connects this library to
the shared `Worker` lifecycle and the existing public APIs.

```text
Client → Dynamo frontend → bidirectional request plane → vLLM sidecar → upstream WebSocket
Client → /v1/audio/speech → raw media request plane     → vLLM sidecar → upstream HTTP
```

Model preprocessing, token feedback, sampling, and audio generation run upstream.
The Python realtime and Omni handlers remain available.

## First Iteration

| Mode | Upstream contract | Public behavior |
| --- | --- | --- |
| `realtime --protocol asr` | vLLM `/v1/realtime`, 16 kHz PCM16, `transcription.*` events | 24 kHz PCM16 transcription sessions, deltas before commit, repeated/queued turns, clear, token usage |
| `realtime --protocol omni` | vLLM-Omni OpenAI `/v1/realtime` | Independent audio turns, output modality selection, concurrent generation with responses forwarded in turn order |
| `realtime --protocol omni-duplex` | vLLM-Omni native `/v1/realtime?duplex=1` | Opt-in native JSON sessions with continuous input/output and upstream-owned playback, interruption, history, and resume |
| `speech` | vLLM-Omni `/v1/audio/speech` | Existing HTTP TTS API, incremental PCM/WAV, legacy frontend aggregation, codecs/speed handled upstream |

The wire contracts were checked against vLLM **v0.31.0** and vLLM-Omni
**v0.31.0rc1**. CPU tests exercise mock services and decode emitted events with the
frontend's pinned protocol types. Exploratory H200 comparisons exercised Qwen3
ASR, TTS, and Omni models against the current handlers and direct upstream APIs.
Broader quality and load testing, repeated-turn GPU tests, and engine-side
cancellation validation remain required before rollout; these are not certified
deployment recipes.

The pinned Omni OpenAI handler buffers input until commit; the adapter then sends
`response.create`. It does not provide streaming inference before commit.
It also does not carry conversation history between turns. As in the existing
Omni bridge, `conversation.item.*` and explicit `response.create` client events
do not change that independent-turn behavior.

Omni's legacy `transcription.*` audio endpoint and its native duplex endpoint
(`duplex=1`) use different contracts. They are not selected by `--protocol omni`.
Use the explicit `omni-duplex` profile below for native sessions. A model's name
does not select or imply these capabilities.

## Native Duplex PoC

This profile targets vLLM-Omni **v0.32.0rc1** with MiniCPM-o-4_5. Start the
upstream with its native duplex deployment configuration, then run:

```bash
dynamo-vllm-sidecar realtime \
  --protocol omni-duplex \
  --http-endpoint "${ENGINE_HTTP_URL:?}" \
  --model "${MODEL_ID:?}"
```

Connect to the matching Dynamo frontend at
`/v1/realtime?duplex=1&model=<URL-encoded-model-id>`. Send native `session.update`
or `session.resume` as the first JSON text frame. The frontend does not synthesize
a greeting: the upstream owns `session.created`, capability negotiation, errors,
and all subsequent events. This is the native Omni protocol, not an OpenAI Realtime
schema extension. The ordinary `/v1/realtime` typed path is unchanged. Both
frontend and sidecar must support this opt-in profile; discovery advertises
`runtime_data.realtime_protocol=omni-duplex` to keep the profiles separate.

The relay preserves JSON fields, omitted values, explicit nulls, event IDs, audio
sample rates, video timestamps, and playback positions. It opens one upstream
WebSocket per client attachment with `duplex=1&autostart=0`. It does not split
turns, resample audio, or insert commit/response events. Upstream owns validation,
reference-media access, voice, session settings, history, and execution of
`playback.ack`, `barge_in`, `response.cancel`, and `conversation.item.truncate`.
Deploy upstream with media-access policies appropriate for the clients; this
profile does not apply the separate speech adapter's reference-media policy.

Client disconnect closes the upstream socket without sending `session.close`,
allowing the engine's own bounded resume lease to apply. Explicit `session.close`
is forwarded, and worker shutdown closes attached sessions. **Use one upstream
worker for this PoC.** Cross-worker resume routing, migration, and recovery after
owner failure are not implemented. Resuming must reach the original engine.

The existing event-count and buffered-output byte bounds apply per attachment;
the output byte limit also bounds each input JSON frame. The frontend additionally
caps native input messages at 8 MiB. Turn-count and decoded
audio limits are rejected in this mode because upstream owns native media and
session limits. The post-commit response timeout does not apply; configure
session lifetime/idle policy upstream. Startup checks model identity and a
native session handshake, including independent input/output capability, then
explicitly closes the probe session. Active health probes check HTTP health and
model identity without consuming session slots. These checks do not validate inference.

LLM text sessions and streaming prefill are outside this iteration. Streaming
prefill will need the corresponding upstream vLLM/vllm-rs APIs.

## Launch Modes

Start the engine with stock `vllm serve` in its own environment. Dynamo is not
required in the engine environment. Set `ENGINE_HTTP_URL` to the engine's base
URL and `MODEL_ID` to the exact ID returned by its `/v1/models`.

With the same discovery and request-plane settings as the frontend:

```bash
dynamo-vllm-sidecar realtime \
  --protocol asr \
  --http-endpoint "${ENGINE_HTTP_URL:?}" \
  --model "${MODEL_ID:?}"
```

Use `--protocol omni` for the Omni OpenAI handler, or start a speech worker:

```bash
dynamo-vllm-sidecar speech \
  --http-endpoint "${ENGINE_HTTP_URL:?}" \
  --model "${MODEL_ID:?}"
```

The embedded launcher accepts the same subcommands:

```bash
python -m dynamo.vllm.sidecar realtime \
  --protocol asr \
  --http-endpoint "${ENGINE_HTTP_URL:?}" \
  --model "${MODEL_ID:?}"
```

The existing `--grpc-endpoint` invocation still starts the LLM gRPC sidecar.
Media modes require aggregated workers, register without tokenizer downloads, and
reject token-pipeline parser, disaggregation, encoder-routing, and RL options.

Use `DYN_SIDECAR_API_KEY` for upstream authentication. HTTP(S) base URLs can contain
a reverse-proxy path prefix. Keys are sent in headers; redirects are not followed
for authenticated upstream requests.

## Session and Delivery Rules

ASR clients begin with:

```json
{
  "type": "session.update",
  "session": {
    "type": "transcription",
    "audio": {
      "input": {
        "format": {"type": "audio/pcm", "rate": 24000},
        "transcription": {"model": "<MODEL_ID>"},
        "turn_detection": null
      }
    }
  }
}
```

Send base64 PCM16 through `input_audio_buffer.append` and close a turn with
`input_audio_buffer.commit`. ASR supports English and explicit client commits,
without prompts, noise reduction, or server VAD. A stateful, band-limited
resampler converts the public 24 kHz stream to vLLM's 16 kHz transport.

Input EOF finishes accepted work. Context stop/kill or dropping the output stream
closes upstream connections. ASR clear cancels the uncommitted turn; committed
turns continue. Omni clear discards the active input buffer without cancelling
committed responses. Each turn gets a separate upstream connection.

Speech requests keep their public delivery and codec fields. The sidecar enables
raw upstream streaming only when `nvext.frontend_accepts_audio_chunks` is true,
the codec is PCM/WAV, speed is unchanged, and delivery is not a URL. Otherwise it
returns one aggregated worker response. Upstream failures never produce a
successful completion.

The upstream speech schema owns codec support; the pinned Omni release does not
support AAC. Sampling passthrough fields `temperature`, `top_p`, and `top_k` are
mapped to upstream `extra_params`. Nested load locations and policy overrides
are rejected before dispatch.

`data_source=url` requires both `--media-output-dir` and
`--media-output-http-url`. Expose that shared directory through the configured
HTTP server and manage retention there. Output names are generated by the worker.
Reference audio is resolved through Dynamo's media-fetch policy before forwarding
a data URI upstream; client-supplied filesystem paths are rejected.
`DYN_MM_ALLOW_INTERNAL=1` enables internal reference-audio sources.

## Bounds and Health

| Option | Default |
| --- | --- |
| `--concurrent-turns` | ASR 1; Omni 8 |
| `--queued-turns` | ASR 1; Omni 0 |
| `--event-capacity` | 64 events per turn |
| `--max-audio-chunk-bytes` | 4 MiB decoded |
| `--max-turn-audio-bytes` | 60 seconds of mono 24 kHz PCM16 |
| `--max-buffered-output-bytes` | 8 MiB per turn, including queued delivery |
| `--connect-timeout-secs` | 30 |
| `--response-timeout-secs` | 120 after realtime commit; 120 per TTS idle interval |
| `--startup-deadline-secs` | 1800 |
| `--startup-retry-secs` | 1 |
| `speech --max-response-bytes` | 256 MiB |
| `speech --tts-ref-audio-max-bytes` | 16 MiB |
| `speech --tts-ref-audio-timeout-secs` | 30 |

Overflow is reported instead of dropping audio silently. Commit and clear do not
consume audio-queue capacity. Output backpressure reaches the upstream socket.

Startup and active canaries check upstream `/health` and model identity.
Realtime canaries also check the WebSocket protocol handshake. These are service
health checks, not inference/quality canaries. Shared worker readiness and
graceful shutdown continue to apply.

## Validation

```bash
cargo test -p dynamo-realtime
cargo test -p dynamo-backend-common -p dynamo-sidecar-common \
  -p dynamo-vllm-sidecar --no-default-features --lib
cargo test -p dynamo-vllm-sidecar --no-default-features --test executable
cargo test -p dynamo-vllm-sidecar --no-default-features --test realtime
```

Before rollout, compare direct-engine, Python-handler, and sidecar runs with the
same model/release and inputs. Check ASR transcript quality, TTS audio quality and
codec handling, first-output latency, realtime factor, and concurrent Omni turn
ordering. Verify that disconnect, clear, and worker shutdown release upstream
engine requests, not only sockets.
