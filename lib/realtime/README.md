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
| `speech` | vLLM-Omni `/v1/audio/speech` | Existing HTTP TTS API, incremental PCM/WAV, legacy frontend aggregation, codecs/speed handled upstream |

The adapters target vLLM **v0.33.0** and vLLM-Omni **v0.33.0rc1**.
Omni native duplex, text-only LLM sessions, and streaming prefill are outside
these launch modes.

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
reject token-pipeline parsers, enabled structural tags, disaggregation,
encoder-routing, and RL options.
Realtime worker selection supports round-robin/random routing for the whole
session; other routing modes are rejected.

Use `DYN_SIDECAR_API_KEY` for upstream authentication. HTTP(S) base URLs can contain
a reverse-proxy path prefix. Keys are sent in headers; redirects are not followed
for authenticated upstream requests.
Native trust roots for secure WebSockets are cached on first use, normally during
realtime startup. Restart the worker to pick up trust-store changes.

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

For Omni (`type: realtime`), the pinned public [audio schema][public-audio-schema]
requires both `input` and `output` when `audio` is present; its output object
requires format, speed and voice. This is a shared frontend constraint for both
worker paths: an input-only Omni audio update fails before reaching either
worker. Transcription sessions use a different schema and accept the input-only
ASR example above. Omitted session fields are preserved by the Omni adapter, but
that does not make every nested partial JSON shape valid at the frontend boundary.

[public-audio-schema]: https://github.com/64bit/async-openai/blob/d453592328fdbb3f59c1ea74f8e6cdf7f9140efb/async-openai/src/types/realtime/session.rs#L239-L294

Send base64 PCM16 through `input_audio_buffer.append` and close a turn with
`input_audio_buffer.commit`. ASR supports English and explicit client commits,
without prompts, noise reduction, or server VAD. A stateful, band-limited
resampler converts the public 24 kHz stream to vLLM's 16 kHz transport.

Omni accepts independent manual-commit turns with mono 24 kHz PCM16 input and
text/audio output modalities. It does not carry conversation history, handle
client `response.cancel`, or support server VAD. `conversation.item.*` and
explicit client `response.create` events do not change this turn behavior.
Audio stays in a bounded local buffer until a nonempty turn commits; only then
does the sidecar connect and upload it upstream. There is no pre-commit inference.

The bundled client defaults to 16 kHz for the Python Omni worker. Select 24 kHz
when sending to the sidecar so the samples retain their original duration:

```bash
python examples/backends/vllm/launch/realtime_audio_client.py \
  --session-type realtime --model "${MODEL_ID:?}" \
  --input-sample-rate 24000 --input-audio speech.wav
```

Omni settings are captured on the first nonempty append. Session updates preserve
omitted fields and affect subsequent turns; locally rejected updates leave the
accepted settings unchanged. The upstream handler validates model-specific
settings when the connection opens.

Input EOF finishes accepted work. Context stop/kill or dropping the output stream
closes upstream connections. ASR clear cancels the uncommitted turn; committed
turns continue, and clear acknowledgements follow their output. Omni clear
discards the local input buffer without cancelling committed responses. Turns
are admitted in input order, and each gets a separate upstream connection.

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
On Unix, published files have mode `0644`; the shared directory must also be
traversable by the HTTP server's user.
Reference audio is resolved through Dynamo's media-fetch policy before forwarding
a data URI upstream; base64 data URIs may contain percent escapes, and
client-supplied filesystem paths are rejected.
Public HTTPS addresses, including IP literals and nondefault ports, are accepted.
`DYN_MM_ALLOW_INTERNAL=1` enables HTTP and internal reference-audio sources.
Otherwise, an applicable egress proxy requires `DYN_MM_TRUST_EGRESS_PROXY=1`,
asserting that the proxy enforces destination policy. Proxy selection honors
`NO_PROXY` on every redirect; direct hops retain connect-time DNS protection.
Downloads follow at most three redirects. The reference-audio timeout covers
the whole download, including redirects and the response body.

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
consume audio-queue capacity. Omni input uses one buffer bounded by the turn byte
limit, then uploads in chunks; `--event-capacity` bounds ASR input and output events.
Output backpressure reaches the upstream socket.
ASR commit and clear can be processed while transcript output is full. A pending
control acknowledgement or validation error can still apply input backpressure.
The realtime response deadline starts once a turn is connected and committed,
excluding time waiting for a turn slot. It covers upload and blocked output sends;
expiry releases the upstream connection before terminal failure delivery can wait
for a slow reader.

Startup and active canaries check upstream `/health` and model identity.
Realtime canaries also check the WebSocket protocol handshake. These are service
health checks, not inference/quality canaries. Local acknowledgements and public
failure events do not mark realtime endpoints ready or defer their canaries.
The sidecar's HTTP health route separately reports runtime dependencies.
Shared worker readiness and graceful shutdown continue to apply.

## Validation

```bash
cargo test --locked \
  -p dynamo-realtime -p dynamo-backend-common -p dynamo-sidecar-common \
  -p dynamo-vllm-sidecar -p dynamo-sglang-sidecar \
  -p dynamo-trtllm-sidecar -p dynamo-lpx-sidecar -p dynamo-sidecar-testkit \
  --lib --test executable --test realtime --test realtime_health
```

Before rollout, compare direct-engine, Python-handler, and sidecar runs with the
same model/release and inputs. Check ASR transcript quality, TTS audio quality and
codec handling, first-output latency, realtime factor, and concurrent Omni turn
ordering. Verify that disconnect, clear, and worker shutdown release upstream
engine requests, not only sockets.
