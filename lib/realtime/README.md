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

The November 19, 2026 compatibility audit uses Dynamo main `51ae4a048d` and
its pinned engines: vLLM **v0.33.0** and vLLM-Omni **v0.33.0rc1**. CPU tests
exercise mock services and decode emitted events with the frontend's pinned
protocol types. The capability assessment below also inspects upstream source;
source support does not imply end-to-end GPU validation.

The November 16 H200 campaign used those engine releases with sidecar commit
`74cdff269b`, Qwen3-ASR, Qwen3-TTS, Qwen3-Omni, and Voxtral Realtime. It is
historical evidence, not a benchmark of this main refresh. ASR quality was the
same across the compared paths, and paced ASR/TTS latency did not show a general
sidecar speedup. Omni did not yield a comparable performance result. Broader
quality/load testing, repeated-turn GPU tests, and engine-side cancellation
validation remain required before rollout; these are not certified deployment
recipes.

The pinned Omni OpenAI handler buffers input until commit; the adapter then sends
`response.create`. It does not provide streaming inference before commit.
It also does not carry conversation history between turns. As in the Python
Omni bridge, `conversation.item.*` and explicit `response.create` client events
do not change that independent-turn behavior. These are adapter restrictions:
the pinned upstream OpenAI handler already implements conversation management,
tools, VAD, and cancellation.

Omni `session.update` events preserve omitted configuration fields. Updates that
pass local validation apply to subsequently opened turns; locally rejected
updates leave the accepted configuration unchanged. The local acknowledgement
does not certify model-specific settings: the upstream handler validates those
when each turn opens its connection.

Omni's legacy `transcription.*` audio endpoint and its native duplex endpoint
(`duplex=1`) use different contracts. They are not selected by `--protocol omni`.
Native playback acknowledgement, overlap/barge-in, session resume, and capability
negotiation need public protocol support and a separate adapter. Some models
already expose these through upstream native duplex; adding them does not
necessarily require new engine APIs. A model's name does not select or imply
these capabilities.

LLM text sessions and streaming prefill are outside this iteration. Streaming
prefill will need the corresponding upstream vLLM/vllm-rs APIs.

## Capability Map

The Python workers call engine internals (`AsyncLLM`/`AsyncOmni`); these adapters
call stock upstream serving APIs. This moves model-specific processing upstream.
It does not automatically expose every Dynamo token-serving feature or every
field accepted by the public realtime schema.
The separate Python [text-only realtime handler](../../components/src/dynamo/vllm/realtime/handlers/text.py)
already supports conversation items, function tools and held-prefill on main;
this map compares audio ASR/Omni and HTTP speech paths.

| Model capability | Python path on audited main | This PR | Follow-up boundary |
| --- | --- | --- | --- |
| ASR incremental transcripts, manual commits, repeated turns | Implemented with internal streaming audio/token feedback | Implemented through vLLM WebSockets, including 24→16 kHz conversion | Validate quality and cancellation per model/release; transport does not fix Qwen3-ASR decoding quality |
| ASR language, prompts, server VAD | Only English, empty prompt, explicit commits | Same restricted public surface | Local VAD is Dynamo work; native language/prompt/segment controls are absent from the [stock ASR socket][asr-socket] |
| TTS complete text → streaming audio | PCM/WAV output streaming already exists; Python constructs model-specific requests | Same delivery modes, with task/codec generation upstream | Broaden checkpoint, codec, speed and voice-cloning validation; forwarding is not proof of model support |
| TTS incremental text input | No bidirectional text-input API in the HTTP speech handler | Not implemented | Separate adapter for existing upstream [`/v1/audio/speech/stream`][tts-stream], available in the pinned release; model-dependent |
| Omni audio → text/audio | Independent utterances through internal `StreamingInput`; ordered concurrent output | Independent committed turns, separate upstream session per turn, ordered concurrent output | Same product scope, different engine input scheduling; no pre-commit inference guarantee |
| Omni history, text/image items, tools | Realtime bridge ignores item mutations and explicit response creation | Same restriction; forwarding session tool settings alone cannot complete a tool round trip | Stateful session adapter using existing [item APIs][omni-items] and [tool construction][omni-tools]; preserve item/response IDs and tool results |
| Omni VAD, client cancellation, barge-in | Clear affects uncommitted input; no public response-cancel handling | Manual commits; teardown cancels upstream, but client `response.cancel` is unsupported | Existing [server/semantic VAD][omni-vad] and [cancel/truncate APIs][omni-cancel] need event mapping and playback cutoff policy |
| Omni realtime voice/speed | Bridge consumes output modalities, not voice/speed | Session fields forwarded; effect is not guaranteed | [Pinned rc1][omni-voice-rc1] lacks newer upstream [speaker mapping][omni-voice-head] and [speed processing][omni-speed-head]; upgrade and effect tests are needed before advertising support |
| Native duplex, playback ACK, resume | Realtime bridge does not use `DuplexOmni` | Not implemented by `--protocol omni` | Separate capability-aware adapter; [Qwen's pinned native plugin][qwen-duplex] already declares interruption, playback truncation and live-session resume |

The Python paths inspected are the [ASR handler](../../components/src/dynamo/vllm/realtime/handlers/transcription.py),
[speech handler](../../components/src/dynamo/vllm/omni/audio_handler.py), and
[Omni realtime handler](../../components/src/dynamo/vllm/omni/realtime_handler.py).
Qwen3-TTS CustomVoice, VoiceDesign, and Base are different checkpoint/task
configurations; support for one does not certify the others. Realtime camera
support is also model/protocol-specific: the Qwen native plugin accepts image
items but does not provide the video-track contract of some other duplex models.
Qwen function tools require upstream tool parsing configuration, including
`--enable-auto-tool-choice --tool-call-parser hermes`.
The upstream TTS input-stream endpoint buffers until `input.done` by default;
its sentence/clause/commit modes make separate synthesis requests. Exposing that
API would not imply one continuously resumable synthesis request.

| Dynamo serving feature | Current realtime Python path and this PR |
| --- | --- |
| Discovery, readiness, shutdown | Both use Dynamo worker discovery and the public frontend. The PR adds upstream health/model/handshake probes and owned connection cleanup. These probes do not execute inference. |
| Routing and affinity | Bidirectional [PushRouter](../runtime/src/pipeline/network/egress/push_router.rs) supports round-robin/random and selects one worker for the entire stream. Least-loaded, P2C, device-aware and KV routing are rejected for this path. Connection affinity does not imply affinity after reconnect. |
| Admission and backpressure | The shared frontend input channel and worker turn concurrency are bounded, but Python Omni's per-turn audio queue is unbounded. The PR also bounds decoded audio, output bytes, queued turns and deadlines. These are per-session limits, not global GPU-capacity admission; main's token scheduler dispatch gate is not used here. |
| ASR worker-loss recovery | Main's frontend [journal/replay](../llm/src/migration/realtime.rs) can recover eligible manual-commit PCM ASR sessions when enabled. The same frontend can serve sidecar workers, but sidecar-specific failure classification and GPU recovery still need validation. Omni is ineligible. This is audio replay, not KV migration or client reconnect resume. |
| Metrics and scaling | Shared lifecycle/health infrastructure remains available. The PR does not export upstream engine/per-stage/KV metrics into Dynamo or provide realtime SLO autoscaling. Session/turn counts, queued audio, first-output latency and coarse load-aware placement are Dynamo follow-ups. |
| Disaggregation and caching | Python realtime Omni runs an aggregated pipeline; the PR explicitly requires aggregated workers and disables KV routing/indexing. Upstream may distribute its own stages behind the endpoint. That does not give Dynamo ownership of Thinker/Talker/codec scheduling or KV handoff. |

The most direct follow-ups are a stateful Omni OpenAI session profile
(history/tools/VAD/cancel), incremental-text TTS, and realtime load/latency
instrumentation. Native duplex is a larger, separate public-protocol adapter:
it needs capability negotiation, playback accounting, owner affinity, resume
tokens and lease handling. Existing upstream APIs cover these transport
features; they do not all require new upstream implementation.

Some capabilities cannot be obtained by adding event forwarding to the current
endpoint:

- **Qwen3-ASR native decoding/latency controls.** The [realtime model][qwen-asr]
  fixes its audio segmentation, and the socket does not expose per-session
  language, prompt, sampling or segment tuning. Shorter Dynamo-side turns or
  whole-utterance HTTP transcription are alternatives with different semantics.
  Fixing the model's streaming quality requires engine/model work.
- **Continuous model-native Qwen3-Omni inference.** Even the native duplex
  [Qwen adapter][qwen-duplex-model] starts a fresh inference request for each
  committed utterance; it has no core resumable request or streaming KV state.
  Concurrent listening and interruptible playback do not supply that capability.
- **Exact recovery after engine loss.** Native [session resume][omni-resume] reattaches to a
  retained live session. Restoring engine/KV state in another process requires
  an export/import or persistent-state contract. Audio/history replay is a
  possible, weaker recovery mechanism.
- **Dynamo token/KV placement and stage disaggregation.** The public audio
  messages lack the token, cache-ownership and stage-handoff information required
  for those features. Additional engine integration is needed; coarse session
  load balancing does not require those internals.

Current upstream already aborts requests during [ASR disconnect cleanup][asr-abort]
and [Omni cancellation][omni-cancel]. GPU resource release still needs deployment
tests, but a missing cancellation API is not the current blocker.

[asr-socket]: https://github.com/vllm-project/vllm/blob/2986b5cb092cb0f1a1fafcb1f031b2a59b1d7448/vllm/entrypoints/speech_to_text/realtime/connection.py#L95-L163
[asr-abort]: https://github.com/vllm-project/vllm/blob/2986b5cb092cb0f1a1fafcb1f031b2a59b1d7448/vllm/entrypoints/speech_to_text/realtime/connection.py#L285-L310
[qwen-asr]: https://github.com/vllm-project/vllm/blob/2986b5cb092cb0f1a1fafcb1f031b2a59b1d7448/vllm/model_executor/models/qwen3_asr_realtime.py#L176-L222
[tts-stream]: https://github.com/vllm-project/vllm-omni/blob/4b63581cf6527dc5c964558650e996c0a36e027a/vllm_omni/entrypoints/openai/serving_speech_stream.py#L4-L59
[omni-items]: https://github.com/vllm-project/vllm-omni/blob/4b63581cf6527dc5c964558650e996c0a36e027a/vllm_omni/entrypoints/openai/realtime/connection.py#L196-L217
[omni-tools]: https://github.com/vllm-project/vllm-omni/blob/4b63581cf6527dc5c964558650e996c0a36e027a/vllm_omni/entrypoints/openai/realtime/connection.py#L604-L653
[omni-vad]: https://github.com/vllm-project/vllm-omni/blob/4b63581cf6527dc5c964558650e996c0a36e027a/vllm_omni/entrypoints/openai/realtime/connection.py#L325-L356
[omni-cancel]: https://github.com/vllm-project/vllm-omni/blob/4b63581cf6527dc5c964558650e996c0a36e027a/vllm_omni/entrypoints/openai/realtime/connection.py#L1511-L1551
[omni-voice-rc1]: https://github.com/vllm-project/vllm-omni/blob/4b63581cf6527dc5c964558650e996c0a36e027a/vllm_omni/entrypoints/openai/realtime/connection.py#L302-L320
[omni-voice-head]: https://github.com/vllm-project/vllm-omni/blob/d189bad1ecda2c0899dac77d30354cd3283a4976/vllm_omni/entrypoints/openai/realtime/connection.py#L2077-L2079
[omni-speed-head]: https://github.com/vllm-project/vllm-omni/blob/d189bad1ecda2c0899dac77d30354cd3283a4976/vllm_omni/entrypoints/openai/realtime/connection.py#L2154-L2163
[qwen-duplex]: https://github.com/vllm-project/vllm-omni/blob/4b63581cf6527dc5c964558650e996c0a36e027a/vllm_omni/model_executor/models/qwen3_omni/duplex/plugin.py#L376-L403
[qwen-duplex-model]: https://github.com/vllm-project/vllm-omni/blob/4b63581cf6527dc5c964558650e996c0a36e027a/vllm_omni/model_executor/models/qwen3_omni/duplex/plugin.py#L3-L6
[omni-resume]: https://github.com/vllm-project/vllm-omni/blob/4b63581cf6527dc5c964558650e996c0a36e027a/vllm_omni/entrypoints/duplex/serving.py#L292-L343

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
cargo test --locked \
  -p dynamo-realtime -p dynamo-backend-common -p dynamo-sidecar-common \
  -p dynamo-vllm-sidecar -p dynamo-sglang-sidecar \
  -p dynamo-trtllm-sidecar -p dynamo-lpx-sidecar -p dynamo-sidecar-testkit \
  --lib --test executable --test realtime
```

Before rollout, compare direct-engine, Python-handler, and sidecar runs with the
same model/release and inputs. Check ASR transcript quality, TTS audio quality and
codec handling, first-output latency, realtime factor, and concurrent Omni turn
ordering. Verify that disconnect, clear, and worker shutdown release upstream
engine requests, not only sockets.
