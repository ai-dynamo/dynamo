---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Speech to Text (STT)
subtitle: Stream audio to a speech recognition model and receive transcripts over the /v1/realtime WebSocket
---

Dynamo serves speech to text, also called automatic speech recognition (ASR),
through the OpenAI Realtime API. A client opens a WebSocket to `/v1/realtime`,
starts a `transcription` session, streams audio, and receives transcript events
while it is still sending. Dynamo does not expose a REST
`/v1/audio/transcriptions` endpoint.

<Warning>
Realtime transcription is experimental. The vLLM path was validated with
English audio.
</Warning>

## Choose a Backend

| Backend | Models | Where it runs |
| --- | --- | --- |
| vLLM | `Qwen/Qwen3-ASR-0.6B` by default, or any vLLM model that implements `SupportsRealtime` | A local launcher or your own deployment |
| Nemotron Speech NIM | Nemotron ASR Streaming (English) behind a Dynamo adapter | Kubernetes example |

SGLang, TensorRT-LLM, and vLLM-Omni do not serve `transcription` sessions.
vLLM-Omni serves audio-in, audio-out models through `session.type="realtime"`;
see [Speech-to-Speech](speech-to-speech.md).

<Tabs>
<Tab title="vLLM">

Set up vLLM as described in
[Voice Pipelines › Set up the backend](voice-pipelines.md#set-up-the-backend).
The worker resamples the 24 kHz input stream with `scipy`, which is one of vLLM's
audio dependencies.

### Launch

```bash
bash examples/backends/vllm/launch/agg_realtime_transcription.sh \
  --max-model-len 4096 --max-num-seqs 2
```

The launcher starts `python -m dynamo.frontend` and a vLLM worker with
`--realtime`. The frontend listens on `DYN_HTTP_PORT` (default `8000`). The default
model is `Qwen/Qwen3-ASR-0.6B`, and for that model the launcher selects the
`Qwen3ASRRealtimeGeneration` architecture for you. Other arguments pass through to
the vLLM worker.

To serve another model, name its realtime architecture explicitly. The launcher
exits with an error if `--hf-overrides` is missing for a non-default model:

```bash
bash examples/backends/vllm/launch/agg_realtime_transcription.sh \
  --model MODEL_ID \
  --hf-overrides '{"architectures":["REALTIME_ARCHITECTURE"]}'
```

The worker refuses to start if the model does not implement vLLM's
`SupportsRealtime` interface. `--realtime` (`DYN_VLLM_REALTIME`) requires an
aggregated worker and cannot be combined with `--embedding-worker`,
`--classify-worker`, multimodal worker flags, `--enable-lora`, `--headless`, or
`--enable-rl`.

### Stream Audio

In another terminal, stream a speech file with the bundled client. Use a file
longer than five seconds to see incremental results:

```bash
python examples/backends/vllm/launch/realtime_audio_client.py \
  --session-type transcription \
  --url ws://localhost:8000/v1/realtime \
  --model Qwen/Qwen3-ASR-0.6B \
  --input-audio speech.wav \
  --require-incremental
```

`--model` must match the name the worker serves. The client converts the file to
24 kHz mono PCM16, sends it in 100 ms chunks (`--chunk-ms`) at real-time pace, and
prints events as they arrive. A successful run prints `delta before commit: True`.
Without `--input-audio`, the client downloads a short sample clip. The client
needs `aiohttp` and `numpy`; install `soundfile` to read formats other than
16-bit PCM WAV.

### Limits

The vLLM worker rejects a `session.update` that asks for anything it does not
support, and replies with an `error` event:

| Setting | Supported |
| --- | --- |
| `audio.input.format` | `{"type": "audio/pcm", "rate": 24000}` only |
| `audio.input.transcription.model` | The served model name |
| `audio.input.transcription.language` | `en` or omitted |
| `audio.input.transcription.prompt` | Not supported |
| `audio.input.noise_reduction` | Not supported |
| `audio.input.turn_detection` | Not supported. Detect speech in the client and send an explicit `input_audio_buffer.commit` |

Each `input_audio_buffer.append` chunk must decode to 4 MiB or less of 16-bit
PCM, and one utterance can hold at most 60 seconds of audio. The worker
runs one transcription at a time per connection and queues at most one more.

The worker feeds audio to vLLM in 100 ms blocks. How often partial text arrives
depends on the model.

</Tab>
<Tab title="Nemotron Speech NIM">

The Nemotron Speech example runs a Nemotron ASR NIM as a sidecar behind a Dynamo
adapter that serves the same `/v1/realtime` transcription protocol. The adapter
is CPU only, and the NIM uses one GPU. The example also deploys an LLM and a TTS
NIM for a cascaded voice agent; see
[Voice Pipelines](voice-pipelines.md).

Follow the
[deployment and smoke-test instructions](https://github.com/ai-dynamo/dynamo/tree/main/examples/nemotron_speech_cascaded_pipeline#deploy-on-kubernetes)
in the example. It requires compatible source-built Dynamo images, so check its
runtime-image requirements before you deploy.

Differences from the vLLM path:

- The adapter registers the model as `nemotron-asr-streaming`. Set
  `session.audio.input.transcription.model` to that name, or to the adapter's
  `--model-name`.
- Input is still 24 kHz PCM16, with at most 60 seconds per utterance.
- `--language-code` defaults to `en-US`. A session that requests a different
  language is rejected.
- `turn_detection` is not supported; commit explicitly.
- On commit, the adapter appends `--commit-padding-ms` of silence so the NIM
  flushes a short utterance. The worker default is `0`, and the example's
  deployment manifest sets `400`.
- `--max-concurrent-turns` bounds active ASR RPCs across all clients, and excess
  turns fail immediately.

The example's `smoke_speech_loop.py` drives the endpoint and reports ASR
first-transcript latency and the final transcript.

</Tab>
</Tabs>

## Protocol

A transcription session follows the OpenAI Realtime API event names:

1. The frontend sends `session.created` as soon as the WebSocket opens.
2. The client sends `session.update` with `"type": "transcription"`. The frontend
   reads `session.audio.input.transcription.model` to pick a worker and forwards
   the event to it. An unknown or unavailable model produces an `error` event with
   code `model_not_found` or `model_unavailable`, and the connection stays open
   so the client can send a corrected `session.update`. The worker replies with
   `session.updated`.
3. The client sends audio as base64 PCM16 in `input_audio_buffer.append` events.
   Frames must be JSON text; binary WebSocket frames are not supported.
4. The client sends `input_audio_buffer.commit` to end the utterance. The worker
   replies with `input_audio_buffer.committed`.
5. The worker sends `conversation.item.input_audio_transcription.delta` events as
   text is recognized, then
   `conversation.item.input_audio_transcription.completed` with the full
   `transcript` and token `usage`. If transcription fails, it sends
   `conversation.item.input_audio_transcription.failed` instead.

`input_audio_buffer.clear` discards the audio buffered for the current utterance
and returns `input_audio_buffer.cleared`.

A minimal `session.update`:

```json
{
  "type": "session.update",
  "session": {
    "type": "transcription",
    "audio": {
      "input": {
        "format": {"type": "audio/pcm", "rate": 24000},
        "transcription": {"model": "Qwen/Qwen3-ASR-0.6B", "language": "en"},
        "noise_reduction": null,
        "turn_detection": null
      }
    }
  }
}
```

Use [`realtime_audio_client.py`](https://github.com/ai-dynamo/dynamo/blob/main/examples/backends/vllm/launch/realtime_audio_client.py)
as the reference client implementation.

## Related Guides

- [Voice Pipelines](voice-pipelines.md): connect ASR to an LLM and TTS.
- [Speech-to-Speech](speech-to-speech.md): one model that takes and returns audio.
- [vLLM multimodal guide](../../developer-guide/knowledge-base/modular-components/backends/vllm/multimodal.md):
  ask questions about audio with an audio-capable chat model through `audio_url`
  in `/v1/chat/completions`.
