---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Speech Recognition (ASR)
subtitle: Stream audio to a vLLM speech recognition model over the /v1/realtime WebSocket
---

Streaming automatic speech recognition (ASR) runs a vLLM worker started with
`--realtime`. Clients stream audio over the OpenAI-compatible `/v1/realtime`
WebSocket and receive transcription events while they are still sending audio.

> [!NOTE]
> Before you begin, set up vLLM with its audio dependencies as described in
> [Voice Pipelines › Set up the backend](voice-pipelines.md#set-up-the-backend).

> [!WARNING]
> Realtime transcription is experimental.

## Launch

The launcher requires `--hf-overrides` to select the model's realtime
architecture explicitly. For Qwen3-ASR:

```bash
bash examples/backends/vllm/launch/agg_realtime_transcription.sh \
  --model Qwen/Qwen3-ASR-0.6B \
  --hf-overrides '{"architectures":["Qwen3ASRRealtimeGeneration"]}' \
  --max-model-len 4096 --max-num-seqs 2
```

The model must implement vLLM's realtime interface (`SupportsRealtime`); the
worker refuses to start otherwise. Other arguments are passed through to the
vLLM worker.

## Stream Audio

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

The client sends audio while it receives transcription events. A successful run
prints `delta before commit: True`. vLLM processes five-second segments and
flushes the remaining audio when the client commits the buffer. Without
`--input-audio`, the client downloads a short sample clip.

The client needs `aiohttp`. Install `soundfile` to read formats other than
16-bit PCM WAV.

## Protocol

A transcription session over `/v1/realtime` follows the OpenAI Realtime API
event names:

1. The client sends `session.update` with `"type": "transcription"` and a
   24 kHz PCM16 input format.
2. The client sends audio as base64 chunks in `input_audio_buffer.append`
   events, then `input_audio_buffer.commit` to flush the end of the utterance.
3. The server sends `conversation.item.input_audio_transcription.delta` events
   as text is recognized, followed by
   `conversation.item.input_audio_transcription.completed` for the utterance.

Use [`realtime_audio_client.py`](https://github.com/ai-dynamo/dynamo/blob/main/examples/backends/vllm/launch/realtime_audio_client.py)
as the reference client implementation.

## Next Steps

- [Voice Pipelines](voice-pipelines.md): connect ASR to an LLM and TTS.
- [Speech-to-Speech](speech-to-speech.md): one model that takes and returns audio.
