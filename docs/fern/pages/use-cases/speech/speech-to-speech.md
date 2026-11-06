---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Speech-to-Speech
subtitle: Stream audio to a vLLM-Omni model and receive spoken responses over /v1/realtime
---

A vLLM-Omni model such as Qwen3-Omni can take spoken input and answer with
synthesized speech in one model, instead of a cascaded ASR, LLM, and TTS
pipeline. Clients stream audio over the `/v1/realtime` WebSocket with
`session.type="realtime"` and receive both the response transcript and the
response audio.

<Note>
Before you begin, set up vLLM-Omni as described in
[Voice Pipelines › Set up the backend](voice-pipelines.md#set-up-the-backend).
</Note>

## Launch

```bash
bash examples/backends/vllm/launch/agg_omni_realtime.sh
```

The launcher serves `Qwen/Qwen3-Omni-30B-A3B-Instruct` with
`python -m dynamo.vllm.omni --realtime --output-modalities audio`. Pass `--model`
to serve a different Omni model.

## Stream Audio

In another terminal, stream an audio file and save the spoken response:

```bash
python examples/backends/vllm/launch/realtime_audio_client.py \
  --session-type realtime \
  --url ws://localhost:8000/v1/realtime \
  --model Qwen/Qwen3-Omni-30B-A3B-Instruct \
  --input-audio question.wav \
  --output-dir dynamo-realtime
```

Omit `--input-audio` to send a short sample clip. The client sends 16 kHz PCM16
audio, prints `response.output_audio_transcript.delta` text as it arrives, and
writes each `response.output_audio.delta` chunk as a WAV file in `--output-dir`.

## Limitations

- Each committed utterance is handled independently. This path does not keep
  multi-turn conversation history; your application must manage it.
- Session recovery across worker failures is not provided.

## Next Steps

- [Voice Pipelines](voice-pipelines.md): build a cascaded ASR, LLM, and TTS pipeline instead.
- [Text-to-Speech](text-to-speech.md): synthesize speech from text.
