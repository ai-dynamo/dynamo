---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Voice Pipelines
subtitle: Run streaming speech models and cascaded voice agents with Dynamo
---

NVIDIA Dynamo serves voice workloads through OpenAI-compatible endpoints. Use
separate automatic speech recognition (ASR), LLM, and text-to-speech (TTS) workers
for a cascaded pipeline, or a vLLM-Omni model for audio input and output.

## Endpoints

| Stage | Endpoint | Output |
| --- | --- | --- |
| Streaming ASR | `/v1/realtime` WebSocket with `session.type="transcription"` | Transcript events |
| LLM | `/v1/chat/completions` with `stream: true` | Text deltas |
| TTS | `/v1/audio/speech` | Audio chunks for supported models and formats |
| Audio-in/audio-out model | `/v1/realtime` WebSocket with `session.type="realtime"` | Text and audio events |

## Run a Pipeline

- **Nemotron Speech cascade on Kubernetes:** Follow the
  [deployment and smoke-test instructions](https://github.com/ai-dynamo/dynamo/tree/main/examples/nemotron_speech_cascaded_pipeline#deploy-on-kubernetes).
  The example deploys Nemotron ASR and Magpie TTS Speech NIMs with Dynamo adapters,
  plus a Nemotron LLM served by vLLM. ASR, LLM, and TTS run in separate pods.
  Follow the example's runtime-image requirements; it currently requires
  compatible source-built Dynamo images, not the published 1.5.0 images.
- **Separate vLLM/vLLM-Omni speech workers:** Use the
  [Voxtral realtime transcription launcher](https://github.com/ai-dynamo/dynamo/blob/main/examples/backends/vllm/launch/agg_realtime_transcription.sh)
  and [Qwen3-TTS launcher](https://github.com/ai-dynamo/dynamo/blob/main/examples/backends/vllm/launch/agg_omni_audio.sh)
  to exercise each endpoint independently before connecting them to an LLM.
- **Audio-in/audio-out with vLLM-Omni:** Run the
  [Qwen3-Omni realtime example](https://github.com/ai-dynamo/dynamo/blob/main/examples/backends/vllm/launch/agg_omni_realtime.sh).
  The launcher prints a client command that streams an audio file and saves the
  response audio. Each committed utterance is handled independently; this path
  does not preserve multi-turn conversation history.

The local launchers require a configured
[vLLM backend environment](../../developer-guide/knowledge-base/modular-components/backends/vllm/overview.md#installation),
with vLLM-Omni installed for the Omni examples.

For a cascaded voice agent, connect an application such as Pipecat to the three
stage endpoints. The application owns conversation history, turn detection,
interruptions, and orchestration; Dynamo serves and routes each stage. See
[external orchestration](https://github.com/ai-dynamo/dynamo/tree/main/examples/nemotron_speech_cascaded_pipeline#external-orchestration)
for client integration details. The deployment smoke test checks speech endpoints,
not a complete browser voice agent.

## Planned Work

[Incremental text input](https://github.com/ai-dynamo/dynamo/issues/14476) is under
development to overlap ASR with LLM prefill. Session recovery across worker
failures is a further goal. Neither capability is provided by the examples above.
