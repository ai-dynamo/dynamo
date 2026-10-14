---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Audio Understanding
subtitle: Send audio clips to audio-capable models through the chat completions endpoint
---

Audio-capable models such as Qwen3-Omni answer questions about audio sent as an
`audio_url` content part in `/v1/chat/completions`. The vLLM worker loads the
audio at its native sample rate, and vLLM's model-specific processor handles
resampling and feature extraction. Omni models accept `image_url`, `video_url`,
and `audio_url` in the same request.

> [!NOTE]
> Before you begin, set up vLLM with its audio dependencies as described in
> [Voice Pipelines › Set up the backend](voice-pipelines.md#set-up-the-backend).

## Support Matrix

| Backend | `audio_url` input |
| --- | --- |
| vLLM | Yes, aggregated and P/D. Audio is not routed to a separate encode worker. |
| SGLang | No |
| TensorRT-LLM | No |

WAV input needs only vLLM's audio dependencies. AAC and other compressed audio also need
the PyAV decoder; see
[Install Additional Decoders](../video/video-decoding.md#install-additional-decoders).

## Launch

Run the aggregated multimodal launcher with an audio-capable model from the root
of a Dynamo repository checkout:

```bash
bash examples/backends/vllm/launch/agg_multimodal.sh --model Qwen/Qwen3-Omni-30B-A3B-Instruct
```

For XPU devices:

```bash
DYN_CHAT_PROCESSOR=vllm \
  bash examples/backends/vllm/launch/xpu/agg_multimodal_xpu.sh --model Qwen/Qwen3-Omni-30B-A3B-Instruct
```

## Send an Audio Request

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-Omni-30B-A3B-Instruct",
    "messages": [{
      "role": "user",
      "content": [
        {"type": "text", "text": "What is said in this recording?"},
        {"type": "audio_url", "audio_url": {"url": "https://raw.githubusercontent.com/yuekaizhang/Triton-ASR-Client/main/datasets/mini_en/wav/1221-135766-0002.wav"}}
      ]
    }],
    "max_tokens": 100
  }'
```

Audio URLs are fetched under the same default-deny policy as images. See
[Image Understanding › Media URL Policy](../image/image-understanding.md#media-url-policy).

For streaming transcription instead of question answering, see
[Speech Recognition](speech-recognition.md).
