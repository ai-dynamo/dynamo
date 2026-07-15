---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Text-to-Audio with vLLM-Omni (TTS)
subtitle: Synthesize speech with vLLM-Omni via the /v1/audio/speech endpoint.
sidebar-title: vLLM-Omni
---

Text-to-audio (TTS) generation runs a vLLM-Omni worker with `--output-modalities audio`. See the [Diffusion Overview](../README.md) for installation and shared configuration.

## Tested Models

| Model | Notes |
|---|---|
| `Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice` | Default model; predefined speakers |
| `Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign` | Describe a voice via `instructions` |

## Launch

Launch using the provided script with `Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice`:

```bash
bash examples/backends/vllm/launch/agg_omni_audio.sh
```

## Generate Speech

<Tabs>
<Tab title="CustomVoice (predefined speaker)">

```bash
curl -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "input": "Hello, how are you?",
    "voice": "vivian",
    "language": "English"
  }' --output output.wav
```

</Tab>
<Tab title="CustomVoice + style">

```bash
curl -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "input": "I am so excited!",
    "voice": "vivian",
    "instructions": "Speak with great enthusiasm"
  }' --output excited.wav
```

</Tab>
<Tab title="VoiceDesign (describe a voice)">

```bash
bash examples/backends/vllm/launch/agg_omni_audio.sh --model Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign

curl -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "input": "Hello world",
    "task_type": "VoiceDesign",
    "instructions": "A warm, friendly female voice with a gentle tone"
  }' --output voicedesign.wav
```

</Tab>
</Tabs>

## Parameters

The `/v1/audio/speech` endpoint follows the [vLLM-Omni](https://docs.vllm.ai/projects/vllm-omni/en/latest/) API format. All TTS-specific parameters are top-level fields:

<ParamField path="input" type="string" required={true}>
  Text to synthesize.
</ParamField>
<ParamField path="model" type="string" default="auto-detected">
  TTS model name.
</ParamField>
<ParamField path="voice" type="string" default="Vivian">
  Speaker name (e.g., vivian, ryan). Validated against model config.
</ParamField>
<ParamField path="response_format" type="wav | mp3 | pcm | flac | aac | opus" default="wav">
  Audio output format.
</ParamField>
<ParamField path="speed" type="float" default="1.0">
  Speed factor (0.25–4.0).
</ParamField>
<ParamField path="task_type" type="CustomVoice | VoiceDesign | Base" default="CustomVoice">
  Synthesis task type (Qwen3-TTS).
</ParamField>
<ParamField path="language" type="string" default="Auto">
  Language code. Validated against model config.
</ParamField>
<ParamField path="instructions" type="string">
  Voice style/emotion description. Required for VoiceDesign.
</ParamField>
<ParamField path="ref_audio" type="string">
  Reference audio URL or base64 data URI. Required for Base.
</ParamField>
<ParamField path="ref_text" type="string">
  Transcript of reference audio (Base task).
</ParamField>
<ParamField path="max_new_tokens" type="int" default="2048">
  Maximum tokens to generate (1–4096).
</ParamField>

Available voices and languages are loaded dynamically from the model's `config.json` at startup. Non-Qwen3-TTS audio models (e.g., MiMo-Audio) use a generic text prompt and ignore TTS-specific parameters.

<Note>
Audio streaming (`stream: true`) and the Base task (voice cloning) are not yet supported.
</Note>

## See Also

- [Diffusion Overview](../README.md)
- [vLLM-Omni Configuration reference](../../../backends/vllm/vllm-omni-config-reference.mdx)
