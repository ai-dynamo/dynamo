---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Text-to-Audio
subtitle: Synthesize speech with vLLM-Omni through the /v1/audio/speech endpoint
---

Text-to-audio (TTS) generation runs a vLLM-Omni worker with `--output-modalities audio`. See the [Diffusion Overview](../overview.md) for installation and shared configuration.

## Tested Models

| Model | Notes |
|---|---|
| `Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice` | Default model; predefined speakers |
| `Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign` | Describe a voice via `instructions` |
| `nvidia/Nemotron-Labs-Audex-2B` | Single built-in voice; optional `cfg_scale` guidance |
| `nvidia/Nemotron-Labs-Audex-30B-A3B` | Same contract as the 2B; needs an explicit stage config |

## Launch

Launch using the provided script with `Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice`:

```bash
bash examples/backends/vllm/launch/agg_omni_audio.sh
```

The same script serves Nemotron Audex, which runs a two-stage pipeline: stage 0 is the thinker that emits speech codec tokens, and stage 1 decodes those tokens to a 16 kHz mono waveform.

```bash
bash examples/backends/vllm/launch/agg_omni_audio.sh --model nvidia/Nemotron-Labs-Audex-2B
```

Both Audex checkpoints report the same model type, so the 30B-A3B needs an explicit stage configuration. Otherwise auto-detection selects the 2B-tuned file. vLLM-Omni ships the configuration, so resolve it from the installed package and pass it through to the worker:

```bash
AUDEX_30B_CFG=$(python -c 'import pathlib, vllm_omni; print(pathlib.Path(vllm_omni.__file__).parent / "deploy/audex_tts_30b.yaml")')
bash examples/backends/vllm/launch/agg_omni_audio.sh \
  --model nvidia/Nemotron-Labs-Audex-30B-A3B \
  --stage-configs-path "$AUDEX_30B_CFG"
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
<Tab title="Audex (guided speech)">

```bash
curl -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "nvidia/Nemotron-Labs-Audex-2B",
    "input": "Hey, this is generated using Dynamo!",
    "cfg_scale": 1.5
  }' --output audex.wav
```

Audex serves a single built-in voice, so it accepts `voice` only when omitted or set to `default`, and rejects `ref_audio` and `ref_text` instead of synthesizing a different-sounding result. The Qwen3-TTS fields `language`, `instructions`, and `task_type` do not apply. Omit `cfg_scale`, or set it to `1.0`, to decode without guidance.

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
<ParamField path="cfg_scale" type="float">
  Classifier-free guidance scale for Nemotron Audex (1.0–10.0). `1.0` disables guidance, and omitting the field decodes unguided. Ignored by other audio models.
</ParamField>

Available voices and languages are loaded dynamically from the model's `config.json` at startup. Nemotron Audex builds its own prompt and accepts only `input`, `model`, `response_format`, `max_new_tokens`, and `cfg_scale`; speech models additionally tolerate `voice` when it is omitted or `default`, while text-to-audio rejects `voice` outright. Other non-Qwen3-TTS audio models (e.g., MiMo-Audio) use a generic text prompt and ignore TTS-specific parameters.

> [!NOTE]
> Audio streaming (`stream: true`) and the Base task (voice cloning) are not yet supported.

## See Also

- [Diffusion Overview](../overview.md)
- [vLLM-Omni Configuration reference](../../../reference/backends/vllm-omni-configuration.mdx)
