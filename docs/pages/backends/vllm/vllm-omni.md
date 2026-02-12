---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
---

# [Experimental] Running Omni Models with vLLM

Dynamo supports omni (multimodal generation) models via the [vLLM-Omni](https://github.com/vllm-project/vllm-omni) backend. This enables multi-stage pipelines for tasks like text-to-text and text-to-image generation through an OpenAI-compatible API.

## Prerequisites

This guide assumes familiarity with deploying Dynamo with vLLM as described in [README.md](/docs/pages/backends/vllm/README.md).

## Quick Start

### Text-to-Text

Launch an aggregated deployment (frontend + omni worker) using the provided script:

```bash
bash examples/backends/vllm/launch/agg_omni.sh
```

This starts `Qwen/Qwen2.5-Omni-7B` with a single-stage thinker config on one GPU. Override the model with:

```bash
bash examples/backends/vllm/launch/agg_omni.sh --model <your-model>
```

Test the deployment:

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen2.5-Omni-7B",
    "messages": [{"role": "user", "content": "What is 2+2?"}],
    "max_tokens": 50,
    "stream": false
  }'
```

### Text-to-Image

Text-to-image uses vLLM-Omni's built-in default stage configs (no custom YAML needed). Launch without a stage config path so vLLM-Omni loads the model's default multi-stage pipeline:

```bash
# Start frontend
python -m dynamo.frontend &

# Start omni worker (vLLM-Omni loads default stage configs for the model)
DYN_SYSTEM_PORT=8081 python -m dynamo.vllm \
  --model <your-text-to-image-model> \
  --omni \
  --connector none
```

Images are returned as base64-encoded PNGs in the response:

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "<your-text-to-image-model>",
    "messages": [{"role": "user", "content": "A cat sitting on a windowsill"}],
    "stream": false
  }'
```

The response contains image data URLs in the content field:

```json
{
  "choices": [{
    "delta": {
      "content": [
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}}
      ]
    }
  }]
}
```

## Key Flags

| Flag | Description |
|------|-------------|
| `--omni` | Enable vLLM-Omni orchestrator (required) |
| `--stage-configs-path <path>` | Path to stage config YAML (optional; vLLM-Omni uses model defaults if omitted) |
| `--connector none` | Disable KV connector (recommended for omni) |

## Stage Configuration

Omni pipelines are configured via YAML stage configs. See [`examples/backends/vllm/launch/stage_configs/single_stage_llm.yaml`](/examples/backends/vllm/launch/stage_configs/single_stage_llm.yaml) for an example. Key fields:

- **`model_stage`**: Pipeline stage name (e.g., `thinker`, `talker`, `code2wav`)
- **`final_output_type`**: Output format — `text` or `image`
- **`is_comprehension`**: Whether this stage processes input text/multimodal content

For full documentation on stage config format, supported fields, and multi-stage pipeline examples, see the [vLLM-Omni Stage Configs documentation](https://docs.vllm.ai/projects/vllm-omni/en/latest/configuration/stage_configs/).

## Current Limitations

- Only text prompts are supported (no multimodal input yet)
- KV cache events are not published for omni workers
