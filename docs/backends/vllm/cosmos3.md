---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Cosmos3
---

Run NVIDIA's **Cosmos3** omni model through Dynamo's
[vLLM-Omni backend](vllm-omni.md) for **text-to-image**, **text-to-video**, and
**image-to-video** generation.

Cosmos3 is a unified world foundation model for Physical AI, built on a
Mixture-of-Transformers architecture. A single `Cosmos3OmniTransformer` runs
a Qwen-style "understanding" stream alongside a "generation" stream joined
by a 3D multimodal RoPE, replacing the separate Predict / Reason / Transfer
models from earlier Cosmos releases. See the
[Cosmos World Foundation Model Platform paper](https://huggingface.co/papers/2501.03575)
for the architectural background, and the
[diffusers Cosmos3 reference](https://huggingface.co/docs/diffusers/main/en/api/pipelines/cosmos3)
for the underlying pipeline.

Cosmos3 support in Dynamo is provided by the native vLLM-Omni pipeline added in
[vllm-project/vllm-omni#3454](https://github.com/vllm-project/vllm-omni/pull/3454).

## Checkpoints

Both checkpoints share the same `Cosmos3OmniPipeline` class and Dynamo flags;
swap the model identifier on the worker (`--model …`) and in request payloads.

| Checkpoint | Description | HF Hub |
|------------|-------------|--------|
| `nvidia/Cosmos3-Nano` | Smaller, faster — default in the Dynamo launch scripts below | [link](https://huggingface.co/nvidia/Cosmos3-Nano) |
| `nvidia/Cosmos3-Super` | Larger, higher quality | [link](https://huggingface.co/nvidia/Cosmos3-Super) |

## Supported modalities

| Task | Endpoint | `--output-modalities` |
|------|----------|-----------------------|
| Text-to-Image | `/v1/images/generations` | `image` |
| Text-to-Video | `/v1/videos` | `video` |
| Image-to-Video | `/v1/videos` (with `input_reference`) | `video` |

## Prerequisites

This guide builds on the [vLLM-Omni backend guide](vllm-omni.md) — see it for general setup, `etcd`/`nats`, and OpenAI-endpoint details.

### Installation

This branch carries Dynamo code changes (the Cosmos3 worker flags and image
output handling) on top of a pinned vLLM-Omni, so run Dynamo **from source on
this branch** — a released `ai-dynamo` wheel will not include the integration.

1. Clone and check out the branch:

   ```bash
   git clone https://github.com/ai-dynamo/dynamo.git
   cd dynamo
   git checkout cosmos3-omni-integration
   ```

2. Create a Python 3.12 environment:

   ```bash
   uv venv --python 3.12 --seed
   source .venv/bin/activate
   ```

3. Build and install Dynamo from source (the branch's Cosmos3 code must be
   live, and the Rust core `ai-dynamo-runtime` isn't published for this dev
   version, so it has to be built locally). See
   [Building from source](../../getting-started/building-from-source.md) for
   prerequisites (Rust toolchain, system deps); the key steps from the repo root:

   ```bash
   uv pip install pip maturin
   (cd lib/bindings/python && maturin develop --uv)   # builds ai-dynamo-runtime
   uv pip install -e lib/gpu_memory_service
   uv pip install -e ".[vllm]"                         # also pulls vllm==0.21.0
   ```

4. Install the Cosmos3-capable vLLM-Omni, pinned to the PR commit (its dynamic
   `setup.py` pulls the matching pipeline deps — `diffusers==0.38`, `torchsde`,
   `x-transformers`):

   ```bash
   uv pip install "vllm-omni @ git+https://github.com/vllm-project/vllm-omni.git@e826f626afb47c8c3c39ccf892ed247f442f6bd2"
   ```

5. Start etcd and NATS:

   ```bash
   docker compose -f dev/docker-compose.yml up -d
   ```

## Serve

Quick start — each script launches the frontend on `:8000` plus a
single-modality worker and prints a sample request:

```bash
examples/backends/vllm/launch/agg_omni_cosmos3_image.sh   # text-to-image
examples/backends/vllm/launch/agg_omni_cosmos3_video.sh   # text-to-video
examples/backends/vllm/launch/agg_omni_cosmos3_i2v.sh     # image-to-video
```

Manual launch:

```bash
python -m dynamo.frontend --http-port 8000 &

python -m dynamo.vllm.omni \
    --model nvidia/Cosmos3-Nano \
    --output-modalities image \            # or: video
    --no-cosmos3-guardrails \              # skip loading the safety guardrail models
    --media-output-fs-url file:///tmp/dynamo_media
```

Cosmos3-specific flags:

| Flag | Purpose |
|------|---------|
| `--no-cosmos3-guardrails` | Disable the Cosmos3 text/video safety guardrails (otherwise loaded at startup). |
| `--flow-shift <float>` | Scheduler flow-shift (image default `3.0`). Launch-time only — not a per-request image parameter. |
| `--media-output-fs-url file://<dir>` | Destination for media when `response_format: "url"`. |

## Requests

### Text-to-image

Run from the repo root; `cosmos3/t2i.json` is the official Cosmos3 t2i payload
(prompt verbatim) mapped to the Dynamo request schema:

```bash
curl -s -X POST http://localhost:8000/v1/images/generations \
  -H 'Content-Type: application/json' \
  --data-binary @examples/backends/vllm/launch/cosmos3/t2i.json \
  | jq -r '.data[0].b64_json' | base64 -d > out.png
```

- `size` must be one of `256x256`, `512x512`, `1024x1024`, `1792x1024`,
  `1024x1792`, `1536x1024`, `1024x1536`, `auto` — the payload uses `1024x1024`
  (the official `960x960` is not an allowed image size).
- Put `num_inference_steps`, `guidance_scale`, `seed`, and `negative_prompt`
  under `nvext` — top-level values are ignored.

### Text-to-video

```bash
curl -s http://localhost:8000/v1/videos \
  -H 'Content-Type: application/json' \
  --data-binary @examples/backends/vllm/launch/cosmos3/t2v.json | jq
```

The official `t2v.json` payload is `1280x720`, `192` frames @ `24` fps (8s).

### Image-to-video

`i2v.json` adds `input_reference` (the official `vision_path` — an http URL;
local paths are rejected, use an http(s) URL or a `data:` base64 URI):

```bash
curl -s http://localhost:8000/v1/videos \
  -H 'Content-Type: application/json' \
  --data-binary @examples/backends/vllm/launch/cosmos3/i2v.json | jq
```
