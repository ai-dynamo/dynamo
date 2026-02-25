---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Examples
---

# SGLang Examples

For quick start instructions, see the [SGLang README](README.md). This document provides all deployment patterns for running SGLang with Dynamo, including LLMs, multimodal, and diffusion models, and Kubernetes deployment.

## Table of Contents

- [Infrastructure Setup](#infrastructure-setup)
- [LLM Serving](#llm-serving)
- [Embedding Models](#embedding-models)
- [Vision Models](#vision-models)
- [Diffusion Models](#diffusion-models)
- [Kubernetes Deployment](#kubernetes-deployment)
- [Testing](#testing)

## Infrastructure Setup

For local/bare-metal development, start etcd and optionally NATS using Docker Compose:

```bash
docker compose -f deploy/docker-compose.yml up -d
```

<Note>
- **etcd** is optional but is the default local discovery backend. You can also use `--discovery-backend file` to use file system based discovery.
- **NATS** is only needed when using KV routing with events (`--kv-events-config`). Use `--no-router-kv-events` on the frontend for prediction-based routing without NATS.
- **On Kubernetes**, neither is required when using the Dynamo operator (`DYN_DISCOVERY_BACKEND=kubernetes`).
</Note>

<Tip>
Each launch script runs the frontend and worker(s) in a single terminal. You can run each command separately in different terminals for testing. For AI agents working with Dynamo, you can run the launch script in the background and use the `curl` commands to test the deployment.
</Tip>

## LLM Serving

### Aggregated Serving

The simplest deployment pattern: a single worker handles both prefill and decode.

```bash
cd $DYNAMO_HOME/examples/backends/sglang
./launch/agg.sh
```

### Aggregated Serving with KV Routing

Two workers behind a [KV-aware router](../../components/router/README.md) that maximizes cache reuse:

```bash
cd $DYNAMO_HOME/examples/backends/sglang
./launch/agg_router.sh
```

This launches the frontend with `--router-mode kv` and two workers with ZMQ-based KV event publishing.

### Disaggregated Serving

Separates prefill and decode into independent workers connected via NIXL for KV cache transfer. Requires 2 GPUs.

```bash
cd $DYNAMO_HOME/examples/backends/sglang
./launch/disagg.sh
```

For details on how SGLang disaggregation works with Dynamo, including the bootstrap mechanism and RDMA transfer flow, see [SGLang Disaggregation](sglang-disaggregation.md).

### Disaggregated Serving with KV-Aware Prefill Routing

Scales to 2 prefill + 2 decode workers with KV-aware routing on both pools. Requires 4 GPUs.

```bash
cd $DYNAMO_HOME/examples/backends/sglang
./launch/disagg_router.sh
```

The frontend uses `--router-mode kv` and automatically detects prefill workers to activate an internal prefill router. Each worker publishes KV events over ZMQ on unique ports.

## Multimodal Serving

### Aggregated Multimodal

Serve multimodal models using SGLang's built-in multimodal support:

```bash
cd $DYNAMO_HOME/examples/backends/sglang
./launch/agg_vision.sh
```

<Accordion title="Verify the deployment">
```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-VL-8B-Instruct",
    "messages": [
      {
        "role": "user",
        "content": [
          {"type": "text", "text": "Describe the image."},
          {"type": "image_url", "image_url": {"url": "http://images.cocodataset.org/test2017/000000155781.jpg"}}
        ]
      }
    ],
    "max_tokens": 50,
    "stream": false
  }' | jq
```
</Accordion>

### Multimodal with Disaggregated Components

For advanced multimodal deployments with separate encoder, prefill, and decode workers (E/PD and E/P/D patterns), see the dedicated [SGLang Multimodal](../../features/multimodal/multimodal-sglang.md) documentation.

| Pattern | Script | Description |
|---------|--------|-------------|
| E/PD | `./launch/multimodal_epd.sh` | Separate vision encoder + combined PD worker |
| E/P/D | `./launch/multimodal_disagg.sh` | Separate encoder, prefill, and decode workers |

## Diffusion Models

### Diffusion LM

Run diffusion language models like [LLaDA2.0](https://github.com/inclusionAI/LLaDA2.0):

```bash
cd $DYNAMO_HOME/examples/backends/sglang
./launch/diffusion_llada.sh
```

### Image Diffusion

Generate images from text prompts using [FLUX](https://huggingface.co/black-forest-labs/FLUX.1-dev) or other diffusion models:

```bash
cd $DYNAMO_HOME/examples/backends/sglang
./launch/image_diffusion.sh
```

Options: `--model-path`, `--fs-url` (local or S3), `--http-url`.

### Video Generation

Generate videos from text prompts using [Wan2.1](https://huggingface.co/Wan-AI) models:

```bash
cd $DYNAMO_HOME/examples/backends/sglang
./launch/text-to-video-diffusion.sh
```

Options: `--wan-size 1b|14b`, `--num-frames`, `--height`, `--width`, `--num-inference-steps`.

For full details on all diffusion worker types (LLM, image, video), see [Diffusion](sglang-diffusion.md).

### Kubernetes Deployment

For complete K8s deployment examples, see:
- [SGLang K8s deployment guide](https://github.com/ai-dynamo/dynamo/tree/main/examples/backends/sglang/deploy)
- [SGLang aggregated router K8s example](https://github.com/ai-dynamo/dynamo/blob/main/examples/backends/sglang/deploy/agg_router.yaml)
- [Kubernetes Deployment Guide](../../kubernetes/README.md)

## Troubleshooting

### CuDNN Version Check Fails

```
RuntimeError: cuDNN frontend 1.8.1 requires cuDNN lib >= 9.5.0
```

Set `SGLANG_DISABLE_CUDNN_CHECK=1` before launching. This is common when PyTorch ships a CuDNN version older than what SGLang's Conv3d models require. Affects vision and diffusion models.

### Model Registration Fails with `config.json` Error

```
unable to extract config.json from directory ...
```

This happens with diffusers models (FLUX.1-dev, Wan2.1, etc.) that use `model_index.json` instead of `config.json`. Ensure you are using the correct worker flag (`--image-diffusion-worker` or `--video-generation-worker`) rather than the standard LLM worker mode. These flags use a registration path that does not require `config.json`.

### GPU OOM on Startup

If a previous run left orphaned GPU processes, the next launch may OOM. Check for zombie processes:

```bash
nvidia-smi  # look for lingering sgl_diffusion::scheduler or python processes
kill -9 <PID>
```

### Disaggregated Workers Cannot Connect

Ensure both prefill and decode workers can reach each other over TCP. The bootstrap mechanism uses `--disaggregation-bootstrap-port` (default: 12345). For multi-node setups, ensure the port is reachable across hosts and set `--host 0.0.0.0`.

## See Also

- **[SGLang README](README.md)**: Quick start and feature overview
- **[Reference Guide](sglang-reference-guide.md)**: Architecture, configuration, and operational details
- **[SGLang Multimodal](../../features/multimodal/multimodal-sglang.md)**: Vision model deployment patterns
- **[SGLang HiCache](../../integrations/sglang-hicache.md)**: Hierarchical cache integration
- **[Benchmarking](../../benchmarks/benchmarking.md)**: Performance benchmarking tools
- **[Tuning Disaggregated Performance](../../performance/tuning.md)**: P/D tuning guide
