---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Examples
---

# vLLM Examples

For quick start instructions, see the [vLLM README](README.md). This document provides all deployment patterns for running vLLM with Dynamo, including aggregated, disaggregated, KV-routed, and expert-parallel configurations.

## Table of Contents

- [Infrastructure Setup](#infrastructure-setup)
- [LLM Serving](#llm-serving)
- [Advanced Examples](#advanced-examples)
- [Kubernetes Deployment](#kubernetes-deployment)
- [Troubleshooting](#troubleshooting)

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
Each launch script runs the frontend and worker(s) in a single terminal. You can run each command separately in different terminals for better log visibility. For AI agents working with Dynamo, you can run the launch script in the background and use the `curl` commands to test the deployment.
</Tip>

## LLM Serving

### Aggregated Serving

The simplest deployment pattern: a single worker handles both prefill and decode. Requires 1 GPU.

```bash
cd $DYNAMO_HOME/examples/backends/vllm
bash launch/agg.sh
```

<Accordion title="Verify the deployment">
```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-0.6B",
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 32
  }'
```
</Accordion>

### Aggregated Serving with KV Routing

Two workers behind a [KV-aware router](../../components/router/README.md) that maximizes cache reuse. Requires 2 GPUs.

```bash
cd $DYNAMO_HOME/examples/backends/vllm
bash launch/agg_router.sh
```

This launches the frontend with `--router-mode kv` and two workers with ZMQ-based KV event publishing.

<Accordion title="Verify the deployment">
```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-0.6B",
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 32
  }'
```
</Accordion>

### Disaggregated Serving

Separates prefill and decode into independent workers connected via NIXL for KV cache transfer. Requires 2 GPUs.

```bash
cd $DYNAMO_HOME/examples/backends/vllm
bash launch/disagg.sh
```

<Accordion title="Verify the deployment">
```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-0.6B",
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 32
  }'
```
</Accordion>

### Disaggregated Serving with KV Routing

Scales to 2 prefill + 2 decode workers with KV-aware routing on both pools. Requires 4 GPUs.

```bash
cd $DYNAMO_HOME/examples/backends/vllm
bash launch/disagg_router.sh
```

The frontend uses `--router-mode kv` and automatically detects prefill workers to activate an internal prefill router. Each worker publishes KV events over ZMQ on unique ports.

<Accordion title="Verify the deployment">
```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-0.6B",
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 32
  }'
```
</Accordion>

### Data Parallel / Expert Parallelism

Launches 4 data-parallel workers with expert parallelism behind a KV-aware router. Uses a Mixture-of-Experts model (`Qwen/Qwen3-30B-A3B`). Requires 4 GPUs.

```bash
cd $DYNAMO_HOME/examples/backends/vllm
bash launch/dep.sh
```

<Accordion title="Verify the deployment">
```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-30B-A3B",
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 32
  }'
```
</Accordion>

<Tip>
Run a disaggregated example and try adding another prefill worker once the setup is running! The system will automatically discover and utilize the new worker.
</Tip>

## Advanced Examples

### Speculative Decoding

Run **Meta-Llama-3.1-8B-Instruct** with **Eagle3** as a draft model for faster inference while maintaining accuracy.

**Guide:** [Speculative Decoding Quickstart](../../features/speculative-decoding/speculative-decoding-vllm.md)

> **See also:** [Speculative Decoding Feature Overview](../../features/speculative-decoding/README.md) for cross-backend documentation.

### Multimodal

Serve multimodal models using the vLLM-Omni integration.

**Guide:** [vLLM-Omni](vllm-omni.md)

### Multi-Node

Deploy vLLM across multiple nodes for larger models.

**Guide:** [Multi-Node Deployment](multi-node.md)

## Kubernetes Deployment

For complete Kubernetes deployment instructions, configurations, and troubleshooting, see the [vLLM Kubernetes Deployment Guide](https://github.com/ai-dynamo/dynamo/tree/main/examples/backends/vllm/deploy/README.md).

See also the [Kubernetes Deployment Guide](../../kubernetes/README.md) for general Dynamo K8s documentation.

## Troubleshooting

### Workers Fail to Start with NIXL Errors

Ensure NIXL is installed and the side-channel ports are not in conflict. Each worker in a multi-worker setup needs a unique `VLLM_NIXL_SIDE_CHANNEL_PORT`.

### KV Router Not Routing Correctly

Ensure `PYTHONHASHSEED=0` is set for all vLLM processes when using KV-aware routing. See [Hashing Consistency](vllm-reference-guide.md#hashing-consistency-for-kv-events) for details.

### GPU OOM on Startup

If a previous run left orphaned GPU processes, the next launch may OOM. Check for zombie processes:

```bash
nvidia-smi  # look for lingering python processes
kill -9 <PID>
```

## See Also

- **[vLLM README](README.md)**: Quick start and feature overview
- **[Reference Guide](vllm-reference-guide.md)**: Configuration, arguments, and operational details
- **[Multi-Node](multi-node.md)**: Multi-node deployment guide
- **[Prometheus](prometheus.md)**: Metrics and monitoring
- **[Benchmarking](../../benchmarks/benchmarking.md)**: Performance benchmarking tools
- **[Tuning Disaggregated Performance](../../performance/tuning.md)**: P/D tuning guide
