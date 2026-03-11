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
- **etcd** is optional but is the default local discovery backend. File-based discovery is also available (see `python -m dynamo.vllm --help` for `--discovery-backend` options).
- **NATS** is only needed when using KV routing with events. Prediction-based routing does not require NATS.
- **On Kubernetes**, neither is required when using the Dynamo operator.
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

### Aggregated Serving with KV Routing

Two workers behind a [KV-aware router](../../components/router/README.md) that maximizes cache reuse. Requires 2 GPUs.

Run on cuda devices:

```bash
cd $DYNAMO_HOME/examples/backends/vllm
bash launch/agg_router.sh
```
Run on XPUs:
```bash
cd $DYNAMO_HOME/examples/backends/vllm
bash launch/agg_router_xpu.sh
```


This launches the frontend in KV routing mode with two workers publishing KV events over ZMQ.

### Disaggregated Serving

Separates prefill and decode into independent workers connected via NIXL for KV cache transfer. Requires 2 GPUs.

```bash
cd $DYNAMO_HOME/examples/backends/vllm
bash launch/disagg.sh
```

### Disaggregated Serving with KV Routing

Scales to 2 prefill + 2 decode workers with KV-aware routing on both pools. Requires 4 GPUs.

```bash
cd $DYNAMO_HOME/examples/backends/vllm
bash launch/disagg_router.sh
```

The frontend runs in KV routing mode and automatically detects prefill workers to activate an internal prefill router.

### Data Parallel / Expert Parallelism

Launches 4 data-parallel workers with expert parallelism behind a KV-aware router. Uses a Mixture-of-Experts model (`Qwen/Qwen3-30B-A3B`). Requires 4 GPUs.

```bash
cd $DYNAMO_HOME/examples/backends/vllm
bash launch/dep.sh
```

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

Deploy vLLM across multiple nodes using Dynamo's distributed capabilities. Multi-node deployments require network connectivity between nodes and firewall rules allowing NATS/ETCD communication.

Start NATS/ETCD on the head node so all worker nodes can reach them:

```bash
# On head node
docker compose -f deploy/docker-compose.yml up -d

# Set on ALL nodes
export HEAD_NODE_IP="<your-head-node-ip>"
export NATS_SERVER="nats://${HEAD_NODE_IP}:4222"
export ETCD_ENDPOINTS="${HEAD_NODE_IP}:2379"
```

For multi-node tensor/pipeline parallelism (when TP x PP exceeds GPUs on a single node), see [`launch/multi_node_tp.sh`](https://github.com/ai-dynamo/dynamo/blob/main/examples/backends/vllm/launch/multi_node_tp.sh). For details on distributed execution, see the [vLLM multiprocessing docs](https://docs.vllm.ai/en/stable/serving/parallelism_scaling/#running-vllm-with-multiprocessing).

### DeepSeek-R1

Dynamo supports DeepSeek R1 with data parallel attention and wide expert parallelism. Each DP attention rank is a separate Dynamo component emitting its own KV events and metrics.

Run on 2 nodes (16 GPUs, dp=16):

```bash
# Node 0
cd $DYNAMO_HOME/examples/backends/vllm
./launch/dsr1_dep.sh --num-nodes 2 --node-rank 0 --gpus-per-node 8 --master-addr <node-0-addr>

# Node 1
./launch/dsr1_dep.sh --num-nodes 2 --node-rank 1 --gpus-per-node 8 --master-addr <node-0-addr>
```

See [`launch/dsr1_dep.sh`](https://github.com/ai-dynamo/dynamo/blob/main/examples/backends/vllm/launch/dsr1_dep.sh) for configurable options.

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
- **[Observability](vllm-observability.md)**: Metrics and monitoring
- **[Benchmarking](../../benchmarks/benchmarking.md)**: Performance benchmarking tools
- **[Tuning Disaggregated Performance](../../performance/tuning.md)**: P/D tuning guide
