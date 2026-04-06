---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Examples
---

For quick start instructions, see the [TensorRT-LLM README](README.md). This document provides all deployment patterns for running TensorRT-LLM with Dynamo, including single-node, multi-node, and Kubernetes deployments.

## Table of Contents

- [Infrastructure Setup](#infrastructure-setup)
- [Single Node Examples](#single-node-examples)
- [Advanced Examples](#advanced-examples)
- [Client](#client)
- [Benchmarking](#benchmarking)

## Infrastructure Setup

For local/bare-metal development, start etcd and optionally NATS using Docker Compose:

```bash
docker compose -f deploy/docker-compose.yml up -d
```

<Note>
- **etcd** is optional but is the default local discovery backend. You can also use `--discovery-backend file` to use file system based discovery.
- **NATS** is optional - only needed if using KV routing with events. Workers must be explicitly configured to publish events. Use `--no-router-kv-events` on the frontend for prediction-based routing without events.
- **On Kubernetes**, neither is required when using the Dynamo operator, which explicitly sets `DYN_DISCOVERY_BACKEND=kubernetes` to enable native K8s service discovery (DynamoWorkerMetadata CRD).
</Note>

<Tip>
Each launch script runs the frontend and worker(s) in a single terminal. You can run each command separately in different terminals for testing. Each shell script simply runs `python3 -m dynamo.frontend <args>` to start up the ingress and `python3 -m dynamo.trtllm <args>` to start up the workers.
</Tip>

For detailed information about the architecture and how KV-aware routing works, see the [Router Guide](../../components/router/router-guide.md).

## Single Node Examples

### Aggregated

```bash
cd $DYNAMO_HOME/examples/backends/trtllm
./launch/agg.sh
```

### Aggregated with KV Routing

```bash
cd $DYNAMO_HOME/examples/backends/trtllm
./launch/agg_router.sh
```

### Disaggregated

```bash
cd $DYNAMO_HOME/examples/backends/trtllm
./launch/disagg.sh
```

### Disaggregated with KV Routing

<Note>
In disaggregated workflow, requests are routed to the prefill worker to maximize KV cache reuse.
</Note>

```bash
cd $DYNAMO_HOME/examples/backends/trtllm
./launch/disagg_router.sh
```

### Aggregated with Multi-Token Prediction (MTP) and DeepSeek R1

```bash
cd $DYNAMO_HOME/examples/backends/trtllm

export AGG_ENGINE_ARGS=./engine_configs/deepseek-r1/agg/mtp/mtp_agg.yaml
export SERVED_MODEL_NAME="nvidia/DeepSeek-R1-FP4"
# nvidia/DeepSeek-R1-FP4 is a large model
export MODEL_PATH="nvidia/DeepSeek-R1-FP4"
./launch/agg.sh
```

<Note>
- There is a noticeable latency for the first two inference requests. Please send warm-up requests before starting the benchmark.
- MTP performance may vary depending on the acceptance rate of predicted tokens, which is dependent on the dataset or queries used while benchmarking. Additionally, `ignore_eos` should generally be omitted or set to `false` when using MTP to avoid speculating garbage outputs and getting unrealistic acceptance rates.
</Note>

## Advanced Examples

### Multinode Deployment

For comprehensive instructions on multinode serving, see the [Multinode Examples](./multinode/trtllm-multinode-examples.md) guide. It provides step-by-step deployment examples and configuration tips for running Dynamo with TensorRT-LLM across multiple nodes. While the walkthrough uses DeepSeek-R1 as the model, you can easily adapt the process for any supported model by updating the relevant configuration files. You can see the [Llama4 + Eagle](./trtllm-llama4-plus-eagle.md) guide to learn how to use these scripts when a single worker fits on a single node.

### Speculative Decoding

- **[Llama 4 Maverick Instruct + Eagle Speculative Decoding](./trtllm-llama4-plus-eagle.md)**

### Model-Specific Guides

- **[Gemma3 with Sliding Window Attention](./trtllm-gemma3-sliding-window-attention.md)**
- **[GPT-OSS-120b](./trtllm-gpt-oss.md)** — Reasoning model with tool calling support

### Kubernetes Deployment

For complete Kubernetes deployment instructions, configurations, and troubleshooting, see the [TensorRT-LLM Kubernetes Deployment Guide](https://github.com/ai-dynamo/dynamo/tree/main/examples/backends/trtllm/deploy/README.md).

## Client

See the [client](../sglang/README.md#testing-the-deployment) section to learn how to send requests to the deployment.

<Note>
To send a request to a multi-node deployment, target the node which is running `python3 -m dynamo.frontend <args>`.
</Note>

## Benchmarking

To benchmark your deployment with AIPerf, see this utility script, configuring the
`model` name and `host` based on your deployment: [perf.sh](https://github.com/ai-dynamo/dynamo/blob/main/benchmarks/llm/perf.sh)
