<!--
SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# LLM Deployment using SGLang

This directory contains an SGLang component for Dynamo and reference implementations for deploying Large Language Models (LLMs) in various configurations using SGLang. SGLang internally uses ZMQ to communicate between the ingress and the engine processes. For Dynamo, we leverage the runtime to communicate directly with the engine processes and handle ingress and pre/post processing on our end.

## Use the Latest Release

We recommend using the latest stable release of dynamo to avoid breaking changes:

[![GitHub Release](https://img.shields.io/github/v/release/ai-dynamo/dynamo)](https://github.com/ai-dynamo/dynamo/releases/latest)

You can find the latest release [here](https://github.com/ai-dynamo/dynamo/releases/latest) and check out the corresponding branch with:

```bash
git checkout $(git describe --tags $(git rev-list --tags --max-count=1))
```

---

## Table of Contents
- [Feature Support Matrix](#feature-support-matrix)
- [Quick Start](#quick-start)
- [Single Node Examples](#run-single-node-examples)
- [Multi-Node and Advanced Examples](#advanced-examples)
- [Deploy on SLURM or Kubernetes](#deployment)

## Feature Support Matrix

### Core Dynamo Features

| Feature | SGLang | Notes |
|---------|--------|-------|
| [**Disaggregated Serving**](../../docs/architecture/disagg_serving.md) | ✅ |  |
| [**Conditional Disaggregation**](../../docs/architecture/disagg_serving.md#conditional-disaggregation) | 🚧 | WIP [PR](https://github.com/sgl-project/sglang/pull/7730) |
| [**KV-Aware Routing**](../../docs/architecture/kv_cache_routing.md) | ✅ |  |
| [**SLA-Based Planner**](../../docs/architecture/sla_planner.md) | ❌ | Planned |
| [**Load Based Planner**](../../docs/architecture/load_planner.md) | ❌ | Planned |
| [**KVBM**](../../docs/architecture/kvbm_architecture.md) | ❌ | Planned |

### Large Scale P/D and WideEP Features

| Feature            | SGLang | Notes                                                                 |
|--------------------|--------|-----------------------------------------------------------------------|
| **WideEP**         | ✅/🚧 | Full support on H100s/GB200 WIP [PR](https://github.com/sgl-project/sglang/pull/7556)                                     |
| **DP Rank Routing**| 🚧    | Direct routing supported. Process per DP rank is not supported        |
| **GB200 Support**  | 🚧    | WIP [PR](https://github.com/sgl-project/sglang/pull/7556) |


## Quick Start

Below we provide a guide that lets you run all of our the common deployment patterns on a single node. See our different [architectures](../llm/README.md#deployment-architectures) for a high level overview of each pattern and the architecture diagram for each.

### Start NATS and ETCD in the background

Start using [Docker Compose](../../deploy/metrics/docker-compose.yml)

```bash
docker compose -f deploy/metrics/docker-compose.yml up -d
```

### Build container

```bash
# pull our pre-build sglang runtime container
docker pull nvcr.io/nvidia/ai-dynamo/sglang-runtime:0.3.2
# or build from source
./container/build.sh --framework sglang
```

### Run container

```bash
./container/run.sh -it --framework sglang
```

## Run Single Node Examples

> [!IMPORTANT]
> Each example corresponds to a simple bash script that runs the OpenAI compatible server, processor, and optional router (written in Rust) and LLM engine (written in Python) in a single terminal. You can easily take each command and run them in separate terminals.
>
> Additionally - because we use sglang's argument parser, you can pass in any argument that sglang supports to the worker!


### Aggregated Serving

```bash
cd $DYNAMO_ROOT/components/backends/sglang
./launch/agg.sh
```

### Aggregated Serving with KV Routing

> [!NOTE]
> The current implementation of `components/backends/sglang/src/dynamo/sglang/worker/main.py` publishes _placeholder_ engine metrics to keep the Dynamo KV-router happy. Real-time metrics will be surfaced directly from the SGLang engine once the following pull requests are merged:
> • Dynamo: [ai-dynamo/dynamo #1465](https://github.com/ai-dynamo/dynamo/pull/1465) – _feat: receive kvmetrics from sglang scheduler_.
>
> After these are in, the TODOs in `main.py` will be resolved and the placeholder logic removed.

```bash
cd $DYNAMO_ROOT/components/backends/sglang
./launch/agg_router.sh
```

### Disaggregated serving

<details>
<summary>Under the hood: SGLang Load Balancer vs Dynamo Discovery</summary>

SGLang uses a mini load balancer to route requests to handle disaggregated serving. The load balancer functions as follows:

1. The load balancer receives a request from the client
2. A random `(prefill, decode)` pair is selected from the pool of available workers
3. Request is sent to both `prefill` and `decode` workers via asyncio tasks
4. Internally disaggregation is done from prefill -> decode

Because Dynamo has a discovery mechanism, we do not use a load balancer. Instead, we first route to a random prefill worker, select a random decode worker, and then send the request to both. Internally, SGLang's bootstrap server (which is a part of the `tokenizer_manager`) is used in conjuction with NIXL to handle the kv transfer.

</details>

> [!IMPORTANT]
> Disaggregated serving in SGLang currently requires each worker to have the same tensor parallel size [unless you are using an MLA based model](https://github.com/sgl-project/sglang/pull/5922)

```bash
cd $DYNAMO_ROOT/components/backends/sglang
./launch/disagg.sh
```

### Disaggregated Serving with Mixture-of-Experts (MoE) models and DP attention

You can use this configuration to test out disaggregated serving with dp attention and expert parallelism on a single node before scaling to the full DeepSeek-R1 model across multiple nodes.

```bash
# note this will require 4 GPUs
cd $DYNAMO_ROOT/components/backends/sglang
./launch/disagg_dp_attn.sh
```

## Advanced Examples

Below we provide a selected list of advanced examples. Please open up an issue if you'd like to see a specific example!

### Run on multi-node
- **[Run a multi-node model](docs/multinode-examples.md)**

### Large scale P/D disaggregation with WideEP
- **[Run DeepSeek-R1 on 104+ H100s](docs/dsr1-wideep-h100.md)**
- **[Run DeepSeek-R1 on GB200s](docs/dsr1-wideep-gb200.md)**

### Speculative Decoding
- **[Deploying DeepSeek-R1 with MTP - coming soon!](.)**

### Structured Output and Tool Calling
- **[Tool calling with Dynamo - coming soon!](.)**

### Supporting SGLang's native endpoints via Dynamo
- **[HTTP Server for native SGLang endpoints](docs/sgl-http-server.md)**

## Deployment

We currently provide deployment examples for Kubernetes (coming soon!) and SLURM

## Kubernetes
- **[Deploying Dynamo with SGLang on Kubernetes - coming soon!](.)**

## SLURM
- **[Deploying Dynamo with SGLang on SLURM](slurm_jobs/README.md)**
