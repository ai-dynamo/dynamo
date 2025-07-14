<!--
SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

# LLM Deployment Examples using SGLang

This directory contains examples and reference implementations for deploying Large Language Models (LLMs) in various configurations using SGLang. SGLang internally uses ZMQ to communicate between the ingress and the engine processes. For Dynamo, we leverage the runtime to communicate directly with the engine processes and handle ingress and pre/post processing on our end.

## Use the Latest Release

We recommend using the latest stable release of dynamo to avoid breaking changes:

[![GitHub Release](https://img.shields.io/github/v/release/ai-dynamo/dynamo)](https://github.com/ai-dynamo/dynamo/releases/latest)

You can find the latest release [here](https://github.com/ai-dynamo/dynamo/releases/latest) and check out the corresponding branch with:

```bash
git checkout $(git describe --tags $(git rev-list --tags --max-count=1))
```

## Feature Support Matrix

### Core Dynamo Features

| Feature | SGLang | Notes | Docs |
|---------|--------|--------|------|
| **Disaggregated Serving** | ✅ | Full support with prefill/decode separation | [doc](https://github.com/sgl-project/sglang/blob/main/docs/disaggregation.md) |
| **Conditional Disaggregation** | 🚧 | Work in progress ([#7730](https://github.com/sgl-project/sglang/pull/7730)) | [PR](https://github.com/sgl-project/sglang/pull/7730) |
| **KV-Aware Routing** | ✅ | Full support | [doc](https://github.com/sgl-project/sglang/blob/main/docs/kv_routing.md) |
| **SLA-Based Planner** | ❌ | Not supported | - |
| **Load Based Planner** | ❌ | Not supported | - |
| **KVBM** | ❌ | Not supported | - |

### Large Scale P/D and WideEP Features

| Feature | SGLang | Notes | Docs |
|---------|--------|------|
| **WideEP** | ✅ | Full Support on H100s and GB200s | [doc](https://github.com/sgl-project/sglang/blob/main/docs/wideep.md) |
| **DP Rank Routing** | 🚧 | Direct routing supported. Process per DP rank is not supported | - |
| **GB200 Support** | 🚧 | Work in progress ([#7556](https://github.com/sgl-project/sglang/pull/7556)) | [PR](https://github.com/sgl-project/sglang/pull/7556) |


## Quick Start 

Below we provide a quick start guide that lets you run all of our the common deployment patterns on a single node. We have a much more comprehensive guide for advanced deployments below. See our different [architectures](../llm/README.md#deployment-architectures) for a high level overview of each pattern and the architecture diagram for each.

### Start NATS and ETCD in the background

Start required services (etcd and NATS) using [Docker Compose](../../deploy/metrics/docker-compose.yml)

```bash
docker compose -f deploy/metrics/docker-compose.yml up -d
```

### Build container

```bash
# On an x86 machine TODO: sglang public image?
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
cd $DYNAMO_ROOT/examples/sglang
./launch/agg.sh
```

### Aggregated Serving with KV Routing

> [!NOTE]
> The current implementation of `examples/sglang/components/worker.py` publishes _placeholder_ engine metrics to keep the Dynamo KV-router happy. Real-time metrics will be surfaced directly from the SGLang engine once the following pull requests are merged:
> • Dynamo: [ai-dynamo/dynamo #1465](https://github.com/ai-dynamo/dynamo/pull/1465) – _feat: receive kvmetrics from sglang scheduler_.
>
> After these are in, the TODOs in `worker.py` will be resolved and the placeholder logic removed.

```bash
cd $DYNAMO_ROOT/examples/sglang
./launch/agg_router.sh
```

### Disaggregated serving

<details>
<summary>SGLang Load Balancer vs Dynamo Discovery</summary>

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
cd $DYNAMO_ROOT/examples/sglang
./launch/disagg.sh
```

### Disaggregated Serving with Mixture-of-Experts (MoE) models and DP attention

You can use this configuration to test out disaggregated serving with dp attention and expert parallelism on a single node before scaling to the full DeepSeek-R1 model across multiple nodes.

```bash
# note this will require 4 GPUs
cd $DYNAMO_ROOT/examples/sglang
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

### SGLang Utilities and Tools
- **[HTTP Server to flush cache and record MoE expert distribution data](docs/sgl-http-server.md)**

## Kubernetes
- **[Deploying Dynamo with SGLang on Kubernetes - coming soon!](.)**