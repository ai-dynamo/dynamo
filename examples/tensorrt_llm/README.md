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

# LLM Deployment Examples using TensorRT-LLM

This directory contains examples and reference implementations for deploying Large Language Models (LLMs) in various configurations using TensorRT-LLM.

# User Documentation

- [Deployment Architectures](#deployment-architectures)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Build docker](#build-docker)
  - [Run container](#run-container)
  - [Run deployment](#run-deployment)
    - [Single Node deployment](#single-node-example-architectures)
    - [Multinode deployment](#multinode-deployment)
  - [Client](#client)
  - [Benchmarking](#benchmarking)
  - [Close Deployment](#close-deployment)
- [KV Cache Transfer](#kv-cache-transfer-in-disaggregated-serving)

# Quick Start

## Use the Latest Release

We recommend using the latest stable release of dynamo to avoid breaking changes:

[![GitHub Release](https://img.shields.io/github/v/release/ai-dynamo/dynamo)](https://github.com/ai-dynamo/dynamo/releases/latest)

You can find the latest release [here](https://github.com/ai-dynamo/dynamo/releases/latest) and check out the corresponding branch with:

```bash
git checkout $(git describe --tags $(git rev-list --tags --max-count=1))
```

## Deployment Architectures

See [deployment architectures](../llm/README.md#deployment-architectures) to learn about the general idea of the architecture. 

Note: TensorRT-LLM disaggregation does not support conditional disaggregation yet. You can configure the deployment to always use either aggregate or disaggregated serving.

## Getting Started

1. Choose a deployment architecture based on your requirements
2. Configure the components as needed
3. Deploy using the provided scripts

### Prerequisites

Start required services (etcd and NATS) using [Docker Compose](../../deploy/metrics/docker-compose.yml)
```bash
docker compose -f deploy/metrics/docker-compose.yml up -d
```

### Build docker

```bash
# TensorRT-LLM uses git-lfs, which needs to be installed in advance.
apt-get update && apt-get -y install git git-lfs

# On an x86 machine:
./container/build.sh --framework tensorrtllm

# On an ARM machine:
./container/build.sh --framework tensorrtllm --platform linux/arm64

# Build the container with the default experimental TensorRT-LLM commit
# WARNING: This is for experimental feature testing only.
# The container should not be used in a production environment.
./container/build.sh --framework tensorrtllm --use-default-experimental-tensorrtllm-commit
```

### Run container

```
./container/run.sh --framework tensorrtllm -it
```
## Run Deployment

This figure shows an overview of the major components to deploy:



```

+------+      +-----------+      +------------------+             +---------------+
| HTTP |----->| processor |----->|      Worker      |------------>|     Prefill   |
|      |<-----|           |<-----|                  |<------------|     Worker    |
+------+      +-----------+      +------------------+             +---------------+
                  |    ^                  |
       query best |    | return           | publish kv events
           worker |    | worker_id        v
                  |    |         +------------------+
                  |    +---------|     kv-router    |
                  +------------->|                  |
                                 +------------------+

```

Note: The above architecture illustrates all the components. The final components
that get spawned depend upon the chosen graph.

### Single-Node example architectures

> [!IMPORTANT]
> Below we provide some simple shell scripts that run the components for each configuration. Each shell script is simply running the `dynamo-run` to start up the ingress and using `python3` to start up the workers. You can easily take each commmand and run them in separate terminals.

#### Aggregated
```bash
cd $DYNAMO_ROOT/examples/tensorrt_llm
./launch/agg.sh
```

#### Aggregated with KV Routing
```bash
cd $DYNAMO_ROOT/examples/tensorrt_llm
./launch/agg_router.sh
```

#### Disaggregated

> [!IMPORTANT]
> Disaggregated serving supports two strategies for request flow: `"prefill_first"` and `"decode_first"`. By default, the script below uses the `"decode_first"` strategy, which can reduce response latency by minimizing extra hops in the return path. You can switch strategies by setting the `DISAGGREGATION_STRATEGY` environment variable.

```bash
cd $DYNAMO_ROOT/examples/tensorrt_llm
./launch/disagg.sh
```

#### Disaggregated with KV Routing

> [!IMPORTANT]
> Disaggregated serving with KV routing uses a "prefill first" workflow by default. Currently, Dynamo supports KV routing to only one endpoint per model. In disaggregated workflow, it is generally more effective to route requests to the prefill worker. If you wish to use a "decode first" workflow instead, you can simply set the `DISAGGREGATION_STRATEGY` environment variable accordingly.

```bash
cd $DYNAMO_ROOT/examples/tensorrt_llm
./launch/disagg_router.sh
```

### Multinode Deployment

For details and instructions on multinode serving, please refer to the [multinode-examples.md](./multinode-examples.md) document. This guide provides step-by-step examples and configuration tips for deploying Dynamo with TensorRT-LLM across multiple nodes.


### Client

See [client](../llm/README.md#client) section to learn how to send request to the deployment.

NOTE: To send a request to a multi-node deployment, target the node which is running `dynamo-run in=http`.

### Benchmarking

To benchmark your deployment with GenAI-Perf, see this utility script, configuring the
`model` name and `host` based on your deployment: [perf.sh](../../benchmarks/llm/perf.sh)

### Close deployment

See [close deployment](../../docs/guides/dynamo_serve.md#close-deployment) section to learn about how to close the deployment.

### KV Cache Transfer in Disaggregated Serving

Dynamo with TensorRT-LLM supports two methods for transferring KV cache in disaggregated serving: UCX (default) and NIXL (experimental). For detailed information and configuration instructions for each method, see the [KV cache transfer guide](./kv-cache-tranfer.md).
