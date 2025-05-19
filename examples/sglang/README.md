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

> [!IMPORTANT]
> In order to run these examples, you will need to install sglang using `uv pip install "sglang[all]>=0.4.6.post4"`. Additionally, SGLang currently does not have pre-built wheels for ARM. If you are on an ARM machine - you will need to install SGLang from source.

## Deployment Architectures

See [deployment architectures](../llm/README.md#deployment-architectures) to learn about the general idea of the architecture. SGLang currently support only aggregated serving but routing and disaggregation support are coming very soon!

## Getting Started

1. Choose a deployment architecture based on your requirements
2. Configure the components as needed
3. Deploy using the provided scripts

### Prerequisites

Start required services (etcd and NATS) using [Docker Compose](../../deploy/docker-compose.yml)

```bash
docker compose -f deploy/docker-compose.yml up -d
```

### Build docker

```bash
# On an x86 machine - sglang does not support ARM yet
./container/build.sh
```

### Run container

```bash
./container/run.sh -it
```

### Example architectures

#### Aggregated

```bash
cd /workspace/examples/sglang
dynamo serve graphs.agg:Frontend -f ./configs/agg.yaml
```

#### Disaggregated

As of `sglang==0.4.6.post4`, SGLang uses a mini load balancer to route requests to handle disaggregated serving. The load balancer functions as follows

1. The load balancer receives a request from the client
2. A random `(prefill, decode)` pair is selected from the pool of available workers
3. Request is sent to both `prefill` and `decode` workers via asyncio tasks
4. Internally disaggregation is done from prefill -> decode

Because Dynamo has a discovery mechanism, we do not use a load balancer. Instead, we first route to a random prefill worker, select a random decode worker, and then send the request to both. Internally, SGLang's bootstrap server (which is a part of the `tokenizer_manager`) is used in conjuction with NIXL to handle the kv transfer.

> [!IMPORTANT]
> Disaggregated serving in SGLang currently requires each worker to have the same tensor parallel size [unless you are using an MLA based model](https://github.com/sgl-project/sglang/pull/5922)

```bash
cd /workspace/examples/sglang
dynamo serve graphs.disagg:Frontend -f ./configs/disagg.yaml
```
