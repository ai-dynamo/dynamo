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

## Deployment Architectures

See [deployment architectures](../llm/README.md#deployment-architectures) to learn about the general idea of the architecture. SGLang currently supports aggregated and disaggregated serving. KV routing support is coming soon!

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
./container/build.sh --framework sglang
```

### Run container

```bash
./container/run.sh -it --framework sglang
```

### Example architectures

#### Aggregated

```bash
cd /workspace/examples/sglang
dynamo serve graphs.agg:Frontend -f ./configs/agg.yaml
```

#### Disaggregated

SGLang uses a mini load balancer to route requests to handle disaggregated serving. The load balancer functions as follows

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

##### Disaggregated with MoE and DP attention

SGLang also supports DP attention for MoE models. We provide an example config for this in `configs/disagg-dp-attention.yaml` which is based on the [DeepSeek-R1-Small-2layers](https://huggingface.co/silence09/DeepSeek-R1-Small-2layers) model. You can use this configuration to test out disaggregated serving on a single node before scaling to the full DeepSeek-R1 model across multiple nodes.

```bash
# note this will require 4 GPUs
cd /workspace/examples/sglang
dynamo serve graphs.disagg:Frontend -f ./configs/disagg-dp-attention.yaml
```

##### Disaggregated with WideEP

Dynamo supports SGLang's implementation of wide expert parallelism and large scale P/D for DeepSeek-R1! You can read their blog post [here](https://www.nvidia.com/en-us/technologies/ai/deepseek-r1-large-scale-p-d-with-wide-expert-parallelism/) for more details. We provide a Dockerfile for this in `container/Dockerfile.sglang-deepep` and configurations to deploy this at scale. In this example, we will run 1 prefill worker on 2 H100 nodes and 1 decode worker on 4 H100 nodes (48 total GPUs). You can easily scale this to 96 GPUs or more by simply changing the configuration files.

Steps to run:

1. Build the SGLang DeepEP container

```bash
git clone https://github.com/sgl-project/sglang.git
cd sglang/docker
docker build -f Dockerfile.deepep -t deepep .
```

You will now have a `deepep:latest` image

2. Build the Dynamo container

```bash
cd $DYNAMO_ROOT
docker build -f container/Dockerfile.sglang-deepep . -t dynamo-deepep --no-cache
```

3. You can run this container on each 8xH100 node using the following command.

> [!IMPORTANT]
> We recommend downloading DeepSeek-R1 and then mounting it to the container. You can find the model [here](https://huggingface.co/deepseek-ai/DeepSeek-R1)

```bash
docker run \
    --gpus all \
    -it \
    --rm \
    --network host \
    --volume /PATH_TO_DSR1_MODEL/:/model/ \
    --shm-size=10G \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    --ulimit nofile=65536:65536 \
    --cap-add CAP_SYS_PTRACE \
    --ipc host \
    dynamo-deepep:latest
```

In each container, you should be in the `/sgl-workspace/dynamo/examples/sglang` directory.

4. On the head prefill node, start `nats-server` and `etcd` using the following commands

```bash
nats-server -js &
etcd --listen-client-urls http://0.0.0.0:2379 \
     --advertise-client-urls http://0.0.0.0:2379 \
     --listen-peer-urls http://0.0.0.0:2380 \
     --initial-cluster default=http://HEAD_PREFILL_NODE_IP:2380 &
```

5. On every other node, go ahead and export the `NATS_SERVER` and `ETCD_ENDPOINTS` environment variables

> [!IMPORTANT]
> You will need the IP address of your head prefill node and head decode node for the configuration files

```bash
# run this on every other node
export NATS_SERVER=nats://HEAD_PREFILL_NODE_IP:4222
export ETCD_ENDPOINTS=http://HEAD_PREFILL_NODE_IP:2379
```

6. Configure each configuration file to use the correct `dist-init-addr`, and `node-rank`

Each container contains the configuration file in `configs/dsr1.yaml`. For our example, we will make the following changes:

On the prefill head node, `vim` into the configs and change the following section of the `SGLangWorker`:

```yaml
SGLangWorker:
    ...
    dist-init-addr: HEAD_PREFILL_NODE_IP
    nnodes: 2
    node-rank: 0
    ...
```

On the other prefill node (since this example has 2 prefill nodes), change the following section of the `SGLangWorker`:

```yaml
SGLangWorker:
    ...
    dist-init-addr: HEAD_PREFILL_NODE_IP
    nnodes: 2
    node-rank: 1
    ...
```

On the decode head node, `vim` into the configs and change the following section of the `SGLangDecodeWorker`:

```yaml
SGLangDecodeWorker:
    ...
    dist-init-addr: HEAD_DECODE_NODE_IP
    nnodes: 4
    node-rank: 0
    ...
```

On the other decode nodes (this example has 4 decode nodes), change the following section of the `SGLangDecodeWorker`:

```yaml
SGLangDecodeWorker:
    ...
    dist-init-addr: HEAD_DECODE_NODE_IP
    nnodes: 4
    # depending on which node this will be 1, 2, and 3
    node-rank: 1
```

7. Start up the workers using the following commands

On prefill head node

```bash
dynamo serve graphs.agg:Frontend -f configs/dsr1.yaml
```

On prefill child node

```bash
dynamo serve graphs.agg:Frontend -f configs/dsr1.yaml --service-name SGLangWorker
```

On all decode nodes

```bash
dynamo serve graphs.disagg:Frontend -f configs/dsr1.yaml --service-name SGLangDecodeWorker
```

8. Run the warmup script to warm up the model

DeepGEMM kernels can sometimes take a while to warm up. Here we provide a small helper script that should help. You can run this as many times as you want before starting inference/benchmarking. You can exec into the head node and run this script standalone - it does not need a container.

```bash
./warmup.sh HEAD_PREFILL_NODE_IP
```
