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


## Deployment Architectures

See [deployment architectures](../llm/README.md#deployment-architectures) to learn about the general idea of the architecture.
Note that this TensorRT-LLM version does not support all the options yet.

Note: TensorRT-LLM disaggregation does not support conditional disaggregation yet. You can only configure the deployment to always use aggregate or disaggregated serving.

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

#### Step 1: Build TensorRT-LLM base container image

Because of the known issue of C++11 ABI compatibility within the NGC pytorch container, we rebuild TensorRT-LLM from source.
See [here](https://nvidia.github.io/TensorRT-LLM/installation/linux.html) for more informantion.

Use the helper script to build a TensorRT-LLM container base image. The script uses a specific commit id from TensorRT-LLM main branch.

```bash
# TensorRT-LLM uses git-lfs, which needs to be installed in advance.
apt-get update && apt-get -y install git git-lfs

# The script uses python packages like docker-squash to squash image
# layers within trtllm base image
DEBIAN_FRONTEND=noninteractive TZ=America/Los_Angeles apt-get -y install python3 python3-pip python3-venv

./container/build_trtllm_base_image.sh
```

For more information see [here](https://nvidia.github.io/TensorRT-LLM/installation/build-from-source-linux.html#option-1-build-tensorrt-llm-in-one-step) for more details on building from source.
If you already have a TensorRT-LLM container image, you can skip this step.

#### Step 2: Build the Dynamo container

```
./container/build.sh --framework tensorrtllm
```

This build script internally points to the base container image built with step 1. If you skipped previous step because you already have the container image available, you can run the build script with that image as a base.


```bash
# Build dynamo image with other TRTLLM base image.
./container/build.sh --framework TENSORRTLLM --base-image <trtllm-base-image> --base-image-tag <trtllm-base-image-tag>
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

### Example architectures

#### Aggregated serving
```bash
cd /workspace/examples/tensorrt_llm
dynamo serve graphs.agg:Frontend -f ./configs/agg.yaml
```

#### Aggregated serving with KV Routing
```bash
cd /workspace/examples/tensorrt_llm
dynamo serve graphs.agg_router:Frontend -f ./configs/agg_router.yaml
```

#### Disaggregated serving
```bash
cd /workspace/examples/tensorrt_llm
dynamo serve graphs.disagg:Frontend -f ./configs/disagg.yaml
```

We are defining TRTLLM_USE_UCX_KVCACHE so that TRTLLM uses UCX for transfering the KV
cache between the context and generation workers.

#### Disaggregated serving with KV Routing
```bash
cd /workspace/examples/tensorrt_llm
dynamo serve graphs.disagg_router:Frontend -f ./configs/disagg_router.yaml
```

We are defining TRTLLM_USE_UCX_KVCACHE so that TRTLLM uses UCX for transfering the KV
cache between the context and generation workers.

#### Multi-node Disaggregated Serving

In the following example, we will demonstrate how to run a Disaggregated Serving
deployment across multiple nodes. For simplicity, we will demonstrate how to
deploy a single Decode worker on one node, and a single Prefill worker on the other node.
However, the instance counts, TP sizes, and other configs, can be customized and deployed
in similar ways.

##### Head Node

Start nats/etcd:
```bash
# NATS data persisted to /tmp/nats/jetstream by default
nats-server -js &

# Persist data to /tmp/etcd, otherwise defaults to ${PWD}/default.etcd if left unspecified
# NOTE: Clearing out the etcd data dir across runs helps to guarantee a clean and reproducible run
etcd --listen-client-urls http://0.0.0.0:2379 --advertise-client-urls http://0.0.0.0:2379 --data-dir /tmp/etcd &
```

FIXME and REMOVEME: Patch dynamo serve for `mpirun` usage on `slurm`, this file comes from this branch:
https://github.com/ai-dynamo/dynamo/compare/main...rmccormick/trtllm/slurm_mpirun_war
```bash
cp /lustre/fsw/core_dlfw_ci/rmccormick/dynamo_trtllm/dynamo_serve_patch.py /usr/local/lib/python3.12/dist-packages/dynamo/sdk/cli/serving.py
```

FIXME and REMOVEME: In slurm based environments specifically, if you reserve a multi-node allocation, for example with `salloc -N 2 ...`
and enter an interactive shell on each node to run commands with `srun -N 1 --jobid=<jobid> ...`, you may
need to edit some slurm environment variables based on the use case:
```bash
# FIXME: This is a hack to avoid mpirun errors when trying to call srun on
# multi-node slurm allocations.
export SLURM_NODELIST=${HOSTNAME}
```

Launch graph of Frontend, (optionally include Router), and Decode worker. Note that
the Prefill worker is intentionally excluded from the graph  because this experiment
will have the Prefill worker on a separate node, so we don't need to launch it on
this node. Therefore, we can just use the aggregated graph defined in `graphs/agg.py`
to launch these components.

```bash
cd /workspace/examples/tensorrt_llm
dynamo serve graphs.agg:Frontend -f ./configs/disagg.yaml &
```

##### Worker Node(s)

Set environment variables pointing at the etcd/nats endpoints on the head node
so the Dynamo Distributed Runtime can orchestrate communication and
discoverability between nodes:
```bash
# if not head node
export HEAD_NODE_IP="<head-node-ip>"
export NATS_SERVER="nats://${HEAD_NODE_IP}:4222"
export ETCD_ENDPOINTS="${HEAD_NODE_IP}:2379"
```

FIXME and REMOVEME: Patch dynamo serve for `mpirun` usage on `slurm`, this file comes from this branch:
https://github.com/ai-dynamo/dynamo/compare/main...rmccormick/trtllm/slurm_mpirun_war
```bash
cp /lustre/fsw/core_dlfw_ci/rmccormick/dynamo_trtllm/dynamo_serve_patch.py /usr/local/lib/python3.12/dist-packages/dynamo/sdk/cli/serving.py
```

FIXME and REMOVEME: In slurm based environments specifically, if you reserve a multi-node allocation, for example with `salloc -N 2 ...`
and enter an interactive shell on each node to run commands with `srun -N 1 --jobid=<jobid> ...`, you may
need to edit some slurm environment variables based on the use case:
```bash
# FIXME: This is a hack to avoid mpirun errors when trying to call srun on
# multi-node slurm allocations.
export SLURM_NODELIST=${HOSTNAME}
```

Deploy a Prefill worker:
```
cd /workspace/examples/tensorrt_llm
dynamo serve components.prefill_worker:TensorRTLLMPrefillWorker -f ./configs/disagg.yaml &
```

Now you have a 2-node deployment with 1 Decode worker on node1, and 1 Prefill worker on node2!

To apply the same concepts to a larger model like DeepSeek R1 FP4 on GB200, you can follow the same
steps but replacing the config files involed, or write your own custom config files:
```bash
# On head node - deploy decode worker, frontend, and processor
cd /workspace/examples/tensorrt_llm
dynamo serve graphs.agg:Frontend -f ./configs/deepseek_r1_4xB200_disagg.yaml &

# On worker node - deploy prefill worker only
cd /workspace/examples/tensorrt_llm
dynamo serve components.prefill_worker:TensorRTLLMPrefillWorker -f ./configs/deepseek_r1_4xB200_disagg.yaml &
```

##### Client

To send a request to the disaggregated deployment, target the head node which deployed the `Frontend`.

### Client

See [client](../llm/README.md#client) section to learn how to send request to the deployment.

### Close deployment

See [close deployment](../../docs/guides/dynamo_serve.md#close-deployment) section to learn about how to close the deployment.

Remaining tasks:

- [x] Add support for the disaggregated serving.
- [ ] Add integration test coverage.
- [ ] Add instructions for benchmarking.
- [ ] Add multi-node support.
- [ ] Merge the code base with llm example to reduce the code duplication.
- [ ] Use processor from dynamo-llm framework.
- [ ] Enable NIXL integration with TensorRT-LLM once available. Currently, TensorRT-LLM uses UCX to transfer KV cache.
