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
```

> [!NOTE]
> Because of a known issue of C++11 ABI compatibility within the NGC pytorch container,
> we rebuild TensorRT-LLM from source. See [here](https://nvidia.github.io/TensorRT-LLM/installation/linux.html)
> for more informantion.
>
> Hence, when running this script for the first time, the time taken by this script can be
> quite long.


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

#### Disaggregated serving with KV Routing
```bash
cd /workspace/examples/tensorrt_llm
dynamo serve graphs.disagg_router:Frontend -f ./configs/disagg_router.yaml
```

#### Multi-Node Disaggregated Serving

In the following example, we will demonstrate how to run a Disaggregated Serving
deployment across multiple nodes. For simplicity, we will demonstrate how to
deploy a single Decode worker on one node, and a single Prefill worker on the other node.
However, the instance counts, TP sizes, other configs, and responsibilities of each node
can be customized and deployed in similar ways.

##### Head Node

Start nats/etcd:
```bash
# NATS data persisted to /tmp/nats/jetstream by default
nats-server -js &

# Persist data to /tmp/etcd, otherwise defaults to ${PWD}/default.etcd if left unspecified
etcd --listen-client-urls http://0.0.0.0:2379 --advertise-client-urls http://0.0.0.0:2379 --data-dir /tmp/etcd &

# NOTE: Clearing out the etcd and nats jetstream data directories across runs
#       helps to guarantee a clean and reproducible results.
```

Launch graph of Frontend, Processor, and TensorRTLLMWorker (decode) on head node:

```bash
cd /workspace/examples/tensorrt_llm
dynamo serve graphs.agg:Frontend -f ./configs/disagg.yaml &
```

Notes:
- The aggregated graph (`graphs.agg`) is chosen here because it also describes
  our desired deployment settings for the head node: launching the utility components
  (Frontend, Processor), and only the decode worker (TensorRTLLMWorker configured with
  `remote-prefill` enabled). We plan to launch the `TensorRTLLMPrefillWorker`
  independently on a separate node in the next step of this demonstration.
  You are free to customize the graph and configuration of components launched on
  each node.
- The disaggregated config `configs/disagg.yaml` is intentionally chosen here as a
  single source of truth to be used for deployments on all of our nodes, describing
  the configurations for all of our components, including both decode and prefill
  workers, but can be customized based on your deployment needs.

##### Worker Node(s)

Set environment variables pointing at the etcd/nats endpoints on the head node
so the Dynamo Distributed Runtime can orchestrate communication and
discoverability between the head node and worker nodes:
```bash
# if not head node
export HEAD_NODE_IP="<head-node-ip>"
export NATS_SERVER="nats://${HEAD_NODE_IP}:4222"
export ETCD_ENDPOINTS="${HEAD_NODE_IP}:2379"
```

Deploy a Prefill worker:
```
cd /workspace/examples/tensorrt_llm
dynamo serve components.prefill_worker:TensorRTLLMPrefillWorker -f ./configs/disagg.yaml --service-name TensorRTLLMPrefillWorker &
```

Now you have a 2-node deployment with 1 Decode worker on the head node, and 1 Prefill worker on a worker node!

##### Additional Notes for Multi-Node Deployments

Notes:
- To include a router in this deployment, change the graph to one that includes the router, such as `graphs.agg_router`,
  and change the config to one that includes the router, such as `configs/disagg_router.yaml`
- This step is assuming you're disaggregated serving and planning to launch prefill workers on separate nodes.
  Howerver, for an aggregated deployment with additional aggregated worker replicas on other nodes, this step
  remains mostly the same. The primary difference between aggregation and disaggregation for this step is
  whether or not the `TensorRTLLMWorker` is configured to do `remote-prefill` or not in the config file
  (ex: `configs/disagg.yaml` vs `configs/agg.yaml`).
- To apply the same concept for launching additional decode workers on worker nodes, you can
  directly start them, similar to the prefill worker step above:
  ```bash
  # Example: deploy decode worker only
  cd /workspace/examples/tensorrt_llm
  dynamo serve components.worker:TensorRTLLMWorker -f ./configs/disagg.yaml --service-name TensorRTLLMWorker &
  ```
- If you see an error about MPI Spawn failing during TRTLLM Worker initialziation on a Slurm-based cluster,
  try unsetting the following environment variables before launching the TRTLLM worker. If you intend to
  run other slurm-based commands or processes on the same node after deploying the TRTLLM worker, you may
  want to save these values into temporary variables and then restore them afterwards.
  ```bash
  # Workaround for error: `mpi4py.MPI.Exception: MPI_ERR_SPAWN: could not spawn processes`
  unset SLURM_JOBID SLURM_JOB_ID SLURM_NODELIST
  ```

### Client

See [client](../llm/README.md#client) section to learn how to send request to the deployment.

NOTE: To send a request to a multi-node deployment, target the node which deployed the `Frontend` component.

### Close deployment

See [close deployment](../../docs/guides/dynamo_serve.md#close-deployment) section to learn about how to close the deployment.

### Benchmarking

To benchmark your deployment with GenAI-Perf, see this utility script, configuring the
`model` name and `host` based on your deployment: [perf.sh](../llm/benchmarks/perf.sh)

### Future Work

Remaining tasks:
- [x] Add support for the disaggregated serving.
- [x] Add multi-node support.
- [x] Add instructions for benchmarking.
- [ ] Add integration test coverage.
- [ ] Merge the code base with llm example to reduce the code duplication.
- [ ] Use processor from dynamo-llm framework.
- [ ] Enable NIXL integration with TensorRT-LLM once available. Currently, TensorRT-LLM uses UCX to transfer KV cache.
