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

## Use the Latest Release

We recommend using the latest stable release of dynamo to avoid breaking changes:

[![GitHub Release](https://img.shields.io/github/v/release/ai-dynamo/dynamo)](https://github.com/ai-dynamo/dynamo/releases/latest)

You can find the latest release [here](https://github.com/ai-dynamo/dynamo/releases/latest) and check out the corresponding branch with:

```bash
git checkout $(git describe --tags $(git rev-list --tags --max-count=1))
```

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

#### Aggregated serving with Multi-Token Prediction (MTP) and DeepSeek R1
```bash
cd /workspace/examples/tensorrt_llm
dynamo serve graphs.agg:Frontend -f configs/deepseek_r1/mtp/mtp_agg.yaml
```

Notes:
- MTP is only available within the container built with the experimental TensorRT-LLM commit. Please add --use-default-experimental-tensorrtllm-commit to the arguments of the build.sh script.

  Example: `./container/build.sh --framework tensorrtllm --use-default-experimental-tensorrtllm-commit`

- There is a noticeable latency for the first two inference requests. Please send warm-up requests before starting the benchmark.
- MTP performance may vary depending on the acceptance rate of predicted tokens, which is dependent on the dataset or queries used while benchmarking. Additionally, `ignore_eos` should generally be omitted or set to `false` when using MTP to avoid speculating garbage outputs and getting unrealistic acceptance rates.

#### Multi-Node Disaggregated Serving

In the following example, we will demonstrate how to run a Disaggregated Serving
deployment across multiple nodes. For simplicity, we will demonstrate how to
deploy a single Decode worker on one node, and a single Prefill worker on the other node.
However, the instance counts, TP sizes, other configs, and responsibilities of each node
can be customized and deployed in similar ways.

For example, to deploy Deepseek R1, you could replace the referenced example
configs (`configs/agg.yaml`, `configs/disagg.yaml`) with corresponding Deepseek R1
example configs (`configs/deepseek_r1/agg.yaml`, `configs/deepseek_r1/disagg.yaml`).
You can find the example Deepseek R1 configs for GB200
[here](configs/deepseek_r1), but the config settings can be customized for testing
other hardware configurations or parallelism strategies.

This "multi-node" example demonstrates how to generally connect dynamo workers from
different nodes, but for simplicity, each worker individually fits on a single node.
For details on how to launch a worker that spans multiple nodes due to sheer model
size, or for features like large scale expert parallelism, see the
[multinode worker example](configs/deepseek_r1/multinode).

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

Launch graph of Frontend and TensorRTLLMWorker (decode) on head node:

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
```bash
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

#### Multi-Node Disaggregated Serving with Multi-Token Prediction (MTP) and DeepSeek R1

Most of the steps remain the same as the above example, but this time we will have `dynamo serve` point to different config files that contains the MTP configurations

##### Head Node

Start nats/etcd
```bash
nats-server -js &
etcd --listen-client-urls http://0.0.0.0:2379 --advertise-client-urls http://0.0.0.0:2379 --data-dir /tmp/etcd &
```

Launch graph of Frontend and TensorRTLLMWorker (decode) on head node:

```bash
cd /workspace/examples/tensorrt_llm
dynamo serve graphs.agg:Frontend -f configs/deepseek_r1/mtp/mtp_disagg.yaml  &
```

##### Worker Node(s)

Set environment variables pointing at the etcd/nats endpoints on the head node.
```bash
export HEAD_NODE_IP="<head-node-ip>"
export NATS_SERVER="nats://${HEAD_NODE_IP}:4222"
export ETCD_ENDPOINTS="${HEAD_NODE_IP}:2379"
```

Deploy a Prefill worker:
```bash
cd /workspace/examples/tensorrt_llm
dynamo serve components.prefill_worker:TensorRTLLMPrefillWorker -f configs/deepseek_r1/mtp/mtp_disagg.yaml --service-name TensorRTLLMPrefillWorker &
```

Notes:
- MTP is only available within the container built with the experimental TensorRT-LLM commit. Please add --use-default-experimental-tensorrtllm-commit to the arguments of the build.sh script.

  Example: `./container/build.sh --framework tensorrtllm --use-default-experimental-tensorrtllm-commit`
- There is a noticeable latency for the first two inference requests. Please send warm-up requests before starting the benchmark.
- MTP performance may vary depending on the acceptance rate of predicted tokens, which is dependent on the dataset or queries used while benchmarking. Additionally, `ignore_eos` should generally be omitted or set to `false` when using MTP to avoid speculating garbage outputs and getting unrealistic acceptance rates.


### Client

See [client](../llm/README.md#client) section to learn how to send request to the deployment.

NOTE: To send a request to a multi-node deployment, target the node which deployed the `Frontend` component.

### Close deployment

See [close deployment](../../docs/guides/dynamo_serve.md#close-deployment) section to learn about how to close the deployment.

### Benchmarking

To benchmark your deployment with GenAI-Perf, see this utility script, configuring the
`model` name and `host` based on your deployment: [perf.sh](../../benchmarks/llm/perf.sh)


### KV Cache Transfer for Disaggregated Serving

In disaggregated serving architectures, KV cache must be transferred between prefill and decode nodes. TensorRT-LLM supports two methods for this transfer:

#### Default Method: UCX
By default, TensorRT-LLM uses UCX (Unified Communication X) for KV cache transfer between prefill and decode nodes. UCX provides high-performance communication optimized for GPU-to-GPU transfers.

#### Experimental Method: NIXL
TensorRT-LLM also provides experimental support for using **NIXL** (NVIDIA Inference Xfer Library) for KV cache transfer. [NIXL](https://github.com/ai-dynamo/nixl) is NVIDIA's high-performance communication library designed for efficient data transfer in distributed GPU environments.

**Note:** NIXL support in TensorRT-LLM is experimental and is not suitable for production environments yet.

#### Using NIXL for KV Cache Transfer

**Note:** NIXL backend for TensorRT-LLM is currently only supported on AMD64 (x86_64) architecture. If you're running on ARM64, you'll need to use the default UCX method for KV cache transfer.

To enable NIXL for KV cache transfer in disaggregated serving:

1. **Build the container with NIXL support:**
   The TensorRT-LLM wheel must be built from source with NIXL support. The `./container/build.sh` script caches previously built TensorRT-LLM wheels to reduce build time. If you have previously built a TensorRT-LLM wheel without NIXL support, you must delete the cached wheel to force a rebuild with NIXL support.

   **Remove cached TensorRT-LLM wheel (only if previously built without NIXL support):**
   ```bash
   rm -rf /tmp/trtllm_wheel
   ```

   **Build the container with NIXL support:**
   ```bash
   ./container/build.sh --framework tensorrtllm \
     --use-default-experimental-tensorrtllm-commit \
     --trtllm-use-nixl-kvcache-experimental
   ```

   **Note:** Both `--use-default-experimental-tensorrtllm-commit` and `--trtllm-use-nixl-kvcache-experimental` flags are required to enable NIXL support.

2. **Run the containerized environment:**
   See [run container](#run-container) section to learn how to start the container image built in previous step.

3. **Start the disaggregated service:**
   See [disaggregated serving](#disaggregated-serving) to see how to start the deployment.

4. **Send the request:**
   See [client](#client) section to learn how to send the request to deployment.

**Important:** Ensure that ETCD and NATS services are running before starting the service.

The container will automatically configure the appropriate environment variables (`TRTLLM_USE_NIXL_KVCACHE=1`) when built with the NIXL flag. The same container image can be used to use UCX for KV cache transfer.
```bash
unset TRTLLM_USE_NIXL_KVCACHE
export TRTLLM_USE_UCX_KVCACHE=1
```

