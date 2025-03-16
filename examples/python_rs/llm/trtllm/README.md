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

# TensorRT-LLM Integration with Dynamo

This example demonstrates how to use Dynamo to serve large language models with the tensorrt_llm engine, enabling efficient model serving with both monolithic and disaggregated deployment options.

## Prerequisites

Start required services (etcd and NATS):

   Option A: Using [Docker Compose](/runtime/rust/docker-compose.yml) (Recommended)
   ```bash
   docker-compose up -d
   ```

   Option B: Manual Setup

    - [NATS.io](https://docs.nats.io/running-a-nats-service/introduction/installation) server with [Jetstream](https://docs.nats.io/nats-concepts/jetstream)
        - example: `nats-server -js --trace`
    - [etcd](https://etcd.io) server
        - follow instructions in [etcd installation](https://etcd.io/docs/v3.5/install/) to start an `etcd-server` locally
        - example: `etcd --listen-client-urls http://0.0.0.0:2379 --advertise-client-urls http://0.0.0.0:2379`


## Building the Environment

### Build the Dynamo container with TensorRT-LLM

#### Step 1: Build TensorRT-LLM base container image

Because of the known issue of C++11 ABI compatibility, we rebuild the TensorRT-LLM from source within NGC pytorch container.
See [here](https://nvidia.github.io/TensorRT-LLM/installation/linux.html) for more informantion.

Use the helper script to build a TensorRT-LLM container base image. The script uses a specific commit id from TensorRT-LLM main branch.

```bash
./container/build_trtllm_base_image.sh
```

For more information see [here](https://nvidia.github.io/TensorRT-LLM/installation/build-from-source-linux.html#option-1-build-tensorrt-llm-in-one-step) for more details on building from source.
If you already have a TensorRT-LLM container image, you can skip this step.

#### Step 2: Build the Dynamo container

```bash
# Build image
./container/build.sh --framework TENSORRTLLM
```

This build script internally points to the base container image built with step 1. If you skipped previous step because you already have the container image available, you can run the build script with that image as a base.

```bash
# Build dynamo image with other TRTLLM base image.
./container/build.sh --framework TENSORRTLLM --base-image <trtllm-base-image> --base-image-tag <trtllm-base-image-tag>
```

## Launching the Environment
```bash
# Run image interactively from with the Dynamo root directory.
./container/run.sh --framework TENSORRTLLM -it
```

## Quick Start using dynamo run

```bash
# Run the server
dynamo run out=pystr:./monolith/dynamo_engine.py -- --engine_args llm_api_config.yaml
```

The above command should load the model specified in `llm_api_config.yaml` and start accepting
text input from the client. For more details on the `dynamo run` command, please refer to the
[dynamo run](/launch/README.md#python-bring-your-own-engine) documentation.

Currently only monolithic deployment option is supported by `dynamo run` for TensorRT-LLM.
Adding support for disaggregated deployment is under development. This does *not* require
any other pre-requisites mentioned in the [Prerequisites](#prerequisites) section.



## Deployment Options

Note: NATS and ETCD servers should be running and accessible from the container as described in the [Prerequisites](#prerequisites) section.

### Monolithic Deployment

#### 1. HTTP Server

Run the server logging (with debug level logging):
```bash
DYN_LOG=DEBUG http &
```
By default the server will run on port 8080.

Add model to the server:
```bash
llmctl http add chat TinyLlama/TinyLlama-1.1B-Chat-v1.0 dynamo.tensorrt-llm.chat/completions
llmctl http add completion TinyLlama/TinyLlama-1.1B-Chat-v1.0 dynamo.tensorrt-llm.completions
```

#### 2. Workers

##### Option 2.1 Single-Node Single-GPU

```bash
# Launch worker
cd /workspace/examples/python_rs/llm/trtllm
mpirun --allow-run-as-root -n 1 --oversubscribe python3 -m monolith.worker --engine_args llm_api_config.yaml 1>agg_worker.log 2>&1 &
```

Upon successful launch, the output should look similar to:

```bash
[TensorRT-LLM][INFO] KV cache block reuse is disabled
[TensorRT-LLM][INFO] Max KV cache pages per sequence: 2048
[TensorRT-LLM][INFO] Number of tokens per block: 64.
[TensorRT-LLM][INFO] [MemUsageChange] Allocated 26.91 GiB for max tokens in paged KV cache (220480).
[02/14/2025-09:38:53] [TRT-LLM] [I] max_seq_len=131072, max_num_requests=2048, max_num_tokens=8192
[02/14/2025-09:38:53] [TRT-LLM] [I] Engine loaded and ready to serve...
```

`nvidia-smi` can be used to check the GPU usage and the model is loaded on single GPU.

##### Option 2.2 Single-Node Multi-GPU

Update `tensor_parallel_size` in the `llm_api_config.yaml` to load the model with the desired number of GPUs.
`nvidia-smi` can be used to check the GPU usage and the model is loaded on 4 GPUs.

When launching the workers, prepend `trtllm-llmapi-launch` to the command.
```bash
# Note: -n should still be 1
trtllm-llmapi-launch mpirun --allow-run-as-root -n 1 --oversubscribe python3 -m monolith.worker ...
```

##### Option 2.3 Multi-Node Multi-GPU

TODO: Add multi-node multi-GPU example

#### 3. Client

```bash
# Chat Completion
curl localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "messages": [
      {"role": "user", "content": "What is the capital of France?"}
    ]
  }'
```

The output should look similar to:
```json
{
  "id": "ab013077-8fb2-433e-bd7d-88133fccd497",
  "choices": [
    {
      "message": {
        "role": "assistant",
        "content": "The capital of France is Paris."
      },
      "index": 0,
      "finish_reason": "stop"
    }
  ],
  "created": 1740617803,
  "model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
  "object": "chat.completion",
  "usage": null,
  "system_fingerprint": null
}
```

```bash
# Completion
curl localhost:8080/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
        "model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "prompt": "The capital of France is",
        "max_tokens": 1,
        "temperature": 0
    }'
```

Output:
```json
{
  "id":"cmpl-e0d75aca1bd540399809c9b609eaf010",
  "choices":[
    {
      "text":"Paris",
      "index":0,
      "finish_reason":"length"
    }
  ],
  "created":1741024639,
  "model":"TinyLlama/TinyLlama-1.1B-Chat-v1.0",
  "object":"text_completion",
  "usage":null
}
```

Note: For KV cache aware routing, please refer to the [KV Aware Routing](./docs/kv_aware_routing.md) section.

### Disaggregated Deployment

**Environment**
This is the latest image with tensorrt_llm supporting distributed serving with pytorch workflow in LLM API.

Run the container interactively with the following command:
```bash
./container/run.sh --image IMAGE -it
```

#### 1. HTTP Server

Run the server logging (with debug level logging):
```bash
DYN_LOG=DEBUG http &
```
By default the server will run on port 8080.

Add model to the server:
```bash
llmctl http add chat TinyLlama/TinyLlama-1.1B-Chat-v1.0 dynamo.disaggregated_server.chat/completions
llmctl http add completion TinyLlama/TinyLlama-1.1B-Chat-v1.0 dynamo.disaggregated_server.completions
```

#### 2. Workers

##### Option 2.1 Single-Node Disaggregated Deployment

**TRTLLM LLMAPI Disaggregated config file**
Define disaggregated config file similar to the example [single_node_config.yaml](disaggregated/llmapi_disaggregated_configs/single_node_config.yaml). The important sections are the model, context_servers and generation_servers.


1. **Launch the workers**

Launch context and generation workers.\
WORLD_SIZE is the total number of workers covering all the servers described in disaggregated configuration.\
For example, 2 TP2 generation servers are 2 workers but 4 mpi executors.

```bash
cd /workspace/examples/python_rs/llm/trtllm/
mpirun --allow-run-as-root --oversubscribe -n WORLD_SIZE python3 -m disaggregated.worker --engine_args llm_api_config.yaml -c disaggregated/llmapi_disaggregated_configs/single_node_config.yaml 1>disagg_workers.log 2>&1 &
```
If using the provided [single_node_config.yaml](disaggregated/llmapi_disaggregated_configs/single_node_config.yaml), WORLD_SIZE should be 2 as it has 1 context servers(TP=1) and 1 generation server(TP=1).

2. **Launch the disaggregated server**

```bash
cd /workspace/examples/python_rs/llm/trtllm/
python3 -m disaggregated.server --engine_args llm_api_config.yaml 1>disagg_server.log 2>&1 &
```

Note: For KV cache aware routing, please refer to the [KV Aware Routing](./docs/kv_aware_routing.md) section.

3. **Send Requests**
Follow the instructions in the [Monolithic Deployment](#3-client) section to send requests to the router.


For more details on the disaggregated deployment, please refer to the [TRT-LLM example](#TODO).


### Multi-Node Disaggregated Deployment

To run the disaggregated deployment across multiple nodes, we need to launch the servers using MPI, pass the correct NATS and etcd endpoints to each server and update the LLMAPI disaggregated config file to use the correct endpoints.

1. Allocate nodes
The following command allocates nodes for the job and returns the allocated nodes.
```bash
salloc -A ACCOUNT -N NUM_NODES -p batch -J JOB_NAME -t HH:MM:SS
```

You can use `squeue -u $USER` to check the URLs of the allocated nodes.

2. Update the TRTLLM LLMAPI disaggregated config file

An example is provided in [multi_node_config.yaml](disaggregated/llmapi_disaggregated_configs/multi_node_config.yaml).

3. Start the NATS and ETCD endpoints

Use the following commands. These commands will require downloading [NATS.io](https://docs.nats.io/running-a-nats-service/introduction/installation) and [ETCD](https://etcd.io/docs/v3.5/install/):
```bash
./nats-server -js --trace
./etcd --listen-client-urls http://0.0.0.0:2379 --advertise-client-urls http://0.0.0.0:2379
```

Export the correct NATS and etcd endpoints.
```bash
export NATS_SERVER="nats://node1:4222"
export ETCD_ENDPOINTS="http://node1:2379,http://node2:2379"
```

4. Launch the workers from node1 or login node. WORLD_SIZE is similar to single node deployment.
```bash
srun --mpi pmix -N NUM_NODES --ntasks WORLD_SIZE --ntasks-per-node=GPUS_PER_NODE --no-container-mount-home --overlap --container-image IMAGE --output batch_%x_%j.log --err batch_%x_%j.err --container-mounts PATH_TO_DYNAMO:/workspace --container-env=NATS_SERVER,ETCD_ENDPOINTS bash -c 'cd /workspace/examples/python_rs/llm/trtllm && python3 -m disaggregated.worker --engine_args llm_api_config.yaml -c disaggregated/llmapi_disaggregated_configs/multi_node_config.yaml' &
```

Once the workers are launched, you should see the output similar to the following in the worker logs.
```
[TensorRT-LLM][INFO] [MemUsageChange] Allocated 18.88 GiB for max tokens in paged KV cache (1800032).
[02/20/2025-07:10:33] [TRT-LLM] [I] max_seq_len=2048, max_num_requests=2048, max_num_tokens=8192
[02/20/2025-07:10:33] [TRT-LLM] [I] Engine loaded and ready to serve...
[02/20/2025-07:10:33] [TRT-LLM] [I] max_seq_len=2048, max_num_requests=2048, max_num_tokens=8192
[TensorRT-LLM][INFO] Number of tokens per block: 32.
[TensorRT-LLM][INFO] [MemUsageChange] Allocated 18.88 GiB for max tokens in paged KV cache (1800032).
[02/20/2025-07:10:33] [TRT-LLM] [I] max_seq_len=2048, max_num_requests=2048, max_num_tokens=8192
[02/20/2025-07:10:33] [TRT-LLM] [I] Engine loaded and ready to serve...
```

4. Launch the disaggregated server from node1 or login node.
```bash
srun --mpi pmix -N 1 --ntasks 1 --ntasks-per-node=1 --overlap --container-image IMAGE --output batch_disagg_server_%x_%j.log --err batch_disagg_server_%x_%j.err --container-mounts PATH_TO_DYNAMO:/workspace  --container-env=NATS_SERVER,ETCD_ENDPOINTS bash -c 'cd /workspace/examples/python_rs/llm/trtllm && python3 -m disaggregated.server --engine_args llm_api_config.yaml --routing-strategy round_robin' &
```

5. Send requests to the disaggregated server.
The disaggregated server will connect to the OAI compatible server. You can send requests to the disaggregated server using the standard OAI format as shown in previous sections.
