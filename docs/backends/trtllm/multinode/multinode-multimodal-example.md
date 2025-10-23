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

# Example: Multi-node TRTLLM Workers with Dynamo on Slurm for multimodal models

> [!IMPORTANT]
> There are some known issues in tensorrt_llm==1.1.0rc5 version for multinode multimodal support. It is important to rebuild the dynamo container with a specific version of tensorrt_llm commit to use multimodal feature.
>
> **Build Container**
> ```bash
> ./container/build.sh --framework trtllm --tensorrtllm-commit b4065d8ca64a64eee9fdc64b39cb66d73d4be47c
> ```
>
> **Run Container**
> ```bash
> ./container/run.sh --framework trtllm -it
> ```
>
> **Update Engine Configuration Files**
>
> Before running the deployment, you must update the engine configuration files to change `backend: DEFAULT` to `backend: default` (lowercase). Run the following command:
> ```bash
> sed -i 's/backend: DEFAULT/backend: default/g' /mnt/recipes/llama4/trtllm/multimodal/prefill.yaml /mnt/recipes/llama4/trtllm/multimodal/decode.yaml
> ```


This guide demonstrates how to deploy large multimodal models that require a multi-node setup. It builds on the general multi-node deployment process described in the main [multinode-examples.md](./multinode-examples.md) guide.

Before you begin, ensure you have completed the initial environment configuration by following the **Setup** section in that guide.

The following sections provide specific instructions for deploying `meta-llama/Llama-4-Maverick-17B-128E-Instruct`, including environment variable setup and launch commands. These steps can be adapted for other large multimodal models.

## Environment Variable Setup

Assuming you have already allocated your nodes via `salloc`, and are
inside an interactive shell on one of the allocated nodes, set the
following environment variables based:
```bash
# NOTE: IMAGE must be set manually for now
# To build an iamge, see the steps here:
# https://github.com/ai-dynamo/dynamo/tree/main/docs/backends/trtllm/README.md#build-container
export IMAGE="<dynamo_trtllm_image>"

# MOUNTS are the host:container path pairs that are mounted into the containers
# launched by each `srun` command.
#
# If you want to reference files, such as $MODEL_PATH below, in a
# different location, you can customize MOUNTS or specify additional
# comma-separated mount pairs here.
#
# NOTE: Currently, this example assumes that the local bash scripts and configs
# referenced are mounted into into /mnt inside the container. If you want to
# customize the location of the scripts, make sure to modify `srun_aggregated.sh`
# accordingly for the new locations of `start_frontend_services.sh` and
# `start_trtllm_worker.sh`.
#
# For example, assuming your cluster had a `/lustre` directory on the host, you
# could add that as a mount like so:
#
# export MOUNTS="${PWD}/../:/mnt,/lustre:/lustre"
export MOUNTS="${PWD}/../:/mnt"

# Can point to local FS as weel
# export MODEL_PATH="/location/to/model"
export MODEL_PATH="meta-llama/Llama-4-Maverick-17B-128E-Instruct"

# The name the model will be served/queried under, matching what's
# returned by the /v1/models endpoint.
#
# By default this is inferred from MODEL_PATH, but when using locally downloaded
# model weights, it can be nice to have explicit control over the name.
export SERVED_MODEL_NAME="meta-llama/Llama-4-Maverick-17B-128E-Instruct"

export MODALITY=${MODALITY:-"multimodal"}
```

## Disaggregated Mode

Assuming you have at least 4 4xGB200 nodes allocated (2 for prefill, 2 for decode)
following the setup above, follow these steps below to launch a **disaggregated**
deployment across 4 nodes:

> [!Tip]
> Make sure you have a fresh environment and don't still have the aggregated
> example above still deployed on the same set of nodes.

```bash
# Defaults set in srun_disaggregated.sh, but can customize here.
# export PREFILL_ENGINE_CONFIG="/mnt/recipes/llama4/trtllm/multimodal/prefill.yaml"
# export DECODE_ENGINE_CONFIG="/mnt/recipes/llama4/trtllm/multimodal/decode.yaml"

# Customize NUM_PREFILL_NODES to match the desired parallelism in PREFILL_ENGINE_CONFIG
# Customize NUM_DECODE_NODES to match the desired parallelism in DECODE_ENGINE_CONFIG
# The products of NUM_PREFILL_NODES*NUM_GPUS_PER_NODE and
# NUM_DECODE_NODES*NUM_GPUS_PER_NODE should match the respective number of
# GPUs necessary to satisfy the requested parallelism in each config.
# export NUM_PREFILL_NODES=2
# export NUM_DECODE_NODES=2

# GB200 nodes have 4 gpus per node, but for other types of nodes you can configure this.
# export NUM_GPUS_PER_NODE=4

# Launches:
# - frontend + etcd/nats on current (head) node.
# - one large prefill trtllm worker across multiple nodes via MPI tasks
# - one large decode trtllm worker across multiple nodes via MPI tasks
./srun_disaggregated.sh
```

## Understanding the Output

1. The `srun_disaggregated.sh` launches three srun jobs instead of two. One for frontend, one for prefill   worker, and one for decode worker.

2. The OpenAI frontend will listen for and dynamically discover workers as
   they register themselves with Dynamo's distributed runtime:
   ```
   0: 2025-06-13T02:36:48.160Z  INFO dynamo_run::input::http: Watching for remote model at models
   0: 2025-06-13T02:36:48.161Z  INFO dynamo_llm::http::service::service_v2: Starting HTTP service on: 0.0.0.0:8000 address="0.0.0.0:8000"
   ```
3. The TRTLLM worker will consist of N (N=8 for TP8) MPI ranks, 1 rank on each
   GPU on each node, which will each output their progress while loading the model.
   You can see each rank's output prefixed with the rank at the start of each log line
   until the model succesfully finishes loading:
    ```
     7: rank7 run mgmn worker node with mpi_world_size: 8 ...
    ```
4. After the model fully finishes loading on all ranks, the worker will register itself,
   and the OpenAI frontend will detect it, signaled by this output:
    ```
    0: 2025-06-13T02:46:35.040Z  INFO dynamo_llm::discovery::watcher: added model model_name="meta-llama/Llama-4-Maverick-17B-128E-Instruct"
    ```
5. At this point, with the worker fully initialized and detected by the frontend,
   it is now ready for inference.

## Example Request

To verify the deployed model is working, send a `curl` request:
```bash
curl localhost:8000/v1/chat/completions -H "Content-Type: application/json" -d '{
    "model": "meta-llama/Llama-4-Maverick-17B-128E-Instruct",
    "messages": [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "Describe the image"
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/inpaint.png"
                    }
                }
            ]
        }
    ],
    "stream": false,
    "max_tokens": 160
}'
```

## Cleanup

To cleanup background `srun` processes launched by `srun_aggregated.sh` or
`srun_disaggregated.sh`, you can run:
```bash
pkill srun
```

## Known Issues

- Loading `meta-llama/Llama-4-Maverick-17B-128E-Instruct` with 8 nodes of H100 with TP=16 is not posssible due to Llama4 Maverick has a config `"num_attention_heads": 40` , trtllm engine asserts on assert `self.num_heads % tp_size == 0`  causing the engine to crash on model loading.