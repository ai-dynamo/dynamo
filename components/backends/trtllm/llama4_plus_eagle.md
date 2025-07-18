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

# Llama 4 Maverick Instruct with Eagle Speculative Decoding on SLURM

This guide demonstrates how to deploy Llama 4 Maverick Instruct with Eagle Speculative Decoding on GB200x4 nodes. We will be following the [multi-node deployment instructions](./multinode/multinode-examples.md) to set up the environment for the following scenarios:

- **Aggregated Serving:**
  Deploy the entire Llama 4 model on a single GB200x4 node for end-to-end serving.

- **Disaggregated Serving:**
  Distribute the workload across two GB200x4 nodes:
    - One node runs the decode worker.
    - The other node runs the prefill worker.

For advanced control over how requests are routed between prefill and decode workers in disaggregated mode, refer to the [Disaggregation Strategy](./README.md#disaggregation-strategy) section.

## Notes
* To run Eagle Speculative Decoding with Llama 4, ensure the container meets the following criteria:
  * Built with a version of TensorRT-LLM based on the 0.21 release [Link](https://github.com/NVIDIA/TensorRT-LLM/tree/release/0.21)
* If you need to download model weights off huggingface, make sure you run the command `huggingface-cli login` and have access to the necessary gated models.


## Setup

Assuming you have already allocated your nodes via `salloc`, and are
inside an interactive shell on one of the allocated nodes, set the
following environment variables based:

```bash
cd $DYNAMO_HOME/components/backends/trtllm

export IMAGE="<dynamo_trtllm_image>"
# export MOUNTS="${PWD}/:/mnt,/lustre:/lustre"
export MOUNTS="${PWD}/:/mnt"
export MODEL_PATH="nvidia/Llama-4-Maverick-17B-128E-Instruct-FP8"
export SERVED_MODEL_NAME="nvidia/Llama-4-Maverick-17B-128E-Instruct-FP8"
```

See [this](./multinode/multinode-examples.md#setup) section from multinode guide to learn more about the above options.


## Aggregated Serving
```bash
export NUM_NODES=1
export ENGINE_CONFIG="/mnt/engine_configs/llama4/eagle/eagle_agg.yaml"
./multinode/srun_aggregated.sh
```
* Known Issue: In Aggregated Serving, setting `max_num_tokens` to higher values (e.g. `max_num_tokens: 8448`) can lead to Out of Memory (OOM) errors. This is being investigated by the TRTLLM team.

## Disaggregated Serving

```bash
export NUM_PREFILL_NODES=1
export PREFILL_ENGINE_CONFIG="/mnt/engine_configs/llama4/eagle/eagle_prefill.yaml"
export NUM_DECODE_NODES=1
export DECODE_ENGINE_CONFIG="/mnt/engine_configs/llama4/eagle/eagle_decode.yaml"
./multinode/srun_disaggregated.sh
```
* Known Issue: In Aggregated Serving, setting `max_num_tokens` to higher values (e.g. `max_num_tokens: 8448`) can lead to Out of Memory (OOM) errors. This is being investigated by the TRTLLM team.


## Example Request

See [here](./multinode/multinode-examples.md#example-request) to learn how to send a request to the deployment.
