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

## Eagle3-one-model
* Eagle3-one-model (`eagle3_one_model=True`) config is added in `engine_configs/llama4/eagle_one_model`. Build dynamo with the latest commit `66f299a` in TRTLLM 1.0.0.rc2 [Link](https://github.com/NVIDIA/TensorRT-LLM/commits/v1.0.0rc2/).
* The configs in `engine_configs/llama4/eagle_one_model` are tested with 8xH100 cluster. Be sure to change the `NUM_GPUS_PER_NODE` accordingly or change TP/EP size in config. 1 8xH100 node for aggregated .yml file, 2 8xH100 for prefill/decode .yml file.
* The current `./multinode/start_frontend_services.sh` may got ran `NUM_GPUS_PER_NODE` times depending on how srun/mpi is launched, beware that the frontend service only needs to be ran once.
* Eagle3-one-model appends the eagle3 layer at the end of the TRTLLM engine, instead of sending base/draft requests between 2 engines. Visit TRTLLM for more information.


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

```
curl localhost:8000/v1/chat/completions -H "Content-Type: application/json" -d '{
        "model": "nvidia/Llama-4-Maverick-17B-128E-Instruct-FP8",
        "messages": [{"role": "user", "content": "Why is NVIDIA a great company?"}],
        "max_tokens": 1024
    }' -w "\n"


# output:
{"id":"cmpl-3e87ea5c-010e-4dd2-bcc4-3298ebd845a8","choices":[{"text":"NVIDIA is considered a great company for several reasons:\n\n1. **Technological Innovation**: NVIDIA is a leader in the field of graphics processing units (GPUs) and has been at the forefront of technological innovation.
...
and the broader tech industry.\n\nThese factors combined have contributed to NVIDIA's status as a great company in the technology sector.","index":0,"logprobs":null,"finish_reason":"stop"}],"created":1753329671,"model":"nvidia/Llama-4-Maverick-17B-128E-Instruct-FP8","system_fingerprint":null,"object":"text_completion","usage":{"prompt_tokens":16,"completion_tokens":562,"total_tokens":578,"prompt_tokens_details":null,"completion_tokens_details":null}}
```
