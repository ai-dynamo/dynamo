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
* Make sure the (`eagle3_one_model: true`) is set in the LLM API config inside the `recipes/llama4/trtllm/eagle` folder.

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
export ENGINE_CONFIG="/mnt/recipes/llama4/trtllm/eagle/eagle_agg.yml"
./multinode/srun_aggregated.sh
```

## Disaggregated Serving

```bash
export NUM_PREFILL_NODES=1
export PREFILL_ENGINE_CONFIG="/mnt/recipes/llama4/trtllm/eagle/eagle_prefill.yaml"
export NUM_DECODE_NODES=1
export DECODE_ENGINE_CONFIG="/mnt/recipes/llama4/trtllm/eagle/eagle_decode.yaml"
./multinode/srun_disaggregated.sh
```

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
