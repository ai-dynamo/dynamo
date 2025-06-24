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

# vLLM Deployment Examples

This directory contains examples for deploying vLLM models in both aggregated and disaggregated configurations.

## Use the Latest Release

We recommend using the latest stable release of dynamo to avoid breaking changes:

[![GitHub Release](https://img.shields.io/github/v/release/ai-dynamo/dynamo)](https://github.com/ai-dynamo/dynamo/releases/latest)

You can find the latest release [here](https://github.com/ai-dynamo/dynamo/releases/latest) and check out the corresponding branch with:

```bash
git checkout $(git describe --tags $(git rev-list --tags --max-count=1))
```

## Prerequisites

1. Run Dynamo vLLM V1 docker:
```bash
./container/build.sh --framework VLLM_V1 --target dev
./container/run.sh --framework VLLM_V1 --target dev -it
```

Or install vLLM manually:

```
uv pip install vllm==0.9.1
```

2. Start required services:
```bash
docker compose -f deploy/metrics/docker-compose.yml up -d
```

## Running the Server

### Aggregated Deployment
```bash
cd examples/vllm_v1
dynamo serve graphs.agg:Frontend -f configs/agg.yaml
```

### Disaggregated Deployment
```bash
cd examples/vllm_v1
dynamo serve graphs.disagg:Frontend -f configs/disagg.yaml
```

## Testing the API

Send a test request using curl:
```bash
curl localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
    "prompt": "In the heart of Eldoria...",
    "stream": false,
    "max_tokens": 30
  }'
```

For more detailed explenations, refer to the main [LLM examples README](../llm/README.md).



## Deepseek R1

To run DSR1 model please first follow the Ray setup from the [multinode documentation](../../docs/examples/multinode.md).

### Aggregated Deployment

```bash
cd examples/vllm_v1
dynamo serve graphs.agg:Frontend -f configs/deepseek_r1/agg.yaml
```


### Disaggregated Deployment

To create frontend with a single decode worker:
```bash
cd examples/vllm_v1
dynamo serve graphs.agg:Frontend -f configs/deepseek_r1/disagg.yaml
```

To create a single decode worker:
```bash
cd examples/vllm_v1
dynamo serve components.worker:VllmDecodeWorker -f configs/deepseek_r1/disagg.yaml
```

To create a single prefill worker:
```bash
cd examples/vllm_v1
dynamo serve components.worker:VllmPrefillWorker -f configs/deepseek_r1/disagg.yaml
```

### Data Parallelism Deployment

Additional configuration steps will be required for enabling DP for DSR1 model,
as it typically requires setting up the DP groups across nodes.
`configs/deepseek_r1/agg_dp.yaml` and `configs/deepseek_r1/disagg_dp.yaml` will be
the replacement for aggregated deployment and disaggregated deployment.
The below demonstration will use deployment of a single worker as an example,
the reader should apply the same to any `dynamo serve` command that will create
a worker.

To create a single decode worker, take note of the IP address, referred as <head-ip> below, of the node, and:
```bash
cd examples/vllm_v1
dynamo serve components.worker:VllmDecodeWorker -f configs/deepseek_r1/disagg_dp.yaml --VllmDecodeWorker.data_parallel_address=<head-ip>
```

The above command will create 1 of the 2 DP groups and the worker will be consdiered
the head of the DP groups. Next we need to create a `VllmDpWorker` to create the rest of the DP groups, one for each group.

```bash
cd examples/vllm_v1
# 'data_parallel_start_rank' == `dp_group_index * data_parallel_size_local`
dynamo serve components.worker:VllmDpWorker -f configs/deepseek_r1/disagg_dp.yaml --VllmDpWorker.data_parallel_address=<head-ip> --VllmDpWorker.data_parallel_start_rank=8

# repeat above until all DP groups are created
```


### Wide EP

If running oustide of Dynamo vLLM V1 container please follow [vLLM guide](https://github.com/vllm-project/vllm/tree/main/tools/ep_kernels) to install EP kernels and install [DeepGEMM](https://github.com/deepseek-ai/DeepGEMM).

To run DSR1 with DEP16 (EP16 MoE and DP16 for other layers) with [DeepEP kernels](https://github.com/deepseek-ai/DeepEP) run on head node:

```
export VLLM_ALL2ALL_BACKEND="deepep_low_latency" # or "deepep_high_throughput"
export VLLM_USE_DEEP_GEMM=1
export GLOO_SOCKET_IFNAME=eth3 # or another non IB interface that you can find with `ifconfig -a`
cd examples/vllm_v1
dynamo serve components.worker:VllmDecodeWorker -f configs/deepseek_r1/disagg_dp.yaml --VllmDecodeWorker.data_parallel_address=<head-ip> --VllmDecodeWorker.enable_expert_parallel=true
```

on 2nd node:

```
export VLLM_ALL2ALL_BACKEND="deepep_low_latency" # or "deepep_high_throughput"
export VLLM_USE_DEEP_GEMM=1
export GLOO_SOCKET_IFNAME=eth3 # or another non IB interface that you can find with `ifconfig -a`
cd examples/vllm_v1
# 'data_parallel_start_rank' == `dp_group_index * data_parallel_size_local`
dynamo serve components.worker:VllmDpWorker -f configs/deepseek_r1/disagg_dp.yaml --VllmDpWorker.data_parallel_address=<head-ip> --VllmDpWorker.data_parallel_start_rank=8 --VllmDpWorker.enable_expert_parallel=true
```

## Testing

Send a test request using curl:
```bash
curl localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "deepseek-ai/DeepSeek-R1",
    "prompt": "In the heart of Eldoria...",
    "stream": false,
    "max_tokens": 30
  }'
```