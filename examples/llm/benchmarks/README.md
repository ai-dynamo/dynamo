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

# LLM Deployment Benchmarking Guide

This guide provides detailed steps on benchmarking Large Language Models (LLMs) in single and multi-node configurations.

> [!NOTE]
> We recommend trying out the [LLM Deployment Examples](./README.md) before benchmarking.

## Prerequisites

H100 80GB x8 node(s) are required for benchmarking.

1\. Build benchmarking image
```bash
./container/build.sh
```

2\. Download model
```bash
huggingface-cli download neuralmagic/DeepSeek-R1-Distill-Llama-70B-FP8-dynamic
```

3\. Start NATS and ETCD
```bash
docker compose -f deploy/docker_compose.yml up -d
```

## Disaggregated Single Node Benchmarking

One H100 80GB x8 node is required for this setup.

With the Dynamo repository, benchmarking image and model available, and **NATS and ETCD started**, perform the following steps:

1\. Run benchmarking container
```bash
./container/run.sh -it \
  -v <huggingface_hub>:/root/.cache/huggingface/hub \
  -v <dynamo_repo>:/workspace
```

2\. Start disaggregated services
```bash
cd /workspace/examples/llm
dynamo serve benchmarks.disagg:Frontend -f benchmarks/disagg.yaml 1> disagg.log 2>&1 &
```
Note: Check the `disagg.log` to make sure the service is fully started before collecting performance numbers.

Key settings:
* **Model**: DeepSeek R1 Distill Llama 70B FP8 (or similar in size).
* **FP8 Model**: Reduces memory usage for both the model and KV cache, easing remote transfer.
* **Block Size 128**: Batches token processing for more efficient chunk transfers to GPUs.
* **PrefillWorker**: Four separate processes (each with 1 GPU) handle the initial prefill (context embedding) phase.
* **VllmWorker**: One process (using 4 GPUs) generates output tokens (the "decode" phase).
* **kv-transfer-config**: Uses the DynamoNixlConnector for remote KV cache transfer.
* **remote-prefill**: KV cache can be passed from prefill to decode workers.
* **No prefix cache**: All requests are unique and cache should not be used.

Collect the performance numbers as shown on the [Collecting Performance Numbers](#collecting-performance-numbers) section below.

## Disaggregated Multi Node Benchmarking

Coming soon...

## vLLM Aggregated Baseline Benchmarking

One (or Two) H100 80GB x8 node is required for this setup.

With the Dynamo repository and the benchmarking image available, perform the following steps:

1\. Run benchmarking container
```bash
./container/run.sh -it \
  -v <huggingface_hub>:/root/.cache/huggingface/hub \
  -v <dynamo_repo>:/workspace
```

2\. Start vLLM serve
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 vllm serve neuralmagic/DeepSeek-R1-Distill-Llama-70B-FP8-dynamic \
  --block-size 128 \
  --max-model-len 3500 \
  --max-num-batched-tokens 3500 \
  --tensor-parallel-size 4 \
  --gpu-memory-utilization 0.95 \
  --disable-log-requests \
  --port 8001 1> vllm_0.log 2>&1 &
CUDA_VISIBLE_DEVICES=4,5,6,7 vllm serve neuralmagic/DeepSeek-R1-Distill-Llama-70B-FP8-dynamic \
  --block-size 128 \
  --max-model-len 3500 \
  --max-num-batched-tokens 3500 \
  --tensor-parallel-size 4 \
  --gpu-memory-utilization 0.95 \
  --disable-log-requests \
  --port 8002 1> vllm_1.log 2>&1 &
```
Notes:
* Check the `vllm_0.log` and `vllm_1.log` to make sure the service is fully started before collecting performance numbers.
* For multi-node benchmarking, use TP=8 and start one instance of `vllm serve` on both node 0 and 1.
* The `vllm serve` configuration should closely match the corresponding disaggregated benchmarking configuration.

3\. Use NGINX as load balancer
```bash
apt update && apt install -y nginx
cp /workspace/examples/llm/benchmarks/nginx.conf /etc/nginx/nginx.conf
service nginx restart
```
Key NGINX configurations:
* **least_conn**: To load balance across vLLM servers.
* **server**: Select all the upstream vLLM servers to load balance across, including those at a different node if applicable.

Collect the performance numbers as shown on the [Collecting Performance Numbers](#collecting-performance-numbers) section below.

## Collecting Performance Numbers

Run the benchmarking script
```bash
bash -x /workspace/examples/llm/benchmarks/perf.sh
```
Key settings:
* **streaming**: Enables token streaming for more realistic performance measurements.
* **concurrency**: Number of simultaneous requests.
* **isl**: Input sequence length at 3000 tokens.
* **osl**: Output sequence length requested at 150 tokens.

## Interpreting Results

Coming soon...
