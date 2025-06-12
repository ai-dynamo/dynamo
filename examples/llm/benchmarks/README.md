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

> [!Important]
> At least one 8xH100-80GB node is required for the following instructions.

 1. Build benchmarking image

    ```bash
    ./container/build.sh
    ```

 2. Download model

    ```bash
    huggingface-cli download neuralmagic/DeepSeek-R1-Distill-Llama-70B-FP8-dynamic
    ```

 3. Start NATS and ETCD

    ```bash
    docker compose -f deploy/docker_compose.yml up -d
    ```

> [!NOTE]
> This guide was tested on node(s) with the following hardware configuration:
>
> * **GPUs**:
>   8xH100-80GB-HBM3 (GPU Memory Bandwidth 3.2 TBs)
>
> * **CPU**:
>   2 x Intel Sapphire Rapids, Intel(R) Xeon(R) Platinum 8480CL E5, 112 cores (56 cores per CPU), 2.00 GHz (Base), 3.8 Ghz (Max boost), PCIe Gen5
>
> * **NVLink**:
>   NVLink 4th Generation, 900 GB/s (GPU to GPU NVLink bidirectional bandwidth), 18 Links per GPU
>
> * **InfiniBand**:
>   8x400Gbit/s (Compute Links), 2x400Gbit/s (Storage Links)
>
> Benchmarking with a different hardware configuration may yield suboptimal results.


## Disaggregated Single Node Benchmarking

> [!Important]
> One 8xH100-80GB node is required for the following instructions.

In the following setup we compare Dynamo disaggregated vLLM performance to
[native vLLM Aggregated Baseline](#vllm-aggregated-baseline-benchmarking) on a single node. These were chosen to optimize
for Output Token Throughput (per sec) when both are performing under similar Inter Token Latency (ms).
For more details on your use case please see the [Performance Tuning Guide](/docs/guides/disagg_perf_tuning.md).

In this setup, we will be using 4 prefill workers and 1 decode worker.
Each prefill worker will use tensor parallel 1 and the decode worker will use tensor parallel 4.

With the Dynamo repository, benchmarking image and model available, and **NATS and ETCD started**, perform the following steps:

 1. Run benchmarking container

    ```bash
    ./container/run.sh --mount-workspace
    ```

    > [!Tip]
    > The huggingface home source mount can be changed by setting `--hf-cache ~/.cache/huggingface`.

 2. Start disaggregated services

    ```bash
    cd /workspace/examples/llm
    dynamo serve benchmarks.disagg:Frontend -f benchmarks/disagg.yaml 1> disagg.log 2>&1 &
    ```

    > [!Tip]
    > Check the `disagg.log` to make sure the service is fully started before collecting performance numbers.

 3. Collect the performance numbers as shown on the [Collecting Performance Numbers](#collecting-performance-numbers) section below.


## Disaggregated Multinode Benchmarking

> [!Important]
> Two 8xH100-80GB nodes are required the following instructions.

In the following steps we compare Dynamo disaggregated vLLM performance to
[native vLLM Aggregated Baseline](#vllm-aggregated-baseline-benchmarking) on two nodes. These were chosen to optimize
for Output Token Throughput (per sec) when both are performing under similar Inter Token Latency (ms).
For more details on your use case please see the [Performance Tuning Guide](/docs/guides/disagg_perf_tuning.md).

In this setup, we will be using 8 prefill workers and 1 decode worker.
Each prefill worker will use tensor parallel 1 and the decode worker will use tensor parallel 8.

With the Dynamo repository, benchmarking image and model available, and **NATS and ETCD started on node 0**, perform the following steps:

 1. Run benchmarking container (nodes 0 & 1)

    ```bash
    ./container/run.sh --mount-workspace
    ```

    > [!Tip]
    > The huggingface home source mount can be changed by setting `--hf-cache ~/.cache/huggingface`.

 2. Config NATS and ETCD (node 1)

    ```bash
    export NATS_SERVER="nats://<node_0_ip_addr>"
    export ETCD_ENDPOINTS="<node_0_ip_addr>:2379"
    ```

    > [!Important]
    > Node 1 must be able to reach Node 0 over the network for the above services.

 3. Start workers (node 0)

    ```bash
    cd /workspace/examples/llm
    dynamo serve benchmarks.disagg_multinode:Frontend -f benchmarks/disagg_multinode.yaml 1> disagg_multinode.log 2>&1 &
    ```

    > [!Tip]
    > Check the `disagg_multinode.log` to make sure the service is fully started before collecting performance numbers.

 4. Start workers (node 1)

    ```bash
    cd /workspace/examples/llm
    dynamo serve components.prefill_worker:PrefillWorker -f benchmarks/disagg_multinode.yaml 1> prefill_multinode.log 2>&1 &
    ```

    > [!Tip]
    > Check the `prefill_multinode.log` to make sure the service is fully started before collecting performance numbers.

 5. Collect the performance numbers as shown on the [Collecting Performance Numbers](#collecting-performance-numbers) section above.


## vLLM Aggregated Baseline Benchmarking

> [!Important]
> One (or two) 8xH100-80GB nodes are required the following instructions.

With the Dynamo repository and the benchmarking image available, perform the following steps:

 1. Run benchmarking container

    ```bash
    ./container/run.sh --mount-workspace
    ```

    > [!Tip]
    > The Hugging Face home source mount can be changed by setting `--hf-cache ~/.cache/huggingface`.

 2. Start vLLM serve

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

    > [!Tip]
    > Check the `vllm_0.log` and `vllm_1.log` to make sure the service is fully started before collecting performance numbers.
    >
    > If benchmarking with two or more nodes, `--tensor-parallel-size 8` should be used and only run one `vllm serve` instance per node.

 3. Use NGINX as load balancer

    ```bash
    apt update && apt install -y nginx
    cp /workspace/benchmarks/llm/nginx.conf /etc/nginx/nginx.conf
    service nginx restart
    ```

    > [!Note]
    > If benchmarking over 2 nodes, the `upstream` configuration will need to be updated to link to the `vllm serve` on the second node.

 4. Collect the performance numbers as shown on the [Collecting Performance Numbers](#collecting-performance-numbers) section below.


## Collecting Performance Numbers

Run the benchmarking script

```bash
bash -x /workspace/benchmarks/llm/perf.sh
```

> [!Tip]
> See [GenAI-Perf tutorial](https://github.com/triton-inference-server/perf_analyzer/blob/main/genai-perf/docs/tutorial.md)
> @ [GitHub](https://github.com/triton-inference-server/perf_analyzer) for additional information about how to run GenAI-Perf
> and how to interpret results.


## Supporting Additional Models

The instructions above can be used for nearly any model desired.
More complex setup instructions might be required for certain models.
The above instruction regarding ETCD, NATS, nginx, dynamo-serve, and GenAI-Perf still apply and can be reused.
The specifics of deploying with different hardware, in a unique environment, or using another model framework can be adapted using the links below.

Regardless of the deployment mechanism, the GenAI-Perf tool will report the same metrics and measurements so long as an accessible endpoint is available for it to interact with. Use the provided [perf.sh](../../../benchmarks/llm/perf.sh) script to automate the measurement of model throughput and latency against multiple request concurrences.

### Deployment Examples

- [Dynamo Multinode Deployments](../../../docs/examples/multinode.md)
- [Dynamo TensorRT LLM Deployments](../../../docs/examples/trtllm.md)
    - [Aggregated Deployment of Very Large Models](../../../docs/examples/multinode.md#aggregated-deployment)
- [Dynamo vLLM Deployments](../../../docs/examples/llm_deployment.md)


## Metrics and Visualization

For instructions on how to acquire per worker metrics and visualize them using Grafana,
please see the provided [Visualization with Prometheus and Grafana](../../../deploy/metrics/README.md).
