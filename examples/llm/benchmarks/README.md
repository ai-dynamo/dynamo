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

 3. Collect the performance numbers:

 ```bash
 bash -x /workspace/benchmarks/llm/perf.sh --mode disaggregated --deployment-kind dynamo_vllm --prefill-tensor-parallelism 1 --prefill-data-parallelism 4 --decode-tensor-parallelism 4 --decode-data-parallelism 1
 ```

 > [!Important]
 > We should be careful in specifying these options in `perf.sh` script. They should closely reflect the deployment config that is being benchmarked. See `perf.sh --help` to learn more about these option. In the above command, we described that our deployment is using disaggregated serving in dynamo with vLLM backend. We have also accurately described that we have 4 prefill workers with TP=1 and 1 decode worker with TP=4

For more information see [Collecting Performance Numbers](#collecting-performance-numbers) section below.

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

 5. Collect the performance numbers:

 ```bash
 bash -x /workspace/benchmarks/llm/perf.sh --mode disaggregated --deployment-kind dynamo_vllm --prefill-tensor-parallelism 1 --prefill-data-parallelism 8 --decode-tensor-parallelism 8 --decode-data-parallelism 1
 ```

 > [!Important]
 > We should be careful in specifying these options in `perf.sh` script. They should closely reflect the deployment config that is being benchmarked. See `perf.sh --help` to learn more about these option. In the above command, we described that our deployment is using disaggregated serving in dynamo with vLLM backend. We have also accurately described that we have 8 prefill workers with TP=1 and 1 decode worker with TP=8

For more information see [Collecting Performance Numbers](#collecting-performance-numbers) section below.


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

 4. Collect the performance numbers:

Single-Node

 ```bash
 bash -x /workspace/benchmarks/llm/perf.sh --mode aggregated --deployment-kind vllm_serve --tensor-parallelism 4 --data-parallelism 2
 ```

 Two Nodes

 ```bash
 bash -x /workspace/benchmarks/llm/perf.sh --mode aggregated --deployment-kind vllm_serve --tensor-parallelism 8 --data-parallelism 2
 ```

 We could also run the benchmarking script and specify the model, input sequence length, output sequence length, and concurrency levels to target for benchmarking:

 ```bash
 bash -x /workspace/benchmarks/llm/perf.sh \
  --mode aggregated \
  --deployment-kind vllm_serve \
  --tensor-parallelism 1 \
  --data-parallelism 1 \
  --model neuralmagic/DeepSeek-R1-Distill-Llama-70B-FP8-dynamic \
  --input-sequence-length 3000 \
  --output-sequence-length 150 \
  --url http://localhost:8000 \
  --concurrency 1,2,4,8,16,32,64,128,256

  # The `--concurrency` option accepts either a single value (e.g., 64) or a comma-separated list (e.g., 1,2,4,8) to specify multiple concurrency levels for benchmarking.
 ```

 > [!Important]
 > We should be careful in specifying these options in `perf.sh` script. They should closely reflect the deployment config that is being benchmarked. See `perf.sh --help` to learn more about these option. In the above command, we described that our deployment is using aggregated serving in `vllm serve`. We have also accurately described that we have 2 workers with TP=4(or TP=8 for two nodes).

For more information see [Collecting Performance Numbers](#collecting-performance-numbers) section below.

## Collecting Performance Numbers

Currently, there is no consistent way of obtaining the configuration of deployment service. Hence, we need to provide this information to the script in form of command line arguments. The benchmarking script `/workspace/benchmarks/llm/perf.sh` uses GenAI-Perf tool to collect the performance numbers at various different request concurrencies. The perf.sh script can be run multiple times to collect numbers for various different deployments. Each script execution will create a new artifacts directory in `artifacts_root` and dump these numbers in it. See [Plotting Pareto Graphs](#plotting-pareto-graphs) to learn how to convert the data from this `artifacts_root` to generate pareto graphs for the performance.

Note: As each `perf.sh` adds a new artifacts directory in the `artifacts_root` always, proper care should be taken that we are starting experiment with clean `artifacts_root` so we include only results from runs that we want to compare.

> [!Tip]
> See [GenAI-Perf tutorial](https://github.com/triton-inference-server/perf_analyzer/blob/main/genai-perf/docs/tutorial.md)
> @ [GitHub](https://github.com/triton-inference-server/perf_analyzer) for additional information about how to run GenAI-Perf
> and how to interpret results.


## Interpreting Results

### Plotting Pareto Graphs

The `artifacts` directory generated by GenAI-Perf contains the raw performance number from the benchmarking.

Using the benchmarking image, install the dependencies for plotting Pareto graph
```bash
pip3 install matplotlib seaborn
```
At the directory where the artifacts are located, plot the Pareto graph

Single-Node:

```bash
python3 /workspace/benchmarks/llm/plot_pareto.py --artifacts-root-dir artifacts_root
```

Two Nodes:

```bash
python3 /workspace/benchmarks/llm/plot_pareto.py --artifacts-root-dir artifacts_root --title "Two Nodes"
```
The graph will be saved to the current directory and named `pareto_plot.png`.

### Interpreting Pareto Graphs

The question we want to answer in this comparison is how much Output Token Throughput can be improved by switching from
aggregated to disaggregated serving when both are performing under similar Inter Token Latency.

For each concurrency benchmarked, it produces a latency and throughput value pair. The x-axis on the Pareto graph is
latency (tokens/s/user), which the latency is lower if the value is higher. The y-axis on the Pareto graph is throughput
(tokens/s/gpu). The latency and throughput value pair forms a dot on the Pareto graph. A line (Pareto Frontier) is
formed when the dots from different concurrency values are plotted on the graph.

With the Pareto Frontiers of the baseline and the disaggregated results plotted on the graph, we can look for the
greatest increase in throughput (along the y-axis) between the baseline and the disaggregated result Pareto Frontier,
over different latencies (along the x-axis).

For example, at 45 tokens/s/user, the increase in tokens/s/gpu is `145 - 80 = 65`, from the orange baseline to the
blue disaggregated line, so the improvement is around 1.44x speed up:
![Example Pareto Plot](./example_plots/single_node_pareto_plot.png)
Note: The above example was collected over a single benchmarking run, the actual number may vary between runs, configurations and hardware.

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


## Monitor Benchmark Startup Status

When running dynamo deployment, you may have multiple instances of the same worker kind for a particular benchmark run.
The deployment can process the workflow as long as at least one worker is ready, in the case where the benchmark is run
as soon as dynamo is responsive to inference request, which may result in inaccurate benchmark result at the beginning of
the benchmark. In such a case, you may additionally deploy benchmark watcher to provide signal on whether the full deployment
is ready. For instance, if you expect the total number of prefill and decode workers to be 10, you can run the below to start
the watcher, which will exit if the total number is less than 10 after timeout. In addition to that, the watcher will create
a HTTP server on port 7001 by default, which you can use to send GET request for readiness to build external benchmarking workflow.

```bash
# start your benchmark deployment
...

# start monitor separately, or it can be part of the deployment above
dynamo serve --service-name Watcher benchmark_watcher:Watcher --Watcher.total-workers=10 --Watcher.timeout=10

# Send curl request to check liveness
curl localhost:7001
127.0.0.1 - - [12/Jun/2025 23:31:52] "GET / HTTP/1.1" 400 -
...
curl localhost:7001
127.0.0.1 - - [12/Jun/2025 23:32:46] "GET / HTTP/1.1" 200 -
```

## Utility for Setting Up Environment

### vLLM
- `vllm_multinode_setup.sh` is a helper script to configure the node for dynamo deployment for
vLLM. Depending on whether environment variable `HEAD_NODE_IP` and `RAY_LEADER_NODE_IP` are set
when the script is invoked, it will:
  - start nats server and etcd on the current node if `HEAD_NODE_IP` is not set, otherwise
  set the environment variables as expected by dynamo.
  - run Ray and connect to the Ray cluster started by `RAY_LEADER_NODE_IP`, otherwise start
  the Ray cluster with current node as the head node.
  - print the command with `HEAD_NODE_IP` and `RAY_LEADER_NODE_IP` set, which can be used in
  another node to setup connectivity with the current node.

  ```bash
  # On node 0
  source vllm_multinode_setup.sh
  ... # starting nats server, etcd and ray cluster

  # script print command
  HEAD_NODE_IP=NODE_0_IP RAY_LEADER_NODE_IP=NODE_0_IP source vllm_multinode_setup.sh

  # On node 1
  HEAD_NODE_IP=NODE_0_IP RAY_LEADER_NODE_IP=NODE_0_IP source vllm_multinode_setup.sh
  ... # connecting to Ray cluster
  ```

## Metrics and Visualization

For instructions on how to acquire per worker metrics and visualize them using Grafana,
please see the provided [Visualization with Prometheus and Grafana](../../../deploy/metrics/README.md).

## Troubleshooting

When benchmarking disaggregation performance, there can be cases where the latency and
throughput number don't match the expectation within some margin. Below is a list of scenarios
that have been encountered, and details on observations and resolutions.

### Interconnect Configuration

Even if the nodes have faster interconnect hardware available, there can be misconfiguration such that
the fastest route may not be selected by NIXL ([example regression](https://github.com/ai-dynamo/dynamo/pull/1314)). NIXL simplifies the interconnect but also hides
selection detail. Therefore this can be the cause if you observe abnormal TTFT increase when
splitting prefill workers and decode workers to different nodes. For example, we have seen instances of ~2 second overhead added to TTFT when TCP is selected over RDMA for KV Cache transfer due to a misconfigured environment.

Currently NIXL doesn't provide utility for reporting which transport is selected. Therefore
you will need to verify if that is the cause by using backend specific debug options.
In the case of UCX backend, you can use `ucx_info -d` to check if the desired interconnect
devices are being recognized. At runtime, `UCX_LOG_LEVEL=debug` and `UCX_PROTO_INFO=y`
can be set as environment variables to provide detailed logs on UCX activities. This will
reveal whether the desired transport is being used.

### The Full Deployment is Configured Correctly

As benchmarking often focuses on configurations where multiple workers are being used,
one may mistakenly consider a deployment ready for benchmarking while there are only a
subset of workers taking requests. For example, in the aggregated baseline benchmarking,
a user can miss updating the ip address to the other node in upstream section of `nginx.conf`.
This could lead to only one of the nodes serving requests. In such a case,
the benchmark can still run to completion, but the result will not reflect the deployment
capacity, because not all the compute resources are being utilized.

Therefore, it is important to verify that the requests can be routed to all workers before
performing the benchmark:
- **Framework-only benchmark** The simplest way is to send sample requests and check
the logs of all workers. Each framework may provide utilities for readiness checks, so please
refer to the framework's documentation for those details.
- **Dynamo based benchmark** Once you start the deployment, you can follow
the instructions in [monitor benchmark startup status](#Monitor-Benchmark-Startup-Status),
which will periodically poll the workers exposed to specific endpoints
and return HTTP 200 code when the expected number of workers are met.
