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

This guide provides detailed steps on benchmarking Large Language Models (LLMs) using the `perf.sh` and `plot_pareto.py` scripts in single and multi-node configurations. These scripts use [AIPerf](https://github.com/triton-inference-server/perf_analyzer) to collect performance metrics and generate Pareto frontier visualizations.

## Overview

The benchmarking tools in this directory help you:

- **Benchmark LLM deployments** at various concurrency levels
- **Compare performance** between aggregated and disaggregated serving modes
- **Generate Pareto plots** to visualize throughput vs latency trade-offs
- **Evaluate different configurations** (tensor parallelism, data parallelism, etc.)

### Scripts

- **`perf.sh`**: Bash script that runs AIPerf benchmarks across multiple concurrency levels
- **`plot_pareto.py`**: Python script that generates Pareto efficiency graphs from benchmark results
- **`nginx.conf`**: NGINX configuration template for load balancing (used for baseline comparisons)

## Prerequisites

> [!Important]
> At least one 8xH100-80GB node is recommended for the following instructions. Different hardware configurations may yield different results.

1. **Dynamo installed** - Follow the [installation guide](../../README.md#installation) to set up Dynamo
2. **Model downloaded** - Download the model you want to benchmark:
   ```bash
   huggingface-cli download Qwen/Qwen3-0.6B
   ```
3. **NATS and etcd running** - Start the required services:

   ```bash
   # Using docker-compose (recommended)
   docker compose -f deploy/docker-compose.yml up -d

   # Or manually:
   # Start etcd: ./etcd
   # Start NATS with JetStream: nats-server -js
   ```

4. **AIPerf installed** - Install AIPerf for benchmarking:
   ```bash
   pip install aiperf
   ```
5. **Python dependencies for plotting**:
   ```bash
   pip install matplotlib seaborn pandas numpy
   ```

> [!NOTE]
> This guide was tested on node(s) with the following hardware configuration:
>
> - **GPUs**: 8xH100-80GB-HBM3 (GPU Memory Bandwidth 3.2 TBs)
> - **CPU**: 2 x Intel Sapphire Rapids, Intel(R) Xeon(R) Platinum 8480CL E5, 112 cores (56 cores per CPU), 2.00 GHz (Base), 3.8 Ghz (Max boost), PCIe Gen5
> - **NVLink**: NVLink 4th Generation, 900 GB/s (GPU to GPU NVLink bidirectional bandwidth), 18 Links per GPU
> - **InfiniBand**: 8x400Gbit/s (Compute Links), 2x400Gbit/s (Storage Links)
>
> Benchmarking with a different hardware configuration may yield different results.

## Deployment Options

You can benchmark Dynamo deployments in two ways:

1. **Kubernetes Deployment** - Deploy using DynamoGraphDeployment (recommended for production)
2. **Local Deployment** - Run components directly on your machine (useful for development/testing)

Choose the method that best fits your use case. The benchmarking scripts work with either approach as long as you have an HTTP endpoint accessible at the specified URL.

## Benchmarking Disaggregated Single Node Deployment

> [!Important]
> One 8xH100-80GB node is required for the following instructions.

In this setup, we compare Dynamo disaggregated vLLM performance to native vLLM aggregated baseline on a single node. We use 4 prefill workers (TP=1 each) and 1 decode worker (TP=4).

### Option 1: Kubernetes Deployment

1. **Deploy DynamoGraphDeployment**:

   ```bash
   # Set namespace
   export NAMESPACE=dynamo-system

   # Create namespace if needed
   kubectl create namespace ${NAMESPACE}

   # Deploy disaggregated configuration
   kubectl apply -f examples/backends/vllm/deploy/disagg.yaml -n ${NAMESPACE}

   # Wait for deployment to be ready
   kubectl wait --for=condition=ready dynamographdeployment/vllm-disagg -n ${NAMESPACE} --timeout=10m
   ```

2. **Port-forward the frontend service**:

   ```bash
   kubectl port-forward -n ${NAMESPACE} svc/vllm-disagg-frontend 8000:8000 > /dev/null 2>&1 &
   ```

3. **Run the benchmark**:
   ```bash
   bash -x benchmarks/llm/perf.sh \
     --mode disaggregated \
     --deployment-kind dynamo_vllm \
     --prefill-tensor-parallelism 1 \
     --prefill-data-parallelism 4 \
     --decode-tensor-parallelism 4 \
     --decode-data-parallelism 1 \
     --url http://localhost:8000
   ```

### Option 2: Local Deployment

1. **Start the frontend**:

   ```bash
   python -m dynamo.frontend --http-port 8000 > frontend.log 2>&1 &
   ```

2. **Start prefill workers** (4 workers, each with TP=1):

   ```bash
   CUDA_VISIBLE_DEVICES=0 python -m dynamo.vllm \
     --model Qwen/Qwen3-0.6B \
     --is-prefill-worker > prefill_0.log 2>&1 &

   CUDA_VISIBLE_DEVICES=1 python -m dynamo.vllm \
     --model Qwen/Qwen3-0.6B \
     --is-prefill-worker > prefill_1.log 2>&1 &

   CUDA_VISIBLE_DEVICES=2 python -m dynamo.vllm \
     --model Qwen/Qwen3-0.6B \
     --is-prefill-worker > prefill_2.log 2>&1 &

   CUDA_VISIBLE_DEVICES=3 python -m dynamo.vllm \
     --model Qwen/Qwen3-0.6B \
     --is-prefill-worker > prefill_3.log 2>&1 &
   ```

3. **Start decode worker** (TP=4):

   ```bash
   CUDA_VISIBLE_DEVICES=4,5,6,7 python -m dynamo.vllm \
     --model Qwen/Qwen3-0.6B > decode.log 2>&1 &
   ```

4. **Wait for services to be ready** - Check the logs to ensure all services are fully started before benchmarking.

5. **Run the benchmark**:
   ```bash
   bash -x benchmarks/llm/perf.sh \
     --mode disaggregated \
     --deployment-kind dynamo_vllm \
     --prefill-tensor-parallelism 1 \
     --prefill-data-parallelism 4 \
     --decode-tensor-parallelism 4 \
     --decode-data-parallelism 1 \
     --url http://localhost:8000
   ```

> [!Important]
> The parallelism settings in `perf.sh` must accurately reflect your deployment configuration. In the above command, we specify:
>
> - 4 prefill workers with TP=1 each (prefill-data-parallelism=4, prefill-tensor-parallelism=1)
> - 1 decode worker with TP=4 (decode-data-parallelism=1, decode-tensor-parallelism=4)
>
> See `perf.sh --help` for more information about these options.

## Benchmarking Disaggregated Multinode Deployment

> [!Important]
> Two 8xH100-80GB nodes are required for the following instructions.

In this setup, we compare Dynamo disaggregated vLLM performance across two nodes. We use 8 prefill workers (TP=1 each) and 1 decode worker (TP=8).

### Setup

1. **On Node 0** - Start NATS and etcd:

   ```bash
   docker compose -f deploy/docker-compose.yml up -d
   # Or start manually: ./etcd and nats-server -js
   ```

2. **On Node 1** - Configure NATS and etcd endpoints:
   ```bash
   export NATS_SERVER="nats://<node_0_ip_addr>:4222"
   export ETCD_ENDPOINTS="<node_0_ip_addr>:2379"
   ```

> [!Important]
> Node 1 must be able to reach Node 0 over the network for the above services.

### Deployment

**Option 1: Kubernetes (Recommended)**

Deploy a multi-node DynamoGraphDeployment following the [multinode deployment guide](../../docs/kubernetes/deployment/multinode-deployment.md), then port-forward and benchmark as shown in the single-node example.

**Option 2: Local**

1. **On Node 0** - Start frontend and decode worker:

   ```bash
   # Start frontend
   python -m dynamo.frontend --http-port 8000 > frontend.log 2>&1 &

   # Start decode worker (TP=8, using all 8 GPUs)
   python -m dynamo.vllm \
     --model Qwen/Qwen3-0.6B > decode.log 2>&1 &
   ```

2. **On Node 1** - Start prefill workers:

   ```bash
   # Set environment variables for Node 0 connectivity
   export NATS_SERVER="nats://<node_0_ip_addr>:4222"
   export ETCD_ENDPOINTS="<node_0_ip_addr>:2379"

   # Start 8 prefill workers (one per GPU)
   for i in {0..7}; do
     CUDA_VISIBLE_DEVICES=$i python -m dynamo.vllm \
       --model Qwen/Qwen3-0.6B \
       --is-prefill-worker > prefill_${i}.log 2>&1 &
   done
   ```

3. **Run the benchmark** (from Node 0 or any machine with access to the frontend):
   ```bash
   bash -x benchmarks/llm/perf.sh \
     --mode disaggregated \
     --deployment-kind dynamo_vllm \
     --prefill-tensor-parallelism 1 \
     --prefill-data-parallelism 8 \
     --decode-tensor-parallelism 8 \
     --decode-data-parallelism 1 \
     --url http://<node_0_ip_addr>:8000
   ```

## Benchmarking vLLM Aggregated Baseline

> [!Important]
> One (or two) 8xH100-80GB nodes are required for the following instructions.

This section shows how to benchmark native vLLM aggregated serving for comparison with Dynamo disaggregated deployments.

### Single Node

1. **Start vLLM servers** (2 instances, each with TP=4):

   ```bash
   CUDA_VISIBLE_DEVICES=0,1,2,3 vllm serve Qwen/Qwen3-0.6B \
     --block-size 128 \
     --max-model-len 3500 \
     --max-num-batched-tokens 3500 \
     --tensor-parallel-size 4 \
     --gpu-memory-utilization 0.95 \
     --disable-log-requests \
     --port 8001 > vllm_0.log 2>&1 &

   CUDA_VISIBLE_DEVICES=4,5,6,7 vllm serve Qwen/Qwen3-0.6B \
     --block-size 128 \
     --max-model-len 3500 \
     --max-num-batched-tokens 3500 \
     --tensor-parallel-size 4 \
     --gpu-memory-utilization 0.95 \
     --disable-log-requests \
     --port 8002 > vllm_1.log 2>&1 &
   ```

2. **Set up NGINX load balancer**:

   ```bash
   sudo apt update && sudo apt install -y nginx
   sudo cp benchmarks/llm/nginx.conf /etc/nginx/nginx.conf
   sudo service nginx restart
   ```

3. **Run the benchmark**:
   ```bash
   bash -x benchmarks/llm/perf.sh \
     --mode aggregated \
     --deployment-kind vllm_serve \
     --tensor-parallelism 4 \
     --data-parallelism 2 \
     --url http://localhost:8000
   ```

### Two Nodes

For two nodes, use `--tensor-parallel-size 8` and run one `vllm serve` instance per node. Update the `nginx.conf` upstream configuration to include the second node's IP address.

## Using perf.sh

The `perf.sh` script runs AIPerf benchmarks across multiple concurrency levels and stores results in an artifacts directory.

### Basic Usage

```bash
bash benchmarks/llm/perf.sh [OPTIONS]
```

### Command-Line Options

```bash
Options:
  --tensor-parallelism, --tp <int>           Tensor parallelism (for aggregated mode)
  --data-parallelism, --dp <int>             Data parallelism (for aggregated mode)
  --prefill-tensor-parallelism, --prefill-tp <int>   Prefill tensor parallelism (for disaggregated mode)
  --prefill-data-parallelism, --prefill-dp <int>     Prefill data parallelism (for disaggregated mode)
  --decode-tensor-parallelism, --decode-tp <int>     Decode tensor parallelism (for disaggregated mode)
  --decode-data-parallelism, --decode-dp <int>       Decode data parallelism (for disaggregated mode)
  --model <model_id>                         Hugging Face model ID to benchmark (default: Qwen/Qwen3-0.6B)
  --input-sequence-length, --isl <int>       Input sequence length (default: 3000)
  --output-sequence-length, --osl <int>      Output sequence length (default: 150)
  --url <http://host:port>                   Target URL for inference requests (default: http://localhost:8000)
  --concurrency <list>                       Comma-separated concurrency levels (default: 1,2,4,8,16,32,64,128,256)
  --mode <aggregated|disaggregated>          Serving mode (default: aggregated)
  --artifacts-root-dir <path>                Root directory to store benchmark results (default: artifacts_root)
  --deployment-kind <type>                   Deployment tag used for pareto chart labels (default: dynamo)
  --help                                     Show this help message and exit
```

### Examples

**Custom model and sequence lengths**:

```bash
bash benchmarks/llm/perf.sh \
  --mode aggregated \
  --deployment-kind vllm_serve \
  --tensor-parallelism 4 \
  --data-parallelism 2 \
  --model Qwen/Qwen3-0.6B \
  --input-sequence-length 2000 \
  --output-sequence-length 256 \
  --concurrency 1,2,4,8,16,32,64
```

**Single concurrency level**:

```bash
bash benchmarks/llm/perf.sh \
  --mode disaggregated \
  --deployment-kind dynamo_vllm \
  --prefill-tensor-parallelism 1 \
  --prefill-data-parallelism 4 \
  --decode-tensor-parallelism 4 \
  --decode-data-parallelism 1 \
  --concurrency 64
```

### Output Structure

The script creates an `artifacts_root` directory (or your specified directory) with the following structure:

```
artifacts_root/
├── artifacts_0/
│   ├── deployment_config.json          # Deployment configuration metadata
│   ├── -concurrency1/
│   │   └── profile_export_aiperf.json   # AIPerf results for concurrency=1
│   ├── -concurrency2/
│   │   └── profile_export_aiperf.json
│   └── ...
├── artifacts_1/                        # Next benchmark run
│   └── ...
```

Each `artifacts_N` directory contains results from one benchmark run. The script automatically increments the index to avoid overwriting previous results.

> [!Tip]
> Start with a clean `artifacts_root` directory when beginning a new comparison experiment to ensure you only include results from the runs you want to compare.

## Using plot_pareto.py

The `plot_pareto.py` script generates Pareto frontier plots from benchmark results, helping you visualize the trade-off between throughput and latency.

### Basic Usage

```bash
python3 benchmarks/llm/plot_pareto.py --artifacts-root-dir artifacts_root
```

### Command-Line Options

```bash
Options:
  --artifacts-root-dir <path>    Root directory containing artifact directories (required)
  --title <string>               Title for the Pareto graph (default: "Single Node")
```

### Examples

**Single node comparison**:

```bash
python3 benchmarks/llm/plot_pareto.py --artifacts-root-dir artifacts_root
```

**Two node comparison**:

```bash
python3 benchmarks/llm/plot_pareto.py \
  --artifacts-root-dir artifacts_root \
  --title "Two Nodes"
```

### Output

The script generates:

- **`pareto_plot.png`**: Pareto frontier visualization
- **`results.csv`**: Detailed results in CSV format

## Interpreting Results

### Understanding Pareto Graphs

Pareto graphs help answer: **How much can output token throughput be improved by switching from aggregated to disaggregated serving when both operate under similar inter-token latency?**

**Axes:**

- **X-axis (tokens/s/user)**: Higher values indicate lower latency per user
- **Y-axis (tokens/s/gpu avg)**: Average throughput per GPU

**Pareto Frontier:**

- The dashed line connects Pareto-efficient points
- A point is Pareto-efficient if no other point has both higher throughput AND lower latency
- Points on the frontier represent optimal trade-offs

**Example Interpretation:**
At 45 tokens/s/user, if the disaggregated line shows 145 tokens/s/gpu and the baseline shows 80 tokens/s/gpu, the improvement is:

- **Absolute improvement**: 145 - 80 = 65 tokens/s/gpu
- **Relative improvement**: 145 / 80 = 1.81x speedup

### Metrics Explained

- **Output Token Throughput**: Total tokens generated per second across all requests
- **Output Token Throughput per User**: Average tokens per second per concurrent user (inverse of latency)
- **Output Token Throughput per GPU**: Average tokens per second per GPU (efficiency metric)
- **Time to First Token (TTFT)**: Latency from request to first token
- **Inter Token Latency**: Average time between consecutive tokens

## Comparing Multiple Deployments

To compare different deployment configurations:

1. **Run benchmarks for each configuration**:

   ```bash
   # Benchmark configuration A
   bash benchmarks/llm/perf.sh --mode aggregated --deployment-kind vllm_serve --tp 4 --dp 2

   # Benchmark configuration B
   bash benchmarks/llm/perf.sh --mode disaggregated --deployment-kind dynamo_vllm --prefill-tp 1 --prefill-dp 4 --decode-tp 4 --decode-dp 1
   ```

2. **Generate comparison plot**:
   ```bash
   python3 benchmarks/llm/plot_pareto.py --artifacts-root-dir artifacts_root
   ```

The plot will show both configurations on the same graph, making it easy to compare their Pareto frontiers.

> [!Important]
> Ensure the `--deployment-kind` values are different for each configuration so they appear as separate series in the plot.

## Supporting Additional Models

The instructions above can be used for nearly any HuggingFace-compatible model. The key requirements are:

1. **Model must be accessible** - Either downloaded locally or accessible via HuggingFace
2. **Deployment must match** - Your deployment configuration must match the parallelism settings specified in `perf.sh`
3. **Endpoint must be accessible** - The HTTP endpoint must be reachable at the specified URL

For more complex setups or different frameworks, refer to:

- [Dynamo Kubernetes Deployment Guide](../../docs/kubernetes/README.md)
- [vLLM Backend Documentation](../../examples/backends/vllm/deploy/README.md)
- [TensorRT-LLM Backend Documentation](../../examples/backends/trtllm/deploy/README.md)
- [SGLang Backend Documentation](../../examples/backends/sglang/deploy/README.md)

## Monitoring Deployment Readiness

When benchmarking, ensure all workers are ready before starting the benchmark. For Dynamo deployments, you can check worker readiness:

**Kubernetes:**

```bash
kubectl get pods -n <namespace> -l app=<deployment-name>
# Check that all pods are in "Running" state
```

**Local:**

- Check the logs of each worker to ensure they've finished loading the model
- Send a test request to verify the endpoint is responding:
  ```bash
  curl http://localhost:8000/v1/models
  ```

## Troubleshooting

### Common Issues

1. **Benchmark fails with connection errors**

   - Verify the endpoint URL is correct and accessible
   - Check that NATS and etcd are running
   - Ensure the frontend service is running and healthy

2. **Incorrect parallelism settings**

   - The parallelism settings in `perf.sh` must match your actual deployment
   - Verify your deployment configuration (Kubernetes YAML or worker command-line args)
   - Check GPU allocation matches your expectations

3. **Plot generation fails**

   - Ensure all required Python packages are installed: `pip install matplotlib seaborn pandas numpy`
   - Verify the artifacts directory contains `deployment_config.json` files
   - Check that `profile_export_aiperf.json` files exist in the concurrency subdirectories

4. **Low throughput or high latency**
   - Verify all workers are actually processing requests (check logs)
   - Ensure network connectivity between nodes (for multinode)
   - Check GPU utilization to confirm resources are being used
   - Review the [Performance Tuning Guide](../../docs/performance/tuning.md) for optimization tips

### Interconnect Configuration (Multinode)

For multinode deployments, ensure the fastest interconnect is being used. Misconfiguration can cause significant latency overhead (e.g., TCP instead of RDMA for KV cache transfer).

- Verify network configuration between nodes
- Check that NIXL is using the optimal transport
- Review backend-specific debug options if experiencing abnormal latency

## Additional Resources

- **[AIPerf Documentation](https://github.com/triton-inference-server/perf_analyzer/blob/main/genai-perf/docs/tutorial.md)** - Learn more about AIPerf benchmarking
- **[Dynamo Benchmarking Guide](../../docs/benchmarks/benchmarking.md)** - General benchmarking framework documentation
- **[Performance Tuning Guide](../../docs/performance/tuning.md)** - Optimize your deployment configuration
- **[Metrics and Visualization](../../deploy/metrics/k8s/README.md)** - Monitor deployments with Prometheus and Grafana
