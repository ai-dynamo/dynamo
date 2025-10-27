<!--
SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# LLM Deployment using vLLM on DGX-SPARK

This directory contains reference implementations for deploying Large Language Models (LLMs) in various configurations using vLLM on **DGX-SPARK systems**. For Dynamo integration, we leverage vLLM's native KV cache events, NIXL based transfer mechanisms, and metric reporting to enable KV-aware routing and P/D disaggregation.

> [!NOTE]
> This guide is specifically tailored for **DGX-SPARK** systems running **ARM64 architecture**. For general x86_64 deployments, refer to the main [README.md](./README.md).

## Use the Latest Release

We recommend using the latest stable release of Dynamo to avoid breaking changes:

[![GitHub Release](https://img.shields.io/github/v/release/ai-dynamo/dynamo)](https://github.com/ai-dynamo/dynamo/releases/latest)

You can find the latest release [here](https://github.com/ai-dynamo/dynamo/releases/latest) and check out the corresponding branch with:

```bash
git checkout $(git describe --tags $(git rev-list --tags --max-count=1))
```

---

## Table of Contents
- [DGX-SPARK Specific Considerations](#dgx-spark-specific-considerations)
- [Feature Support Matrix](#feature-support-matrix)
- [Quick Start](#quick-start)
- [Single Node Examples](#run-single-node-examples)
- [Advanced Examples](#advanced-examples)
- [Deploy on Kubernetes](#kubernetes-deployment)
- [Configuration](#configuration)

## DGX-SPARK Specific Considerations

### Architecture Requirements

DGX-SPARK systems run on **ARM64 architecture** (`aarch64`), which requires specific build configurations:

- **Platform**: `linux/arm64`
- **Architecture**: `arm64` / `aarch64`
- **Base Images**: ARM64-compatible NVIDIA CUDA images

### Build Requirements

When building containers for DGX-SPARK, you **must** specify the ARM64 platform:

```bash
# Correct build command for DGX-SPARK
./container/build.sh --framework VLLM --platform linux/arm64
```

> [!WARNING]
> **Do not use the default build command** (`./container/build.sh --framework VLLM`) as it defaults to `linux/amd64` and will cause `exec /bin/sh: exec format error` on ARM64 systems.

### Performance Considerations

DGX-SPARK systems may have different performance characteristics compared to x86_64 systems:

- **Memory bandwidth**: ARM64 systems may have different memory access patterns
- **GPU utilization**: Ensure proper GPU affinity and NUMA awareness
- **Container overhead**: ARM64 containers may have slightly different resource usage

## Feature Support Matrix

### Core Dynamo Features

| Feature | vLLM on DGX-SPARK | Notes |
|---------|-------------------|-------|
| [**Disaggregated Serving**](../../../docs/design_docs/disagg_serving.md) | ✅ | Fully supported on ARM64 |
| [**Conditional Disaggregation**](../../../docs/design_docs/disagg_serving.md#conditional-disaggregation) | 🚧 | WIP - ARM64 compatibility verified |
| [**KV-Aware Routing**](../../../docs/router/kv_cache_routing.md) | ✅ | ARM64 optimized |
| [**SLA-Based Planner**](../../../docs/planner/sla_planner.md) | ✅ | Platform agnostic |
| [**Load Based Planner**](../../../docs/planner/load_planner.md) | 🚧 | WIP - ARM64 testing in progress |
| [**KVBM**](../../../docs/kvbm/kvbm_architecture.md) | ✅ | ARM64 compatible |
| [**LMCache**](./LMCache_Integration.md) | ✅ | ARM64 supported |

### Large Scale P/D and WideEP Features

| Feature            | vLLM on DGX-SPARK | Notes                                                                 |
|--------------------|-------------------|-----------------------------------------------------------------------|
| **WideEP**         | ✅                | Support for PPLX / DeepEP verified on ARM64                          |
| **DP Rank Routing**| ✅                | Supported via external control of DP ranks - ARM64 optimized          |
| **GB200 Support**  | 🚧                | Container functional on main - ARM64 compatibility testing ongoing   |

## vLLM Quick Start

Below we provide a guide that lets you run all of our common deployment patterns on a single DGX-SPARK node.

### Start NATS and ETCD in the background

Start using [Docker Compose](../../../deploy/docker-compose.yml)

```bash
docker compose -f deploy/docker-compose.yml up -d
```

### Pull or build container

We have public images available on [NGC Catalog](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/ai-dynamo/collections/ai-dynamo/artifacts). 

> [!IMPORTANT]
> **For DGX-SPARK systems**, you must build your own container with ARM64 support:

```bash
# Build for ARM64 architecture (DGX-SPARK)
./container/build.sh --framework VLLM --platform linux/arm64

# Or explicitly with --dgx-spark flag (also forces linux/arm64)
./container/build.sh --framework VLLM --dgx-spark
```

> [!NOTE]
> The build script automatically (when using `--dgx-spark` or ARM64 platform):
> - Detects ARM64 platform and sets the correct architecture arguments (`ARCH=arm64`, `ARCH_ALT=aarch64`)
> - Uses NIXL 0.7.0 (CUDA 13.0 support) instead of 0.6.0 when `--dgx-spark` flag is set
> - Uses a special `Dockerfile.vllm.dgx-spark` that leverages NVIDIA's pre-built vLLM container (`nvcr.io/nvidia/vllm:25.09-py3`)
> - This container already includes DGX Spark functional support (Blackwell GPU compute_121) and fixes the `nvcc fatal: Unsupported gpu architecture 'compute_121a'` error

### Run container

```bash
./container/run.sh -it --framework VLLM [--mount-workspace]
```

> [!NOTE]
> **`--mount-workspace` is optional** and mounts your local Dynamo project directory into the container at `/workspace`. Use it for:
> - **Development**: When you need to edit source files and see changes immediately
> - **Local testing**: When running examples that need access to project files
> 
> Skip `--mount-workspace` for production deployments or when you don't need to modify source code.

#### What happens when you run the container?

When you run `./container/run.sh -it --framework VLLM`, it:

1. **Starts a Docker container** using the `dynamo:latest-vllm` image
2. **Runs interactively** (`-it` flag) with a bash shell
3. **Does NOT automatically start any vLLM service** - it just gives you a shell inside the container
4. **No model is loaded by default** - you're just in an empty container environment

The container uses `/opt/nvidia/nvidia_entrypoint.sh` as the entrypoint, which typically just starts a bash shell when no specific command is provided.

#### Setting up different serving modes

The serving modes are controlled by **launch scripts** that you run **inside the container**. Here's how:

**Aggregated Serving (Single GPU):**
```bash
# Start the container
./container/run.sh -it --framework VLLM

# Inside the container, run:
cd components/backends/vllm
bash launch/agg.sh
```

**Disaggregated Serving (Two GPUs):**
```bash
# Start the container
./container/run.sh -it --framework VLLM

# Inside the container, run:
cd components/backends/vllm
bash launch/disagg.sh
```

**Other available modes:**
- `agg_router.sh` - Aggregated serving with KV routing (2 GPUs)
- `disagg_router.sh` - Disaggregated serving with KV routing (3 GPUs)
- `dep.sh` - Data Parallel Attention / Expert Parallelism (4 GPUs)

#### Complete workflow example

Here's a complete example for disaggregated serving:

```bash
# 1. Build the container (if not already built)
./container/build.sh --framework VLLM --platform linux/arm64

# 2. Start the container
./container/run.sh -it --framework VLLM

# 3. Inside the container, start disaggregated serving
cd components/backends/vllm
bash launch/disagg.sh

# 4. Test the API
curl -X POST "http://localhost:8000/v1/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-0.6B",
    "prompt": "Hello, how are you?",
    "max_tokens": 50
  }'
```

> [!TIP]
> **Key Points:**
> - **No model is loaded by default** - the container just gives you a shell
> - **Models are specified in the launch scripts** (currently `Qwen/Qwen3-0.6B`)
> - **You can modify the launch scripts** to use different models
> - **The `--enforce-eager` flag** is for quick testing (remove for production)
> - **GPU assignment** is handled by `CUDA_VISIBLE_DEVICES` in the scripts

This includes the specific commit [vllm-project/vllm#19790](https://github.com/vllm-project/vllm/pull/19790) which enables support for external control of the DP ranks.

## Run Single Node Examples

> [!IMPORTANT]
> Below we provide simple shell scripts that run the components for each configuration. Each shell script runs `python3 -m dynamo.frontend` to start the ingress and uses `python3 -m dynamo.vllm` to start the vLLM workers. You can also run each command in separate terminals for better log visibility.

This figure shows an overview of the major components to deploy:

```
+------+      +-----------+      +------------------+             +---------------+
| HTTP |----->| dynamo    |----->|   vLLM Worker    |------------>|  vLLM Prefill |
|      |<-----| ingress   |<-----|                  |<------------|    Worker     |
+------+      +-----------+      +------------------+             +---------------+
                  |    ^                  |
       query best |    | return           | publish kv events
           worker |    | worker_id        v
                  |    |         +------------------+
                  |    +---------|     kv-router    |
                  +------------->|                  |
                                 +------------------+
```

Note: The above architecture illustrates all the components. The final components that get spawned depend upon the chosen deployment pattern.

### Aggregated Serving

```bash
# requires one gpu
cd components/backends/vllm
bash launch/agg.sh
```

### Aggregated Serving with KV Routing

```bash
# requires two gpus
cd components/backends/vllm
bash launch/agg_router.sh
```

### Disaggregated Serving

```bash
# requires two gpus
cd components/backends/vllm
bash launch/disagg.sh
```

### Disaggregated Serving with KV Routing

```bash
# requires three gpus
cd components/backends/vllm
bash launch/disagg_router.sh
```

### Single Node Data Parallel Attention / Expert Parallelism

This example is not meant to be performant but showcases Dynamo routing to data parallel workers

```bash
# requires four gpus
cd components/backends/vllm
bash launch/dep.sh
```

> [!TIP]
> Run a disaggregated example and try adding another prefill worker once the setup is running! The system will automatically discover and utilize the new worker.

## Advanced Examples

Below we provide a selected list of advanced deployments. Please open up an issue if you'd like to see a specific example!

### Multi-Node Disaggregated Serving

DGX-SPARK systems are perfect for multi-node deployments, especially when paired with other GPU servers. This allows you to optimize resource utilization across different hardware configurations.

#### Example: DGX-SPARK + RTX 3090 Setup

**Hardware Configuration:**
- **DGX-SPARK (ARM64)**: Prefill worker (prompt processing) + Frontend
- **RTX 3090 Server (x86_64)**: Decode worker (token generation)

**Why this works well:**
- DGX-SPARK: High compute power (~100 TFLOPs FP16) for compute-intensive prefill
- RTX 3090: Handles decode after receiving KV cache from DGX Spark
- Network efficiency: Only KV cache data transferred, not full model weights

#### Step-by-Step Multi-Node Setup

**1. Infrastructure Setup**

**On DGX-SPARK (Head Node):**
```bash
# Start NATS and ETCD services
docker compose -f deploy/docker-compose.yml up -d

# Set environment variables
export HEAD_NODE_IP="<dgx-spark-ip>"
export NATS_SERVER="nats://${HEAD_NODE_IP}:4222"
export ETCD_ENDPOINTS="${HEAD_NODE_IP}:2379"
```

**On RTX 3090 Server:**
```bash
# Set the same environment variables
export HEAD_NODE_IP="<dgx-spark-ip>"
export NATS_SERVER="nats://${HEAD_NODE_IP}:4222"
export ETCD_ENDPOINTS="${HEAD_NODE_IP}:2379"
```

**2. Build Containers**

**On DGX-SPARK:**
```bash
# Option 1: Use platform flag (recommended)
./container/build.sh --framework VLLM --platform linux/arm64

# Option 2: Use dgx-spark flag (also uses NIXL 0.7.0 with CUDA 13.0)
./container/build.sh --framework VLLM --dgx-spark
```

**On RTX 3090 Server:**
```bash
./container/build.sh --framework VLLM --platform linux/amd64
```

**3. Deploy Workers**

> [!NOTE]
> **Worker flags**: Prefill worker uses `--is-prefill-worker` flag. Decode worker runs without any flag (just regular vllm command).

**On DGX-SPARK (Prefill Worker + Frontend):**
```bash
# Start the container
./container/run.sh -it --framework VLLM

# Inside the container, set environment variables:
export HEAD_NODE_IP="<dgx-spark-ip>"
export NATS_SERVER="nats://${HEAD_NODE_IP}:4222"
export ETCD_ENDPOINTS="${HEAD_NODE_IP}:2379"

cd components/backends/vllm

# Start frontend with KV routing
python -m dynamo.frontend \
    --router-mode kv \
    --http-port 8000 &

# Start prefill worker (handles prompt processing - compute-intensive)
python -m dynamo.vllm \
    --model Qwen/Qwen3-0.6B \
    --enforce-eager \
    --is-prefill-worker \
    --connector nixl
```

**On RTX 3090 Server (Decode Worker):**
```bash
# Start the container
./container/run.sh -it --framework VLLM

# Inside the container, set environment variables:
export HEAD_NODE_IP="<dgx-spark-ip>"
export NATS_SERVER="nats://${HEAD_NODE_IP}:4222"
export ETCD_ENDPOINTS="${HEAD_NODE_IP}:2379"

cd components/backends/vllm

# Start decode worker (handles token generation - memory-intensive)
python -m dynamo.vllm \
    --model Qwen/Qwen3-0.6B \
    --enforce-eager \
    --connector nixl
```

#### How Multi-Node Disaggregation Works

1. **Request Flow:**
   - Client sends request to DGX-SPARK frontend (port 8000)
   - Frontend routes prefill work to DGX-SPARK itself (high compute for prefill)
   - DGX-SPARK processes the prompt and builds the KV cache (compute-intensive)
   - KV cache is transferred to the RTX 3090 server
   - RTX 3090 server generates tokens using the KV cache (memory-intensive)

2. **KV Cache Transfer:**
   - Uses NIXL connector for efficient KV cache transfer over network
   - Layer-by-layer streaming: Each layer's KV cache can be transferred as it's computed
   - DGX-SPARK sends processed KV cache to decode server
   - Decode server uses the KV cache for efficient token generation

3. **Resource Optimization:**
   - DGX-SPARK: Handles compute-intensive prefill (~100 TFLOPs compute)
   - RTX 3090: Handles memory-intensive decode after receiving KV cache

#### Network Requirements

Ensure your firewall allows:
- **Port 4222**: NATS communication
- **Port 2379**: ETCD communication  
- **Port 8000**: HTTP API (if accessing from external clients)

#### Testing Multi-Node Setup

Once both workers are running, test from any machine:

```bash
curl -X POST "http://<dgx-spark-ip>:8000/v1/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-0.6B",
    "prompt": "Hello, how are you?",
    "max_tokens": 50
  }'
```

> [!TIP]
> **Multi-Node Benefits:**
> - **Resource optimization**: Use each machine's strengths
> - **Scalability**: Add more prefill or decode workers as needed
> - **Cost efficiency**: Leverage existing hardware optimally
> - **Network efficiency**: Only KV cache data transferred, not full model weights

### Kubernetes Deployment

For complete Kubernetes deployment instructions, configurations, and troubleshooting, see [vLLM Kubernetes Deployment Guide](../../../components/backends/vllm/deploy/README.md)

> [!NOTE]
> When deploying on Kubernetes with DGX-SPARK nodes, ensure your cluster nodes are properly labeled with ARM64 architecture and use ARM64-compatible base images.

## Configuration

vLLM workers are configured through command-line arguments. Key parameters include:

- `--model`: Model to serve (e.g., `Qwen/Qwen3-0.6B`)
- `--is-prefill-worker`: Enable prefill-only mode for disaggregated serving
- `--metrics-endpoint-port`: Port for publishing KV metrics to Dynamo
- `--connector`: Specify which kv_transfer_config you want vllm to use `[nixl, lmcache, kvbm, none]`. This is a helper flag which overwrites the engines KVTransferConfig.

See `args.py` for the full list of configuration options and their defaults.

The [documentation](https://docs.vllm.ai/en/v0.9.2/configuration/serve_args.html?h=serve+arg) for the vLLM CLI args points to running 'vllm serve --help' to see what CLI args can be added. We use the same argument parser as vLLM.

### Hashing Consistency for KV Events

When using KV-aware routing, ensure deterministic hashing across processes to avoid radix tree mismatches. Choose one of the following:

- Set `PYTHONHASHSEED=0` for all vLLM processes when relying on Python's builtin hashing for prefix caching.
- If your vLLM version supports it, configure a deterministic prefix caching algorithm, for example:

```bash
vllm serve ... --enable-prefix-caching --prefix-caching-algo sha256
```
See the high-level notes in [KV Cache Routing](../../../docs/router/kv_cache_routing.md) on deterministic event IDs.

## Request Migration

You can enable [request migration](../../../docs/fault_tolerance/request_migration.md) to handle worker failures gracefully. Use the `--migration-limit` flag to specify how many times a request can be migrated to another worker:

```bash
python3 -m dynamo.vllm ... --migration-limit=3
```

This allows a request to be migrated up to 3 times before failing. See the [Request Migration Architecture](../../../docs/fault_tolerance/request_migration.md) documentation for details on how this works.

## Request Cancellation

When a user cancels a request (e.g., by disconnecting from the frontend), the request is automatically cancelled across all workers, freeing compute resources for other requests.

### Cancellation Support Matrix

| | Prefill | Decode |
|-|---------|--------|
| **Aggregated** | ✅ | ✅ |
| **Disaggregated** | ✅ | ✅ |

For more details, see the [Request Cancellation Architecture](../../../docs/fault_tolerance/request_cancellation.md) documentation.

## Troubleshooting

### Common Issues on DGX-SPARK

#### Build Errors

**Error**: `exec /bin/sh: exec format error`

**Solution**: Ensure you're building with the correct platform:
```bash
./container/build.sh --framework VLLM --platform linux/arm64
```

**Error**: `nvcc fatal: Unsupported gpu architecture 'compute_121a'` (Blackwell GPU)

**Solution**: This error occurs when trying to build vLLM from source with an older CUDA toolchain that doesn't support Blackwell GPUs. The build script automatically uses `Dockerfile.vllm.dgx-spark` which leverages NVIDIA's pre-built vLLM container (`nvcr.io/nvidia/vllm:25.09-py3`) with native DGX Spark support:
```bash
./container/build.sh --framework VLLM --platform linux/arm64
```
The special Dockerfile skips building vLLM from source and uses NVIDIA's container that already includes compute_121 (Blackwell) support.

#### Performance Issues

- **Memory bandwidth**: Monitor memory usage patterns specific to ARM64
- **GPU utilization**: Check GPU affinity settings and NUMA topology
- **Container overhead**: ARM64 containers may have different resource requirements

#### Architecture Detection

To verify your system architecture:
```bash
uname -m  # Should return 'aarch64' for DGX-SPARK
```

### Getting Help

For DGX-SPARK specific issues:
1. Check this README for ARM64-specific considerations
2. Verify you're using the correct build platform (`linux/arm64`)
3. Review the main [README.md](./README.md) for general troubleshooting
4. Open an issue with `DGX-SPARK` and `ARM64` tags for specific support
