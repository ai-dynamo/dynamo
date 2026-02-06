<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# FlexKV Integration in Dynamo

## Introduction

[FlexKV](https://github.com/taco-project/FlexKV) is a scalable, distributed runtime for KV cache offloading developed by Tencent Cloud's TACO team in collaboration with the community. It acts as a unified KV caching layer for inference engines like vLLM, TensorRT-LLM, and SGLang.

### Key Features

- **Multi-level caching**: CPU memory, local SSD, and scalable storage (cloud storage) for KV cache offloading
- **Distributed KV cache reuse**: Share KV cache across multiple nodes using distributed RadixTree
- **High-performance I/O**: Supports io_uring and GPU Direct Storage (GDS) for accelerated data transfer
- **Asynchronous operations**: Get and put operations can overlap with computation through prefetching


## Prerequisites

1. **Dynamo installed** with vLLM support
2. **Infrastructure services running**:
   ```bash
   docker compose -f deploy/docker-compose.yml up -d
   ```
3. **FlexKV dependencies** (for SSD offloading):
   ```bash
   apt install liburing-dev libxxhash-dev
   ```

## Quick Start

### Enable FlexKV

Set the `DYNAMO_USE_FLEXKV` environment variable and use the `--connector flexkv` flag:

```bash
export DYNAMO_USE_FLEXKV=1
python -m dynamo.vllm --model Qwen/Qwen3-0.6B --connector flexkv
```

## Aggregated Serving

### Basic Setup

```bash
# Terminal 1: Start frontend
python -m dynamo.frontend &

# Terminal 2: Start vLLM worker with FlexKV
DYNAMO_USE_FLEXKV=1 \
FLEXKV_CPU_CACHE_GB=32 \
  python -m dynamo.vllm --model Qwen/Qwen3-0.6B --connector flexkv
```

### With KV-Aware Routing

For multi-worker deployments with KV-aware routing to maximize cache reuse:

```bash
# Terminal 1: Start frontend with KV router
python -m dynamo.frontend \
    --router-mode kv \
    --router-reset-states &

# Terminal 2: Worker 1
DYNAMO_USE_FLEXKV=1 \
FLEXKV_CPU_CACHE_GB=32 \
FLEXKV_SERVER_RECV_PORT="ipc:///tmp/flexkv_server_0" \
CUDA_VISIBLE_DEVICES=0 \
python -m dynamo.vllm \
    --model Qwen/Qwen3-0.6B \
    --connector flexkv \
    --gpu-memory-utilization 0.2 \
    --kv-events-config '{"publisher":"zmq","topic":"kv-events","endpoint":"tcp://*:20080","enable_kv_cache_events":true}' &

# Terminal 3: Worker 2
DYNAMO_USE_FLEXKV=1 \
FLEXKV_CPU_CACHE_GB=32 \
FLEXKV_SERVER_RECV_PORT="ipc:///tmp/flexkv_server_1" \
CUDA_VISIBLE_DEVICES=1 \
python -m dynamo.vllm \
    --model Qwen/Qwen3-0.6B \
    --connector flexkv \
    --gpu-memory-utilization 0.2 \
    --kv-events-config '{"publisher":"zmq","topic":"kv-events","endpoint":"tcp://*:20081","enable_kv_cache_events":true}'
```

## Disaggregated Serving

FlexKV can be used with disaggregated prefill/decode serving. The prefill worker uses FlexKV for KV cache offloading, while NIXL handles KV transfer between prefill and decode workers.

```bash
# Terminal 1: Start frontend
python -m dynamo.frontend &

# Terminal 2: Decode worker (without FlexKV)
CUDA_VISIBLE_DEVICES=0 python -m dynamo.vllm --model Qwen/Qwen3-0.6B --connector nixl &

# Terminal 3: Prefill worker (with FlexKV)
DYN_VLLM_KV_EVENT_PORT=20081 \
VLLM_NIXL_SIDE_CHANNEL_PORT=20097 \
DYNAMO_USE_FLEXKV=1 \
FLEXKV_CPU_CACHE_GB=32 \
CUDA_VISIBLE_DEVICES=1 \
  python -m dynamo.vllm \
  --model Qwen/Qwen3-0.6B \
  --is-prefill-worker \
  --connector nixl flexkv
```

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `DYNAMO_USE_FLEXKV` | Enable FlexKV integration | `0` (disabled) |
| `FLEXKV_CPU_CACHE_GB` | CPU memory cache size in GB | Required |
| `FLEXKV_CONFIG_PATH` | Path to FlexKV YAML config file | Not set |
| `FLEXKV_SERVER_RECV_PORT` | IPC port for FlexKV server | Auto |

### CPU-Only Offloading

For simple CPU memory offloading:

```bash
unset FLEXKV_CONFIG_PATH
export FLEXKV_CPU_CACHE_GB=32
```

### CPU + SSD Tiered Offloading

For multi-tier offloading with SSD storage, create a configuration file:

```bash
cat > ./flexkv_config.yml <<EOF
cpu_cache_gb: 32
ssd_cache_gb: 1024
ssd_cache_dir: /data0/flexkv_ssd/;/data1/flexkv_ssd/
enable_gds: false
EOF

export FLEXKV_CONFIG_PATH="./flexkv_config.yml"
```

### Configuration Options

| Option | Description |
|--------|-------------|
| `cpu_cache_gb` | CPU memory cache size in GB |
| `ssd_cache_gb` | SSD cache size in GB |
| `ssd_cache_dir` | SSD cache directories (semicolon-separated for multiple SSDs) |
| `enable_gds` | Enable GPU Direct Storage for SSD I/O |

> **Note:** For full configuration options, see the [FlexKV Configuration Reference](https://github.com/taco-project/FlexKV/blob/main/docs/flexkv_config_reference/README_en.md).

## Distributed KV Cache Reuse

FlexKV supports distributed KV cache reuse to share cache across multiple nodes. This enables:

- **Distributed RadixTree**: Each node maintains a local snapshot of the global index
- **Lease Mechanism**: Ensures data validity during cross-node transfers
- **RDMA-based Transfer**: Uses Mooncake Transfer Engine for high-performance KV cache transfer

For setup instructions, see the [FlexKV Distributed Reuse Guide](https://github.com/taco-project/FlexKV/blob/main/docs/dist_reuse/README_en.md).

## Architecture

FlexKV consists of three core modules:

### StorageEngine

Initializes the three-level cache (GPU → CPU → SSD/Cloud). It groups multiple tokens into blocks and stores KV cache at the block level, maintaining the same KV shape as in GPU memory.

### GlobalCacheEngine

The control plane that determines data transfer direction and identifies source/destination block IDs. Includes:
- RadixTree for prefix matching
- Memory pool to track space usage and trigger eviction

### TransferEngine

The data plane that executes data transfers:
- Multi-threading for parallel transfers
- High-performance I/O (io_uring, GDS)
- Asynchronous operations overlapping with computation

## Verify Deployment

```bash
curl localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-0.6B",
    "messages": [{"role": "user", "content": "Hello!"}],
    "stream": false,
    "max_tokens": 30
  }'
```

## See Also

- [FlexKV GitHub Repository](https://github.com/taco-project/FlexKV)
- [FlexKV vLLM Adapter Documentation](https://github.com/taco-project/FlexKV/blob/main/docs/vllm_adapter/README_en.md)

