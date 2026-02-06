<!--
SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

# Dynamo KVBM

The Dynamo KVBM is a distributed KV-cache block management system designed for scalable LLM inference. It cleanly separates memory management from inference runtimes (vLLM, TensorRT-LLM, and SGLang), enabling GPU↔CPU↔Disk/Remote tiering, asynchronous block offload/onboard, and efficient block reuse.

![A block diagram showing a layered architecture view of Dynamo KV Block manager.](../../../docs/images/kvbm-architecture.png)


## Feature Highlights

- **Distributed KV-Cache Management:** Unified GPU↔CPU↔Disk↔Remote tiering for scalable LLM inference.
- **Async Offload & Reuse:** Seamlessly move KV blocks between memory tiers using GDS-accelerated transfers powered by NIXL, without recomputation.
- **Runtime-Agnostic:** Works out-of-the-box with vLLM, TensorRT-LLM, and SGLang via lightweight connectors.
- **Memory-Safe & Modular:** RAII lifecycle and pluggable design for reliability, portability, and backend extensibility.

## Installation

```bash
pip install kvbm
```

See the [support matrix](../../../docs/reference/support-matrix.md) for version compatibility questions.

## Build from Source

The pip wheel is built through a Docker build process:

```bash
# Build the Docker image with KVBM enabled (from the dynamo repo root)
./container/build.sh --framework none --enable-kvbm --tag local-kvbm
```

Once built, you can either:

**Option 1: Run and use the container directly**
```bash
./container/run.sh --framework none -it
```

**Option 2: Extract the wheel file to your local filesystem**
```bash
# Create a temporary container from the built image
docker create --name temp-kvbm-container local-kvbm:latest

# Copy the KVBM wheel to your current directory
docker cp temp-kvbm-container:/opt/dynamo/wheelhouse/ ./dynamo_wheelhouse

# Clean up the temporary container
docker rm temp-kvbm-container

# Install the wheel locally
pip install ./dynamo_wheelhouse/kvbm*.whl
```

Note that the default pip wheel built is not compatible with CUDA 13 at the moment.


## Integrations

### Environment Variables

| Variable | Description | Default |
|-----------|--------------|----------|
| `DYN_KVBM_CPU_CACHE_GB` | CPU pinned memory cache size (GB) | required |
| `DYN_KVBM_DISK_CACHE_GB` | SSD Disk/Storage system cache size (GB) | optional |
| `DYN_KVBM_DISK_CACHE_DIR` | Disk cache directory | `/tmp/` |
| `DYN_KVBM_DISK_ZEROFILL_FALLBACK` | Enable zero-fill when `fallocate()` unsupported (e.g., Lustre) | `false` |
| `DYN_KVBM_DISK_DISABLE_O_DIRECT` | Disable O_DIRECT for disk I/O (debug/compatibility) | `false` |
| `DYN_KVBM_LEADER_WORKER_INIT_TIMEOUT_SECS` | Timeout (in seconds) for the KVBM leader and worker to synchronize and allocate the required memory and storage. Increase this value if allocating large amounts of memory or storage. | 120 |
| `DYN_KVBM_METRICS` | Enable metrics endpoint | `false` |
| `DYN_KVBM_METRICS_PORT` | Metrics port | `6880` |
| `DYN_KVBM_DISABLE_DISK_OFFLOAD_FILTER` | Disable disk offload filtering to remove SSD lifespan protection | `false` |
| `DYN_KVBM_HOST_OFFLOAD_PREFIX_MIN_PRIORITY` | Minimum priority (0-100) for CPU offload with contiguous (prefix) semantics: offloading stops at the first block below threshold, and all subsequent blocks are also skipped. Used for priority-based filtering. | `0` (no filtering) |

#### Disk Storage Configuration

**Why special configuration may be needed:**

Some filesystems (e.g., Lustre, certain network filesystems) don't support `fallocate()`, which KVBM uses for fast disk space allocation. Additionally, KVBM uses O_DIRECT I/O for GPU DirectStorage (GDS) performance, which requires strict 4096-byte alignment.

**Setup for filesystems without fallocate() support:**
```bash
export DYN_KVBM_DISK_CACHE_DIR=/mnt/storage/kvbm_cache
export DYN_KVBM_DISK_ZEROFILL_FALLBACK=true  # Enables zero-fill fallback when fallocate() unsupported
```

**What happens:**
- Without `ZEROFILL_FALLBACK=true`: Disk cache allocation may fail with "Operation not supported"
- With `ZEROFILL_FALLBACK=true`: KVBM writes zeros using page-aligned buffers compatible with O_DIRECT requirements

**Troubleshooting:** If you encounter "write all error" or EINVAL (errno 22), try disabling O_DIRECT: `export DYN_KVBM_DISK_DISABLE_O_DIRECT=true`

### vLLM

```bash
DYN_KVBM_CPU_CACHE_GB=100 vllm serve \
  --kv-transfer-config '{"kv_connector":"DynamoConnector","kv_role":"kv_both","kv_connector_module_path":"kvbm.vllm_integration.connector"}' \
  Qwen/Qwen3-8B
```

For more detailed integration with dynamo, disaggregated serving support and benchmarking, please check [vllm-setup](../../../docs/kvbm/kvbm_guide.md#run-kvbm-in-dynamo-with-vllm)

### TensorRT-LLM

```bash
cat >/tmp/kvbm_llm_api_config.yaml <<EOF
cuda_graph_config: null
kv_cache_config:
  enable_partial_reuse: false
  free_gpu_memory_fraction: 0.80
kv_connector_config:
  connector_module: kvbm.trtllm_integration.connector
  connector_scheduler_class: DynamoKVBMConnectorLeader
  connector_worker_class: DynamoKVBMConnectorWorker
EOF

DYN_KVBM_CPU_CACHE_GB=100 trtllm-serve Qwen/Qwen3-8B \
  --host localhost --port 8000 \
  --backend pytorch \
  --extra_llm_api_options /tmp/kvbm_llm_api_config.yaml
```

For more detailed integration with dynamo and benchmarking, please check [trtllm-setup](../../../docs/kvbm/kvbm_guide.md#run-kvbm-in-dynamo-with-tensorrt-llm)


## 📚 Docs

- [Architecture](../../../docs/kvbm/README.md#architecture)
- [Design Deepdive](../../../docs/kvbm/kvbm_design.md)
- [NIXL Overview](https://github.com/ai-dynamo/nixl/blob/main/docs/nixl.md)
