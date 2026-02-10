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

# KVBM Guide
The Dynamo KV Block Manager (KVBM) is a scalable runtime component designed to handle memory allocation, management, and remote sharing of Key-Value (KV) blocks for inference tasks across heterogeneous and distributed environments. It acts as a unified memory layer for frameworks like vLLM and TensorRT-LLM.

KVBM is modular and can be used standalone via `pip install kvbm` or as the memory management component in the full Dynamo stack. This guide covers installation, configuration, and deployment of the Dynamo KV Block Manager (KVBM) and other KV cache management systems.

## Table of Contents

- [Quick Start](#quick-start)
- [Run KVBM Standalone](#run-kvbm-standalone)
- [Run KVBM in Dynamo with vLLM](#run-kvbm-in-dynamo-with-vllm)
- [Run KVBM in Dynamo with TensorRT-LLM](#run-kvbm-in-dynamo-with-tensorrt-llm)
- [Run Dynamo with SGLang HiCache](#run-dynamo-with-sglang-hicache)
- [Disaggregated Serving with KVBM](#disaggregated-serving-with-kvbm)
- [Configuration](#configuration)
- [Enable and View KVBM Metrics](#enable-and-view-kvbm-metrics)
- [Benchmarking KVBM](#benchmarking-kvbm)
- [Troubleshooting](#troubleshooting)
- [Developing Locally](#developing-locally)

## Quick Start

## Run KVBM Standalone

KVBM can be used independently without using the rest of the Dynamo stack:

```bash
pip install kvbm
```

See the [support matrix](../../reference/support-matrix.md) for version compatibility.

### Build from Source

To build KVBM from source, see the detailed instructions in the [KVBM bindings README](../../../lib/bindings/kvbm/README.md#build-from-source).

## Run KVBM in Dynamo with vLLM

### Docker Setup

```bash
# Start up etcd for KVBM leader/worker registration and discovery
docker compose -f deploy/docker-compose.yml up -d

# Build a dynamo vLLM container (KVBM is built in by default)
./container/build.sh --framework vllm

# Launch the container
./container/run.sh --framework vllm -it --mount-workspace --use-nixl-gds
```

### Aggregated Serving

```bash
cd $DYNAMO_HOME/examples/backends/vllm
./launch/agg_kvbm.sh
```

#### Verify Deployment

```bash
curl localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-0.6B",
    "messages": [{"role": "user", "content": "Hello, how are you?"}],
    "stream": false,
    "max_tokens": 10
  }'
```

#### Alternative: Using Direct vllm serve

You can also use `vllm serve` directly with KVBM:

```bash
vllm serve --kv-transfer-config '{"kv_connector":"DynamoConnector","kv_role":"kv_both", "kv_connector_module_path": "kvbm.vllm_integration.connector"}' Qwen/Qwen3-0.6B
```

## Run KVBM in Dynamo with TensorRT-LLM

> [!NOTE]
> **Prerequisites:**
> - Ensure `etcd` and `nats` are running before starting
> - KVBM only supports TensorRT-LLM's PyTorch backend
> - Disable partial reuse (`enable_partial_reuse: false`) to increase offloading cache hits
> - KVBM requires TensorRT-LLM v1.2.0rc2 or newer

### Docker Setup

```bash
# Start up etcd for KVBM leader/worker registration and discovery
docker compose -f deploy/docker-compose.yml up -d

# Build a dynamo TRTLLM container (KVBM is built in by default)
./container/build.sh --framework trtllm

# Launch the container
./container/run.sh --framework trtllm -it --mount-workspace --use-nixl-gds
```

### Aggregated Serving

```bash
# Write the LLM API config
cat > "/tmp/kvbm_llm_api_config.yaml" <<EOF
backend: pytorch
cuda_graph_config: null
kv_cache_config:
  enable_partial_reuse: false
  free_gpu_memory_fraction: 0.80
kv_connector_config:
  connector_module: kvbm.trtllm_integration.connector
  connector_scheduler_class: DynamoKVBMConnectorLeader
  connector_worker_class: DynamoKVBMConnectorWorker
EOF

# Start dynamo frontend
python3 -m dynamo.frontend --http-port 8000 &

# Serve the model with KVBM
python3 -m dynamo.trtllm \
  --model-path Qwen/Qwen3-0.6B \
  --served-model-name Qwen/Qwen3-0.6B \
  --extra-engine-args /tmp/kvbm_llm_api_config.yaml &
```

#### Verify Deployment

```bash
curl localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-0.6B",
    "messages": [{"role": "user", "content": "Hello, how are you?"}],
    "stream": false,
    "max_tokens": 30
  }'
```

#### Alternative: Using trtllm-serve

```bash
trtllm-serve Qwen/Qwen3-0.6B --host localhost --port 8000 --backend pytorch --extra_llm_api_options /tmp/kvbm_llm_api_config.yaml
```

## Run Dynamo with SGLang HiCache

SGLang's Hierarchical Cache (HiCache) extends KV cache storage beyond GPU memory to include host CPU memory. When using NIXL as the storage backend, HiCache integrates with Dynamo's memory infrastructure.

### Quick Start

```bash
# Start SGLang worker with HiCache enabled
python -m dynamo.sglang \
  --model-path Qwen/Qwen3-0.6B \
  --host 0.0.0.0 --port 8000 \
  --enable-hierarchical-cache \
  --hicache-ratio 2 \
  --hicache-write-policy write_through \
  --hicache-storage-backend nixl

# In a separate terminal, start the frontend
python -m dynamo.frontend --http-port 8000

# Send a test request
curl localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-0.6B",
    "messages": [{"role": "user", "content": "Hello!"}],
    "stream": false,
    "max_tokens": 30
  }'
```

> **Learn more:** See the [SGLang HiCache Integration Guide](../../integrations/sglang_hicache.md) for detailed configuration, deployment examples, and troubleshooting.

## Disaggregated Serving with KVBM

KVBM supports disaggregated serving where prefill and decode operations run on separate workers. KVBM is enabled on the prefill worker to offload KV cache.

### Disaggregated Serving with vLLM

```bash
# 1P1D - one prefill worker and one decode worker
# NOTE: requires at least 2 GPUs
cd $DYNAMO_HOME/examples/backends/vllm
./launch/disagg_kvbm.sh

# 2P2D - two prefill workers and two decode workers
# NOTE: requires at least 4 GPUs
cd $DYNAMO_HOME/examples/backends/vllm
./launch/disagg_kvbm_2p2d.sh
```

### Disaggregated Serving with TRT-LLM

> [!NOTE]
> The latest TensorRT-LLM release (1.3.0rc1) is currently experiencing a request hang when running disaggregated serving with KVBM.
> Please include the TensorRT-LLM commit id `18e611da773026a55d187870ebcfa95ff00c8482` when building the Dynamo TensorRT-LLM runtime image to test the KVBM + disaggregated serving feature.

```bash
# Build the Dynamo TensorRT-LLM container using commit ID 18e611da773026a55d187870ebcfa95ff00c8482. Note: This build can take a long time.
./container/build.sh --framework trtllm --tensorrtllm-commit 18e611da773026a55d187870ebcfa95ff00c8482 --tensorrtllm-git-url https://github.com/NVIDIA/TensorRT-LLM.git

# Launch the container
./container/run.sh --framework trtllm -it --mount-workspace --use-nixl-gds
```
> [!NOTE]
> Important: After logging into the Dynamo TensorRT-LLM runtime container, copy the Triton kernels into the container’s virtual environment as a separate Python module.

```bash
# Clone the TensorRT-LLM repo and copy the triton_kernels folder into the container as a Python module.
git clone https://github.com/NVIDIA/TensorRT-LLM.git /tmp/TensorRT-LLM && \
cd /tmp/TensorRT-LLM && \
git checkout 18e611da773026a55d187870ebcfa95ff00c8482 && \
cp -r triton_kernels /opt/dynamo/venv/lib/python3.12/site-packages/ && \
cd /workspace && \
rm -rf /tmp/TensorRT-LLM
```

```bash
# Launch prefill worker with KVBM
python3 -m dynamo.trtllm \
  --model-path Qwen/Qwen3-0.6B \
  --served-model-name Qwen/Qwen3-0.6B \
  --extra-engine-args /tmp/kvbm_llm_api_config.yaml \
  --disaggregation-mode prefill &
```

## Configuration

### Cache Tier Configuration

Configure KVBM cache tiers using environment variables:

```bash
# Option 1: CPU cache only (GPU -> CPU offloading)
export DYN_KVBM_CPU_CACHE_GB=4  # 4GB of pinned CPU memory

# Option 2: Both CPU and Disk cache (GPU -> CPU -> Disk tiered offloading)
export DYN_KVBM_CPU_CACHE_GB=4
export DYN_KVBM_DISK_CACHE_GB=8  # 8GB of disk

# [Experimental] Option 3: Disk cache only (GPU -> Disk direct offloading)
# NOTE: Experimental, may not provide optimal performance
# NOTE: Disk offload filtering not supported with this option
export DYN_KVBM_DISK_CACHE_GB=8
```

You can also specify exact block counts instead of GB:
- `DYN_KVBM_CPU_CACHE_OVERRIDE_NUM_BLOCKS`
- `DYN_KVBM_DISK_CACHE_OVERRIDE_NUM_BLOCKS`

### SSD Lifespan Protection

When disk offloading is enabled, disk offload filtering is enabled by default to extend SSD lifespan. The current policy only offloads KV blocks from CPU to disk if the blocks have frequency ≥ 2. Frequency doubles on cache hit (initialized at 1) and decrements by 1 on each time decay step.

To disable disk offload filtering:

```bash
export DYN_KVBM_DISABLE_DISK_OFFLOAD_FILTER=true
```

## Enable and View KVBM Metrics

### Setup Monitoring Stack

```bash
# Start basic services (etcd & natsd), along with Prometheus and Grafana
docker compose -f deploy/docker-observability.yml up -d
```

### Enable Metrics for vLLM

```bash
DYN_KVBM_METRICS=true \
DYN_KVBM_CPU_CACHE_GB=20 \
python -m dynamo.vllm \
    --model Qwen/Qwen3-0.6B \
    --enforce-eager \
    --connector kvbm
```

### Enable Metrics for TensorRT-LLM

```bash
DYN_KVBM_METRICS=true \
DYN_KVBM_CPU_CACHE_GB=20 \
python3 -m dynamo.trtllm \
  --model-path Qwen/Qwen3-0.6B \
  --served-model-name Qwen/Qwen3-0.6B \
  --extra-engine-args /tmp/kvbm_llm_api_config.yaml &
```

### Firewall Configuration (Optional)

```bash
# If firewall blocks KVBM metrics ports
sudo ufw allow 6880/tcp
```

### View Metrics

Access Grafana at http://localhost:3000 (default login: `dynamo`/`dynamo`) and look for the **KVBM Dashboard**.

### Available Metrics

| Metric | Description |
|--------|-------------|
| `kvbm_matched_tokens` | Number of matched tokens |
| `kvbm_offload_blocks_d2h` | Offload blocks from device to host |
| `kvbm_offload_blocks_h2d` | Offload blocks from host to disk |
| `kvbm_offload_blocks_d2d` | Offload blocks from device to disk (bypassing host) |
| `kvbm_onboard_blocks_d2d` | Onboard blocks from disk to device |
| `kvbm_onboard_blocks_h2d` | Onboard blocks from host to device |
| `kvbm_host_cache_hit_rate` | Host cache hit rate (0.0-1.0) |
| `kvbm_disk_cache_hit_rate` | Disk cache hit rate (0.0-1.0) |

## Benchmarking KVBM

Use [LMBenchmark](https://github.com/LMCache/LMBenchmark) to evaluate KVBM performance.

### Setup

```bash
git clone https://github.com/LMCache/LMBenchmark.git
cd LMBenchmark/synthetic-multi-round-qa
```

### Run Benchmark

```bash
# Synthetic multi-turn chat dataset
# Arguments: model, endpoint, output prefix, qps
./long_input_short_output_run.sh \
    "Qwen/Qwen3-0.6B" \
    "http://localhost:8000" \
    "benchmark_kvbm" \
    1
```

Average TTFT and other performance numbers will be in the output.

> **TIP:** If metrics are enabled, observe KV offloading and onboarding in the Grafana dashboard.

### Baseline Comparison

#### vLLM Baseline (without KVBM)

```bash
vllm serve Qwen/Qwen3-0.6B
```

#### TensorRT-LLM Baseline (without KVBM)

```bash
# Create config without kv_connector_config
cat > "/tmp/llm_api_config.yaml" <<EOF
backend: pytorch
cuda_graph_config: null
kv_cache_config:
  enable_partial_reuse: false
  free_gpu_memory_fraction: 0.80
EOF

trtllm-serve Qwen/Qwen3-0.6B --host localhost --port 8000 --backend pytorch --extra_llm_api_options /tmp/llm_api_config.yaml
```

## Troubleshooting

### No TTFT Performance Gain

**Symptom:** Enabling KVBM does not show TTFT improvement or causes performance degradation.

**Cause:** Not enough prefix cache hits on KVBM to reuse offloaded KV blocks.

**Solution:** Enable KVBM metrics and check the Grafana dashboard for `Onboard Blocks - Host to Device` and `Onboard Blocks - Disk to Device`. Large numbers of onboarded KV blocks indicate good cache reuse:

![Grafana Example](../../images/kvbm_metrics_grafana.png)

### KVBM Worker Initialization Timeout

**Symptom:** KVBM fails to start when allocating large memory or disk storage.

**Solution:** Increase the leader-worker initialization timeout (default: 1800 seconds):

```bash
export DYN_KVBM_LEADER_WORKER_INIT_TIMEOUT_SECS=3600  # 1 hour
```

### Disk Offload Fails to Start

**Symptom:** KVBM fails to start when disk offloading is enabled.

**Cause:** `fallocate()` is not supported on the filesystem (e.g., Lustre, certain network filesystems).

**Solution:** Enable disk zerofill fallback:

```bash
export DYN_KVBM_DISK_ZEROFILL_FALLBACK=true
```

If you encounter "write all error" or EINVAL (errno 22), also try:

```bash
export DYN_KVBM_DISK_DISABLE_O_DIRECT=true
```

## Developing Locally

Inside the Dynamo container, after changing KVBM-related code (Rust and/or Python):

```bash
cd /workspace/lib/bindings/kvbm
uv pip install maturin[patchelf]
maturin build --release --out /workspace/dist
uv pip install --upgrade --force-reinstall --no-deps /workspace/dist/kvbm*.whl
```

## See Also

- [KVBM Overview](README.md) for a quick overview of KV Caching, KVBM and its architecture
- [KVBM Design](../../design_docs/kvbm_design.md) for a deep dive into KVBM architecture
- [LMCache Integration](../../integrations/lmcache_integration.md)
- [FlexKV Integration](../../integrations/flexkv_integration.md)
- [SGLang HiCache](../../integrations/sglang_hicache.md)
