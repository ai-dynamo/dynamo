---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: LMCache
---

## Introduction

LMCache is a high-performance KV cache layer that supercharges LLM serving by enabling **prefill-once, reuse-everywhere** semantics. As described in the [official documentation](https://docs.lmcache.ai/index.html), LMCache lets LLMs prefill each text only once by storing the KV caches of all reusable texts, allowing reuse of KV caches for any reused text (not necessarily prefix) across any serving engine instance.

This document describes how LMCache is integrated into Dynamo's vLLM backend to provide enhanced performance and memory efficiency.

## Installation Notes

The Dynamo vLLM runtime expects LMCache to come from the underlying vLLM container or Python environment. As of vLLM `v0.19.1`, the upstream `vllm/vllm-openai` CUDA 12 x86_64 image includes a working LMCache wheel. The CUDA 13 x86_64 image includes `lmcache`, but its compiled `lmcache.c_ops` extension is linked against `libcudart.so.12`, while the image provides CUDA 13. LMCache also does not currently publish aarch64 wheels, so a wheel install is not enough for arm64/aarch64 containers.

For CUDA 13 or arm64/aarch64 images, build LMCache from source in a dev image or temporary builder stage that matches the runtime image's Python, PyTorch, and CUDA stack, then install the resulting wheel into the runtime image. Follow the official [LMCache installation guide](https://docs.lmcache.ai/getting_started/installation.html), and keep build isolation disabled so the build does not pull a different torch stack:

```bash
git clone --depth 1 --branch v0.4.3 https://github.com/LMCache/LMCache.git /tmp/lmcache
cd /tmp/lmcache

# In the dev/builder image, install matching CUDA development headers first.
# Example Debian package names for CUDA 13.0 images:
apt-get update
apt-get install -y --no-install-recommends \
  build-essential \
  libcublas-dev-13-0 \
  libcusolver-dev-13-0 \
  libcusparse-dev-13-0

uv pip install --system --requirement requirements/build.txt

# Choose architectures supported by the CUDA toolkit in the image.
export TORCH_CUDA_ARCH_LIST="8.0 8.9 9.0 10.0 12.0"
CUDA_HOME=/usr/local/cuda \
uv build --wheel --no-build-isolation --out-dir /tmp/lmcache-wheel

# In the runtime image, install the wheel without changing the base dependency solve.
uv pip install --system --no-deps --force-reinstall /tmp/lmcache-wheel/lmcache-*.whl
```

Once LMCache publishes compatible CUDA 13 and/or aarch64 wheels, prefer those wheels over a source build.

## Aggregated Serving

### Configuration

LMCache is enabled using the `--kv-transfer-config` flag:

```bash
python -m dynamo.vllm --model <model_name> --kv-transfer-config '{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_both"}'
```

### Customization

LMCache configuration can be customized via environment variables listed [here](https://docs.lmcache.ai/api_reference/configurations.html).

For advanced configurations, LMCache supports multiple [storage backends](https://docs.lmcache.ai/index.html):

- **CPU RAM**: Fast local memory offloading
- **Local Storage**: Disk-based persistence
- **Redis**: Distributed cache sharing
- **GDS Backend**: GPU Direct Storage for high throughput
- **InfiniStore/Mooncake**: Cloud-native storage solutions

### Deployment

Use the provided launch script for quick setup:

```bash
./examples/backends/vllm/launch/agg_lmcache.sh
```

This will:
1. Start the Dynamo frontend
2. Launch a single vLLM worker with LMCache enabled

### Architecture for Aggregated Mode

In aggregated mode, the system uses:

- **KV Connector**: `LMCacheConnectorV1`
- **KV Role**: `kv_both` (handles both reading and writing)

## Disaggregated Serving

Disaggregated serving separates prefill and decode operations into dedicated workers. This provides better resource utilization and scalability for production deployments.

### Deployment

Use the provided disaggregated launch script (requires at least 2 GPUs):

```bash
./examples/backends/vllm/launch/disagg_lmcache.sh
```

This will:
1. Start the Dynamo frontend
2. Launch a decode worker on GPU 0
3. Wait for initialization
4. Launch a prefill worker on GPU 1 with LMCache enabled

### Worker Roles

#### Decode Worker

- **Purpose**: Handles token generation (decode phase)
- **GPU Assignment**: CUDA_VISIBLE_DEVICES=0
- **LMCache Config**: Uses `NixlConnector` only for KV transfer between prefill and decode workers

#### Prefill Worker

- **Purpose**: Handles prompt processing (prefill phase)
- **GPU Assignment**: CUDA_VISIBLE_DEVICES=1
- **LMCache Config**: Uses `MultiConnector` with both LMCache and NIXL connectors. This enables prefill worker to use LMCache for KV offloading and use NIXL for KV transfer between prefill and decode workers.
- **Flag**: `--disaggregation-mode prefill`

## Architecture

### KV Transfer Configuration

The system automatically configures KV transfer based on the deployment mode and worker type:

#### Prefill Worker (Disaggregated Mode)

```python
kv_transfer_config = KVTransferConfig(
    kv_connector="PdConnector",
    kv_role="kv_both",
    kv_connector_extra_config={
        "connectors": [
            {"kv_connector": "LMCacheConnectorV1", "kv_role": "kv_both"},
            {"kv_connector": "NixlConnector", "kv_role": "kv_both"}
        ]
    }
)
```

#### Decode Worker or Aggregated Mode

```python
kv_transfer_config = KVTransferConfig(
    kv_connector="LMCacheConnectorV1",
    kv_role="kv_both"
)
```

#### Fallback (No LMCache)

```python
kv_transfer_config = KVTransferConfig(
    kv_connector="NixlConnector",
    kv_role="kv_both"
)
```

### Integration Points

1. **Argument Parsing** (`args.py`):
   - Configures appropriate KV transfer settings
   - Sets up connector configurations based on worker type

2. **Engine Setup** (`main.py`):
   - Initializes LMCache environment variables
   - Creates vLLM engine with proper KV transfer config
   - Handles both aggregated and disaggregated modes

### Best Practices

1. **Chunk Size Tuning**: Adjust `LMCACHE_CHUNK_SIZE` based on your use case:
   - Smaller chunks (128-256): Better reuse granularity for varied content
   - Larger chunks (512-1024): More efficient for repetitive content patterns

2. **Memory Allocation**: Set `LMCACHE_MAX_LOCAL_CPU_SIZE` conservatively:
   - Leave sufficient RAM for other system processes
   - Monitor memory usage during peak loads

3. **Workload Optimization**: LMCache performs best with:
   - Repeated prompt patterns (RAG, multi-turn conversations)
   - Shared context across sessions
   - Long-running services with warm caches

## Metrics and Monitoring

When LMCache is enabled with `--kv-transfer-config '{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_both"}'` and `DYN_SYSTEM_PORT` is set, LMCache metrics are automatically exposed via Dynamo's `/metrics` endpoint alongside vLLM and Dynamo metrics.

**Requirements to access LMCache metrics:**

- `--kv-transfer-config '{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_both"}'` - Enables LMCache
- `DYN_SYSTEM_PORT=8081` - Enables metrics HTTP endpoint
- `PROMETHEUS_MULTIPROC_DIR` (optional) - If not set, Dynamo manages it internally

For detailed information on LMCache metrics, including the complete list of available metrics and how to access them, see the **[LMCache Metrics section](../backends/vllm/vllm-observability.md#lmcache-metrics)** in the vLLM Prometheus Metrics Guide.

## Troubleshooting

### LMCache log: `PrometheusLogger instance already created with different metadata`

You may see an error like:

```text
LMCache ERROR: PrometheusLogger instance already created with different metadata. This should not happen except in test
```

**Version note**: We reproduced this behavior with **vLLM v0.12.0**. We have not reproduced it with **vLLM v0.11.0**, so it may be specific to (or introduced in) v0.12.0.

This is emitted by LMCache when the LMCache connector is initialized more than once in the same process (for example, once for a `WORKER` role and later for a `SCHEDULER` role). LMCache uses a process-global singleton for its Prometheus logger, so the second initialization can log this warning if its metadata differs.

- **Impact**: This is a log-only error; in our testing it does not prevent vLLM/Dynamo from serving requests. If you care about LMCache metric labels, be aware the logger singleton uses the first-seen metadata.
- **Repro without Dynamo** (vLLM v0.12.0):

```bash
vllm serve Qwen/Qwen3-0.6B \
  --host 127.0.0.1 --port 18000 \
  --gpu-memory-utilization 0.24 \
  --enforce-eager \
  --no-enable-prefix-caching \
  --max-num-seqs 2 \
  --kv-offloading-backend lmcache \
  --kv-offloading-size 1 \
  --disable-hybrid-kv-cache-manager
```

- **Mitigation (silence)**: set `LMCACHE_LOG_LEVEL=CRITICAL`.
- **Upstream issue**: [vLLM issue #30996](https://github.com/vllm-project/vllm/issues/30996).

### vLLM log: `Found PROMETHEUS_MULTIPROC_DIR was set by user`

vLLM v1 uses `prometheus_client.multiprocess` and stores intermediate metric values in `PROMETHEUS_MULTIPROC_DIR`.

- If you **set `PROMETHEUS_MULTIPROC_DIR` yourself**, vLLM warns that the directory must be wiped between runs to avoid stale/incorrect metrics.
- When running via Dynamo, the vLLM wrapper may set `PROMETHEUS_MULTIPROC_DIR` internally to a temporary directory to avoid vLLM cleanup issues. If you still see the warning, confirm you are not exporting `PROMETHEUS_MULTIPROC_DIR` in your shell or container environment.

## References and Additional Resources

- [LMCache Documentation](https://docs.lmcache.ai/index.html) - Comprehensive guide and API reference
- [Configuration Reference](https://docs.lmcache.ai/api_reference/configurations.html) - Detailed configuration options
- [LMCache Observability Guide](https://docs.lmcache.ai/production/observability/vllm_endpoint.html) - Metrics and monitoring details
