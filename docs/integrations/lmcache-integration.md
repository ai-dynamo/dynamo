---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: LMCache
---

## Introduction

LMCache is a high-performance KV cache layer that supercharges LLM serving by enabling **prefill-once, reuse-everywhere** semantics. As described in the [official documentation](https://docs.lmcache.ai/index.html), LMCache lets LLMs prefill each text only once by storing the KV caches of all reusable texts, allowing reuse of KV caches for any reused text (not necessarily prefix) across any serving engine instance.

This document describes how LMCache is integrated into Dynamo's vLLM backend to provide enhanced performance and memory efficiency.

## Installation Notes

Dynamo's vLLM runtime expects LMCache to be present in the same Python environment. On supported environments (x86_64, Python 3.10-3.13, PyTorch built against CUDA 12.x), the published wheel installs directly:

```bash
uv pip install lmcache
```

LMCache only publishes x86_64 manylinux wheels linked against CUDA 12. For aarch64 hosts, or hosts running PyTorch built against a different CUDA major version, build LMCache from source against your matching torch + CUDA stack — see the official [LMCache installation guide](https://docs.lmcache.ai/getting_started/installation.html).

LMCache is compatible with vLLM 0.20.0 after [PR](https://github.com/LMCache/LMCache/pull/3282).

## Aggregated Serving

### Configuration

LMCache runs the cache engine as an out-of-process sidecar (`lmcache server`); the Dynamo worker connects to it via the `LMCacheMPConnector`. Start the sidecar, then launch the worker:

```bash
lmcache server --l1-size-gb 100 --eviction-policy LRU &

python -m dynamo.vllm \
  --model <model_name> \
  --disable-hybrid-kv-cache-manager \
  --kv-transfer-config '{"kv_connector":"LMCacheMPConnector","kv_role":"kv_both"}'
```

### Customization

The LMCache MP server is configured via CLI arguments. See the [Configuration Reference](https://docs.lmcache.ai/mp/configuration.html) for the full list of `lmcache server` flags.

LMCache MP uses a two-tier storage architecture: an in-memory L1 cache (sized with `--l1-size-gb`) plus optional persistent L2 adapters configured with `--l2-adapter`. The supported [L2 storage backends](https://docs.lmcache.ai/mp/l2_storage.html) are:

- **POSIX**: Standard POSIX file I/O on any file system
- **GDS** / **GDS_MT**: NVIDIA GPU Direct Storage (single- and multi-threaded), bypassing the CPU for NVMe SSDs that support GDS
- **HF3FS**: Distributed / shared file-system backend
- **OBJ**: Object store backend
- **AZURE_BLOB**: Azure Blob Storage

### Deployment

Use the provided launch script for quick setup:

```bash
./examples/backends/vllm/launch/agg_lmcache_mp.sh
```

This will:
1. Start the LMCache MP server
2. Start the Dynamo frontend
3. Launch a single vLLM worker with `LMCacheMPConnector` connected to the sidecar

### Architecture for Aggregated Mode

In aggregated mode, the system uses:

- **KV Connector**: `LMCacheMPConnector`
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

#### Aggregated Mode

```python
kv_transfer_config = KVTransferConfig(
    kv_connector="LMCacheMPConnector",
    kv_role="kv_both"
)
```

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

#### Decode Worker (Disaggregated Mode)

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
   - Creates vLLM engine with proper KV transfer config
   - Handles both aggregated and disaggregated modes

3. **Sidecar Lifecycle** (launch script):
   - Starts the `lmcache server` process before the Dynamo worker
   - Tears it down on exit via the script's cleanup trap

### Best Practices

1. **Chunk Size Tuning**: Pass `--chunk-size` to `lmcache server` based on your use case:
   - Smaller chunks (128-256): Better reuse granularity for varied content
   - Larger chunks (512-1024): More efficient for repetitive content patterns

2. **Memory Allocation**: Set `--l1-size-gb` on `lmcache server` conservatively:
   - Leave sufficient RAM for other system processes
   - Monitor memory usage during peak loads

3. **Workload Optimization**: LMCache performs best with:
   - Repeated prompt patterns (RAG, multi-turn conversations)
   - Shared context across sessions
   - Long-running services with warm caches

## Metrics and Monitoring

The LMCache MP server records metrics through the OpenTelemetry SDK and exposes them on its own HTTP admin port (default `:8080/metrics`), prefixed `lmcache_mp_`:

```bash
curl -s localhost:8080/metrics | grep '^lmcache_mp_'
```

vLLM and Dynamo metrics remain on Dynamo's `:8081/metrics` (set `DYN_SYSTEM_PORT=8081` on the worker to enable that endpoint).

For detailed information on LMCache metrics, including the complete list of available metrics and how to access them, see the **[LMCache Metrics section](../backends/vllm/vllm-observability.md#lmcache-metrics)** in the vLLM Prometheus Metrics Guide.

## Troubleshooting

### vLLM log: `Found PROMETHEUS_MULTIPROC_DIR was set by user`

vLLM v1 uses `prometheus_client.multiprocess` and stores intermediate metric values in `PROMETHEUS_MULTIPROC_DIR`.

- If you **set `PROMETHEUS_MULTIPROC_DIR` yourself**, vLLM warns that the directory must be wiped between runs to avoid stale/incorrect metrics.
- When running via Dynamo, the vLLM wrapper may set `PROMETHEUS_MULTIPROC_DIR` internally to a temporary directory to avoid vLLM cleanup issues. If you still see the warning, confirm you are not exporting `PROMETHEUS_MULTIPROC_DIR` in your shell or container environment.

## References and Additional Resources

- [LMCache Documentation](https://docs.lmcache.ai/index.html) - Comprehensive guide and API reference
- [Configuration Reference](https://docs.lmcache.ai/mp/configuration.html) - `lmcache server` CLI arguments
- [LMCache Observability Guide](https://docs.lmcache.ai/mp/observability.html) - Metrics and monitoring details
