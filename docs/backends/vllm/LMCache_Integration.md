# LMCache Integration in Dynamo

## Introduction

LMCache is a high-performance KV cache layer that supercharges LLM serving by enabling **prefill-once, reuse-everywhere** semantics. As described in the [official documentation](https://docs.lmcache.ai/index.html), LMCache lets LLMs prefill each text only once by storing the KV caches of all reusable texts, allowing reuse of KV caches for any reused text (not necessarily prefix) across any serving engine instance.

This document describes how LMCache is integrated into Dynamo's vLLM backend to provide enhanced performance and memory efficiency.

### Key Benefits
- **Reduced Time to First Token (TTFT)**: Eliminates redundant prefill computations
- **Memory Offloading**: Intelligent KV cache placement across CPU/GPU/storage tiers
- **Improved Throughput**: Reduced GPU memory pressure enables higher batch sizes

## Platform Support

**Important Note**: LMCache integration currently only supports x86 architecture. ARM64 is not supported at this time.

## Aggregated Serving


### Configuration

LMCache is enabled using the `--connector lmcache` flag:

```bash
python -m dynamo.vllm --model <model_name> --connector lmcache
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
1. Start the dynamo frontend
2. Launch a single vLLM worker with LMCache enabled

### Architecture for Aggregated Mode

In aggregated mode, the system uses:
- **KV Connector**: `LMCacheConnectorV1`
- **KV Role**: `kv_both` (handles both reading and writing)

## Disaggregated Serving

Disaggregated serving separates prefill and decode operations into dedicated workers. This provides better resource utilization and scalability for production deployments.

### Deployment

Use the provided disaggregated launch script(the script requires at least 2 GPUs):

```bash
./examples/backends/vllm/launch/disagg_lmcache.sh
```

This will:
1. Start the dynamo frontend
2. Launch a decode worker on GPU 0
3. Wait for initialization
4. Launch a prefill worker on GPU 1 with LMCache enabled

### Worker Roles

#### Decode Worker
- **Purpose**: Handles token generation (decode phase)
- **GPU Assignment**: CUDA_VISIBLE_DEVICES=0
- **LMCache Config**: Uses `NixlConnector` only for kv transfer between prefill and decode workers

#### Prefill Worker
- **Purpose**: Handles prompt processing (prefill phase)
- **GPU Assignment**: CUDA_VISIBLE_DEVICES=1
- **LMCache Config**: Uses `MultiConnector` with both LMCache and NIXL connectors. This enables prefill worker to use LMCache for kv offloading and use NIXL for kv transfer between prefill and decode workers.
- **Flag**: `--is-prefill-worker`

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

When LMCache is enabled with `--connector lmcache` and `DYN_SYSTEM_PORT` is set, LMCache metrics are automatically exposed via Dynamo's `/metrics` endpoint alongside vLLM and Dynamo metrics.

**Requirements to access LMCache metrics:**
- `--connector lmcache` - Enables LMCache
- `DYN_SYSTEM_PORT=8081` - Enables metrics HTTP endpoint

For detailed information on LMCache metrics, including the complete list of available metrics and how to access them, see the **[LMCache Metrics section](prometheus.md#lmcache-metrics)** in the vLLM Prometheus Metrics Guide.

## References and Additional Resources

- [LMCache Documentation](https://docs.lmcache.ai/index.html) - Comprehensive guide and API reference
- [Configuration Reference](https://docs.lmcache.ai/api_reference/configurations.html) - Detailed configuration options
- [LMCache Observability Guide](https://docs.lmcache.ai/production/observability/vllm_endpoint.html) - Metrics and monitoring details

