# kvbm-config

Configuration library for KVBM (KV Block Manager). Provides centralized, validated configuration for all KVBM components including Tokio, Rayon, Nova transport, NixL backends, cache tiers, and offload policies.

## Quick Start

### Using Environment Variables

```bash
# Set cache size
export KVBM_CACHE_HOST_SIZE_GB=4.0

# Set Tokio threads
export KVBM_TOKIO_WORKER_THREADS=4

# Load from a custom config file
export KVBM_CONFIG_PATH=/path/to/my-kvbm.toml
```

### Using TOML Config File

Create `/opt/dynamo/etc/kvbm.toml` or set `KVBM_CONFIG_PATH`:

```toml
[tokio]
worker_threads = 4

[cache.host]
cache_size_gb = 4.0

[offload.g2_to_g3.presence_lfu]
min_lfu_count = 16
```

### Using JSON (vLLM Integration)

Pass JSON to `kv_connector_extra_config` with `leader` and `worker` profile keys:

```python
extra_config = {
    "leader": {
        "cache": {"host": {"cache_size_gb": 2.0}},
        "tokio": {"worker_threads": 2},
        "nova": {
            "discovery": {
                "type": "filesystem",
                "path": "/tmp/nova-discovery/cluster.json"
            }
        },
        "object": {
            "client": {
                "type": "s3",
                "endpoint_url": "http://minio:9000",
                "bucket": "kvbm-blocks",
                "region": "us-east-1",
                "force_path_style": True,
                "max_concurrent_requests": 16
            }
        }
    },
    "worker": {
        "nixl": {"backends": {"UCX": {}, "POSIX": {}}},
        "tokio": {"worker_threads": 1}
    }
}
```

## Sample Config Files

| File | Description |
|------|-------------|
| [`kvbm.example.toml`](kvbm.example.toml) | Full TOML config with all options documented |
| [`kvbm.example.json`](kvbm.example.json) | Minimal JSON config showing defaults |
| [`kvbm.full.example.json`](kvbm.full.example.json) | Comprehensive JSON reference with all options |

## Configuration Loading Priority

Configuration sources are merged in this order (lowest to highest priority):

1. **Code defaults** - Built-in Rust struct defaults
2. **System config** - `/opt/dynamo/etc/kvbm.toml`
3. **User config** - File at `KVBM_CONFIG_PATH`
4. **Environment variables** - `KVBM_*` prefixed
5. **JSON overrides** - From `kv_connector_extra_config` or programmatic

## Configuration Reference

### Tokio Runtime

Async runtime configuration for Nova transport and background tasks.

| Field | Type | Default | Env Var | Description |
|-------|------|---------|---------|-------------|
| `worker_threads` | `usize` | `1` | `KVBM_TOKIO_WORKER_THREADS` | Number of async worker threads |
| `max_blocking_threads` | `usize` | `512` | `KVBM_TOKIO_MAX_BLOCKING_THREADS` | Max blocking thread pool size |

### Rayon Thread Pool

CPU-bound parallel work (tensor operations, compression).

| Field | Type | Default | Env Var | Description |
|-------|------|---------|---------|-------------|
| `num_threads` | `usize` | CPU count | `KVBM_RAYON_NUM_THREADS` | Number of Rayon threads |

### Nova Transport

High-performance RPC transport for KVBM.

| Field | Type | Default | Env Var | Description |
|-------|------|---------|---------|-------------|
| `backend.tcp_port` | `u16` | `0` | `KVBM_NOVA_BACKEND_TCP_PORT` | TCP port (0 = OS-assigned) |
| `backend.tcp_addr` | `string` | `None` | `KVBM_NOVA_BACKEND_TCP_ADDR` | IP address to bind |
| `backend.tcp_interface` | `string` | `None` | `KVBM_NOVA_BACKEND_TCP_INTERFACE` | Network interface name |

> **Note:** `tcp_addr` and `tcp_interface` are mutually exclusive. If neither is set, binds to `0.0.0.0`.

#### Discovery (Optional)

Choose one discovery method for multi-node setups:

**Etcd Discovery:**
```toml
[nova.discovery.etcd]
cluster_id = "my-cluster"
endpoints = ["http://localhost:2379"]
ttl_secs = 60
```

**P2P Discovery:**
```toml
[nova.discovery.p2p]
cluster_id = "my-cluster"
bootstrap_peers = ["192.168.1.10:5000"]
```

**Filesystem Discovery:**
```toml
[nova.discovery.filesystem]
path = "/tmp/kvbm-discovery"
```

### Cache Configuration

#### Host Cache (G2 Tier)

CPU memory cache for KV blocks offloaded from GPU.

| Field | Type | Default | Env Var | Description |
|-------|------|---------|---------|-------------|
| `cache_size_gb` | `f64` | `None` | `KVBM_CACHE_HOST_SIZE_GB` | Cache size in GB |
| `num_blocks` | `usize` | `None` | `KVBM_CACHE_HOST_NUM_BLOCKS` | Explicit block count (priority) |

> **Note:** `num_blocks` takes priority over `cache_size_gb` if both are set.

#### Disk Cache (G3 Tier)

Local storage cache for overflow from host memory.

| Field | Type | Default | Env Var | Description |
|-------|------|---------|---------|-------------|
| `cache_size_gb` | `f64` | `None` | `KVBM_CACHE_DISK_SIZE_GB` | Cache size in GB |
| `num_blocks` | `usize` | `None` | `KVBM_CACHE_DISK_NUM_BLOCKS` | Explicit block count (priority) |
| `use_gds` | `bool` | `false` | - | Use GPUDirect Storage |
| `storage_path` | `path` | `None` | - | Directory for cache files |

### Offload Policies

Controls how blocks move between storage tiers. Policies are evaluated in order with AND logic (all must pass).

#### Available Policies

| Policy | Description |
|--------|-------------|
| `pass_all` | No filtering, all blocks pass |
| `presence` | Skip blocks already in destination tier |
| `presence_lfu` | Presence check + minimum access count threshold |

#### G1 → G2 (GPU → Host)

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `policies` | `list` | `["presence"]` | Policies to apply |

**Default behavior:** Prevents duplicate transfers when the same sequence is enqueued multiple times.

#### G2 → G3 (Host → Disk)

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `policies` | `list` | `["presence_lfu"]` | Policies to apply |
| `presence_lfu.min_lfu_count` | `u32` | `8` | Minimum access count for offload |

**Default behavior:** Only offloads "hot" blocks that have been accessed at least 8 times, preventing disk thrashing for rarely-used blocks.

### Object Storage (G4 Tier)

Remote object storage for persistent KV cache sharing across instances.

#### S3 Configuration (JSON)

```json
{
  "object": {
    "client": {
      "type": "s3",
      "endpoint_url": "http://minio:9000",
      "bucket": "kvbm-blocks",
      "region": "us-east-1",
      "force_path_style": true,
      "max_concurrent_requests": 16
    }
  }
}
```

#### S3 Configuration (TOML)

```toml
[object.client]
type = "s3"
bucket = "kvbm-blocks"
region = "us-east-1"
# endpoint_url = "http://localhost:9000"  # For MinIO
# force_path_style = true                  # Required for MinIO
max_concurrent_requests = 16
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `type` | `string` | - | Client type: `"s3"` or `"nixl"` |
| `endpoint_url` | `string` | `None` | S3 endpoint (None = AWS S3) |
| `bucket` | `string` | `"kvbm-blocks"` | S3 bucket name |
| `region` | `string` | `"us-east-1"` | AWS region |
| `force_path_style` | `bool` | `false` | Use path-style URLs (for MinIO) |
| `max_concurrent_requests` | `usize` | `16` | Max concurrent S3 requests |

### NixL Backends

High-performance data transfer using NixL.

```toml
[nixl.backends]
UCX = {}      # Unified Communication X
POSIX = {}    # Standard POSIX I/O
# GDS = {}    # GPUDirect Storage
```

**Default backends:** `UCX` and `POSIX` are enabled by default.

## Profile-Based Configuration

vLLM uses profile-based configuration with `leader` and `worker` top-level keys. The leader process manages coordination, discovery, and object storage offload. Workers handle data transfers using NixL backends.

### Typical Leader vs Worker Differences

| Setting | Leader | Worker |
|---------|--------|--------|
| `cache.host` | Larger (manages metadata) | Smaller or same |
| `tokio.worker_threads` | Fewer (coordination) | More (data transfer) |
| `nixl.backends` | Optional | Required (UCX, POSIX) |
| `nova.discovery` | Required | Often not needed |
| `object` | Required (S3 config) | Inherited or separate |

### Complete Example

```json
{
  "leader": {
    "cache": { "host": { "cache_size_gb": 4.0 } },
    "tokio": { "worker_threads": 2 },
    "nova": {
      "discovery": {
        "type": "filesystem",
        "path": "/tmp/nova-discovery/cluster.json"
      }
    },
    "offload": {
      "g1_to_g2": { "policies": ["presence"] },
      "g2_to_g3": {
        "policies": ["presence_lfu"],
        "presence_lfu": { "min_lfu_count": 8 }
      }
    },
    "object": {
      "client": {
        "type": "s3",
        "endpoint_url": "http://minio:9000",
        "bucket": "kvbm-blocks",
        "region": "us-east-1",
        "force_path_style": true,
        "max_concurrent_requests": 16
      }
    }
  },
  "worker": {
    "nixl": { "backends": { "UCX": {}, "POSIX": {} } },
    "tokio": { "worker_threads": 1 },
    "cache": { "host": { "cache_size_gb": 2.0 } }
  },
  "default": {
    "nova": { "backend": { "tcp_port": 0 } }
  }
}
```

### Loading in Rust

```rust
// Leader gets leader profile values
let config = KvbmConfig::from_figment_with_json_for_leader(json)?;

// Worker gets worker profile values
let config = KvbmConfig::from_figment_with_json_for_worker(json)?;
```

## Validation

All configuration is validated on load:

| Config | Field | Constraint |
|--------|-------|------------|
| Tokio | `worker_threads` | 1 ≤ x ≤ CPU count |
| Tokio | `max_blocking_threads` | ≥ 1 |
| Rayon | `num_threads` | ≥ 1 |
| Offload | `min_lfu_count` | ≥ 1 |
| Etcd | `ttl_secs` | 10 ≤ x ≤ 600 |
| Etcd | `max_retries` | 0 ≤ x ≤ 10 |

## Environment Variable Reference

| Variable | Maps To |
|----------|---------|
| `KVBM_CONFIG_PATH` | Path to TOML config file |
| `KVBM_TOKIO_WORKER_THREADS` | `tokio.worker_threads` |
| `KVBM_TOKIO_MAX_BLOCKING_THREADS` | `tokio.max_blocking_threads` |
| `KVBM_RAYON_NUM_THREADS` | `rayon.num_threads` |
| `KVBM_NOVA_BACKEND_TCP_PORT` | `nova.backend.tcp_port` |
| `KVBM_NOVA_BACKEND_TCP_ADDR` | `nova.backend.tcp_addr` |
| `KVBM_NOVA_BACKEND_TCP_INTERFACE` | `nova.backend.tcp_interface` |
| `KVBM_CACHE_HOST_SIZE_GB` | `cache.host.cache_size_gb` |
| `KVBM_CACHE_HOST_NUM_BLOCKS` | `cache.host.num_blocks` |
| `KVBM_CACHE_DISK_SIZE_GB` | `cache.disk.cache_size_gb` |
| `KVBM_CACHE_DISK_NUM_BLOCKS` | `cache.disk.num_blocks` |
