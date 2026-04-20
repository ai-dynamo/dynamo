# kvbm-config

Centralized configuration for the KVBM (Key-Value Block Manager) v2 runtime.
Uses [Figment](https://docs.rs/figment) for hierarchical configuration merging
with support for TOML files, environment variables, JSON overrides, and
backward-compatible v1 `DYN_KVBM_*` environment variables.

## Configuration Priority

Sources are merged lowest-to-highest priority:

| Priority | Source | Description |
|----------|--------|-------------|
| 1 | Code defaults | `KvbmConfig::default()` |
| 2 | V1 env vars | `DYN_KVBM_*` compat layer (see below) |
| 3 | System TOML | `/opt/dynamo/etc/kvbm.toml` |
| 4 | User TOML | Path from `KVBM_CONFIG_PATH` env var |
| 5 | V2 env vars | `KVBM_*` prefixed (see below) |
| 6 | JSON override | Passed from Python via `from_figment_with_json()` |

## V2 Configuration Reference

Every config option, its type, default, and corresponding v2 environment variable.

### Tokio Runtime

| Config Path | Type | Default | Env Var | Description |
|---|---|---|---|---|
| `tokio.worker_threads` | `Option<usize>` | `1` | `KVBM_TOKIO_WORKER_THREADS` | Async worker threads. None = logical CPU count |
| `tokio.max_blocking_threads` | `Option<usize>` | None | `KVBM_TOKIO_MAX_BLOCKING_THREADS` | Blocking thread pool cap. None = Tokio default (512) |

### Rayon Thread Pool

| Config Path | Type | Default | Env Var | Description |
|---|---|---|---|---|
| `rayon.num_threads` | `Option<usize>` | None | `KVBM_RAYON_NUM_THREADS` | Rayon pool size. None = logical CPU count |

### Messenger (Transport + Discovery)

| Config Path | Type | Default | Env Var | Description |
|---|---|---|---|---|
| `messenger.init_timeout_secs` | `u64` | `1800` | `KVBM_MESSENGER_INIT_TIMEOUT_SECS` | Leader-worker init timeout (seconds) |
| `messenger.backend.tcp_addr` | `Option<String>` | None | `KVBM_MESSENGER_BACKEND_TCP_ADDR` | Bind IP address (mutually exclusive with tcp_interface) |
| `messenger.backend.tcp_interface` | `Option<String>` | None | `KVBM_MESSENGER_BACKEND_TCP_INTERFACE` | Bind network interface name (mutually exclusive with tcp_addr) |
| `messenger.backend.tcp_port` | `u16` | `0` | `KVBM_MESSENGER_BACKEND_TCP_PORT` | TCP port. 0 = OS-assigned ephemeral |
| `messenger.discovery` | `Option<DiscoveryConfig>` | None | | None = discovery disabled. Tagged enum: `etcd`, `p2p`, `filesystem` |

#### Etcd Discovery (`messenger.discovery.type = "etcd"`)

| Config Path | Type | Default | Env Var | Description |
|---|---|---|---|---|
| `messenger.discovery.cluster_id` | `String` | required | `KVBM_MESSENGER_DISCOVERY_CLUSTER_ID` | Key prefix for etcd discovery |
| `messenger.discovery.endpoints` | `Vec<String>` | `["http://localhost:2379"]` | | Etcd endpoint URLs |
| `messenger.discovery.ttl_secs` | `u64` | `60` | `KVBM_MESSENGER_DISCOVERY_TTL_SECS` | Lease TTL (10-600 seconds) |
| `messenger.discovery.operation_timeout_secs` | `u64` | `30` | | Per-operation timeout |
| `messenger.discovery.max_retries` | `u32` | `3` | | Operation retry count (0-10) |

#### P2P Discovery (`messenger.discovery.type = "p2p"`)

| Config Path | Type | Default | Description |
|---|---|---|---|
| `messenger.discovery.cluster_id` | `String` | required | Swarm key |
| `messenger.discovery.listen_port` | `u16` | `0` | 0 = OS-assigned |
| `messenger.discovery.bootstrap_peers` | `Vec<String>` | `[]` | Bootstrap peer addresses |
| `messenger.discovery.replication_factor` | `usize` | `3` | DHT replication factor |
| `messenger.discovery.enable_mdns` | `bool` | `false` | mDNS for local discovery |
| `messenger.discovery.record_ttl_secs` | `u64` | `600` | DHT record TTL |

#### Filesystem Discovery (`messenger.discovery.type = "filesystem"`)

| Config Path | Type | Default | Description |
|---|---|---|---|
| `messenger.discovery.path` | `PathBuf` | required | Path to discovery JSON file |

### NixL (RDMA Transfers)

Omit entire `[nixl]` section to disable NixL.

| Config Path | Type | Default | Env Var | Description |
|---|---|---|---|---|
| `nixl` | `Option<NixlConfig>` | None | | None = NixL disabled entirely |
| `nixl.backends` | `HashMap<String, HashMap<String, String>>` | `{UCX: {}, POSIX: {}}` | `KVBM_NIXL_BACKENDS` | Map of backend name (uppercase) to optional params |

Supported backends: `UCX`, `POSIX`, `GDS`, `GDS_MT`

### Cache Tiers

| Config Path | Type | Default | Env Var | Description |
|---|---|---|---|---|
| `cache.parallelism` | `ParallelismMode` | `tensor_parallel` | `KVBM_CACHE_PARALLELISM` | `tensor_parallel` (sharded) or `replicated_data` (MLA) |

#### Host Cache (G2 - Pinned CPU Memory)

| Config Path | Type | Default | Env Var | Description |
|---|---|---|---|---|
| `cache.host.cache_size_gb` | `Option<f64>` | None | `KVBM_CACHE_HOST_CACHE_SIZE_GB` | Cache size in GB. Converted to blocks |
| `cache.host.num_blocks` | `Option<usize>` | None | `KVBM_CACHE_HOST_NUM_BLOCKS` | Explicit block count (overrides cache_size_gb) |

#### Disk Cache (G3 - Persistent Storage)

Omit entire `[cache.disk]` section to disable disk caching.

| Config Path | Type | Default | Env Var | Description |
|---|---|---|---|---|
| `cache.disk` | `Option<DiskCacheConfig>` | None | | None = disk tier disabled |
| `cache.disk.cache_size_gb` | `Option<f64>` | None | `KVBM_CACHE_DISK_CACHE_SIZE_GB` | Cache size in GB. Converted to blocks |
| `cache.disk.num_blocks` | `Option<usize>` | None | `KVBM_CACHE_DISK_NUM_BLOCKS` | Explicit block count (overrides cache_size_gb) |
| `cache.disk.use_gds` | `bool` | `false` | | Enable GPUDirect Storage for direct GPU-disk transfers |
| `cache.disk.storage_path` | `Option<PathBuf>` | None | | Directory for disk cache files |

### Offload Policies

Policies control which blocks transfer between tiers. Multiple policies in the list use AND logic.

Policy types: `pass_all`, `presence`, `presence_lfu`

#### G1 to G2 (GPU to Host)

| Config Path | Type | Default | Env Var | Description |
|---|---|---|---|---|
| `offload.g1_to_g2.policies` | `Vec<PolicyType>` | `[]` | `KVBM_OFFLOAD_G1_TO_G2_POLICIES` | Empty = engine applies tier-specific defaults |
| `offload.g1_to_g2.presence` | `PresenceFilterConfig` | `{}` | | Config for `presence` policy (currently no params) |
| `offload.g1_to_g2.presence_lfu.min_lfu_count` | `u32` | `8` | | Min access count before offload |
| `offload.g1_to_g2.min_priority` | `i32` | `0` | `KVBM_OFFLOAD_G1_TO_G2_MIN_PRIORITY` | Prefix-contiguous offload priority threshold |
| `offload.g1_to_g2.max_concurrent_transfers` | `Option<usize>` | None | `KVBM_OFFLOAD_G1_TO_G2_MAX_CONCURRENT_TRANSFERS` | None = engine default (4) |
| `offload.g1_to_g2.max_batch_size` | `Option<usize>` | None | `KVBM_OFFLOAD_G1_TO_G2_MAX_BATCH_SIZE` | None = engine default (16) |

#### G2 to G3 (Host to Disk)

| Config Path | Type | Default | Env Var | Description |
|---|---|---|---|---|
| `offload.g2_to_g3.policies` | `Vec<PolicyType>` | `[]` | `KVBM_OFFLOAD_G2_TO_G3_POLICIES` | Empty = engine applies tier-specific defaults |
| `offload.g2_to_g3.presence` | `PresenceFilterConfig` | `{}` | | Config for `presence` policy |
| `offload.g2_to_g3.presence_lfu.min_lfu_count` | `u32` | `8` | | Min access count before offload |
| `offload.g2_to_g3.min_priority` | `i32` | `0` | `KVBM_OFFLOAD_G2_TO_G3_MIN_PRIORITY` | Prefix-contiguous offload priority threshold |
| `offload.g2_to_g3.max_concurrent_transfers` | `Option<usize>` | None | `KVBM_OFFLOAD_G2_TO_G3_MAX_CONCURRENT_TRANSFERS` | None = engine default |
| `offload.g2_to_g3.max_batch_size` | `Option<usize>` | None | `KVBM_OFFLOAD_G2_TO_G3_MAX_BATCH_SIZE` | None = engine default |

### Onboard Strategy

| Config Path | Type | Default | Env Var | Description |
|---|---|---|---|---|
| `onboard.mode` | `OnboardMode` | `inter` | `KVBM_ONBOARD_MODE` | `inter` (async between passes) or `intra` (sync per-layer) |

### Object Storage (G4 Tier)

Omit entire `[object]` section to disable object storage.

| Config Path | Type | Default | Description |
|---|---|---|---|
| `object` | `Option<ObjectConfig>` | None | None = object storage disabled |
| `object.client` | `ObjectClientConfig` | | Tagged enum: `s3` or `nixl` |

#### S3 Client (`object.client.type = "s3"`)

| Config Path | Type | Default | Description |
|---|---|---|---|
| `object.client.bucket` | `String` | `"kvbm-blocks"` | S3 bucket name |
| `object.client.region` | `String` | `"us-east-1"` | AWS region |
| `object.client.endpoint_url` | `Option<String>` | None | Custom endpoint (MinIO, etc.). None = AWS S3 |
| `object.client.force_path_style` | `bool` | `false` | Path-style URLs (required for MinIO) |
| `object.client.max_concurrent_requests` | `usize` | `16` | Concurrent S3 request limit |

#### NixL Client (`object.client.type = "nixl"`)

| Config Path | Type | Default | Description |
|---|---|---|---|
| `object.client.backend` | | | Tagged enum: `s3` (same fields as S3 Client above) |

### Events

| Config Path | Type | Default | Env Var | Description |
|---|---|---|---|---|
| `events.enabled` | `bool` | `false` | `KVBM_EVENTS_ENABLED` | Enable event publishing |
| `events.subject` | `String` | `"kvbm.events"` | `KVBM_EVENTS_SUBJECT` | NATS subject for publishing |
| `events.channel_capacity` | `usize` | `1024` | `KVBM_EVENTS_CHANNEL_CAPACITY` | Broadcast channel buffer (16-65536) |
| `events.policy` | `EventPolicyConfig` | `power_of_two` | `KVBM_EVENTS_POLICY` | `power_of_two` (sparse sampling) or `all` |
| `events.batching.window_duration_ms` | `u64` | `10` | `KVBM_EVENTS_BATCHING_WINDOW_DURATION_MS` | Flush interval (1-10000 ms) |
| `events.batching.max_batch_size` | `usize` | `1024` | `KVBM_EVENTS_BATCHING_MAX_BATCH_SIZE` | Max events per batch (1-65536) |

### Metrics

| Config Path | Type | Default | Env Var | Description |
|---|---|---|---|---|
| `metrics.enabled` | `bool` | `false` | `KVBM_METRICS_ENABLED` | Enable metrics endpoint |
| `metrics.port` | `u16` | `6880` | `KVBM_METRICS_PORT` | Metrics endpoint port |
| `metrics.cache_stats_max_requests` | `usize` | `1000` | `KVBM_METRICS_CACHE_STATS_MAX_REQUESTS` | Sliding window size for cache hit rate |
| `metrics.cache_stats_log_interval_secs` | `u64` | `5` | `KVBM_METRICS_CACHE_STATS_LOG_INTERVAL_SECS` | Stats logging interval |

### Debug

| Config Path | Type | Default | Env Var | Description |
|---|---|---|---|---|
| `debug.recording` | `bool` | `false` | `KVBM_DEBUG_RECORDING` | Enable KVBM recording for replay/debugging |

## TOML Example

```toml
[tokio]
worker_threads = 4

[messenger]
init_timeout_secs = 600

[messenger.backend]
tcp_addr = "0.0.0.0"
tcp_port = 9000

[messenger.discovery]
type = "filesystem"
path = "/tmp/kvbm-discovery.json"

[nixl.backends.UCX]
[nixl.backends.POSIX]

[cache]
parallelism = "tensor_parallel"

[cache.host]
cache_size_gb = 4.0

[cache.disk]
cache_size_gb = 20.0
storage_path = "/mnt/nvme/kv_cache"

[offload.g1_to_g2]
policies = ["presence"]

[offload.g2_to_g3]
policies = ["presence_lfu"]
[offload.g2_to_g3.presence_lfu]
min_lfu_count = 4

[onboard]
mode = "inter"

[events]
enabled = true
subject = "kvbm.events"

[metrics]
enabled = true
port = 9090
```

## JSON Override (from Python)

The primary integration point for vLLM's `kv_connector_extra_config`:

```python
extra_config = {
    "tokio": {"worker_threads": 8},
    "cache": {
        "host": {"cache_size_gb": 4.0},
        "parallelism": "replicated_data"
    },
    "messenger": {
        "backend": {"tcp_port": 9000}
    }
}
```

Profile-based JSON (leader vs worker get different values):

```python
extra_config = {
    "default": {"cache": {"host": {"cache_size_gb": 4.0}}},
    "leader": {"tokio": {"worker_threads": 2}},
    "worker": {"tokio": {"worker_threads": 8}}
}
```

## V1 Environment Variable Compatibility

The `V1EnvCompat` provider reads legacy `DYN_KVBM_*` environment variables and maps
them to the v2 config structure. This allows v2 to run with existing v1 deployments
without changing any environment configuration.

V1 env vars have **lower priority** than v2 env vars, TOML files, and JSON overrides.
If both `DYN_KVBM_CPU_CACHE_GB` and `KVBM_CACHE_HOST_CACHE_SIZE_GB` are set, the
v2 variable wins.

### Direct Mappings

| V1 Env Var | V2 Config Path | Type |
|---|---|---|
| `DYN_KVBM_CPU_CACHE_GB` | `cache.host.cache_size_gb` | f64 |
| `DYN_KVBM_CPU_CACHE_OVERRIDE_NUM_BLOCKS` | `cache.host.num_blocks` | usize |
| `DYN_KVBM_DISK_CACHE_GB` | `cache.disk.cache_size_gb` | f64 |
| `DYN_KVBM_DISK_CACHE_OVERRIDE_NUM_BLOCKS` | `cache.disk.num_blocks` | usize |
| `DYN_KVBM_DISK_CACHE_DIR` | `cache.disk.storage_path` | string |
| `DYN_KVBM_METRICS` | `metrics.enabled` | bool |
| `DYN_KVBM_METRICS_PORT` | `metrics.port` | u16 |
| `DYN_KVBM_CACHE_STATS_MAX_REQUESTS` | `metrics.cache_stats_max_requests` | usize |
| `DYN_KVBM_CACHE_STATS_LOG_INTERVAL_SECS` | `metrics.cache_stats_log_interval_secs` | u64 |
| `DYN_KVBM_ENABLE_RECORD` | `debug.recording` | bool |
| `DYN_KVBM_LEADER_WORKER_INIT_TIMEOUT_SECS` | `messenger.init_timeout_secs` | u64 |
| `DYN_KVBM_HOST_OFFLOAD_PREFIX_MIN_PRIORITY` | `offload.g1_to_g2.min_priority` | i32 |
| `DYN_KVBM_MAX_CONCURRENT_TRANSFERS` | `offload.g1_to_g2.max_concurrent_transfers` | usize |
| `DYN_KVBM_MAX_TRANSFER_BATCH_SIZE` | `offload.g1_to_g2.max_batch_size` | usize |
| `DYN_KVBM_TRANSFER_BATCH_SIZE` | `offload.g1_to_g2.max_batch_size` | usize (fallback) |
| `DYN_KVBM_KV_EVENTS_ENABLE_CONSOLIDATOR` | `events.enabled` | bool |

### Semantic Mappings

| V1 Env Var | V2 Config Path | Transformation |
|---|---|---|
| `DYN_KVBM_NCCL_MLA_MODE=true` | `cache.parallelism` | `true` sets `"replicated_data"` |
| `DYN_KVBM_DISABLE_DISK_OFFLOAD_FILTER=true` | `offload.g1_to_g2.policies` | `true` sets `["pass_all"]` |
| `DYN_KVBM_NIXL_BACKEND_<NAME>=true` | `nixl.backends` | Each `true` backend is added |
| `DYN_KVBM_OBJECT_ENABLED=1` | `object` | Synthesizes `ObjectConfig` from `_BUCKET`, `_ENDPOINT`, `_REGION` |

### Deprecated (Ignored with Warning)

These v1 env vars have no v2 equivalent — ZMQ transport was replaced by Velo:

- `DYN_KVBM_LEADER_ZMQ_HOST`
- `DYN_KVBM_LEADER_ZMQ_PUB_PORT`
- `DYN_KVBM_LEADER_ZMQ_ACK_PORT`
- `DYN_KVBM_TRTLLM_ZMQ_PORT`

## Rust API

```rust
use kvbm_config::KvbmConfig;

// Load from env + files (standard path)
let config = KvbmConfig::from_env()?;

// Load with JSON override (from Python)
let config = KvbmConfig::from_figment_with_json(json_str)?;

// Load with profile selection (leader vs worker)
let leader = KvbmConfig::from_figment_with_json_for_leader(json_str)?;
let worker = KvbmConfig::from_figment_with_json_for_worker(json_str)?;

// Custom figment composition
let config = KvbmConfig::extract_from(
    KvbmConfig::figment()
        .merge(("cache.host.cache_size_gb", 4.0f64))
)?;
```
