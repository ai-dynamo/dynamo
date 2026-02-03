# AGENTS.md - KVBM Component

This file provides guidance to AI agents when working with the KVBM (KV Block Manager) component.

## Overview

KVBM is Dynamo's distributed KV cache block management system for scalable LLM inference. It cleanly separates memory management from inference runtimes (vLLM, TensorRT-LLM), enabling GPU↔CPU↔Disk/Remote tiering, asynchronous block offload/onboard, and efficient block reuse. KVBM uses NIXL for high-performance data transfers and integrates with etcd for leader/worker coordination.

## Code Location

```
lib/bindings/kvbm/
├── Cargo.toml                  # Rust package manifest
├── Cargo.lock                  # Rust dependency lock
├── pyproject.toml              # Python package configuration (maturin build)
├── README.md                   # Standalone KVBM usage guide
├── LICENSE                     # Apache-2.0 license
├── python/
│   └── kvbm/
│       ├── __init__.py         # Public API (BlockManager, KvbmLeader, KvbmWorker)
│       ├── _core.pyi           # Type stubs for Rust bindings
│       ├── utils.py            # Utility functions (runtime detection)
│       ├── vllm_integration/   # vLLM connector implementation
│       │   ├── __init__.py
│       │   ├── connector/
│       │   │   ├── __init__.py
│       │   │   ├── dynamo_connector.py    # Main vLLM KVConnectorBase_V1 implementation
│       │   │   └── pd_connector.py        # Prefill/Decode multi-connector
│       │   ├── connector_leader.py        # Scheduler-side KVBM logic
│       │   ├── connector_worker.py        # Worker-side KVBM logic
│       │   ├── consolidator_config.py     # KV events consolidator config
│       │   ├── kv_cache_manager.py        # KV cache manager interface
│       │   ├── kv_cache_utils.py          # Cache block utilities
│       │   └── rust.py                    # Rust binding wrappers
│       └── trtllm_integration/  # TensorRT-LLM connector implementation
│           ├── __init__.py
│           ├── connector/
│           │   ├── __init__.py
│           │   ├── kvbm_connector_leader.py   # TRT-LLM scheduler connector
│           │   └── kvbm_connector_worker.py   # TRT-LLM worker connector
│           ├── consolidator_config.py
│           └── rust.py
└── src/                        # Rust core implementation
    ├── lib.rs                  # Library entry point, PyO3 bindings
    ├── block_manager.rs        # Main BlockManager module
    └── block_manager/
        ├── block.rs            # Block data structures and state machine
        ├── block_list.rs       # Block list management
        ├── cache_stats.rs      # Cache statistics and metrics
        ├── controller.rs       # Block controller logic
        ├── dlpack.rs           # DLPack tensor interop
        ├── layer.rs            # Layer-level block operations
        ├── distributed.rs      # Distributed coordination
        ├── distributed/
        │   ├── leader.rs       # Distributed leader logic
        │   ├── worker.rs       # Distributed worker logic
        │   └── utils.rs        # Distributed utilities
        ├── vllm.rs             # vLLM-specific block management
        └── vllm/
            ├── block_list.rs   # vLLM block list
            ├── request.rs      # Request handling
            ├── slot.rs         # Slot management
            ├── connector.rs    # Connector base
            └── connector/
                ├── leader.rs           # Leader connector
                ├── worker.rs           # Worker connector
                ├── trtllm_leader.rs    # TRT-LLM leader
                ├── trtllm_worker.rs    # TRT-LLM worker
                └── leader/
                    ├── recorder.rs     # Event recording
                    └── slot.rs         # Slot state
```

## Key Commands

### Running Tests
```bash
# KVBM integration tests
pytest tests/kvbm_integration/

# Specific test file
pytest tests/kvbm_integration/test_kvbm.py -v

# Determinism tests (requires GPU)
pytest tests/kvbm_integration/test_determinism_agg.py -v

# With coverage
pytest tests/kvbm_integration/ --cov=kvbm
```

### Building KVBM
```bash
# Build from source (inside Dynamo container)
cd /workspace/lib/bindings/kvbm
uv pip install maturin[patchelf]
maturin build --release --out /workspace/dist
uv pip install --upgrade --force-reinstall --no-deps /workspace/dist/kvbm*.whl

# Build via Docker (from repo root)
./container/build.sh --framework none --enable-kvbm --tag local-kvbm
```

### Running KVBM with vLLM
```bash
# Start etcd for leader/worker coordination
docker compose -f deploy/docker-compose.yml up -d

# Set cache tier configuration
export DYN_KVBM_CPU_CACHE_GB=4

# Run vLLM with KVBM connector
vllm serve --kv-transfer-config '{"kv_connector":"DynamoConnector","kv_role":"kv_both","kv_connector_module_path":"kvbm.vllm_integration.connector"}' Qwen/Qwen3-0.6B

# Or via Dynamo
python -m dynamo.vllm --model Qwen/Qwen3-0.6B --connector kvbm
```

### Running KVBM with TensorRT-LLM
```bash
# Create LLM API config
cat > /tmp/kvbm_llm_api_config.yaml <<EOF
backend: pytorch
kv_cache_config:
  enable_partial_reuse: false
  free_gpu_memory_fraction: 0.80
kv_connector_config:
  connector_module: kvbm.trtllm_integration.connector
  connector_scheduler_class: DynamoKVBMConnectorLeader
  connector_worker_class: DynamoKVBMConnectorWorker
EOF

# Set cache configuration
export DYN_KVBM_CPU_CACHE_GB=4

# Run TRT-LLM with KVBM
trtllm-serve Qwen/Qwen3-0.6B --host localhost --port 8000 --backend pytorch --extra_llm_api_options /tmp/kvbm_llm_api_config.yaml
```

## Architecture

### Entry Flow (vLLM)
```
DynamoConnector (dynamo_connector.py)
    ├── role == SCHEDULER → KvConnectorLeader (connector_leader.py)
    │   ├── Creates KvbmLeader (Rust binding)
    │   ├── Manages slot creation/tracking
    │   ├── get_num_new_matched_tokens() → Check for cache hits
    │   ├── update_state_after_alloc() → Track allocated blocks
    │   ├── build_connector_meta() → Serialize metadata for workers
    │   └── request_finished() → Trigger offload decisions
    └── role == WORKER → KvConnectorWorker (connector_worker.py)
        ├── Creates KvbmWorker (Rust binding)
        ├── register_kv_caches() → Register GPU memory with NIXL
        ├── bind_connector_metadata() → Receive scheduler decisions
        ├── start_load_kv() → Begin async onboarding from CPU/disk
        └── save_kv_layer() → Offload KV data to CPU/disk
```

### Entry Flow (TensorRT-LLM)
```
DynamoKVBMConnectorLeader (kvbm_connector_leader.py)
    ├── Implements KvCacheConnectorScheduler interface
    ├── Creates KvbmLeader + RustKvConnectorLeader
    ├── get_num_new_matched_tokens() → Check cache hits
    ├── update_state_after_alloc() → Track blocks
    ├── build_connector_meta() → Build worker metadata
    └── request_finished() → Trigger offload

DynamoKVBMConnectorWorker (kvbm_connector_worker.py)
    ├── Implements KvCacheConnectorWorker interface
    ├── Creates KvbmWorker + RustKvConnectorWorker
    ├── register_kv_caches() → Register memory
    └── load_kv() / save_kv() → Data transfer
```

### Memory Tier Architecture
```
┌─────────────────────────────────────────────────────────────┐
│                    Inference Engine                         │
│              (vLLM / TensorRT-LLM / SGLang)                │
└─────────────────────────┬───────────────────────────────────┘
                          │ Connector API
┌─────────────────────────▼───────────────────────────────────┐
│                    KVBM Logic Layer                         │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────────────┐   │
│  │ BlockManager│ │ Slot Manager│ │ Transfer Manager    │   │
│  │ (lifecycle) │ │ (requests)  │ │ (async D2H/H2D/D2D) │   │
│  └─────────────┘ └─────────────┘ └─────────────────────┘   │
└─────────────────────────┬───────────────────────────────────┘
                          │ NIXL API
┌─────────────────────────▼───────────────────────────────────┐
│                      NIXL Layer                             │
│  ┌────────┐  ┌────────┐  ┌────────┐  ┌────────────────┐    │
│  │GPU HBM │  │Host RAM│  │Local   │  │Remote Storage  │    │
│  │(G1)    │  │(G2)    │  │SSD (G3)│  │(G4)            │    │
│  └────────┘  └────────┘  └────────┘  └────────────────┘    │
└─────────────────────────────────────────────────────────────┘
```

### Block State Machine
```
Reset → Partial → Complete → Registered → Reset
  │        │         │           │
  │        │         │           └── drop() triggers Remove event
  │        │         └── register() makes visible for reuse
  │        └── commit() when block is full
  └── init_sequence() starts filling
```

## Key Design Decisions

1. **Rust core with Python bindings.** Core block management logic is in Rust for performance; Python bindings via PyO3/maturin expose `BlockManager`, `KvbmLeader`, `KvbmWorker`.

2. **Leader/Worker architecture.** Leader (scheduler-side) makes offload/onboard decisions; Worker (GPU-side) executes transfers. They communicate via serialized metadata.

3. **NIXL for data transfer.** All GPU↔CPU↔Disk transfers go through NIXL, which handles RDMA, GDS, and cross-node memory sharing.

4. **etcd for coordination.** Leader/worker discovery and synchronization uses etcd (required service).

5. **Async transfers.** Offload and onboard operations are asynchronous to overlap with computation. `save_kv_layer()` and `start_load_kv()` return immediately.

6. **Block deduplication.** Blocks are identified by sequence hash for prefix cache reuse. The RadixTree enables fast prefix matching.

7. **SSD lifespan protection.** Disk offload filtering (enabled by default) only offloads frequently-accessed blocks to extend SSD lifespan.

## Environment Variables

| Variable | Purpose | Default |
|----------|---------|---------|
| `DYN_KVBM_CPU_CACHE_GB` | CPU pinned memory cache size (GB) | Required |
| `DYN_KVBM_DISK_CACHE_GB` | SSD cache size (GB) | Optional |
| `DYN_KVBM_DISK_CACHE_DIR` | Disk cache directory | `/tmp/` |
| `DYN_KVBM_DISK_ZEROFILL_FALLBACK` | Enable zero-fill fallback for unsupported filesystems | `false` |
| `DYN_KVBM_DISK_DISABLE_O_DIRECT` | Disable O_DIRECT for disk I/O | `false` |
| `DYN_KVBM_LEADER_WORKER_INIT_TIMEOUT_SECS` | Timeout for leader/worker sync | `120` |
| `DYN_KVBM_METRICS` | Enable Prometheus metrics endpoint | `false` |
| `DYN_KVBM_METRICS_PORT` | Metrics port | `6880` |
| `DYN_KVBM_DISABLE_DISK_OFFLOAD_FILTER` | Disable SSD lifespan protection | `false` |
| `DYN_KVBM_HOST_OFFLOAD_PREFIX_MIN_PRIORITY` | Min priority for CPU offload | `0` |
| `DYN_KVBM_CPU_CACHE_OVERRIDE_NUM_BLOCKS` | Override CPU cache block count | - |
| `DYN_KVBM_DISK_CACHE_OVERRIDE_NUM_BLOCKS` | Override disk cache block count | - |

## Common Modification Patterns

### Adding a New Storage Backend
1. Extend NIXL storage interface (separate NIXL repo)
2. Register backend in `NixlStorage` (Rust side)
3. Add environment variables for backend configuration
4. Update `LayoutConfig` if backend requires special layout handling

### Adding a New Metric
1. Add metric definition in `src/block_manager/cache_stats.rs`
2. Emit metric in relevant Rust code paths
3. Expose via Prometheus endpoint at `/metrics` port `DYN_KVBM_METRICS_PORT`

### Modifying Block State Machine
1. Update `BlockState` enum in `src/block_manager/block.rs`
2. Update transition logic in same file
3. Ensure RAII handles (`PublishHandle`, `RegistrationHandle`) are updated
4. Update event emission for new states

## Error Handling

Common error scenarios and debugging:

- **Leader/worker timeout**: Increase `DYN_KVBM_LEADER_WORKER_INIT_TIMEOUT_SECS`
- **Disk allocation failure**: Enable `DYN_KVBM_DISK_ZEROFILL_FALLBACK=true`
- **No cache hits**: Check metrics for `kvbm_matched_tokens`; ensure prefix caching is enabled in the inference engine
- **Performance degradation**: Check `kvbm_onboard_blocks_h2d` vs `kvbm_offload_blocks_d2h` ratio

## Documentation

| Document | Purpose |
|----------|---------|
| [KVBM Overview](../../../docs/kvbm/README.md) | Quick start, feature matrix, architecture overview |
| [KVBM Guide](../../../docs/kvbm/kvbm_guide.md) | Installation, configuration, deployment instructions |
| [KVBM Design](../../../docs/kvbm/kvbm_design.md) | Architecture deep dive, components, data flows |
| [LMCache Integration](../../../docs/integrations/lmcache_integration.md) | Using LMCache with Dynamo |
| [FlexKV Integration](../../../docs/integrations/flexkv_integration.md) | Using FlexKV with Dynamo |
| [SGLang HiCache](../../../docs/integrations/sglang_hicache.md) | SGLang's hierarchical cache with NIXL |

## Testing Notes

- Integration tests are in `tests/kvbm_integration/`
- Tests require etcd and nats services: `docker compose -f deploy/docker-compose.yml up -d`
- Determinism tests validate that KVBM doesn't introduce non-determinism under load
- Use `DYN_KVBM_METRICS=true` to enable metrics for debugging
- `common.py` provides `ApiTester`, `DeterminismTester`, and KVBM metric parsing utilities
