<!--
SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
All rights reserved.
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

# KVBM G4 Object Storage Support

This document describes the G4 (Object Storage) tier in the Dynamo KV Block Manager, enabling KV cache offloading to remote object storage backends.

## Overview

The KVBM memory hierarchy consists of four tiers:

| Tier | Storage Type | Description |
|------|--------------|-------------|
| **G1** | Device (GPU) | High-bandwidth GPU VRAM for active inference |
| **G2** | Host (CPU) | Pinned CPU memory for staging and fast access |
| **G3** | Disk (NVMe/SSD) | Local persistent storage for overflow |
| **G4** | Object Storage | Remote object storage for large-scale KV cache persistence |

G4 extends the KVBM tiering model to include remote object storage, enabling virtually unlimited KV cache capacity beyond local storage constraints.

## Architecture

### Data Flow

G4 sits behind G2 (host memory). Host memory acts as a staging/bounce buffer for all object storage transfers:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           OFFLOAD FLOW                                  │
│                                                                         │
│   G1 (Device)  ───────►  G2 (Host)  ───────►  G4 (Object Storage)       │
│       GPU                CPU Memory              Remote Storage         │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│                           ONBOARD FLOW                                  │
│                                                                         │
│   G4 (Object Storage)  ───────►  G2 (Host)  ───────►  G1 (Device)       │
│       Remote Storage            CPU Memory               GPU            │
└─────────────────────────────────────────────────────────────────────────┘
```

### Block Addressing

Each tier uses a different addressing model:

| Tier | Addressing Model | NIXL Backend |
|------|------------------|--------------|
| G1 | GPU pointer + block index | UCX |
| G2 | Host pointer + block index | UCX/POSIX |
| G3 | File path + offset | GDS_MT |
| G4 | Bucket + sequence hash (key) | OBJ |

### Key Differences from Other Tiers

Object storage (G4) differs from block-addressable tiers (G1/G2/G3):

| Aspect | G1/G2/G3 Tiers | G4 Object Storage |
|--------|----------------|-------------------|
| Addressing | Contiguous memory/file regions | Key-value (hash → blob) |
| Registration | Pre-registered with NIXL | Ephemeral per-transfer |
| Block lifetime | Pool-managed | Externally managed (cloud persistence) |

## Configuration

### Environment Variables

Configure G4 object storage using these environment variables:

| Variable | Description | Default |
|----------|-------------|---------|
| `DYN_KVBM_OBJECT_ENABLED` | Enable G4 object storage (`1` or `true`) | `0` |
| `DYN_KVBM_OBJECT_BUCKET` | Bucket name template. Supports `{worker_id}` substitution | Required |
| `DYN_KVBM_OBJECT_ENDPOINT` | Object storage endpoint URL | None |
| `DYN_KVBM_OBJECT_REGION` | Storage region | None |
| `DYN_KVBM_OBJECT_ACCESS_KEY` | Access key for authentication | None |
| `DYN_KVBM_OBJECT_SECRET_KEY` | Secret key for authentication | None |
| `DYN_KVBM_OBJECT_WRITE_THROUGH` | Keep blocks in host cache after G4 offload | `1` (enabled) |
| `DYN_KVBM_OBJECT_NUM_BLOCKS` | Maximum blocks tracked in G4 registry | 100,000 |

### Bucket Template

The bucket template supports `{worker_id}` substitution for per-worker buckets:

```bash
# Single shared bucket
export DYN_KVBM_OBJECT_BUCKET="kv-cache"

# Per-worker buckets
export DYN_KVBM_OBJECT_BUCKET="kv-cache-worker-{worker_id}"
```

### Example Configuration

```bash
# Enable G4 object storage
export DYN_KVBM_OBJECT_ENABLED=1
export DYN_KVBM_OBJECT_BUCKET="my-kv-cache-{worker_id}"
export DYN_KVBM_OBJECT_ENDPOINT="http://object-storage.example.com:9000"
export DYN_KVBM_OBJECT_REGION="us-east-1"
export DYN_KVBM_OBJECT_ACCESS_KEY="your-access-key"
export DYN_KVBM_OBJECT_SECRET_KEY="your-secret-key"
```

## Write-Through Caching

By default, G4 uses **write-through caching**: blocks offloaded to object storage are also retained in host memory (G2). This provides:

- **Fast retrieval**: Hot blocks can be served from host cache without network round-trip
- **Graceful degradation**: If object storage is unavailable, cached blocks remain accessible
- **Reduced latency**: Subsequent requests for the same blocks avoid object storage latency

To disable write-through (blocks are evicted from host after G4 offload):

```bash
export DYN_KVBM_OBJECT_WRITE_THROUGH=0
```

## NIXL Dependencies and Build

G4 object storage support requires NIXL (NVIDIA Inference Xfer Library) with the OBJ backend plugin enabled.

### NIXL Version Requirements

| Component | Version | Notes |
|-----------|---------|-------|
| **NIXL** | 0.8.0+ | Required for OBJ backend support |
| **UCX** | 1.20.x | Tested and recommended version |
| **CUDA** | 12.x or 13.x | GPU memory transfers |
| **Python** | 3.10+ | For Python bindings |


### Building from Source

#### Prerequisites

**Ubuntu:**
```bash
sudo apt install build-essential cmake pkg-config
```

**Python dependencies:**
```bash
pip3 install meson ninja pybind11 tomlkit
```

#### Building NIXL With the Provided Makefile

A Makefile is provided at `lib/bindings/kvbm/Makefile` for convenient rebuilding of NIXL and KVBM:

```bash
# Navigate to the kvbm directory
cd lib/bindings/kvbm

# Rebuild both NIXL and kvbm
make

# Rebuild NIXL only
make nixl

# Rebuild kvbm only (most common for code changes)
make kvbm

# Quick rebuild (kvbm only)
make quick

# Show installed package info
make info

# Clean build artifacts
make clean

# Show all available targets
make help
```

**Makefile targets:**

| Target | Description |
|--------|-------------|
| `make` / `make rebuild` | Rebuild both NIXL and kvbm |
| `make nixl` | Rebuild and install NIXL only |
| `make kvbm` | Rebuild and install kvbm only |
| `make quick` | Rebuild kvbm only (for Rust code changes) |
| `make nixl-build` | Build NIXL (without install) |
| `make nixl-install` | Install NIXL |
| `make kvbm-build` | Build kvbm wheel |
| `make kvbm-install` | Install kvbm (no deps) |
| `make clean` | Clean build artifacts |
| `make info` | Show installed package info |
| `make shutdown` | Terminate all vLLM processes |


### Verifying NIXL Installation

```bash
# Check NIXL library
ls -lh /opt/nvidia/nvda_nixl/lib/x86_64-linux-gnu/libnixl.so

# Check available plugins (should include OBJ for G4 support)
ls -lh /opt/nvidia/nvda_nixl/lib/x86_64-linux-gnu/plugins/
```

## NIXL Integration

G4 transfers are handled through NIXL's OBJ backend. The transfer flow:

1. **Offload (Host → Object Storage)**:
   - Blocks are copied from host pinned memory
   - NIXL OBJ backend handles transfers
   - Sequence hash serves as the object key

2. **Onboard (Object Storage → Host)**:
   - Registry lookup identifies available blocks by sequence hash
   - NIXL OBJ backend downloads objects to host memory
   - Blocks are then available for G2→G1 transfer

### NIXL Backend Requirements

Ensure the NIXL OBJ backend is available:

```rust
if !nixl_agent.has_backend("OBJ") {
    // Object storage transfers not available
}
```

## Distributed Object Registry

The distributed registry enables cross-worker coordination for G4 object storage operations. It tracks which KV cache blocks have been stored in object storage, enabling deduplication and efficient lookups across multiple workers.

### Registry Architecture

The registry uses a **hub-and-spoke** model:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         REGISTRY SERVICE                                 │
│                                                                          │
│                      ┌─────────────────────┐                             │
│                      │   Registry Hub      │                             │
│                      │   (Coordinator)     │                             │
│                      └──────────┬──────────┘                             │
│                                 │                                        │
│            ┌────────────────────┼────────────────────┐                   │
│            ▼                    ▼                    ▼                   │
│   ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐         │
│   │   Worker 0      │  │   Worker 1      │  │   Worker N      │         │
│   │ (Registry       │  │ (Registry       │  │ (Registry       │         │
│   │  Client)        │  │  Client)        │  │  Client)        │         │
│   └─────────────────┘  └─────────────────┘  └─────────────────┘         │
└─────────────────────────────────────────────────────────────────────────┘
```

- **Hub**: Single coordinator holding the registry
- **Clients**: Workers connect to the hub to query and register entries
- **Transport**: ZMQ-based communication (REQ/REP for queries, PUB/SUB for registrations)

### Registry Features

| Feature | Description |
|---------|-------------|
| **Deduplication** | Avoid redundant object writes by checking what already exists |
| **Matching** | Find which blocks can be loaded from object storage |
| **Registration** | Track newly stored blocks across all workers |
| **TinyLFU Eviction** | Intelligent eviction combining frequency and recency |
| **Lease System** | Prevent race conditions during concurrent offloads |

### Building the Registry

The registry hub is provided as an example binary in the workspace:

```bash
# Navigate to the registry example
cd examples/kvbm/distributed/object-registry

# Build the registry binary
cargo build --release

# Or build from workspace root
cargo build --manifest-path examples/kvbm/distributed/object-registry/Cargo.toml --release
```

### Running the Registry Hub

Start the registry hub before launching workers:

```bash
# Run with default settings
cd examples/kvbm/distributed/object-registry
cargo run --release

# Or run with custom settings
DYN_REGISTRY_HUB_CAPACITY=10000000 \
DYN_REGISTRY_HUB_QUERY_ADDR=tcp://*:6000 \
DYN_REGISTRY_HUB_REGISTER_ADDR=tcp://*:6001 \
cargo run --release
```

On startup, the hub displays its configuration:

```
╔══════════════════════════════════════════════════════════════╗
║           Distributed Object Registry                        ║
╠══════════════════════════════════════════════════════════════╣
║  Capacity:        1000000 entries                            ║
║  Query Addr:      tcp://*:5555                               ║
║  Register Addr:   tcp://*:5556                               ║
║  Lease Timeout:   30 secs                                    ║
╚══════════════════════════════════════════════════════════════╝
```

### Registry Hub Configuration

| Variable | Description | Default |
|----------|-------------|---------|
| `DYN_REGISTRY_HUB_CAPACITY` | Registry capacity (number of entries) | `1000000` |
| `DYN_REGISTRY_HUB_QUERY_ADDR` | ZMQ REP socket for queries | `tcp://*:5555` |
| `DYN_REGISTRY_HUB_REGISTER_ADDR` | ZMQ SUB socket for registrations | `tcp://*:5556` |
| `DYN_REGISTRY_HUB_LEASE_TIMEOUT_SECS` | Lease timeout for `can_offload` claims | `30` |


### Configuring Workers (Registry Clients)

Enable workers to connect to the distributed registry:

```bash
# Enable distributed registry
export DYN_REGISTRY_ENABLE=1

# Hub addresses (replace 'leader' with actual hostname/IP)
export DYN_REGISTRY_CLIENT_QUERY_ADDR=tcp://leader:5555
export DYN_REGISTRY_CLIENT_REGISTER_ADDR=tcp://leader:5556
```

### Registry Client Configuration

| Variable | Description | Default |
|----------|-------------|---------|
| `DYN_REGISTRY_ENABLE` | Enable distributed registry (`1` or `true`) | `0` |
| `DYN_REGISTRY_CLIENT_QUERY_ADDR` | Hub query address | `tcp://localhost:5555` |
| `DYN_REGISTRY_CLIENT_REGISTER_ADDR` | Hub register address | `tcp://localhost:5556` |
| `DYN_REGISTRY_CLIENT_LOCAL_CACHE` | Local cache capacity (0 = disabled) | `0` |

### Complete G4 + Registry Setup Example

```bash
# ============================================
# On the leader/coordinator node
# ============================================

# Start the registry hub
cd examples/kvbm/distributed/object-registry
DYN_REGISTRY_HUB_CAPACITY=5000000 \
cargo run --release &

# ============================================
# On all worker nodes
# ============================================

# Enable G4 object storage
export DYN_KVBM_OBJECT_ENABLED=1
export DYN_KVBM_OBJECT_BUCKET="kv-cache-{worker_id}"
export DYN_KVBM_OBJECT_ENDPOINT="http://object-storage.example.com:9000"

# Enable distributed registry (for cross-worker deduplication)
export DYN_REGISTRY_ENABLE=1
export DYN_REGISTRY_CLIENT_QUERY_ADDR=tcp://leader:5555
export DYN_REGISTRY_CLIENT_REGISTER_ADDR=tcp://leader:5556

# Start your inference worker
# ...
```

### Registry Operations

The registry supports these operations:

| Operation | Description |
|-----------|-------------|
| `can_offload(hashes)` | Check which hashes can be offloaded (not already stored) |
| `register(hashes)` | Register hashes as stored in object storage |
| `match_sequence_hashes(hashes)` | Find which hashes exist in object storage |
| `unregister(hashes)` | Remove hashes from registry (does NOT delete objects) |

### Local Registry (Testing)

For single-node testing or when distributed registry is disabled, KVBM uses a local in-process registry:

```rust
use dynamo_llm::block_manager::distributed::registry::LocalRegistry;

let registry = LocalRegistry::new(100_000);
registry.register(&[hash1, hash2]).await?;

let result = registry.can_offload(&[hash1, hash2, hash3]).await?;
// result.can_offload == vec![hash3]  (hash1, hash2 already registered)
```

## Metrics

G4 operations are tracked with these metrics:

| Metric | Description |
|--------|-------------|
| `offload_blocks_h2obj` | Blocks offloaded from Host to Object Storage |
| `onboard_blocks_obj2h` | Blocks onboarded from Object Storage to Host |
| `g4_lookup_hits` | Registry lookup hits |
| `g4_lookup_misses` | Registry lookup misses |

