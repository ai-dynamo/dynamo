---
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: KVBM v2 Guide
---

> [!NOTE]
> **KVBM v2 is currently under active development on the [`ryan/kvbm-bindings`](https://github.com/ai-dynamo/dynamo/tree/ryan/kvbm-bindings) branch.** The instructions below require building from that branch.

## Run KVBM v2 in Dynamo with vLLM

### Docker Setup

Unlike v1, KVBM v2 does **not** require etcd or NATS. No `docker compose` step is needed.

```bash
# Build from the ryan/kvbm-bindings branch
git checkout ryan/kvbm-bindings

# Build a dynamo vLLM container (KVBM v2 is built in)
python container/render.py --framework vllm --target runtime --output-short-filename
docker build -t dynamo:latest-vllm-v2-runtime -f container/rendered.Dockerfile .

# Launch the container
container/run.sh --image dynamo:latest-vllm-v2-runtime -it --mount-workspace --use-nixl-gds
```

### Aggregated Serving

```bash
# run ingress
python -m dynamo.frontend &

# run worker with KVBM v2 enabled
# NOTE: remove --enforce-eager for production use
# NOTE: DYN_KVBM_NIXL_BACKEND_UCX=true is required to enable NIXL for RDMA transfers
DYN_KVBM_NIXL_BACKEND_UCX=true \
DYN_KVBM_CPU_CACHE_GB=20 \
  python -m dynamo.vllm \
    --model Qwen/Qwen3-0.6B \
    --kv-transfer-config '{"kv_connector":"DynamoConnector","kv_connector_module_path":"kvbm.v2.vllm.connector","kv_role":"kv_both"}' \
    --enforce-eager
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

```bash
DYN_KVBM_NIXL_BACKEND_UCX=true \
DYN_KVBM_CPU_CACHE_GB=20 \
vllm serve --kv-transfer-config '{"kv_connector":"DynamoConnector","kv_role":"kv_both","kv_connector_module_path":"kvbm.v2.vllm.connector"}' Qwen/Qwen3-0.6B
```

## Configuration

### Cache Tier Configuration

Configure KVBM v2 cache tiers using environment variables. The v2 binding maps these env vars through the v1-compat layer onto the resolved cache config, so the user-facing UX matches v1 exactly:

```bash
# Option 1: CPU cache only (GPU → CPU offloading)
export DYN_KVBM_CPU_CACHE_GB=4    # 4GB of pinned CPU memory

# Option 2: CPU + Disk cache (GPU → CPU → Disk tiered offloading)
export DYN_KVBM_CPU_CACHE_GB=4
export DYN_KVBM_DISK_CACHE_GB=8   # 8GB of disk

# Option 3: Disk cache only — enables host-bypass mode
#          (GPU ↔ Disk direct via GDS, bypassing CPU memory)
export DYN_KVBM_DISK_CACHE_GB=8
```

You can also specify exact block counts instead of GB:

- `DYN_KVBM_CPU_CACHE_OVERRIDE_NUM_BLOCKS`
- `DYN_KVBM_DISK_CACHE_OVERRIDE_NUM_BLOCKS`

> [!NOTE]
> KVBM is a write-through cache; capacities should grow with each enabled tier — `DYN_KVBM_CPU_CACHE_GB ≥ <GPU KV cache size>` and `DYN_KVBM_DISK_CACHE_GB ≥ DYN_KVBM_CPU_CACHE_GB`. Misconfiguring the CPU cache below the GPU cache size causes offload churn rather than a benefit.

#### Host-Bypass Mode (Disk-Only)

Setting `DYN_KVBM_DISK_CACHE_GB` *without* `DYN_KVBM_CPU_CACHE_GB` (or with an explicit `DYN_KVBM_CPU_CACHE_GB=0`) enables host-bypass mode. In this mode:

- **Offload** moves blocks directly **G1 (GPU) → G3 (Disk)** via GDS — no G2 staging.
- **Onboard** moves disk hits directly **G3 → G1** via GDS — no G3→G2→G1 staging.
- **GDS support is required.** KVBM probes for GDS at startup; if the probe fails, the first transfer errors loudly so the misconfiguration is visible immediately.

**Current scope limits in host-bypass mode:**

- **Onboard mode must be `Inter`** (the default). `Intra` mode uses a layer-by-layer CudaStream load that expects G2 sources, so it cannot serve directly from disk. Combining `KVBM_ONBOARD_MODE=intra` with bypass is rejected at startup with a clear error — use `Inter` whenever bypass is in effect.
- **No remote search.** Bypass is local-only. The remote-search protocol assumes G2 destinations for staging; a deployment that engages both falls through to the standard staging path, which will surface a missing-G2-destination error rather than silently misroute disk traffic.

### Onboard Mode

KVBM v2 supports two onboarding modes that control how KV cache blocks are loaded from host (G2) to device (G1) memory:

```bash
# Inter-pass onboarding (default)
# Blocks are loaded asynchronously between scheduler passes via Nova messages.
# The forward pass does not block on transfers.
export KVBM_ONBOARD_MODE=inter

# Intra-pass onboarding
# Blocks are loaded synchronously, layer-by-layer during the forward pass.
# Each layer waits for its KV cache load to complete before proceeding.
export KVBM_ONBOARD_MODE=intra
```

Full example with intra mode:

```bash
KVBM_ONBOARD_MODE=intra \
DYN_KVBM_NIXL_BACKEND_UCX=true \
DYN_KVBM_CPU_CACHE_GB=20 \
  python -m dynamo.vllm \
    --model Qwen/Qwen3-0.6B \
    --kv-transfer-config '{"kv_connector":"DynamoConnector","kv_connector_module_path":"kvbm.v2.vllm.connector","kv_role":"kv_both"}' \
    --enforce-eager
```

### Offload Policies

KVBM v2 runs two independent offload pipelines; each filters blocks through a list of policies (implicit AND). Defaults are:

| Tier | Default | Rationale |
|------|---------|-----------|
| G1→G2 | `presence` | Offload any GPU block not already on host and not already in flight. |
| G2→G3 | `presence` | Offload any host block not already on disk and not already in flight. |

Available policy types:

- `pass_all` — no filtering, every block is offloaded
- `presence` — skip blocks already in the destination tier or currently in flight
- `presence_lfu` — `presence` plus a TinyLFU access-count threshold; offloads when `count > min_lfu_count` (default `1`)

Override per tier via env vars (the value is JSON):

```bash
# Opt into LFU-on-admission for G2→G3 (useful when disk write bandwidth is precious, or SSD lifespan is a concern)
KVBM_OFFLOAD_G2_TO_G3_POLICIES='["presence_lfu"]'

# Tune the LFU threshold (default 1; e.g. require 4+ hits before offload)
KVBM_OFFLOAD_G2_TO_G3_PRESENCE_LFU_MIN_LFU_COUNT=3

# Force every block through (helpful for smoke-testing or observing G3 traffic)
KVBM_OFFLOAD_G2_TO_G3_POLICIES='["pass_all"]'
```

## Enable and View KVBM Metrics

### Setup Monitoring Stack

```bash
# Start Prometheus and Grafana (KVBM v2 itself does not require etcd or NATS,
# but the bundled observability stack is shared with v1 and reused here).
docker compose -f deploy/docker-observability.yml up -d
```

### Enable Metrics for vLLM

Set `DYN_KVBM_METRICS=true` when launching the worker. The KVBM metrics endpoint binds to port `6880`; the bundled Prometheus stack scrapes it automatically.

```bash
DYN_KVBM_METRICS=true \
DYN_KVBM_CPU_CACHE_GB=20 \
  python -m dynamo.vllm \
    --model Qwen/Qwen3-0.6B \
    --kv-transfer-config '{"kv_connector":"DynamoConnector","kv_connector_module_path":"kvbm.v2.vllm.connector","kv_role":"kv_both"}' \
    --enforce-eager
```

### Firewall Configuration (Optional)

```bash
# If a firewall blocks the KVBM metrics port
sudo ufw allow 6880/tcp
```

### View Metrics

Access Grafana at http://localhost:3000 (default login: `dynamo`/`dynamo`) and look for the **KVBM Dashboard**.

### Available Metrics

| Metric | Description |
|--------|-------------|
| `kvbm_matched_tokens` | Number of matched tokens |
| `kvbm_offload_blocks_d2h` | Offload blocks from device to host (G1→G2) |
| `kvbm_offload_blocks_h2d` | Offload blocks from host to disk (G2→G3) |
| `kvbm_offload_blocks_d2d` | Offload blocks from device to disk, bypassing host (G1→G3) |
| `kvbm_onboard_blocks_d2h` | Onboard blocks from disk to host (G3→G2 staging step, v2 only) |
| `kvbm_onboard_blocks_h2d` | Onboard blocks from host to device (G2→G1) |
| `kvbm_onboard_blocks_d2d` | Onboard blocks from disk to device direct (G3→G1, e.g. via GDS) |
| `kvbm_host_cache_hit_rate` | Host cache hit rate (0.0–1.0) |
| `kvbm_disk_cache_hit_rate` | Disk cache hit rate (0.0–1.0) |

## Benchmarking

Use [LMBenchmark](https://github.com/LMCache/LMBenchmark) to evaluate KVBM v2 performance, same as v1:

```bash
git clone https://github.com/LMCache/LMBenchmark.git
cd LMBenchmark/synthetic-multi-round-qa

./long_input_short_output_run.sh \
    "Qwen/Qwen3-0.6B" \
    "http://localhost:8000" \
    "benchmark_kvbm_v2" \
    1
```
