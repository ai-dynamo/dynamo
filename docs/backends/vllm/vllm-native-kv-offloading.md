---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Native KV Offloading
subtitle: vLLM's OffloadingConnector with KV-aware routing in Dynamo
---

This guide shows how to run vLLM's native CPU KV cache offloading
(`OffloadingConnector`) with Dynamo's KV router. For Dynamo-side offloading
backends such as KVBM, LMCache, and FlexKV, see
[KV Cache Offloading](vllm-kv-offloading.md).

## Support Matrix

Status legend: ✅ validated end to end · ⚠️ available but not yet validated end to end · 🚧 Dynamo integration in progress · ❌ not yet supported.

| Combination | Status | Notes |
| --- | --- | --- |
| Aggregated serving + event-driven KV routing (`--router-mode kv`) | ✅ | Requires the versions and flags below. |
| Chunked offloading (offload `block_size` larger than the GPU block size) | ✅ | Validated with 256-token offload blocks over a 16-token GPU block size. |
| Approximate KV routing (`--no-router-kv-events`) | ✅ | Validated with two workers. The router predicts GPU-tier reuse only, so lower-tier weights have no effect. |
| Disaggregated serving (`MultiConnector`: `NixlConnector` + `OffloadingConnector`) | ⚠️ | Validated end to end with the pending Dynamo fix in [#11219](https://github.com/ai-dynamo/dynamo/pull/11219). |
| Tensor parallelism (TP > 1) | ✅ | Validated with TP=2, including GPU eviction, CPU reload, and one event stream per engine. |
| Models with sliding-window or Mamba/SSM layers | ✅ | Validated with Gemma-2 and Falcon-H1. The router sees only full-attention KV cache groups. |
| Disk and multi-tier offloading (`TieringOffloadingSpec`) | 🚧 | vLLM main emits FS and OBJ events; Dynamo tier mapping is in progress. |
| Shared-pool routing | 🚧 | vLLM main emits optional locality metadata; Dynamo shared-pool indexing is in progress. |

## Requirements

> [!IMPORTANT]
> Router-visible CPU offloading requires **vLLM v0.24.0 or later** and **Dynamo 1.3.0 or later**.
>
> - vLLM v0.24.0 adds self-describing KV events for the `OffloadingConnector`, opt-in via `"self_describing_kv_events": true`. Earlier versions publish placeholder CPU events with no token payload, which the router cannot index.
> - Dynamo 1.3.0 maps vLLM's `medium=CPU` events to the router's host tier and adds `--router-host-cache-hit-weight`.

Three settings are required, and none is on by default:

1. On every worker, enable KV event publishing with `--kv-events-config` — `--router-mode kv` on the frontend does not enable it (see [KV Routing Requirements](README.md#kv-routing-requirements)).
2. On every worker, set `"self_describing_kv_events": true` inside `kv_connector_extra_config`.
3. On the frontend, select event-driven KV routing with `--router-mode kv`.

Prefix caching must stay enabled on the workers; `python -m dynamo.vllm` enables it by default.

## Setup

**vLLM workers** — native CPU offloading with router-usable events:

```bash
export PYTHONHASHSEED=0   # deterministic block hashes across workers

CUDA_VISIBLE_DEVICES=0 python -m dynamo.vllm \
  --model Qwen/Qwen3-0.6B \
  --block-size 16 \
  --kv-transfer-config '{"kv_connector":"OffloadingConnector","kv_role":"kv_both","kv_connector_extra_config":{"cpu_bytes_to_use":17179869184,"block_size":256,"self_describing_kv_events":true}}' \
  --kv-events-config '{"publisher":"zmq","topic":"kv-events","endpoint":"tcp://*:20080","enable_kv_cache_events":true}'
```

Launch additional workers with their own `CUDA_VISIBLE_DEVICES` and a unique `--kv-events-config` endpoint port per worker on the same host (for example `tcp://*:20081`).

**Dynamo frontend** — event-driven KV routing:

```bash
python -m dynamo.frontend \
  --http-port 8000 \
  --router-mode kv \
  --router-host-cache-hit-weight 1.0 \
  --router-prefill-load-scale 10
```

The router tuning flags are optional; see [Configuration](#configuration).

> [!NOTE]
> The `--kv-transfer-config` and `--kv-events-config` JSON is vLLM-native — Dynamo passes engine flags through unchanged. `cpu_bytes_to_use` (16 GiB in this example) is the total pinned host memory for the CPU tier, shared across the engine's workers rather than per rank.

## Configuration

These are the main settings for the common CPU-offloading path. For all
connector options, see
[vLLM's KV offloading guide](https://docs.vllm.ai/en/latest/features/kv_offloading_usage/).

**Worker side (`kv_connector_extra_config` keys):**

| Key | Default | Description |
| --- | --- | --- |
| `self_describing_kv_events` | `false` | Publish CPU-tier events that the router can index. |
| `cpu_bytes_to_use` | — (required) | Total pinned host bytes for the CPU tier, shared across the engine's workers. |
| `block_size` | GPU block size | Offload granularity in tokens; must be a multiple of the GPU `--block-size`. Larger blocks reduce event volume and transfer overhead. |

**Frontend side (Dynamo router):**

| Flag | Env var | Default | Description |
| --- | --- | --- | --- |
| `--router-host-cache-hit-weight` | `DYN_ROUTER_HOST_CACHE_HIT_WEIGHT` | `0.75` | Credit for CPU-tier prefix overlap relative to GPU overlap. |
| `--router-prefill-load-scale` | `DYN_ROUTER_PREFILL_LOAD_SCALE` | `1.0` | How strongly cache overlap affects worker selection. |

Tune these values for your workload only after [verifying](#verification) that
CPU-tier events reach the router. See
[Configuration and Tuning](../../components/router/router-configuration.md)
for the full router reference.

## How Routing Works

vLLM copies sealed GPU KV blocks to pinned CPU memory. Dynamo adds:

- **Tier-aware KV routing.** The router credits CPU-tier prefix overlap when
  choosing a worker, weighted by `--router-host-cache-hit-weight`.
- **Event relay.** Each worker relays vLLM's GPU and CPU KV events from the
  local ZMQ publisher to the Dynamo event plane.

The GPU and CPU copies then evict independently:

| Transition | Event emitted |
| --- | --- |
| Prefill seals a GPU block | `store(GPU)` |
| Async GPU → CPU copy completes | `store(CPU)` |
| GPU copy evicted (block still resident on CPU) | `remove(GPU)` |
| CPU copy evicted | `remove(CPU)` |

Without this event wiring, offloading still works inside each worker, but the
router treats CPU-resident prefixes as cache misses.

## Verification

Check that the router applies KV events:

```bash
curl -s localhost:8000/metrics | grep kv_cache_events_applied
```

The `event_type="stored",status="ok"` counter should increase as requests run.

To confirm CPU-tier routing, run the frontend with debug logging and repeat a
long prefix after it has been offloaded:

```bash
DYN_LOG=debug python -m dynamo.frontend --http-port 8000 --router-mode kv 2>&1 | grep "lower-tier"
```

A line like this confirms that the router consulted the CPU tier:

```text
Queried lower-tier indexer storage_tier=HostPinned queried_workers=2 matched_workers=1
```

## Troubleshooting

| Symptom | Likely cause | Fix |
| --- | --- | --- |
| `kv_cache_events_applied` stays at zero | Workers started without `--kv-events-config` | Pass it on every aggregated worker with `"enable_kv_cache_events": true`. |
| GPU events apply, but no `storage_tier=HostPinned` activity | vLLM older than v0.24.0, or `self_describing_kv_events` not set | Upgrade vLLM and set the flag; neither misconfiguration raises an error. |
| No CPU store events although the flag is set | No complete offload block, or `cpu_bytes_to_use` is too small | Test with a prompt longer than the offload `block_size`, then increase the CPU pool if needed. |
| Prefix matches never span workers | Non-deterministic Python block hashing | Export `PYTHONHASHSEED=0` for every worker process. |

## Further Reading

- [Offloading Support Matrix](../../components/router/router-offloading.md) — cross-framework support matrix for KV routing with offloading
- [vLLM KV offloading guide](https://docs.vllm.ai/en/latest/features/kv_offloading_usage/) — connector configuration reference
- [Configuration and Tuning](../../components/router/router-configuration.md) — full router flag reference, including lower-tier cache-hit weights
- [Router Guide](../../components/router/router-guide.md) — routing modes and deployment matrix
- [KV Cache Offloading](vllm-kv-offloading.md) — KVBM, LMCache, and FlexKV offloading backends for vLLM
- [Using HiCache](../sglang/sglang-hicache.md) — the SGLang counterpart: tier-aware routing with HiCache
