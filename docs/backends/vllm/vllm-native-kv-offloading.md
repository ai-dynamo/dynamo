---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Native KV Offloading
subtitle: vLLM's OffloadingConnector with KV-aware routing in Dynamo
---

This guide covers running vLLM's native CPU KV cache offloading (the `OffloadingConnector`) with Dynamo, and the flags and versions required for the Dynamo KV router to account for the CPU cache tier when selecting workers. For Dynamo-side offloading backends (KVBM, LMCache, FlexKV), see [KV Cache Offloading](vllm-kv-offloading.md).

## Overview

vLLM's `OffloadingConnector` extends the prefix cache with a pinned CPU memory tier: sealed GPU KV blocks are copied to host memory asynchronously, and later requests whose prefix was evicted from GPU reload it from CPU instead of recomputing. The connector itself — sizing, eviction, platform support — is a vLLM feature; see [vLLM's KV offloading guide](https://docs.vllm.ai/en/latest/features/kv_offloading_usage/) for the full reference.

What Dynamo adds on top:

- **Tier-aware KV routing.** The KV router tracks per-tier residency for each worker and credits CPU-tier prefix overlap when scoring candidates, weighted by `--router-host-cache-hit-weight`.
- **Event relay.** Each worker relays vLLM's KV cache events for both tiers (`medium=GPU` / `medium=CPU`) from vLLM's local ZMQ publisher onto the Dynamo event plane, where the router consumes them.

Without the event wiring described below, offloading still works inside each worker, but the router treats CPU-resident prefixes as cache misses: a worker holding a full prefix in CPU memory looks identical to a cold worker, and multi-worker deployments route away from reusable cache.

## Support Matrix

Status legend: ✅ validated end to end · ⚠️ available but not yet validated end to end · 🚧 Dynamo integration in progress · ❌ not yet supported.

| Combination | Status | Notes |
| --- | --- | --- |
| Aggregated serving + event-driven KV routing (`--router-mode kv`) | ✅ | Requires the versions and flags below. |
| Chunked offloading (offload `block_size` larger than the GPU block size) | ✅ | Validated with 256-token offload blocks over a 16-token GPU block size; cuts KV event volume substantially at equal serving latency. |
| Approximate KV routing (`--no-router-kv-events`) | ⚠️ | Runs, but the router predicts device-tier reuse only; lower-tier weights have no effect. |
| Disaggregated serving (`MultiConnector`: `NixlConnector` + `OffloadingConnector`) | ⚠️ | vLLM tests this composition upstream; not yet validated behind the Dynamo KV router. |
| Tensor parallelism (TP > 1) | ⚠️ | Events are published once per engine and the CPU pool is sharded across TP workers by design; not yet validated end to end. |
| Models with sliding-window or Mamba/SSM layers | ⚠️ | Offloading works; self-describing events cover full-attention KV cache groups only, so the router sees a subset of CPU-resident blocks. |
| Disk and multi-tier offloading (`TieringOffloadingSpec`) | ⚠️ | Router-usable events are available on the vLLM main branch. Use vLLM v0.26.0 or later once released. |
| Shared-pool routing | 🚧 | The vLLM main branch publishes optional locality metadata on FS and OBJ events. Dynamo shared-pool indexing is in progress. |

## Requirements

> [!IMPORTANT]
> Router-visible CPU offloading requires **vLLM v0.24.0 or later** and **Dynamo 1.3.0 or later**.
>
> - vLLM v0.24.0 adds self-describing KV events for the `OffloadingConnector`, opt-in via `"self_describing_kv_events": true`. Earlier versions publish placeholder CPU events with no token payload, which the router cannot index.
> - The vLLM main branch adds router-usable multi-tier events and optional FS/OBJ locality metadata. Use vLLM v0.26.0 or later once released.
> - Dynamo 1.3.0 maps vLLM's `medium=CPU` events to the router's host tier and adds the `--router-host-cache-hit-weight` and `--router-disk-cache-hit-weight` flags.

Running an older vLLM or an older Dynamo fails silently: serving continues normally and no error is raised — the router just never learns about the CPU tier. Use [Verification](#verification) to confirm the wiring end to end.

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

The two `--router-*` weights are optional tuning; see [Configuration](#configuration).

> [!NOTE]
> The `--kv-transfer-config` and `--kv-events-config` JSON is vLLM-native — Dynamo passes engine flags through unchanged. `cpu_bytes_to_use` (16 GiB in this example) is the total pinned host memory for the CPU tier, shared across the engine's workers rather than per rank.

## Configuration

The knobs relevant to the router integration are listed below. For the full connector reference see [vLLM's KV offloading guide](https://docs.vllm.ai/en/latest/features/kv_offloading_usage/); for the full router reference see [Configuration and Tuning](../../components/router/router-configuration.md).

**Worker side (`kv_connector_extra_config` keys):**

| Key | Default | Description |
| --- | --- | --- |
| `self_describing_kv_events` | `false` | Publish full per-block payloads for CPU-tier events so external consumers can index them. Required for router tier awareness; has no effect unless KV events are enabled. |
| `cpu_bytes_to_use` | — (required) | Total pinned host bytes for the CPU tier, shared across the engine's workers. |
| `block_size` | GPU block size | Offload granularity in tokens; must be a multiple of the GPU `--block-size`. Larger blocks reduce event volume and transfer overhead. |
| `offload_prompt_only` | `true` | Offload prompt-phase blocks only; decode-generated blocks produce no CPU events. |

**Frontend side (Dynamo router):**

| Flag | Env var | Default | Description |
| --- | --- | --- | --- |
| `--router-host-cache-hit-weight` | `DYN_ROUTER_HOST_CACHE_HIT_WEIGHT` | `0.75` | Credit for CPU-tier prefix overlap relative to GPU overlap (0.0–1.0). `1.0` values a CPU-resident prefix as highly as a GPU-resident one. |
| `--router-disk-cache-hit-weight` | `DYN_ROUTER_DISK_CACHE_HIT_WEIGHT` | `0.25` | Same semantics for disk-tier overlap; unused for CPU-only offloading. |
| `--router-prefill-load-scale` | `DYN_ROUTER_PREFILL_LOAD_SCALE` | `1.0` | Weight of predicted prefill cost, after cache credits, relative to decode load. Higher values make cache overlap dominate worker selection. |

### Tuning

The defaults are conservative for cache-heavy workloads. At `--router-host-cache-hit-weight 0.75`, a worker holding 100% of a prefix in CPU ties with a worker holding 75% of it on GPU. At `--router-prefill-load-scale 1.0`, one uncached block costs the same as one block of decode load, so small load imbalances can outweigh large cache-overlap differences. In validation on long-prefix multi-turn workloads, `--router-host-cache-hit-weight 1.0` with `--router-prefill-load-scale 10` performed close to the optimum; the best values are workload-dependent.

> [!WARNING]
> Verify events end to end before raising `--router-prefill-load-scale`. A higher scale amplifies whatever the router believes about cache state — if CPU-tier events are missing, it amplifies wrong beliefs and can perform worse than the defaults.

## Event Model

The `OffloadingConnector` is write-through: sealed GPU blocks are copied to CPU during prefill, and the two tiers then evict independently. Each transition publishes a KV event, which the Dynamo worker relays from vLLM's ZMQ publisher onto the Dynamo event plane:

| Transition | Event emitted |
| --- | --- |
| Prefill seals a GPU block | `store(GPU)` |
| Async GPU → CPU copy completes | `store(CPU)` |
| GPU copy evicted (block still resident on CPU) | `remove(GPU)` |
| CPU copy evicted | `remove(CPU)` |

Only whole offload blocks are announced; a prefix tail shorter than `block_size` stays GPU-only. When offload blocks overlap on a shared prefix, vLLM re-announces the shared constituent blocks and Dynamo deduplicates them — no configuration is needed. The router maps `medium=CPU` to its host tier and credits host-tier overlap per [Configuration and Tuning](../../components/router/router-configuration.md).

## Verification

**Events reach the router.** The frontend exposes the router's applied-event counter on its HTTP metrics endpoint:

```bash
curl -s localhost:8000/metrics | grep kv_cache_events_applied
```

The `event_type="stored",status="ok"` counter should increase as requests run. If it stays at zero, no events are flowing at all — check `--kv-events-config` on the workers.

**The router sees the CPU tier.** Run the frontend with debug logging and send a request whose prefix has been offloaded (for example, repeat a long prompt after the GPU cache has filled):

```bash
DYN_LOG=debug python -m dynamo.frontend --http-port 8000 --router-mode kv 2>&1 | grep "lower-tier"
```

A line like the following confirms CPU-tier events were received and consulted during routing:

```text
Queried lower-tier indexer storage_tier=HostPinned queried_workers=2 matched_workers=1
```

If device-tier events apply but this line never appears, the CPU events are placeholders — check the vLLM version and the `self_describing_kv_events` flag.

## Troubleshooting

| Symptom | Likely cause | Fix |
| --- | --- | --- |
| `kv_cache_events_applied` stays at zero | Workers started without `--kv-events-config` | Pass it on every aggregated worker with `"enable_kv_cache_events": true`. |
| GPU events apply, but no `storage_tier=HostPinned` activity | vLLM older than v0.24.0, or `self_describing_kv_events` not set | Upgrade vLLM and set the flag; neither misconfiguration raises an error. |
| No CPU store events although the flag is set | `cpu_bytes_to_use` too small to hold a request's offload blocks | Raise it. A 256-token offload block is tens of megabytes for a 30B-class model, and offloading skips requests that do not fit. |
| Throughput drops after raising `--router-prefill-load-scale` | Router amplifying wrong cache beliefs because CPU-tier events are missing or broken | Confirm both verification checks pass before tuning routing policy. |
| Prefix matches never span workers | Non-deterministic Python block hashing | Export `PYTHONHASHSEED=0` for every worker process. |
| No CPU events for decode-generated blocks | `offload_prompt_only` defaults to `true` | Expected. Set it to `false` to offload decode-phase blocks as well. |

## Further Reading

- [Offloading Support Matrix](../../components/router/router-offloading.md) — cross-framework support matrix for KV routing with offloading
- [vLLM KV offloading guide](https://docs.vllm.ai/en/latest/features/kv_offloading_usage/) — connector configuration reference
- [Configuration and Tuning](../../components/router/router-configuration.md) — full router flag reference, including lower-tier cache-hit weights
- [Router Guide](../../components/router/router-guide.md) — routing modes and deployment matrix
- [KV Cache Offloading](vllm-kv-offloading.md) — KVBM, LMCache, and FlexKV offloading backends for vLLM
- [Using HiCache](../sglang/sglang-hicache.md) — the SGLang counterpart: tier-aware routing with HiCache
