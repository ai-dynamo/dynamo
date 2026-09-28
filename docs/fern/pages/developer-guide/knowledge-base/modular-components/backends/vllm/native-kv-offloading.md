---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Native KV Offloading
subtitle: vLLM native CPU offloading and experimental Mooncake Store hints for the KV router
---

This guide covers vLLM's native CPU KV cache offloading (`OffloadingConnector`) with NVIDIA Dynamo's KV router. [Mooncake Store shared-cache routing](#mooncake-store-shared-cache-routing) is a separate, experimental path for `MooncakeStoreConnector`. For Dynamo-side offloading backends such as LMCache and FlexKV, see [KV Cache Offloading](kv-cache-offloading.md).

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
| Disk and multi-tier offloading (`TieringOffloadingSpec`) | ⚠️ | vLLM main emits a unified `STORAGE` medium (FS/OBJ media removed in [vLLM #48123](https://github.com/vllm-project/vllm/pull/48123)); Dynamo maps `STORAGE` to the Disk lower tier. Cache-salted requests: see [Known Limitations](#known-limitations). |
| Shared-pool routing with `OffloadingConnector` | Not supported | Per-event locality stays worker-local when `LOCAL` or absent; `REMOTE` or unknown locality is dropped. These events do not populate the Mooncake Store index. |
| Shared-pool routing with `MooncakeStoreConnector` | Experimental | Separate pinned adapter, worker metadata, and Mooncake events required. Full-attention, sliding-window, and Mamba-align layouts are supported subject to the [contract below](#mooncake-store-shared-cache-routing). GPU validation remains pending. |

## Requirements

> [!IMPORTANT]
> Router-visible CPU offloading requires **vLLM v0.24.0 or later** and **Dynamo 1.3.0 or later**.
>
> - vLLM v0.24.0 adds self-describing KV events for the `OffloadingConnector`, opt-in via `"self_describing_kv_events": true`. Earlier versions publish placeholder CPU events with no token payload, which the router cannot index.
> - Dynamo 1.3.0 maps vLLM's `medium=CPU` events to the router's host tier and adds `--router-host-cache-hit-weight`.

Three settings are required, and none is on by default:

1. On every worker, enable KV event publishing with `--kv-events-config` — `--router-mode kv` on the frontend does not enable it (see [KV-Aware Routing](overview.md#feature-support-matrix)).
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
[Configuration and Tuning](../../router/configuration-and-tuning.md)
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

## Known Limitations

- **Cache-salted requests are not routable through `STORAGE`-tier events.**
  vLLM's self-describing offload events do not yet carry `extra_keys` (the
  per-block `cache_salt` and multimodal identifiers), so Dynamo cannot recover
  the cache namespace and indexes `STORAGE` blocks under their unsalted hash. A
  salted request then computes a salted query hash that never matches those
  entries, so the `STORAGE`-tier copy is missed by routing — a lost cache-hit
  opportunity, not a correctness problem (no wrong KV is returned). Populating
  `extra_keys` on offload events is a known upstream deferral tracked in
  vLLM RFC [#49413](https://github.com/vllm-project/vllm/issues/49413).

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

## Mooncake Store Shared-Cache Routing

**Experimental.** This path estimates reusable KV prefixes in a shared Mooncake Store for vLLM's `MooncakeStoreConnector`. It is separate from `OffloadingConnector` CPU and `STORAGE` events. The estimate changes shared-cache scoring only; it does not authorize KV reuse, move data, or change scheduler accounting. The worker still checks actual store availability. GPU acceptance for this integration remains unverified.

### Dependency and Worker Contract

The initial adapter supports vLLM commit [`1085b64425a9e6f5ca52876ad32e55fda5665f4e`](https://github.com/vllm-project/vllm/tree/1085b64425a9e6f5ca52876ad32e55fda5665f4e), not every release at or above v0.29. The installed build must identify that revision through SCM version metadata or PEP 610 provenance and retain the adapter's checked source fingerprints. A version override cannot make a different private API compatible.

Mooncake source verification uses commit [`ffe013517eaafa8f33e5e0ee034fd6b8f5561e92`](https://github.com/kvcache-ai/Mooncake/tree/ffe013517eaafa8f33e5e0ee034fd6b8f5561e92). The build must include per-medium eviction and clear semantics from [Mooncake #3818](https://github.com/kvcache-ai/Mooncake/pull/3818), merge commit `7e8ca11d87fad66dc75cb40d8ebe2ed95ef15cd1`. Older builds, including `v0.3.13.post1`, do not meet this contract.

Configure every participating worker as follows:

| Setting | Required value or behavior |
| --- | --- |
| Worker extension | `--worker-extension-cls dynamo.vllm.mooncake_store_runtime.MooncakeStoreWorkerExtension`. Enable it explicitly; the frontend flag does not reach workers. |
| Connector | Exactly one initialized `MooncakeStoreConnector`, directly or within `MultiConnector`, with reusable KV storage rather than capacity-only mode. Keep `enable_lookup=true` (the default) in `kv_connector_extra_config`. |
| Cache namespace | Set a nonempty, deployment-specific `cache_prefix` in `kv_connector_extra_config`. Keep it identical across compatible workers and distinct across model revisions or incompatible producers. A model basename does not isolate a deployment. |
| Hash policy | Set `PYTHONHASHSEED=0` in the launcher and every rank, keep prefix caching enabled, and use `--prefix-caching-hash-algo sha256`. |
| GPU events | Set `VLLM_KV_EVENTS_USE_INT_BLOCK_HASHES=1` and enable `--kv-events-config` publication. Store keys use the full 64-character SHA-256 digest; GPU events use its low 64 bits. |
| Inputs | Use a text-generation model without prompt embeddings or encoder-decoder inputs. Existing Dynamo-normalized cache namespaces and LoRA are supported. Multimodal inputs, arbitrary extra keys, speculative decoding, and Eagle are not supported by this hint. |
| Scheduler | Use the default vLLM scheduler or Dynamo's known `InstrumentedScheduler`; arbitrary scheduler overrides cannot establish compatible hash semantics. |
| Parallelism | Tensor and pipeline parallelism are supported using the connector's exact required object prefixes. Keep prefill and decode context parallelism at one. Use one data-parallel rank per endpoint; the pinned public RPC does not expose all ranks for an internal or hybrid multi-DP endpoint. |
| Hybrid geometry | Use supported full-attention groups, optionally combined with sliding-window groups or Mamba in `align` mode. The adapter validates concrete group specifications and observable token boundaries. Unsupported geometry disables the hint. |

After initialization, Dynamo calls a bounded named worker RPC and compares the resolved descriptors from the relevant ranks. It publishes the supported contract as optional runtime metadata. It does not reconstruct object membership from CLI topology. If forward pass metrics (FPM) also need the GC worker extension, Dynamo selects the known composite extension; unrelated extension conflicts are rejected instead of overwritten.

These worker flags illustrate the event and hash requirements; add them to a deployment with the supported Store connector and its `MOONCAKE_CONFIG_PATH` configuration:

```bash
export PYTHONHASHSEED=0
export VLLM_KV_EVENTS_USE_INT_BLOCK_HASHES=1

python -m dynamo.vllm \
  --model Qwen/Qwen3-0.6B \
  --worker-extension-cls dynamo.vllm.mooncake_store_runtime.MooncakeStoreWorkerExtension \
  --prefix-caching-hash-algo sha256 \
  --kv-transfer-config '{"kv_connector":"MooncakeStoreConnector","kv_role":"kv_both","kv_connector_extra_config":{"cache_prefix":"qwen3-0.6b-revision-a"}}' \
  --kv-events-config '{"publisher":"zmq","topic":"kv-events","endpoint":"tcp://*:20080","enable_kv_cache_events":true}'
```

Use a unique GPU event port for each worker endpoint on the same host. Keep the workers' configured Mooncake master and default tenant associated with the frontend's publisher endpoint for the deployment lifetime. The adapter cannot verify a live master or tenant through the connector, so these are deployment requirements, not resolved runtime facts.

### Mooncake Publisher and Frontend

Build Mooncake with `ENABLE_KV_EVENTS=ON`. Configure the master publisher with these options:

| Master option | Requirement |
| --- | --- |
| `enable_kv_events` | `true` |
| `kv_events_bind_endpoint` | Bind the endpoint that the frontend can reach, for example `tcp://*:5557`. |
| `kv_events_backend_id` | Use a fixed backend identity for the configured master. Do not multiplex unrelated backends into the endpoint. |
| `kv_events_emit_object_key` | `true`; disabling it suppresses `stored` and `removed` events entirely. |
| `kv_events_emit_legacy_compat` | Keep the default `true`; the subscriber also accepts the current event names. |

The publisher's model, block-size, LoRA, salt, and data-parallel context must describe the associated deployment. See the [pinned Mooncake publisher contract](https://github.com/kvcache-ai/Mooncake/blob/ffe013517eaafa8f33e5e0ee034fd6b8f5561e92/docs/source/design/kv-event/publisher-design.md) for `kv_events_model_name`, `kv_events_block_size`, `kv_events_lora_name`, `kv_events_additional_salt`, and `kv_events_dp_rank`.

Select the backend on a supporting frontend:

```bash
DYN_MOONCAKE_KV_EVENTS_ENDPOINT=tcp://127.0.0.1:5557 \
python -m dynamo.frontend \
  --router-mode kv \
  --shared-cache-type mooncake-store \
  --shared-cache-multiplier 0.5
```

`DYN_MOONCAKE_KV_EVENTS_ENDPOINT` is the only frontend endpoint source. The feature requires local GPU event learning and positive device-overlap credit. It rejects `--load-aware`, `--no-router-kv-events`, `--use-remote-indexer`, non-KV router modes, and zero `--router-kv-overlap-score-credit`. Existing worker-role and custom-selector cache-input gates still apply; a decode-only hop does not gain a store subscriber. See [Frontend Configuration](../../../../../reference/components/frontend-configuration.mdx#kv-scoring-and-cache-locality) for the canonical flag reference.

### Readiness and Limitations

A reusable candidate requires all required object prefixes for each participating group and digest. CPU and disk copies count as independent readable media; removing one does not erase the other. Duplicate stores do not establish a different rank's object, and `group_id` is an identity, not a completion barrier.

The router joins learned GPU identities with store events and returns only proven common token boundaries strictly shorter than the request. Full-attention groups require every block through the boundary. Sliding-window groups require the active window; Mamba-align groups require the state at that boundary. Missing older window blocks or Mamba states therefore need not prevent a later candidate from succeeding. Finer-boundary reuse can remain uncounted even with complete event history.

The main GPU event span must divide every full-attention group's physical span. A sliding-window group's span must either be a multiple of that event span or need only the terminal block to cover its window. Candidate boundaries align to the least common multiple of the main event span, coordinator alignment, and every participating group span. For example, a 32-token full-attention group plus an 8-token sliding-window group with an 8-token window is observable, but a 16-token window needs an interior hash that the 32-token event stream does not expose. The latter geometry disables the hint instead of inventing that hash.

- **Cold state:** A cold router needs both learned GPU identity and observed Store residency. Publisher radix-tree replay retains positive input-eligibility provenance where available. Concurrent radix-tree and other hash-only snapshots are conservatively ineligible for shared identity learning until eligible live GPU `Stored` events arrive. GPU replay alone cannot reconstruct a warm Mooncake Store.
- **Event-only estimate:** There is no authoritative query, startup snapshot, replay service, or heartbeat. Detected sequence discontinuities, malformed relevant events, and surfaced stream failures invalidate residency. Silent reconnects and undetected event loss, including a lost final removal, can leave stale estimates.
- **Bounded state:** Capacity limits can discard evidence and reduce shared-cache hints. Contract changes discard learned identity and residency; compatible membership changes retain the compatible domain.
- **Optional fallback:** Missing or incompatible metadata from any admitted worker, an unsupported private API, or unsupported inputs disables only the shared hint. Normal GPU routing and worker admission remain available. The adapter does not support Mooncake HA failover or non-default tenants.
- **Mixed versions:** Default deployments retain the N-2 worker/frontend compatibility contract when optional metadata is absent. Explicitly selecting `mooncake-store` requires a frontend that recognizes this value.

## Further Reading

- [Offloading Support Matrix](../../router/offloading-support-matrix.md) — cross-framework support matrix for KV routing with offloading
- [vLLM KV offloading guide](https://docs.vllm.ai/en/latest/features/kv_offloading_usage/) — connector configuration reference
- [Configuration and Tuning](../../router/configuration-and-tuning.md) — full router flag reference, including lower-tier cache-hit weights
- [Router Guide](../../router/router-guide.md) — routing modes and deployment topologies
- [KV Cache Offloading](kv-cache-offloading.md) — LMCache and FlexKV offloading backends for vLLM
- [Using HiCache](../sglang/hicache.md) — the SGLang counterpart: tier-aware routing with HiCache
