---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Native KV Offloading
subtitle: vLLM's OffloadingConnector with KV-aware routing in Dynamo
---

This guide shows how to run vLLM's native CPU KV cache offloading
(`OffloadingConnector`) with Dynamo's KV router. For Dynamo-side offloading
backends such as LMCache and FlexKV, see
[KV Cache Offloading](kv-cache-offloading.md).

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
| Shared-pool routing | 🚧 | Dynamo consumes per-event locality (`LOCAL` or absent → worker-local; `REMOTE` or unknown → dropped). A shared-pool index is still planned. |

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

## Retained GPU Decode-to-Prefill Reuse

**Experimental.** `GPUOffloadingConnector` with `GPUOffloadingSpec` retains sealed
KV blocks in a dedicated CUDA allocation. Decode-to-prefill reuse copies from
the decode GPU allocation into a retained allocation on the prefill GPU through
KVCR and NIXL CUDA IPC, including an opt-in multi-node NVLink (MNNVL) profile.
Local model-cache stores and loads use device-to-device
copies. CPU memory carries control messages and pointer/size descriptors; it
does not hold KV payloads. Budget additional VRAM on both workers.

The implementation requires one engine per role, tensor parallel size one or two,
and identical raw cache layouts. TP2 requires a single local multiprocessing
engine with data parallel size one and no pipeline or context parallelism.
The default transport uses a same-host GPU pair
with working CUDA IPC. The MNNVL profile supports hosts in the same healthy
NVLink fabric and clique. Canonical layouts, replicated layouts, RDMA, CPU/disk payload
pools and the versioned state-agent ingress are unsupported. Use the native
worker event publisher and a compatible vLLM offloading/tiering API.

For TP2, set `gpu_rank_devices` to the explicit CUDA device ordinal for each
rank, such as `[0, 1]`, in the connector's `kv_connector_extra_config`. Each
worker imports only its own bank and verifies the physical GPU UUID. The
`gpu_bytes_to_use` budget counts both rank banks together: a 2-GiB budget
allocates at most 1 GiB on each GPU. KVCR transfers both labeled rank parts in
one operation, and ready inventory waits for both model workers to complete
their stores. A TP2 compatibility suffix separates this inventory from TP1;
the certified base domain must still identify the exact model and raw layout.

Configure the GPU secondary tier with both
`"type": "GPUKVCRSecondaryTierManager"` and
`"module_path": "kvbm.vllm_integration.gpu_offloading"`. The module path is
required by the tier factory when building Prometheus metric definitions.
The connector and spec also use the out-of-tree module
`kvbm.vllm_integration.gpu_offloading`.

On vLLM versions whose `MultiConnector` reconstructs statistics by connector
name, register the out-of-tree connector in each application/engine process
before statistics are reconstructed. A connector's `kv_connector_module_path`
alone does not populate that process-local registry. Use vLLM's public factory
API in the application bootstrap:

```python
from vllm.distributed.kv_transfer.kv_connector.factory import KVConnectorFactory

KVConnectorFactory.register_connector(
    "GPUOffloadingConnector",
    "kvbm.vllm_integration.gpu_offloading",
    "GPUOffloadingConnector",
)
```

Run this once per process. If the name is already registered, verify that
`get_connector_class_by_name("GPUOffloadingConnector")` resolves to the same
class; reject a conflicting registration. Spawned processes need the same
bootstrap registration. Keep statistics enabled during validation.

The retained tier publishes `gpu_direct` events and has an independent lifetime
from the model's normal GPU cache. Native cache clears preserve retained
inventory; retained-cache removals revoke it explicitly. The router requires
`router_hint_gpu_direct`, GPU source endpoints and a matching certified
`router_hint_transfer_domain` before emitting `kv.fetch_gpu@1.0`. The domain is
an operator-supplied layout contract, not automatic model compatibility proof.

With the default same-host profile, both workers require a native NIXL/UCX implementation of the strict
`NIXL_UCX_CUDA_IPC_ONLY=1` policy with `UCX_TLS=cuda_ipc,posix,self` and actual
CUDA memory support. A stock installation that rejects this profile is not
compatible. See the [KVCR GPU endpoint contract](https://github.com/ai-dynamo/kvcr#direct-gpu-memory-endpoints)
for registration, dependency and lifetime requirements.

### MNNVL Profile

**Experimental.** Set `gpu_transport` to `"mnnvl"` in both the connector's
`kv_connector_extra_config` and its `GPUKVCRSecondaryTierManager` entry under
`secondary_tiers`. Set the tier's `ucx_tls` to
`["cuda_ipc", "tcp", "self"]`. Keep `router_hint_transfer_domain` and the tier's
`compatibility_domain` equal to the same certified base layout domain. Dynamo
observes the assigned GPU's completed NVML fabric state and incorporates the
cluster UUID and clique ID into the effective capability and inventory domain.
Workers in another fabric or transport profile cannot use that inventory.

The retained banks use physical CUDA VMM allocations with
`CU_MEM_HANDLE_TYPE_FABRIC`. Torch aliases the allocation and retains its owner;
local rank imports use FABRIC handles. Both endpoints need an accessible IMEX
channel and a NIXL/UCX build with KVCR's additive MNNVL dependency patches.
For Slurm sites using `switch/nvidia_imex`, reserve a shared channel for the job's
segment. Sites supporting the unique-channel option can use
`--segment=2 --network=unique-channel-per-segment` for a two-host test; propagate
the job's private IMEX device namespace into every worker container.

Do not set `NIXL_UCX_CUDA_IPC_ONLY=1` for this profile. KVCR applies the transport
policy to its dedicated backend context: CUDA IPC FABRIC GPU RMA, TCP control
messages, and no host-payload fallback. This leaves the separate vLLM
prefill-to-decode backend's context unchanged. Ordinary CUDA malloc and managed
allocations are rejected as retained MNNVL banks. Validate actual cross-host
bytes, CUDA copy activity and quiescent teardown before measuring performance.

Enable the separate GPU router option only after measuring its acquisition,
transfer, onboarding and prefill costs. CPU coefficients do not qualify it.
See [Decode-to-Prefill Reuse](../../router/configuration-and-tuning.md#decode-to-prefill-reuse)
for selection behavior. Missing capabilities or inventory leave the normal
request path available. A deadline does not cancel native DMA: uncertain source
pins and target allocations remain retained until actual quiescence. The strict
GPU profiles do not provide native peer-death recovery.

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

## Further Reading

- [Offloading Support Matrix](../../router/offloading-support-matrix.md) — cross-framework support matrix for KV routing with offloading
- [vLLM KV offloading guide](https://docs.vllm.ai/en/latest/features/kv_offloading_usage/) — connector configuration reference
- [Configuration and Tuning](../../router/configuration-and-tuning.md) — full router flag reference, including lower-tier cache-hit weights
- [Router Guide](../../router/router-guide.md) — routing modes and deployment topologies
- [KV Cache Offloading](kv-cache-offloading.md) — LMCache and FlexKV offloading backends for vLLM
- [Using HiCache](../sglang/hicache.md) — the SGLang counterpart: tier-aware routing with HiCache
