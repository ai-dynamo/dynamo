<!-- # SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0 -->

# SGLang

See [docs/backends/sglang/](../../../../docs/fern/pages/developer-guide/knowledge-base/modular-components/backends/sglang/overview.md) for documentation.

## Recurrent checkpoint events

For hybrid models (attention plus Mamba, Gated DeltaNet, KDA or short-conv layers),
`DYN_SGLANG_RECURRENT_KV_EVENTS=1` makes the worker ask SGLang to publish one
tagged `BlockStored` per cached recurrent state and a tagged `BlockRemoved` when
the state is evicted (`KVEventsConfig.publish_recurrent_checkpoints`,
sgl-project/sglang#51647). SGLang keeps one recurrent group, so the worker's
join of group events publishes each state as one `recurrent_checkpoint` event,
exactly as it does for vLLM's three groups. The router then credits the worker
only up to its deepest checkpoint instead of every cached attention page. The
default remains attention-only publication; enable the option only after every
router and event consumer understands `kv_cache_group` events.

The option needs `--kv-events-config` with a ZMQ publisher, the radix cache, the
local indexer, a model SGLang serves with its Mamba radix-cache component, and a
page size such that `page_size * dcp_size` is a multiple of the Mamba chunk size
(64 tokens for Gated DeltaNet). SGLang keeps states on that grid, and the router
scores checkpoints only at whole indexer blocks, so other page sizes fail at
startup. Dense models, SGLang builds without the field, and the state-agent event
source fail at startup as well. The runtime metadata publishes the layout under
the `sglang_mamba_last_page_v1` contract. Host-tier (HiCache) checkpoints are not
published yet.

## KVCR-owned inventory

> [!IMPORTANT]
> Dynamo 1.7.0 uses [KVCR 0.1.1](https://github.com/ai-dynamo/kvcr/tree/v0.1.1)
> as its KVCR baseline. KVCR 0.1.1 does not support the SGLang HiCacheStorage or
> direct linker engine integration described in this section. KVCR 0.1.1 supports
> SGLang only for the KV hint protocol, and hint protocol support does not provide
> engine integration. SGLang engine integration is proposed in SGLang pull requests
> [#45647](https://github.com/sgl-project/sglang/pull/45647) and
> [#45650](https://github.com/sgl-project/sglang/pull/45650).
> The instructions below are preview instructions. No KVCR version that Dynamo
> pins includes SGLang engine integration, and no qualified Dynamo, SGLang, and
> KVCR version combination is recorded for them.
> Ordinary SGLang serving without KVCR is not affected.

KVCR inventory publication is opt-in for the in-process `python -m dynamo.sglang`
HiCacheStorage and direct linker paths. It requires a matching SGLang backend that
reports the `kvcr_inventory` capability and engine-lifetime cache-owner metadata.

For HiCacheStorage, configure the KVCR backend with
`"publish_inventory": true`, remote hints enabled, and an explicit reachable
`control_advertise_host`.

For the experimental direct linker, use TP1 and configure the worker with:

```bash
--tp-size 1 \
--enable-unified-cache-external-linker \
--unified-cache-external-linker-backend kvcr \
--hicache-storage-backend-extra-config '{"publish_inventory": true, "kvcr_service_socket_path": null, "advertise_host": "192.0.2.10"}'
```

Replace `192.0.2.10` with a non-loopback address reachable by peers. Set distinct
control and NIXL ports for colocated workers. Do not use HiCacheStorage options
such as `control_advertise_host` in the linker configuration. See the
[experimental linker quick start](https://github.com/ai-dynamo/kvcr/pull/129) for
a complete example.

For either path, enable KV events with a ZMQ publisher and `"format": "dynamo"`.
Start a state-agent host in the workers' namespace before starting the workers:

```bash
DYN_NAMESPACE=<worker-namespace> python -m dynamo.kv_state_agent
```

The worker reads the raw-event endpoint, incarnation-specific topic, cache owner,
and peer-control endpoint from SGLang startup metadata. It attaches before model
registration and uses `state_agent_v2` instead of the legacy event publisher.
The router can then distinguish framework GPU/HiCache-host residency from actual
KVCR-owned DRAM residency, even though both host caches use CPU memory.

Upgrade the Dynamo frontend, worker, and state-agent host together when enabling
this mode. Existing deployments without the capability keep the legacy path;
vLLM attachment descriptors without an ingress protocol keep their existing
default. The new attachment requires a single-host LLM engine, PP=1, and one
tokenizer worker; the direct linker additionally requires TP1 and DP1. Incomplete
source metadata is rejected at startup.

This is a live process-local inventory, not durable recovery: detaching the
engine clears its KVCR owner, and a detected raw-event sequence gap withdraws
readiness until the worker is restarted. Router hints remain advisory; storage
lookup and completed delivery still determine usable pages.
