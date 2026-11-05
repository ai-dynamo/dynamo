<!-- # SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0 -->

# SGLang

See [docs/backends/sglang/](../../../../docs/fern/pages/developer-guide/knowledge-base/modular-components/backends/sglang/overview.md) for documentation.

## KVCR-owned inventory

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
