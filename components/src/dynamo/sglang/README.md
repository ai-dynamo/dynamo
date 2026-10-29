<!-- # SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0 -->

# SGLang

See [docs/backends/sglang/](../../../../docs/fern/pages/developer-guide/knowledge-base/modular-components/backends/sglang/overview.md) for documentation.

## KVCR-owned HiCache inventory

KVCR inventory publication is opt-in for the in-process `python -m dynamo.sglang`
HiCacheStorage path. It requires a matching SGLang backend that reports the
`kvcr_inventory` capability and engine-lifetime cache-owner metadata.

Keep KV events enabled and configure the KVCR backend with
`"publish_inventory": true`, remote hints enabled, and an explicit reachable
`control_advertise_host`. Use a ZMQ KV-event publisher with `"format": "dynamo"`.
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
tokenizer worker; incomplete source metadata is rejected at startup.

This is a live process-local inventory, not durable recovery: detaching the
engine clears its KVCR owner, and a detected raw-event sequence gap withdraws
readiness until the worker is restarted. Router hints remain advisory; storage
lookup and completed delivery still determine usable pages. This integration
does not enable direct-device linker mode.
