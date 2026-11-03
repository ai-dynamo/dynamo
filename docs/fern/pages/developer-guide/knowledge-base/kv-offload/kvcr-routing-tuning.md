---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Tune Routing for KVCR
subtitle: Recommended router settings for KVCR, tuning the cache weights, and keeping fetches on fast links
---

The KV router scores each worker by the prefill work it would have to do. Cached KV lowers that cost. Three settings control how much credit KV outside the GPU gets:

| Setting | Default | Applies to |
|---|---|---|
| `host_cache_hit_weight` | `0.75` | Prefix blocks the worker holds in host memory |
| `disk_cache_hit_weight` | `0.25` | Prefix blocks the worker holds on disk |
| `shared_cache_multiplier` | `0` (`0.5` with `--shared-cache-type hicache`) | Prefix blocks the worker could fetch from a peer through KVCR, and shared-cache hits |

`0` ignores those blocks; `1` weights them the same as GPU hits.

> [!IMPORTANT]
> With the default `shared_cache_multiplier` of `0`, the router does not prefer an idle worker that could fetch a prefix over a busy worker that holds it. KVCR still fetches when the router picks a worker for other reasons. Credit for KVCR-fetchable blocks requires Dynamo with [#16701](https://github.com/ai-dynamo/dynamo/pull/16701); on Dynamo v1.6, `shared_cache_multiplier` is accepted but credits only HiCache shared-cache hits.

## Recommended Routing

Combine three settings so sessions stay on their worker until it is busy, then move to a worker that can fetch their prefix:

- **Soft session affinity.** With the default policy, a session goes to the worker it is bound to regardless of load until the worker crosses the busy threshold, so most turns reuse that worker's GPU cache. Unlike hard affinity, a soft binding is dropped for a request when the worker is overloaded, and the request goes through normal KV-aware selection. With hard affinity, an overloaded bound worker fails the request instead of moving it.
- **A busy threshold** (`--active-decode-blocks-threshold`). This is what marks a worker overloaded. Without one, no worker is ever overloaded and soft affinity never moves a session.
- **Fetchable-block credit** (`shared_cache_multiplier`). When a session moves, this lets the router prefer a worker that can fetch the prefix through KVCR, instead of treating the prefix as lost. Use it with a raised `prefill_load_scale`, so the router weighs prefill work strongly enough not to move sessions between workers for small gains.

Fetchable-block credit needs Dynamo with [#16701](https://github.com/ai-dynamo/dynamo/pull/16701) (not in v1.6.x). The values below come from simulations in that PR: a 24-hour agentic trace on 8 workers with KVCR fetches restricted to the same network domain ([`required`](#keep-fetches-on-fast-links)). There, a multiplier of `0.75` with a `prefill_load_scale` of 16 to 64 beat no credit. If fetches may cross network domains (`preferred`), leave `shared_cache_multiplier` at `0`: the router credits a cross-domain source at the full multiplier, and that did worse than no credit.

1. Create `policy.yaml`:

   ```yaml
   router:
     host_cache_hit_weight: 0.75
     disk_cache_hit_weight: 0.25
   worker_selection:
     aggregated: tuned
     instances:
       - name: tuned
         type: dynamo-default-cost-fn
         parameters:
           shared_cache_multiplier: 0.75
           prefill_load_scale: 16
   ```

   `worker_selection` names the policy for aggregated workers; `dynamo-default-cost-fn` is the default policy, so this keeps default routing and sets only these two parameters. For disaggregated serving, set `prefill` and `decode` instead of `aggregated`.

2. Start the frontend with the policy, session affinity, and a busy threshold:

   ```bash
   python3 -m dynamo.frontend --router-mode kv \
     --router-policy-config policy.yaml \
     --router-session-affinity-ttl-secs 86400 \
     --router-session-affinity-mode soft \
     --active-decode-blocks-threshold 0.9
   ```

   `--router-session-affinity-ttl-secs` turns affinity on and sets how long an idle session keeps its worker. Agent sessions can sit idle for a long time between turns, so use a long TTL; in the same simulations, 1 hour did worse than 24 hours. `--active-decode-blocks-threshold 0.9` marks a worker overloaded above 90% KV cache use; tune it for your load, at runtime if needed through the frontend's `/busy_threshold` endpoint.

3. Send `X-Dynamo-Session-ID: <session-id>` on every request of a session. Requests without it are routed normally.

The policy file is read at startup; restart the frontend after changing it. A frontend started with `shared_cache_multiplier` at `0` does not track fetchable blocks, and per-request overrides cannot turn it on. On Kubernetes, mount the file from a `ConfigMap` on the frontend; see [Swap the Worker-Selection Policy](../../../kubernetes/kv-aware-routing/dynamo-frontend.md#swap-the-worker-selection-policy). For details, see [Session Affinity](../modular-components/router/configuration-and-tuning.md#session-affinity) and [Busy Thresholds](../modular-components/router/worker-filtering.md#busy-thresholds).

## Tune the Weights

For the host and disk weights alone, the frontend flags `--router-host-cache-hit-weight` and `--router-disk-cache-hit-weight` also work. Set `shared_cache_multiplier` only in the policy file; its `--shared-cache-multiplier` flag is deprecated.

Lower `host_cache_hit_weight` if host-memory hits are slower than recomputing on an idle worker. Raise `disk_cache_hit_weight` toward the host weight for fast local NVMe. Lower `shared_cache_multiplier` if fetches across your network are slow. Set all three to `0` for a baseline that routes on GPU cache and load only.

## Keep Fetches on Fast Links

A KVCR fetch over a slow link can be slower than recomputing. Restrict fetch sources to the target worker's network domain, such as a host or an RDMA rack:

```bash
DYN_TOPOLOGY_ENABLED=true
DYN_KV_TRANSFER_DOMAIN=rack           # topology key; its value is read from /etc/dynamo/topology/rack
DYN_KV_TRANSFER_ENFORCEMENT=required  # or preferred
```

Set these on each worker. Use `required` for KVCR: it allows only sources in the same domain. `preferred` also uses another domain when no same-domain source has a longer prefix. On Kubernetes, the operator sets these from the DGD's `KvTransferPolicy`; see [Topology-Aware KV Transfer](../modular-components/router/topology-aware-kv-transfer.md).

## Related Topics

- [KV Cache Runner (KVCR)](kvcr.md)
- [Configure KVCR for vLLM](configure-kvcr-vllm.md)
- [Configuration and Tuning](../modular-components/router/configuration-and-tuning.md)
