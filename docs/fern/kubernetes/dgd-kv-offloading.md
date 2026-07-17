---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Set up KV Cache Offloading
subtitle: Add a KV cache offloading connector to a worker in a DynamoGraphDeployment so KV blocks spill to host memory or disk.
---

KV cache offloading lets a worker keep more KV cache than fits in GPU memory by spilling blocks to host (CPU) memory or local disk. This serves longer contexts and reuses cached prefixes across requests. This page shows how to turn it on **inside a DynamoGraphDeployment (DGD)** — the engine-internals and local-CLI details live in the per-backend pages linked at the end.

This is a [how-to](dgd-guide.md) for an existing deployment. If you have not authored a DGD yet, start with the [DGD Guide](dgd-guide.md).

## The pattern

Offloading is configured **on the worker**, not the Frontend. For vLLM workers, you pass a `--kv-transfer-config` JSON argument that names a connector, then size the offload tiers with environment variables. The shape is always the same:

```yaml
spec:
  components:
  - name: VllmDecodeWorker
    type: worker
    podTemplate:
      spec:
        containers:
        - name: main
          command:
          - python3
          - -m
          - dynamo.vllm
          args:
          - --model
          - Qwen/Qwen3-32B
          - --kv-transfer-config
          - '{"kv_connector":"<Connector>","kv_role":"kv_both", ...}'
          env:
          - name: <TIER_SIZE_ENV>
            value: "<gb>"
```

## Choose a connector

Each offloading backend uses the same `--kv-transfer-config` hook with a different `kv_connector`. Pick one — they are alternatives, not layers.

| Backend | `kv_connector` | Sizing | Best for |
|---|---|---|---|
| **KVBM** (KV Block Manager) | `DynamoConnector` (with `kv_connector_module_path: kvbm.vllm_integration.connector`) | `DYN_KVBM_CPU_CACHE_GB`, `DYN_KVBM_DISK_CACHE_GB` (env) | Dynamo-native CPU/disk tiering across vLLM and TensorRT-LLM |
| **LMCache** | `LMCacheMPConnector` | LMCache server config | Prefill-once / reuse-everywhere across a fleet |
| **FlexKV** | `FlexKVConnectorV1` | `DYNAMO_USE_FLEXKV=1`, `FLEXKV_CPU_CACHE_GB` (env) | Distributed KV offloading runtime |
| **SGLang HiCache** | n/a — use `--enable-hierarchical-cache` and `DYN_SHARED_CACHE_TYPE` instead | `DYN_SHARED_CACHE_MULTIPLIER` | SGLang hierarchical cache with a tier-aware router |

> [!NOTE]
> SGLang HiCache does not use the `--kv-transfer-config` connector mechanism. On an SGLang worker, set `--enable-hierarchical-cache` in `args` and `DYN_SHARED_CACHE_TYPE` in the container `env`. See [Using HiCache](../backends/sglang/sglang-hicache.md).

## Worked example: KVBM on a vLLM worker

This aggregated Qwen3-32B worker offloads KV blocks to 100 GB of host memory. It is the worker section of the [agg-kvbm recipe](https://github.com/ai-dynamo/dynamo/blob/main/recipes/qwen3-32b/vllm/agg-kvbm/deploy.yaml):

```yaml
spec:
  components:
  - name: VllmDecodeWorker
    type: worker
    replicas: 1
    podTemplate:
      spec:
        containers:
        - name: main
          image: nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.2.1
          command:
          - python3
          - -m
          - dynamo.vllm
          args:
          - --model
          - Qwen/Qwen3-32B
          - --kv-transfer-config
          - '{"kv_connector":"DynamoConnector","kv_role":"kv_both","kv_connector_module_path":"kvbm.vllm_integration.connector"}'
          envFrom:
          - secretRef:
              name: hf-token-secret
          env:
          # Tune the CPU cache size for your node's available host memory.
          - name: DYN_KVBM_CPU_CACHE_GB
            value: "100"
          resources:
            limits:
              nvidia.com/gpu: "1"
              memory: 200Gi        # host memory must hold the CPU cache tier
            requests:
              nvidia.com/gpu: "1"
              memory: 150Gi
```

Two things to size together:

- `DYN_KVBM_CPU_CACHE_GB` (and optionally `DYN_KVBM_DISK_CACHE_GB`) set the offload tier sizes.
- `resources.limits.memory` must be large enough to hold the CPU cache tier plus the engine's normal footprint. The recipe requests 150–200 GB for a 100 GB CPU cache.

To add a disk tier, set `DYN_KVBM_DISK_CACHE_GB` and mount a volume for it. To export offloading metrics to Prometheus, set `DYN_KVBM_METRICS=true`.

## Verify

After the worker is `Running`, send repeated requests that share a long prefix and confirm the second request has lower TTFT. With `DYN_KVBM_METRICS=true`, check the offload hit-rate metrics — see [Metrics](observability/metrics.md).

## Related pages

These cover engine internals, the local-CLI workflow, and tuning for each backend:

- [KVBM Guide](../components/kvbm/kvbm-guide.md) — KV Block Manager architecture and the full env-var reference.
- [LMCache Integration](../integrations/lmcache-integration.md) — LMCache server and connector details.
- [FlexKV Integration](../integrations/flexkv-integration.md) — FlexKV runtime configuration.
- [Using HiCache](../backends/sglang/sglang-hicache.md) — SGLang hierarchical cache.
- [Custom KV Events](../integrations/kv-events-custom-engines.md) — publish KV events from a custom engine.
