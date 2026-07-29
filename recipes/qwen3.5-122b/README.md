<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Qwen3.5-122B-A10B-FP8 Recipes (H200)

Recipes for [Qwen/Qwen3.5-122B-A10B-FP8](https://huggingface.co/Qwen/Qwen3.5-122B-A10B-FP8),
the FP8 checkpoint of [Qwen/Qwen3.5-122B-A10B](https://huggingface.co/Qwen/Qwen3.5-122B-A10B)
(122B total / 10B active hybrid MoE — Gated DeltaNet linear attention + MoE with full
attention every 4th layer). The FP8 weights fit a single 143 GB H200 at the full
262,144-token context.

## Configurations

Dynamo + vLLM deployment profiles for the agentic workload on **H200**.

|                          | H200 aggregated agentic (tp1 + MTP)                 | H200 disaggregated agentic (1P2D)                 |
| ------------------------ | --------------------------------------------------- | ------------------------------------------------- |
| **GPU**                  | 1x H200 per worker; scale via `replicas`            | 1x H200 prefill + 2x H200 decode (3x total)       |
| **Mode**                 | Aggregated                                          | Prefill/decode disaggregated (1P2D)               |
| **Framework**            | vLLM (runtime `1.3.0`)                              | vLLM (runtime `1.3.0`)                            |
| **Precision**            | FP8 weights + FP8 KV                                | FP8 weights + FP8 KV                             |
| **Parallelism**          | TP1                                                 | TP1 (per worker)                                 |
| **MoE backend**          | `triton` (Hopper FP8)                               | `triton` (Hopper FP8)                            |
| **KV cache manager**     | Hybrid (DeltaNet SSM + attention)                   | Hybrid (DeltaNet SSM + attention)                |
| **Routing**              | KV-aware (`DYN_ROUTER_MODE=kv`) + worker KV events  | KV-aware + worker KV events                      |
| **Speculative decoding** | MTP, `num_speculative_tokens=3`                     | None — see Limitations                           |
| **Context length**       | 262,144 (model default)                             | 262,144 (model default)                          |
| **KV transfer**          | N/A (aggregated)                                    | NIXL/UCX over InfiniBand                          |

### Why TP1 + replicas + KV routing (aggregated)

Every multi-GPU engine layout measured (TP2, TP4, TP8, DP+EP) delivered less output
throughput **per GPU** than independent TP1 replicas at the agentic SLA. The winning
layout is one engine per GPU, scaled horizontally behind the KV-aware router. KV routing
is load-bearing: the replicas are independent engines and agentic requests share
~57k-token prefixes, so the router must land each request on the replica that already
holds its prefix. The DGD ships `replicas: 2` (minimal KV-router validation); a full 8x
H200 node runs `replicas: 8`.

### Why 1P2D (disaggregated)

Disaggregation splits prefill and decode onto separate GPUs joined by NIXL KV transfer
over InfiniBand. Across a topology sweep at the agentic SLA, **1 prefill : 2 decode**
gave the best output tok/s per GPU: system throughput is set by the decode-worker count
(2 decode workers are needed to meet the SLA — a single decode worker cannot hold the
concurrent long-context KV), while one prefill worker keeps up with the ~90%-cache-hit
prefill load. Adding more decode or prefill raises total throughput but lowers per-GPU
throughput at this fixed operating point. MTP is not used (see Limitations).

## Supported features

- Modalities: Text, Image, Video
- Reasoning (`--dyn-reasoning-parser qwen3`)
- Tool calling (`--dyn-tool-call-parser qwen3_coder`)

## Prerequisites

1. **Dynamo Platform installed** on the cluster with DGD CRDs served.
2. **NGC/nvcr image pull access** for `nvcr.io/nvidia/ai-dynamo` — the deploy manifests do
   not set `imagePullSecrets`, so create `nvcr-secret` and attach it to the namespace's
   default service account (see Quick Start note).
3. **Hugging Face token** with access to `Qwen/Qwen3.5-122B-A10B-FP8` (public, Apache-2.0),
   stored as `hf-token-secret` — used by the model-download Job.
4. **`model-cache` PVC** (ReadWriteMany), populated via `model-cache/`.
5. **(disaggregated only)** GPU-local RDMA NICs exposed to pods (e.g. an `rdma/ib` device
   plugin giving each worker its topology-local NIC) for NIXL KV transfer.

## Quick Start

### 1. Namespace + HF secret
```bash
export NAMESPACE=your-namespace
kubectl create namespace ${NAMESPACE}
kubectl create secret generic hf-token-secret --from-literal=HF_TOKEN="your-token" -n ${NAMESPACE}
```
> [!NOTE]
> If the namespace lacks nvcr pull access:
> ```bash
> kubectl create secret docker-registry nvcr-secret \
>   --docker-server=nvcr.io --docker-username='$oauthtoken' \
>   --docker-password="<your-NGC-API-key>" -n ${NAMESPACE}
> kubectl patch serviceaccount default -n ${NAMESPACE} \
>   -p '{"imagePullSecrets":[{"name":"nvcr-secret"}]}'
> ```

### 2. Storage
> [!NOTE]
> Edit `model-cache/model-cache.yaml` — set `storageClassName` to a ReadWriteMany class on your cluster.
```bash
kubectl apply -f model-cache/model-cache.yaml -n ${NAMESPACE}
```

### 3. Download the model
```bash
kubectl apply -f model-cache/model-download.yaml -n ${NAMESPACE}
kubectl wait --for=condition=Complete job/model-download -n ${NAMESPACE} --timeout=7200s
```

### 4. Deploy the DGD
```bash
MODE=agg # or disagg
kubectl apply -f vllm/${MODE}-h200-agentic/deploy.yaml -n ${NAMESPACE}
```
Aggregated: scale to a full node by editing `spec.components[agg].replicas` to `8`.

### 5. Smoke test
```bash
kubectl port-forward svc/$(kubectl get svc -o name -n ${NAMESPACE} | grep frontend | head -1 | cut -d/ -f2) 8000:8000 -n ${NAMESPACE} &
curl http://localhost:8000/v1/models
curl http://localhost:8000/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "Qwen/Qwen3.5-122B-A10B",
  "messages": [{"role": "user", "content": "Hello"}],
  "max_tokens": 32
}'
```

### 6. Benchmark
See [perf/README.md](perf/README.md) — mooncake agentic trace replay for both profiles.

## Optimization targets

| Workload | Median ISL | Median OSL | KV cache hit rate | User output tok/s |
| -------- | ---------- | ---------- | ----------------- | ----------------- |
| Agentic  | 64k        | 400        | 90%               | 50                |

## Performance results

Measured on H200 against the **real** 15% agentic mooncake trace (3,541 requests, block
512, closed-loop concurrency 8; SLA = P50 TTFT < 5 s **and** ≥ 50 output tok/s/user).
Headline metric is system output tok/s per GPU.

| Recipe                       | GPUs | tok/s/GPU @ SLA | user tok/s (P50) | TTFT (P50) |
| ---------------------------- | ---- | --------------- | ---------------- | ---------- |
| Aggregated TP1 + MTP (kv)    | 2    | ~380            | 139              | 0.72 s     |
| Disaggregated 1P2D (kv)      | 3    | ~184            | 87               | 0.94 s     |

KV-aware routing beats `round_robin` by ~19–22% output tok/s (and ~2.5x lower TTFT) on
both profiles, by landing each request on the worker that already holds its ~57k-token
shared prefix.

## Known issues

- **MTP + disaggregation is not supported on this arch.** Disaggregation needs
  `VLLM_SSM_CONV_STATE_LAYOUT=DS` for NIXL's Mamba conv-state transfer, which conflicts
  with the `mamba_cache_mode='align'` that MTP + prefix caching forces (vLLM
  [#38898](https://github.com/vllm-project/vllm/issues/38898)) — the decode engine crashes
  under concurrent long-context traffic. The **aggregated** profile ships MTP
  (`num_speculative_tokens=3`); the **disaggregated** profile runs without it.
