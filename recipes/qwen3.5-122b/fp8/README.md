<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Qwen3.5-122B-A10B-FP8 Recipes (H200)

[Qwen/Qwen3.5-122B-A10B-FP8](https://huggingface.co/Qwen/Qwen3.5-122B-A10B-FP8) — 122B total /
10B active hybrid MoE (Gated DeltaNet + full attention every 4th layer). FP8 weights fit one
143 GB H200 at the full 262,144-token context.

## Configurations

|                          | Aggregated (tp1 + MTP)         | Disaggregated (1P2D)          |
| ------------------------ | ------------------------------ | ----------------------------- |
| **GPU**                  | 1x H200 per replica, `replicas: 2` | 1x prefill + 2x decode (3x)   |
| **Framework**            | Dynamo 1.3.0 / vLLM 0.23       | Dynamo 1.3.0 / vLLM 0.23      |
| **Precision**            | FP8 weights + FP8 KV           | FP8 weights + FP8 KV          |
| **Parallelism**          | TP1                            | TP1 per worker                |
| **MoE backend**          | `triton`                       | `triton`                      |
| **KV cache manager**     | Hybrid (DeltaNet SSM + attention) | Hybrid                     |
| **Routing**              | KV-aware + worker KV events    | KV-aware + worker KV events   |
| **Speculative decoding** | MTP, `num_speculative_tokens=3` | None — see Known issues      |
| **Context length**       | 262,144                        | 262,144                       |
| **KV transfer**          | n/a                            | NIXL/UCX over InfiniBand      |
| **Async scheduling**     | enabled                        | disabled on decode — see Known issues |

Scale aggregated to a full node with `replicas: 8`.

## Supported features

- Modality: text, image, video
- Reasoning (`--dyn-reasoning-parser qwen3`)
- Tool / function calling (`--dyn-tool-call-parser qwen3_coder`)

## Prerequisites

1. Dynamo Platform installed with DGD CRDs served.
2. nvcr pull access for `nvcr.io/nvidia/ai-dynamo`. The manifests set no
   `imagePullSecrets` — create `nvcr-secret` and attach it to the default service account.
3. Hugging Face token as `hf-token-secret` (model is public, Apache-2.0).
4. `model-cache` PVC (ReadWriteMany), populated via `model-cache/`.
5. Disaggregated only: GPU-local RDMA NICs exposed to pods (`rdma/ib` device plugin) for
   NIXL KV transfer.

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
> Edit `model-cache/model-cache.yaml` — set `storageClassName` to a ReadWriteMany class.
```bash
kubectl apply -f model-cache/model-cache.yaml -n ${NAMESPACE}
```

### 3. Download the model
```bash
kubectl apply -f model-cache/model-download.yaml -n ${NAMESPACE}
kubectl wait --for=condition=Complete job/model-download -n ${NAMESPACE} --timeout=7200s
```

### 4. Deploy
```bash
MODE=agg # or disagg
kubectl apply -f vllm/${MODE}-h200-agentic/deploy.yaml -n ${NAMESPACE}
```

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
See [perf/README.md](perf/README.md).

## Optimization targets

| Workload | Median ISL | Median OSL | KV cache hit rate | User output tok/s |
| -------- | ---------- | ---------- | ----------------- | ----------------- |
| Agentic  | 64k        | 400        | 90%               | 50                |

## Performance results

Measured 2026-08-05 on the first 1,500 requests of the agentic Mooncake trace (see
[perf/README.md](perf/README.md)), block size 512, closed-loop, both profiles on the same
trace. SLA: P50 TTFT < 5 s and P50 output >= 50 tok/s/user. Per-GPU = system throughput /
GPUs. Each profile is reported at its highest SLA-passing concurrency. Aggregated runs
`replicas: 2`.

| config (shipped) | GPUs | conc | tok/s/GPU | tok/s/user | TTFT (P50) | note |
| ---------------- | ---- | ---- | --------- | ---------- | ---------- | ---- |
| aggregated — 2x TP1 + MTP | 2 | c8 | 375.2 | 137.07 | 708 ms | KV-bound; cannot reach the 50 tok/s/user floor — TTFT 14.7 s at c16 |
| **disaggregated** — 1P2D | 3 | c18 | 244.8 | 51.32 | 3770 ms | at the 50 tok/s/user floor |

## Known issues

1. Disaggregated decode requires `--no-async-scheduling` on vLLM < 0.26.0. The KV-block
   zeroing kernel is not ordered against the NIXL RDMA write and can erase transferred KV.
   It is silent — no error, no failed transfer — but IFEval drops 84.66 → 51.02 and
   GPQA-Diamond 83.84 → 19.19, below the 25% random baseline. Fixed by
   [vllm#45357](https://github.com/vllm-project/vllm/pull/45357) and
   [vllm#48481](https://github.com/vllm-project/vllm/pull/48481). The flag costs ~7%
   throughput; drop it on the Dynamo 1.4.0 runtime (vLLM 0.26.0).
2. MTP is not supported with disaggregation on this architecture. NIXL's Mamba conv-state
   transfer needs `VLLM_SSM_CONV_STATE_LAYOUT=DS`, which conflicts with the
   `mamba_cache_mode='align'` that MTP + prefix caching forces
   ([vllm#38898](https://github.com/vllm-project/vllm/issues/38898)). Aggregated ships MTP;
   disaggregated runs without it.
