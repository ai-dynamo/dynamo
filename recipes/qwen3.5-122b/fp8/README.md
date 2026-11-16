<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Qwen3.5-122B-A10B-FP8 Recipes (H200)

[Qwen/Qwen3.5-122B-A10B-FP8](https://huggingface.co/Qwen/Qwen3.5-122B-A10B-FP8) — 122B total /
10B active hybrid MoE (Gated DeltaNet + full attention every 4th layer). FP8 weights fit one
143 GB H200 at the full 262,144-token context.

## Configurations

|                          | Aggregated (tp2 + MTP)         | Disaggregated (1P2D)          |
| ------------------------ | ------------------------------ | ----------------------------- |
| **GPU**                  | 2x H200 per replica, `replicas: 2` (4x) | 1x prefill + 2x decode (3x)   |
| **Framework**            | Dynamo 1.3.0 / vLLM 0.23       | Dynamo 1.3.0 / vLLM 0.23      |
| **Precision**            | FP8 weights + BF16 KV (`auto`) | FP8 weights + FP8 KV          |
| **Parallelism**          | TP2                            | TP1 per worker                |
| **MoE backend**          | `triton`                       | `triton`                      |
| **KV cache manager**     | Hybrid (DeltaNet SSM + attention) | Hybrid                     |
| **Routing**              | KV-aware + worker KV events    | KV-aware + worker KV events   |
| **Speculative decoding** | MTP, `num_speculative_tokens=3` | None — see Known issues      |
| **Context length**       | 262,144                        | 262,144                       |
| **KV transfer**          | n/a                            | NIXL/UCX over InfiniBand      |
| **Async scheduling**     | enabled                        | disabled on decode — see Known issues |

Scale aggregated to a full node with `replicas: 4`.

## Supported features

- Modality: text, image, video
- Reasoning (`--dyn-reasoning-parser qwen3`)
- Tool / function calling (`--dyn-tool-call-parser qwen3_coder`)

## Prerequisites

1. Dynamo Platform installed with DGD CRDs served.
2. Hugging Face token as `hf-token-secret` (model is public, Apache-2.0).
3. `model-cache` PVC (ReadWriteMany), populated via `model-cache/`.
4. Disaggregated only: GPU-local RDMA NICs exposed to pods (`rdma/ib` device plugin) for
   NIXL KV transfer.

## Quick Start

### 1. Namespace + HF secret
```bash
export NAMESPACE=your-namespace
kubectl create namespace ${NAMESPACE}
kubectl create secret generic hf-token-secret --from-literal=HF_TOKEN="your-token" -n ${NAMESPACE}
```
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

## Benchmark Results

Measured 2026-08-06 on the 3,541-request agentic Mooncake trace (see
[perf/README.md](perf/README.md)), block size 512, closed-loop, both profiles on the same
trace. SLA: P50 TTFT < 5 s and P50 output >= 50 tok/s/user; each profile is reported at its
highest SLA-passing concurrency. `System output tok/s/GPU` is system throughput / GPUs.
Aggregated runs `replicas: 2`.

| Recipe | GPU | Topology | Workload | MTP | Concurrency | User output tok/s | TTFT (P50) | System output tok/s/GPU |
|--------|-----|----------|----------|-----|-------------|-------------------|------------|-------------------------|
| `vllm/agg-h200-agentic/deploy.yaml` | H200 | AGG, 2x TP2 (4 GPU) | agentic | yes | 64 | 52.9 | 313 ms | 720.3 |
| `vllm/disagg-h200-agentic/deploy.yaml` | H200 | 1P2D (3 GPU) | agentic | no | 18 | 52.5 | 3087 ms | 256.5 |

**Aggregated is the recommended profile.** Disaggregated is 2.8x lower per GPU on this
workload and is provided as a functional reference, not a throughput recommendation. Two
reasons, both specific to this workload rather than to disaggregation in general: at 90%
KV-cache hit the prefill worker has little to do, yet it emits no output tokens, so a third
of the fleet is unproductive; and MTP cannot run disaggregated on this architecture (see
Known Issues), so decode gives up speculative decoding. Use it when prefill and decode must
scale independently, or on a workload with a lower cache-hit rate where the prefill GPU
earns its place.

## Known Issues

1. Disaggregated decode requires `--no-async-scheduling` on vLLM < 0.26.0. Without it the
   KV-block zeroing kernel races the NIXL RDMA write and silently erases transferred KV —
   IFEval 84.66 → 51.02, GPQA-Diamond 83.84 → 19.19. Fixed by
   [vllm#45357](https://github.com/vllm-project/vllm/pull/45357) and
   [vllm#48481](https://github.com/vllm-project/vllm/pull/48481); the flag costs ~7%
   throughput and can be dropped on a runtime shipping vLLM >= 0.26.0.
2. Prefill and decode must use the same tensor-parallel size when disaggregated —
   asymmetric sharding fails NIXL transfer with a block-size mismatch.
3. MTP is not supported with disaggregation on this architecture — NIXL's Mamba conv-state
   transfer needs `VLLM_SSM_CONV_STATE_LAYOUT=DS`, which conflicts with the
   `mamba_cache_mode='align'` that MTP + prefix caching forces
   ([vllm#38898](https://github.com/vllm-project/vllm/issues/38898)).
