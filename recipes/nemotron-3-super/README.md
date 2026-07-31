<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Nemotron-3-Super Recipes

Recipes for **nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4** (B200) and **nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-FP8** (H200) — a ~120B hybrid Mamba/Attention/MoE model (~12B active).

We ship Dynamo + vLLM deployment profiles across two GPU SKUs and two serving modes.

## Configurations

|                          | B200 chat                    | H200 chat                     | B200 agentic                 | H200 agentic                 |
|--------------------------|------------------------------|-------------------------------|------------------------------|------------------------------|
| **GPU** (per worker)     | 4× B200                      | 4× H200                       | 4× B200                      | 4× H200                      |
| **Mode**                 | aggregated                   | aggregated                    | aggregated                   | aggregated                   |
| **Framework**            | vLLM 0.21.0                  | vLLM 0.21.0                   | vLLM 0.21.0                  | vLLM 0.21.0                  |
| **Precision**            | NVFP4 + FP8 KV               | FP8 + FP8 KV                  | NVFP4 + FP8 KV               | FP8 + FP8 KV                 |
| **Parallelism**          | TP4 + EP                     | TP4 + EP                      | TP4 + EP                     | TP4 + EP                     |
| **MoE backend**          | FLASHINFER_TRTLLM            | FLASHINFER_CUTLASS (default)  | FLASHINFER_TRTLLM            | FLASHINFER_CUTLASS (default) |
| **Attention backend**    | FLASH_ATTN                   | FLASH_ATTN (default)          | FLASH_ATTN                   | FLASH_ATTN (default)         |
| **AllReduce backend**    | FlashInfer TRTLLM            | FlashInfer TRTLLM (default)   | FlashInfer TRTLLM            | FlashInfer TRTLLM (default)  |
| **All2All backend**      | DeepEP high-throughput       | FlashInfer NVLink one-sided   | DeepEP low-latency           | DeepEP high-throughput       |
| **Routing**              | KV-aware                     | KV-aware                      | KV-aware                     | KV-aware                     |
| **Speculative decoding** | MTP (DL=3)                   | MTP (DL=3)                    | MTP (DL=3)                   | MTP (DL=3)                   |

## Supported features

- Text-only chat
- Reasoning (`enable_thinking: true|false` via `chat_template_kwargs`)
- Tool calling
- Function calling with JSON arguments

## Prerequisites

1. **Dynamo Platform installed** on the target cluster (DGD CRDs registered with `nvidia.com/v1beta1` served).
2. **Namespace labeled for KAI**:
   ```bash
   export NAMESPACE=your-namespace
   kubectl create namespace ${NAMESPACE}
   kubectl label namespace ${NAMESPACE} kai.scheduler/enabled=true
   ```
   Without this label, pods sit `SchedulingGated` indefinitely because KAI's `pod-grouper` filters by namespace label.
3. **HuggingFace token secret** with access to `nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4` (B200) and/or `nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-FP8` (H200):
   ```bash
   kubectl create secret generic hf-token-secret \
     --from-literal=HF_TOKEN="$HF_TOKEN" \
     -n ${NAMESPACE}
   ```

## Quick Start

### 1. Create storage

> **Note:** edit `model-cache/model-cache.yaml` first and set `storageClassName` to match your cluster (`kubectl get storageclass`).

```bash
kubectl apply -f model-cache/model-cache.yaml -n ${NAMESPACE}
```

### 2. Download model

Two Jobs share the same PVC. Each pulls into the default HF cache layout (`HF_HOME=/model-cache`, files end up at `/model-cache/hub/models--nvidia--<repo>/snapshots/<sha>/`). Apply only the one your target SKU needs (or both — note PVC sizing).

```bash
# B200 — NVFP4 checkpoint (~80 GB, ~80 s on Vast)
kubectl apply -f model-cache/model-download.yaml -n ${NAMESPACE}
kubectl wait --for=condition=Complete job/model-download -n ${NAMESPACE} --timeout=1800s

# H200 — FP8 checkpoint (~120 GB)
kubectl apply -f model-cache/model-download-fp8.yaml -n ${NAMESPACE}
kubectl wait --for=condition=Complete job/model-download-fp8 -n ${NAMESPACE} --timeout=3600s
```

> **Note:** running both Jobs lands ~200 GB on the PVC, which is the default size in `model-cache.yaml`. Bump `storage:` in that file before downloading both.

### 3. Deploy the DGD

Pick the use-case variant and apply:

```bash
SKU=b200       # or h200
USECASE=chat   # or agentic

kubectl apply -f vllm/agg-${SKU}-${USECASE}/deploy.yaml -n ${NAMESPACE}
kubectl get dgd nemotron-3-super-${SKU}-${USECASE} -n ${NAMESPACE} -w
```

Two worker replicas, 4× B200/H200 each (half a node). First-time boot per worker ≈ 6–9 min (image pull + vLLM engine init + Inductor + CUDA graph capture up to size 512).

### 4. Smoke test

```bash
kubectl port-forward svc/nemotron-3-super-${SKU}-${USECASE}-frontend 8000:8000 -n ${NAMESPACE}

# B200 uses the NVFP4 model id; H200 uses the FP8 model id
MODEL_ID=nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4   # or -FP8 for H200

curl http://localhost:8000/v1/models
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d "{\"model\":\"${MODEL_ID}\",
       \"messages\":[{\"role\":\"user\",\"content\":\"Hello!\"}],
       \"max_tokens\":64,
       \"chat_template_kwargs\":{\"enable_thinking\":false}}"
```

### 5. Benchmark

See [`perf/README.md`](perf/README.md) for the full benchmark workflow — staging Mooncake-format traces on the PVC, running the AIPerf trace-replay Job ([`perf/perf.yaml`](perf/perf.yaml)), running a concurrency sweep, and fetching artifacts.

## Performance results

| Recipe               | SKU  | # of worker replicas | Concurrency | User output tok/s | System output tok/s/gpu |
|----------------------|------|----------------------|-------------|-------------------|-------------------------|
| Chat (15% subset)    | B200 | 2                    | 128         | 61.25             | 844.5                   |
| Agentic (15% subset) | B200 | 2                    | 192         | 63.16             | 1388.4                  |
| Chat (15% subset)    | H200 | 2                    | 64          | 56.07             | 404.6                   |
| Agentic (15% subset) | H200 | 2                    | 128         | 62.94             | 851.0                   |

## Spec-dec toggle (B200)

B200 chat and agentic ship with **MTP spec-dec ON by default** — DL=3, `moe_backend=triton`, stripped `compilation-config`, and `MAX_NUM_BATCHED_TOKENS=65536`.

To turn MTP off, remove the `- --speculative-config=$(SPECULATIVE_CONFIG)` line from worker args. With the extra memory headroom freed up, you can optionally bump `MAX_NUM_BATCHED_TOKENS` to `"131072"` and switch the `COMPILATION_CONFIG` env to the `compilation-config-fused` ConfigMap key (which enables `use_inductor_graph_partition` + `fuse_allreduce_rms` + `fuse_attn_quant`) for better throughput.

## Known issues

1. Some 400 HTTP errors raised by the workers on invalid inputs are surfaced as **500** errors through the Dynamo frontend (the proxy does not always preserve the worker's original status code).

## File layout

```text
recipes/nemotron-3-super/
  README.md
  model-cache/
    model-cache.yaml          # PVC (RWX, 200Gi on storageClass vast)
    model-download.yaml       # Job: hf download NVFP4 checkpoint (B200)
    model-download-fp8.yaml   # Job: hf download FP8 checkpoint (H200)
  vllm/
    agg-b200-chat/deploy.yaml      # DGD: B200×4, NVFP4, DeepEP high-throughput
    agg-b200-agentic/deploy.yaml   # DGD: B200×4, NVFP4, DeepEP low-latency
    agg-h200-chat/deploy.yaml      # DGD: H200×4, FP8, FlashInfer NVLink one-sided, MTP spec-dec
    agg-h200-agentic/deploy.yaml   # DGD: H200×4, FP8, DeepEP high-throughput, MTP spec-dec
  perf/
    README.md                 # benchmark workflow
    perf.yaml                 # AIPerf trace-replay Job
```
