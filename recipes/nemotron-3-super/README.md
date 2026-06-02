<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Nemotron-3-Super Turbo Recipes

NIM Turbo deployment recipes for **nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4** (B200) and **nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-FP8** (H200) — a ~120B hybrid Mamba/Attention/MoE model (~12B active).

We ship Dynamo + vLLM deployment profiles across two GPU SKUs and two serving modes.

## Configurations

|                              | B200 chat | B200 agentic | H200 chat | H200 agentic |
|------------------------------|-----------|--------------|-----------|--------------|
| **GPU**                      | 4× B200 (per worker), 2 replicas | 4× B200 (per worker), 2 replicas | 4× H200 (per worker), 2 replicas | 4× H200 (per worker), 2 replicas |
| **Mode**                     | aggregated | aggregated | aggregated | aggregated |
| **Framework**                | vLLM 0.21 | vLLM 0.21 | vLLM 0.21 | vLLM 0.21 |
| **HF checkpoint**            | `nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4` | NVFP4 (same) | `nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-FP8` | FP8 (same) |
| **Precision**                | NVFP4 weights + FP8 KV (`fp8_e4m3`) | NVFP4 weights + FP8 KV | FP8 weights + FP8 KV | FP8 weights + FP8 KV |
| **Mamba SSM cache dtype**    | `float16` | `float16` | `float16` | `float16` |
| **Parallelism**              | TP=4, Expert Parallel | TP=4, Expert Parallel | TP=4, Expert Parallel | TP=4, Expert Parallel |
| **MoE GEMM backend**         | FlashInfer TRT-LLM (`--moe-backend flashinfer_trtllm`) + FlashInfer autotune | same | vLLM default (Hopper) + FlashInfer autotune | same |
| **All-to-All backend**       | DeepEP high-throughput | DeepEP low-latency | FlashInfer NVLink one-sided | DeepEP high-throughput |
| **AllReduce backend**        | FlashInfer pinned to TRTLLM (`VLLM_FLASHINFER_ALLREDUCE_BACKEND=trtllm`); NCCL default (NVLS / symm-mem are disabled on the MTP path) | same | NCCL default (Hopper) | NCCL default |
| **Compilation**              | stripped (`max_cudagraph_capture_size=512` only — MTP path; fused passes available as `compilation-config-fused` for the MTP-OFF path) | same | `use_inductor_graph_partition=true`; `fuse_allreduce_rms` + `fuse_attn_quant`; `max_cudagraph_capture_size=512` | same |
| **Prefix caching**           | `--enable-prefix-caching` + chunked prefill | same | same | same |
| **Routing**                  | KV-aware (`DYN_ROUTER_MODE=kv`) | same | same | same |
| **Speculative decoding**     | MTP (DL=3, `moe_backend=triton`) | MTP (DL=3, `moe_backend=triton`) | MTP (DL=3) | MTP (DL=3) |
| **KV cache offloading**      | none | none | none | none |
| **Max model length**         | 131072 (128k) | 131072 (128k) | 131072 (128k) | 131072 (128k) |
| **Max batched tokens**       | 65536 (MTP-ON default; bump to 131072 with MTP off) | 65536 (MTP-ON default; bump to 131072 with MTP off) | 16384 | 16384 |
| **Tool / reasoning parser**  | `nemotron_nano` | `nemotron_nano` | `nemotron_nano` | `nemotron_nano` |
| **Workload**                 | General chat / reasoning (throughput) | Agentic coding (latency) | General chat / reasoning + MTP | Agentic coding + MTP (currently identical to chat — diverge as needed) |

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
4. **`nvcr.io` pull access on the namespace's default ServiceAccount.** The recipes no longer specify `imagePullSecrets` inline, so the cluster's default SA in your namespace must already carry a docker-registry secret for `nvcr.io/nvstaging` (typically provisioned by cluster ops). If your namespace doesn't have one, create one and patch it onto the default SA, e.g.:
   ```bash
   kubectl create secret docker-registry nvcr-imagepullsecret \
     --docker-server=nvcr.io \
     --docker-username='$oauthtoken' \
     --docker-password="$NGC_API_KEY" \
     -n ${NAMESPACE}
   kubectl patch serviceaccount default -n ${NAMESPACE} \
     -p '{"imagePullSecrets":[{"name":"nvcr-imagepullsecret"}]}'
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

kubectl apply -f vllm/turbo_nemotron_3_super_agg_${SKU}_${USECASE}.yaml -n ${NAMESPACE}
kubectl get dgd turbo-nemotron-3-super-${SKU}-${USECASE} -n ${NAMESPACE} -w
```

Two worker replicas, 4× B200/H200 each (half a node). First-time boot per worker ≈ 6–9 min (image pull + vLLM engine init + Inductor + CUDA graph capture up to size 512).

### 4. Smoke test

```bash
kubectl port-forward svc/turbo-nemotron-3-super-${SKU}-${USECASE}-frontend 8000:8000 -n ${NAMESPACE}

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

For H200 MTP benchmarking with a fixed acceptance length, switch the worker's `SPECULATIVE_CONFIG` env to point at the `speculative-config-synthetic` key in the H200 ConfigMap (already provided alongside the default `speculative-config`).

## Spec-dec toggle (B200)

B200 chat and agentic ship with **MTP spec-dec ON by default** — DL=3, `moe_backend=triton`, stripped `compilation-config`, and `MAX_NUM_BATCHED_TOKENS=65536`.

To turn MTP off, flip three knobs in the worker pod spec:

1. Remove the `- --speculative-config=$(SPECULATIVE_CONFIG)` line from worker args.
2. Switch the `COMPILATION_CONFIG` env's `configMapKeyRef.key` from `compilation-config` → `compilation-config-fused` (turns on `use_inductor_graph_partition` + `fuse_allreduce_rms` + `fuse_attn_quant`).
3. Bump the `MAX_NUM_BATCHED_TOKENS` env value from `"65536"` → `"131072"` (no draft model means there's headroom for a larger chunked-prefill chunk).

Both ConfigMap keys (`compilation-config` and `compilation-config-fused`) ship in the same DGD; flipping is a YAML edit, no re-render.

> Fusions and MTP can't be on at the same time on B200: Inductor's first-time Triton autotune calls `torch.cuda.synchronize()` inside the active cudagraph capture, which aborts with `cudaErrorStreamCaptureUnsupported` / `cudaErrorStreamCaptureInvalidated` (vLLM #40742). That's why fusions are only enabled in the MTP-OFF path.

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
    turbo_nemotron_3_super_agg_b200_chat.yaml      # DGD: B200×4, NVFP4, DeepEP high-throughput
    turbo_nemotron_3_super_agg_b200_agentic.yaml   # DGD: B200×4, NVFP4, DeepEP low-latency
    turbo_nemotron_3_super_agg_h200_chat.yaml      # DGD: H200×4, FP8, FlashInfer NVLink one-sided, MTP spec-dec
    turbo_nemotron_3_super_agg_h200_agentic.yaml   # DGD: H200×4, FP8, DeepEP high-throughput, MTP spec-dec
  perf/
    README.md                 # benchmark workflow
    perf.yaml                 # AIPerf trace-replay Job
```
