# Qwen3.5-122B-A10B-NVFP4 Recipes

Recipes for [Qwen3.5-122B-A10B-NVFP4](https://huggingface.co/nvidia/Qwen3.5-122B-A10B-NVFP4),
the NVFP4 quantization of [Qwen/Qwen3.5-122B-A10B](https://huggingface.co/Qwen/Qwen3.5-122B-A10B)
(122B total / 10B active hybrid MoE — Gated DeltaNet linear attention + MoE with
full attention every 4th layer).

## Configurations

Dynamo + vLLM deployment profiles for the agentic workload. This set covers
**B200**; H200 profiles are tracked as a follow-up.

|                          | B200 aggregated agentic                     | B200 disaggregated agentic                   |
| ------------------------ | ------------------------------------------- | -------------------------------------------- |
| **GPU** (per worker)     | 1x B200                                     | 1x B200 prefill + 1x B200 decode             |
| **Mode**                 | Aggregated                                  | Prefill/decode disaggregated (1P2D)          |
| **Framework**            | vLLM                                        | vLLM                                         |
| **Precision**            | NVFP4 + FP8 KV                              | NVFP4 + FP8 KV                               |
| **Parallelism**          | TP1                                         | TP1 (per worker)                             |
| **MoE backend**          | FLASHINFER_TRTLLM                           | FLASHINFER_TRTLLM                            |
| **KV cache manager**     | Hybrid (DeltaNet SSM + attention)           | Hybrid (DeltaNet SSM + attention)            |
| **Routing**              | KV-aware (workers publish KV events)        | KV-aware (workers publish KV events)         |
| **Speculative decoding** | None — see Limitations                      | None — see Limitations                       |
| **Context length**       | 262,144 (model default)                     | 262,144 (model default)                      |
| **KV transfer**          | N/A                                         | NIXL/UCX over InfiniBand                     |

## Supported features

- Modalities: Text
- Reasoning
- Tool calling

## Prerequisites

1. **Dynamo Platform installed** — see [Kubernetes Deployment Guide](../../docs/kubernetes/README.md).
2. **Hugging Face token** with access to `nvidia/Qwen3.5-122B-A10B-NVFP4`.
3. **(disaggregated only)** GPU-local RDMA NICs exposed to pods (e.g. an
   `rdma/ib` device plugin) for NIXL KV transfer.

## Quick Start

### 1. Create namespace and secret

```bash
export NAMESPACE=your-namespace
kubectl create namespace ${NAMESPACE}
kubectl create secret generic hf-token-secret \
  --from-literal=HF_TOKEN="your-token" \
  -n ${NAMESPACE}
```

### 2. Create storage

> [!NOTE]
> Edit `model-cache/model-cache.yaml` and set `storageClassName` to a
> ReadWriteMany storage class available on the target cluster.

```bash
kubectl apply -f model-cache/model-cache.yaml -n ${NAMESPACE}
```

### 3. Download the model

```bash
kubectl apply -f model-cache/model-download.yaml -n ${NAMESPACE}
kubectl wait --for=condition=Complete job/model-download -n ${NAMESPACE} --timeout=3600s
```

### 4. Deploy the DGD

```bash
SKU=b200 # H200 profiles to follow
MODE=agg # or disagg
kubectl apply -f vllm/${MODE}-${SKU}-agentic/deploy.yaml -n ${NAMESPACE}
```

### 5. Benchmark

See [perf/README.md](perf/README.md) for the full benchmark workflow — trace
staging on the PVC, running the AIPerf trace-replay Job, running a concurrency
sweep, and fetching artifacts.

## Optimization targets

| Workload | Median ISL | Median OSL | KV cache hit rate | User output tok/s |
| -------- | ---------- | ---------- | ----------------- | ----------------- |
| Agentic  | 64k        | 400        | 90%               | 50                |

## Performance results

Measured on B200 against the **real** 15% agentic mooncake trace (closed-loop
concurrency; SLA = P50 TTFT < 5 s **and** ≥ 50 output tok/s/user). Headline metric
is system output tok/s per GPU at the best SLA-passing concurrency.

| Recipe                    | GPUs | tok/s/GPU @ SLA | user tok/s (P50) | TTFT (P50) |
| ------------------------- | ---- | --------------- | ---------------- | ---------- |
| Aggregated TP1            | 1    | ~1,067          | 80               | 0.6 s      |
| Disaggregated 1P2D+seq128 | 3    | ~1,112          | 62               | 2.9 s      |

## Limitations

- **Speculative decoding (MTP) + disaggregation is mutually exclusive on this
  arch.** Disaggregation requires `VLLM_SSM_CONV_STATE_LAYOUT=DS` (for NIXL's
  3-read Mamba conv-state transfer), but MTP + prefix caching forces
  `mamba_cache_mode='align'`, whose DS conv-state copy path is unimplemented for
  `num_accepted_tokens > 1`. The decode `EngineCore` crashes (`NotImplementedError`
  → `EngineDeadError`) on the first concurrent batch of real long-context traffic.
  Upstream: vLLM [#38898](https://github.com/vllm-project/vllm/issues/38898) (open)
  and PR [#40454](https://github.com/vllm-project/vllm/pull/40454) (which defaults
  spec decode to `align` mode); tracked in NVBug 6442165 / DYN-864. A synthetic
  forced-acceptance sweep does **not** hit this (it skips the real conv-state
  copy), which is why the combination looked viable in earlier numbers.
- **MTP on aggregation** is likewise not shipped: it requires DS unset and, on this
  workload, MTP-heavy decode starves prefill on the shared GPU (TTFT regressions)
  for no throughput win.
