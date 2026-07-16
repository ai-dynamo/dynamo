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
| **GPU**                  | 1x B200                                     | 1x B200 prefill + 2x B200 decode (3x total)  |
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

- Modalities: Text, Image, Video
- Reasoning
- Tool calling

## Prerequisites

1. **Dynamo Platform installed** on the target cluster with DGD CRDs served —
   see [Kubernetes Deployment Guide](../../docs/kubernetes/README.md).
2. **NGC/nvcr image pull access** — an NGC pull secret named `nvcr-secret`
   attached to the namespace's default service account (the deploy manifests pull
   from `nvcr.io/nvstaging/ai-dynamo`).
3. **Hugging Face token** with access to `nvidia/Qwen3.5-122B-A10B-NVFP4`, stored
   as `hf-token-secret` — used by both the model-download Job and the serving
   workers.
4. **`model-cache` PVC** (ReadWriteMany) populated with the model, or permission
   to create and populate it via the manifests in `model-cache/`.
5. **(disaggregated only)** GPU-local RDMA NICs exposed to pods (e.g. an
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

> [!NOTE]
> The deploy manifests pull the runtime image from `nvcr.io/nvstaging/ai-dynamo`
> and do not set `imagePullSecrets`, so the target namespace must already have
> nvcr/NGC pull access. If the cluster does not inject a default pull secret,
> create one and attach it to the namespace's default service account:
>
> ```bash
> kubectl create secret docker-registry nvcr-secret \
>   --docker-server=nvcr.io --docker-username='$oauthtoken' \
>   --docker-password="<your-NGC-API-key>" -n ${NAMESPACE}
> kubectl patch serviceaccount default -n ${NAMESPACE} \
>   -p '{"imagePullSecrets":[{"name":"nvcr-secret"}]}'
> ```

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

- **Speculative decoding (MTP) + disaggregation is not shipped on this arch.**
  Disaggregation requires `VLLM_SSM_CONV_STATE_LAYOUT=DS` (for NIXL's 3-read Mamba
  conv-state transfer), but MTP + prefix caching forces `mamba_cache_mode='align'`,
  whose DS conv-state copy path is unimplemented for `num_accepted_tokens > 1` —
  so the decode `EngineCore` first crashes (`NotImplementedError` → `EngineDeadError`)
  on the first concurrent batch of real long-context traffic.
  **Patching the crash** (vLLM [#45473](https://github.com/vllm-project/vllm/pull/45473)'s
  `ds_conv_tail_copy` kernel) then exposes a **silent quality regression**: MTP is
  enabled on the decode worker but not on prefill, so the two workers disagree on
  spec-decode conv-state metadata. Decode reserves extra conv-window rows and
  speculative blocks (conv state `(dim, conv_kernel-1 + num_speculative_tokens)`,
  `num_speculative_blocks > 0`); prefill does not (`(dim, conv_kernel-1)`,
  `num_speculative_blocks = 0`). The NIXL transfer then misplaces prefill's smaller
  conv state into decode's larger slot, so decode resumes from corrupted recurrent
  state and emits prompt-independent garbage. Matching `--speculative-config` on
  both prefill and decode realigns the geometry and restores correct output, but MTP
  is not a throughput win on this workload (see below), so the combination is not
  shipped. Upstream: vLLM [#38898](https://github.com/vllm-project/vllm/issues/38898)
  and PR [#40454](https://github.com/vllm-project/vllm/pull/40454); tracked in
  NVBug 6442165 / DYN-864. A synthetic forced-acceptance sweep hits neither failure
  (it skips the real conv-state copy), which is why the combination looked viable in
  earlier numbers.
- **MTP on aggregation** is likewise not shipped: it requires DS unset and, on this
  workload, MTP-heavy decode starves prefill on the shared GPU (TTFT regressions)
  for no throughput win.
