# GLM-5.2 Recipes

Recipes for `[GLM-5.2](https://huggingface.co/zai-org/GLM-5.2)`.

## Configurations

Dynamo + SGLang deployment profiles for the B200 agentic workload:


|                          | B200 aggregated agentic                    | B200 disaggregated agentic                 |
| ------------------------ | ------------------------------------------ | ------------------------------------------ |
| **GPU** (per worker)     | 4x B200                                    | 4x B200 prefill + 8x B200 decode           |
| **Mode**                 | Aggregated                                 | Prefill/decode disaggregated               |
| **Framework**            | SGLang                                     | SGLang                                     |
| **Precision**            | NVFP4 + FP8 KV                             | NVFP4 + FP8 KV                             |
| **Parallelism**          | DTP4                                       | DEP4 / DTP8                                |
| **Routing**              | KV-aware                                   | KV-aware                                   |
| **Speculative decoding** | EAGLE-style MTP (DL=3, SpeedBench AL=2.69) | EAGLE-style MTP (DL=3, SpeedBench AL=2.69) |
| **Context length**       | 500,000                                    | 500,000                                    |
| **KV cache offloading**  | HiCache CPU                                | HiCache CPU                                |
| **KV transfer**          | N/A                                        | NIXL/UCX over IB                           |


## Supported features

- Modalities: Text
- Reasoning
- Tool calling

## Prerequisites

1. **Dynamo Platform installed** — see[Kubernetes Deployment Guide](../../docs/kubernetes/README.md).
2. **Hugging Face token** with access to `nvidia/GLM-5.2-NVFP4`:
  ```bash
   export NAMESPACE=your-namespace
   kubectl create secret generic hf-token-secret \
     --from-literal=HF_TOKEN="your-token" \
     -n ${NAMESPACE}
  ```

## Quick Start

### 1. Create namespace

```bash
export NAMESPACE=your-namespace
kubectl create namespace ${NAMESPACE}
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
kubectl wait --for=condition=Complete job/model-download -n ${NAMESPACE} --timeout=7200s
```

### 4. Deploy the DGD

Deploy the target DGD:

```bash
SKU=b200
MODE=agg # or disagg
kubectl apply -f sglang/${MODE}-${SKU}-agentic/deploy.yaml -n ${NAMESPACE}
```



### 5. Benchmark

See `[perf/README.md](perf/README.md)` for the full benchmark workflow — trace staging on the PVC, running the AIPerf trace-replay Job, running a concurrency sweep, and fetching artifacts.

## Optimization targets


| Workload | Median ISL | Median OSL | KV cache hit rate | User output tok/s |
| -------- | ---------- | ---------- | ----------------- | ----------------- |
| Agentic  | 64k        | 400        | 90%               | 50                |


Modified Mooncake traces are provided to showcase the value of KV-aware routing and CPU offloading, see `[perf/README.md](perf/README.md)` for details.

## Performance results


| Workload             | Recipe                 | SKU  | Concurrency | System output tok/s/gpu | User output tok/s (P50) | TTFT (P50) |
| -------------------- | ---------------------- | ---- | ----------- | ----------------------- | ----------------------- | ---------- |
| Agentic (15% subset) | Aggregated (4 workers) | B200 | 64          | 176.420                 | 57.493                  | 355.555    |
| Agentic (15% subset) | Disaggregated (3P1D)   | B200 | 128         | 320.907                 | 65.105                  | 1938.059   |




## Limitations

- B200 recipes support up to 500K context lengths, and H200 recipes up to 250K. The full 1M context length is not supported out of the box.

