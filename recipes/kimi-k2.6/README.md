# Kimi-K2.6 Recipes

Recipes for **moonshotai/Kimi-K2.6**.

## Configurations

Dynamo + vLLM deployment profiles across two GPU SKUs and two target workloads:

|                          | B200 chat                             | H200 chat                             | B200 agentic                         | H200 agentic                         |
|--------------------------|---------------------------------------|---------------------------------------|--------------------------------------|--------------------------------------|
| **GPU** (per worker)                  | 4x B200                               | 8x H200                               | 4x B200                              | 8x H200                              |
| **Mode**                 | aggregated                            | aggregated                            | aggregated                           | aggregated                           |
| **Framework**            | vLLM 0.21.0                           | vLLM 0.21.0                           | vLLM 0.21.0                          | vLLM 0.21.0                          |
| **Precision**            | NVFP4 + FP8 KV                        | INT4                                  | NVFP4 + FP8 KV                       | INT4                                 |
| **Parallelism**          | TP4                                   | TP8                                   | TP4                                  | TP8                                  |
| **MoE backend**          | FLASHINFER_TRTLLM                     | MARLIN                                | FLASHINFER_TRTLLM                    | MARLIN                               |
| **Attention backend**    | TOKENSPEED_MLA                        | FLASH_ATTN_MLA                        | TOKENSPEED_MLA                       | FLASH_ATTN_MLA                       |
| **AllReduce backend**    | NCCL symmetric memory                 | NCCL                                  | NCCL symmetric memory                | NCCL                                 |
| **All2All backend**      | N/A                                   | N/A                                   | N/A                                  | N/A                                  |
| **Routing**              | KV-aware                              | KV-aware                              | KV-aware                             | KV-aware                             |
| **Speculative decoding** | EAGLE3 MLA (DL=3, SpeedBench AL=2.49) | EAGLE3 MLA (DL=3, SpeedBench AL=2.49) | EAGLE3 MLA (DL=3, SpeedBench AL=2.49) | EAGLE3 MLA (DL=3, SpeedBench AL=2.49) |
| **KV cache offloading**  | LMCache CPU                           | LMCache CPU                           | LMCache CPU                          | LMCache CPU                          |


## Supported features

- Modalities: Text + Image
- Reasoning
- Tool calling


## Prerequisites

1. **Dynamo Platform installed** — see [Kubernetes Deployment Guide](../../docs/kubernetes/README.md).
2. **HuggingFace token** with access to `nvidia/Kimi-K2.6-NVFP4`, `moonshotai/Kimi-K2.6` and `lightseekorg/kimi-k2.6-eagle3-mla`:
   ```bash
   export NAMESPACE=your-namespace
   kubectl create secret generic hf-token-secret \
     --from-literal=HF_TOKEN="your-token" \
     -n ${NAMESPACE}
   ```


## Quick Start

### 1. Create namespace

```
export NAMESPACE=your-namespace
kubectl create namespace ${NAMESPACE}
```

### 2. Create Storage

> **Note:** Edit `model-cache/model-cache.yaml` first and update `storageClassName` to match your cluster (`kubectl get storageclass`).

```bash
kubectl apply -f model-cache/model-cache.yaml -n ${NAMESPACE}
```

### 3. Download model + EAGLE3 head

> **Note:** Edit `model-cache/model-download.yaml` first and remove the `hf download` lines that do not apply to your deployment (For H200, remove the NVFP4 download, for B200, remove the native INT4 download).

```bash
kubectl apply -f model-cache/model-download.yaml -n ${NAMESPACE}
kubectl wait --for=condition=Complete job/model-download -n ${NAMESPACE} --timeout=3600s
```

### 4. Deploy the DGD

Deploy the target DGD:

```bash
SKU=b200 # or h200
USECASE=chat # or agentic

kubectl apply -f vllm/agg-${SKU}-${USECASE}/deploy.yaml -n ${NAMESPACE}
```


### 5. Benchmark

See [`perf/README.md`](perf/README.md) for the full benchmark workflow — trace staging on the PVC, running the AIPerf trace-replay Job ([`perf/perf.yaml`](perf/perf.yaml)), running a concurrency sweep, and fetching artifacts.


## Optimization targets

Recipes are optimized for the following configurations, at the target user interactivity:

| Workload                 | Median ISL | Median OSL | KV cache hit rate | User output tok/s |
|------------------------|------------|------------|----------------|------------|
| Chat                   |      1k      | 1k        |  70%       | 50        |
| Agentic                  |      64k      | 400        | 90%        | 50        |


Modified Mooncake traces are provided to showcase the value of KV-aware routing and CPU offloading, see [perf/README.md](./perf/README.md) for details.


## Performance results

| Recipe                 | SKU | # of worker replicas | Concurrency | User output tok/s | System output tok/s/gpu |
|------------------------|------------|------------|----------------|------------|------------|
| Chat (15% subset)                   |      B200      | 4        |   48      |    	49.86     |  107.8 |
| Agentic (15% subset)                 |      B200      | 4        | 64        |  55.50       | 166.5 |
| Chat (15% subset)                  |      H200      | 4        |    32     |   	54.86      | 38.7 |
| Agentic (15% subset)                 |      H200      | 4        |    48     |   	56.06       | 66.5 |


## Known issues

1. Dynamo's KV cache router does not support all LMCache KV events, so routing can be sub-optimal
2. Some 400 HTTP errors from the workers on invalid inputs can be raised as 500 errors through the frontend
