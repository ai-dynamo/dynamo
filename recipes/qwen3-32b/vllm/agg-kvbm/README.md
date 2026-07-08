# Qwen3-32B: Aggregated + KVBM (single GPU)

Single-GPU aggregated deployment of `Qwen/Qwen3-32B` with the KV Block Manager
(KVBM) enabled. KVBM offloads cold KV cache blocks to host memory so the
effective cache footprint extends beyond GPU HBM, which improves prefix-reuse
hit rate on long or repeated prompts without adding GPUs.

## Hardware

- **1x NVIDIA H200 (141 GB) or B200 (192 GB)**. Qwen3-32B in BF16 is ~64 GB of
  weights plus KV cache and activations, so 80 GB H100 leaves very little room
  and is likely to OOM under real load. If you only have H100 80 GB, see
  `../../../qwen3-32b-fp8/` for the FP8 variant.
- **≥ ~150 GiB of host memory on the node**. `DYN_KVBM_CPU_CACHE_GB=100` is
  pinned as page-locked host memory for KVBM's G2 tier. The worker declares
  `resources.requests.memory: 150Gi` and `resources.limits.memory: 200Gi`
  (100 GiB pinned KV pool + ~50 GiB headroom for Python, weight-loader
  working memory, and CUDA/NCCL buffers). If you raise `DYN_KVBM_CPU_CACHE_GB`,
  scale these up by roughly the same delta.

## Prerequisites

Same as the sibling recipes in this directory:

1. **Dynamo Platform installed** — see the [Kubernetes Deployment Guide](../../../../docs/kubernetes/README.md).
2. **Pre-existing `model-cache` and `compilation-cache` PVCs** — see
   [`../../model-cache/cache.yaml`](../../model-cache/cache.yaml) and
   [`../../model-cache/model-download.yaml`](../../model-cache/model-download.yaml).
3. **HuggingFace token Secret** named `hf-token-secret` in your namespace.

```bash
export NAMESPACE=your-namespace
kubectl create secret generic hf-token-secret \
  --from-literal=HF_TOKEN="your-token" \
  -n ${NAMESPACE}
```

## Deploy

```bash
kubectl apply -f deploy.yaml -n ${NAMESPACE}

kubectl wait --for=condition=ready pod \
  -l nvidia.com/dynamo-graph-deployment-name=agg-kvbm-qwen3-32b \
  -n ${NAMESPACE} --timeout=1200s
```

## Verify

```bash
kubectl port-forward svc/agg-kvbm-qwen3-32b-frontend 8000:8000 -n ${NAMESPACE}

curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-32B",
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 64
  }'
```

## KVBM configuration

The connector is selected through the worker's `--kv-transfer-config`:

```json
{"kv_connector":"DynamoConnector","kv_role":"kv_both","kv_connector_module_path":"kvbm.vllm_integration.connector"}
```

Worker env var set by this recipe:

| Variable | Default in this recipe | Description |
|---|---|---|
| `DYN_KVBM_CPU_CACHE_GB` | `100` | CPU memory reserved for offloaded KV blocks. Raise for longer contexts or higher reuse; if you change this, also bump `resources.requests.memory` / `limits.memory` on the worker by roughly the same delta. |

### (Optional) Prometheus metrics

Metrics are **off** by default. To expose them, add the following to the
worker's `env` and `mainContainer.ports` in `deploy.yaml`:

```yaml
env:
  - name: DYN_KVBM_METRICS
    value: "true"
  - name: DYN_KVBM_METRICS_PORT
    value: "6880"
ports:
  - name: kvbm
    containerPort: 6880
```

Once enabled, scrape `:6880/metrics` for counters like
`kvbm_offload_blocks_d2h`, `kvbm_onboard_blocks_h2d`, `kvbm_matched_tokens`,
`kvbm_host_cache_hit_rate`, plus per-route transfer counters. If you run the
Prometheus Operator, add a `PodMonitor` selecting pods with
`nvidia.com/dynamo-component-type: worker` and port `kvbm`.

## Cleanup

```bash
kubectl delete dynamographdeployment agg-kvbm-qwen3-32b -n ${NAMESPACE}
```
