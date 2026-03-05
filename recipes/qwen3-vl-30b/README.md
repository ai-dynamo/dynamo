# Qwen3-VL-30B-A3B: Encoder Cache in Disaggregated E/PD

This recipe demonstrates throughput/latency improvements from enabling multimodal encoder cache in disaggregated serving under different image re-use configurations.

## Results

### `agg`

| Reuse Tag | P90 TTFT (No Encoder Cache, ms) | P90 TTFT (With Encoder Cache, ms) | TTFT Improvement (%) |
|---|---:|---:|---:|
| `r00` | TBD | TBD | TBD |
| `r10` | TBD | TBD | TBD |
| `r25` | TBD | TBD | TBD |
| `r50` | TBD | TBD | TBD |
| `r75` | TBD | TBD | TBD |

### `disagg_ep_d`

| Reuse Tag | P90 TTFT (No Encoder Cache, ms) | P90 TTFT (With Encoder Cache, ms) | TTFT Improvement (%) |
|---|---:|---:|---:|
| `r00` | TBD | TBD | TBD |
| `r10` | TBD | TBD | TBD |
| `r25` | TBD | TBD | TBD |
| `r50` | TBD | TBD | TBD |
| `r75` | TBD | TBD | TBD |

### `disagg_e_pd`

| Reuse Tag | P90 TTFT (No Encoder Cache, ms) | P90 TTFT (With Encoder Cache, ms) | TTFT Improvement (%) |
|---|---:|---:|---:|
| `r00` | TBD | TBD | TBD |
| `r10` | TBD | TBD | TBD |
| `r25` | TBD | TBD | TBD |
| `r50` | TBD | TBD | TBD |
| `r75` | TBD | TBD | TBD |

## Pre-requisites

1. **Dynamo Platform installed** - See [Kubernetes Deployment Guide](../../docs/kubernetes/README.md)
2. **8x B200 GPUs**
3. **HuggingFace token** configured:
   ```bash
   export NAMESPACE=your-namespace
   kubectl create secret generic hf-token-secret \
     --from-literal=HF_TOKEN="your-token" \
     -n ${NAMESPACE}
   ```

## Experiment Overview

We compare the impact of encoder cache toggled on vs off for three deployment modes:
- `agg TP=1 x8 replicas`
- `disagg E/PD x1 encode worker, x7 pd workers`
- `disagg EP/D x4 EP workers, x4 decode workers`

To test this, the `data-gen/generate-datasets-job.yaml` creates 5 datasets of synthetic text + image data each with varying levels of image per request overlap. They are tagged as `r_{##}` representing the ratio of total slots to image slots. For example, r_{50} refers to a dataset where the image pool is half the size of the total slots. In other words, you would need to traverse half of the entire dataset to have seen every image. Refer to jsonl [documention](https://github.com/ai-dynamo/dynamo/tree/main/benchmarks/multimodal/jsonl) for more details on data generation.

Each dataset is hardcoded to have 1000 requests, 1000 tokens of user-input text.

### Notes:

1. Exact cache hit rates cannot be explicitly controlled via dataset due potential LRU encoder cache eviction policies; however, decreasing the image pool relative to the number of requests allows for proportionally higher probabilities of seeing duplicate images and cache hits.

(2) Agg encoder cache requires `ec_both` ECConnector role in vLLM, but that functionality was merged post 1.0.0 release. If you see an error such as `Input should be 'ec_producer' or 'ec_consumer' [type=literal_error, input_value='ec_both', input_type=str]`, you can use the `patches/patch_vllm_agg_encoder_cache.sh` script to re-tag your dynamo image with the patch applied. See [mulitmodal-vllm.md](https://github.com/ai-dynamo/dynamo/blob/main/docs/features/multimodal/multimodal-vllm.md#embedding-cache) for more details.

## Quick Start

### 1. Set Namespace and Create Storage

```bash
export NAMESPACE=your-namespace
kubectl apply -f model-cache/model-cache.yaml -n ${NAMESPACE}
kubectl get pvc -n ${NAMESPACE}
```

### 2. Download Model and Generate Datasets

```bash
kubectl apply -f model-cache/model-download.yaml -n ${NAMESPACE}
kubectl wait --for=condition=Complete job/model-download -n ${NAMESPACE} --timeout=3600s

kubectl apply -f data-gen/generate-datasets-job.yaml -n ${NAMESPACE}
kubectl wait --for=condition=Complete job/qwen3-vl-30b-generate-datasets -n ${NAMESPACE} --timeout=3600s
kubectl logs job/qwen3-vl-30b-generate-datasets -n ${NAMESPACE}
```

### 3. Deploy and Benchmark

**Option A: Aggregated (`agg`)**

```bash
# Cache OFF
kubectl apply -f vllm/agg/deploy-cache-off.yaml -n ${NAMESPACE}
kubectl delete pod qwen3-vl-30b-agg-benchmark-cache-off -n ${NAMESPACE} --ignore-not-found=true
kubectl apply -f vllm/agg/perf-cache-off.yaml -n ${NAMESPACE}
```

For cache ON with `agg`, use the patched image first:

```bash
cd patches
./patch_vllm_agg_encoder_cache.sh \
  --base-image <your dynamo image>:<your tag> \
  --output-image <your dynamo image>:<your tag>-agg-ec-patched
```

```bash
# Cache ON
kubectl apply -f vllm/agg/deploy-cache-on.yaml -n ${NAMESPACE}
kubectl delete pod qwen3-vl-30b-agg-benchmark-cache-on -n ${NAMESPACE} --ignore-not-found=true
kubectl apply -f vllm/agg/perf-cache-on.yaml -n ${NAMESPACE}
```

**Option B: Disaggregated EP/D (`disagg_ep_d`)**

```bash
# Cache OFF
kubectl apply -f vllm/disagg-ep-d/deploy-cache-off.yaml -n ${NAMESPACE}
kubectl apply -f vllm/disagg-ep-d/perf-cache-off.yaml -n ${NAMESPACE}

# Cache ON
kubectl apply -f vllm/disagg-ep-d/deploy-cache-on.yaml -n ${NAMESPACE}
kubectl apply -f vllm/disagg-ep-d/perf-cache-on.yaml -n ${NAMESPACE}
```

**Option C: Disaggregated E/PD (`disagg_e_pd`)**

```bash
# Cache OFF
kubectl apply -f vllm/disagg-e-pd/deploy-cache-off.yaml -n ${NAMESPACE}
kubectl apply -f vllm/disagg-e-pd/perf-cache-off.yaml -n ${NAMESPACE}

# Cache ON
kubectl apply -f vllm/disagg-e-pd/deploy-cache-on.yaml -n ${NAMESPACE}
kubectl apply -f vllm/disagg-e-pd/perf-cache-on.yaml -n ${NAMESPACE}
```

### 4. Monitor Benchmark Progress

```bash
kubectl get pods -n ${NAMESPACE} -l app=benchmark

# Follow one benchmark pod at a time
kubectl logs -f qwen3-vl-30b-agg-benchmark-cache-off -n ${NAMESPACE}
kubectl logs -f qwen3-vl-30b-agg-benchmark-cache-on -n ${NAMESPACE}
```

Wait for `All runs complete. Artifacts in /perf-cache/artifacts/qwen3_vl_30b_encoder_cache/<config>`.

### 5. Run Analysis and View Results

```bash
# Aggregated
kubectl delete job qwen3-vl-30b-agg-analysis -n ${NAMESPACE} --ignore-not-found=true
kubectl apply -f vllm/agg/analysis.yaml -n ${NAMESPACE}
kubectl wait --for=condition=Complete job/qwen3-vl-30b-agg-analysis -n ${NAMESPACE} --timeout=600s
kubectl logs job/qwen3-vl-30b-agg-analysis -n ${NAMESPACE}

# EP/D
kubectl delete job qwen3-vl-30b-disagg-ep-d-analysis -n ${NAMESPACE} --ignore-not-found=true
kubectl apply -f vllm/disagg-ep-d/analysis.yaml -n ${NAMESPACE}
kubectl wait --for=condition=Complete job/qwen3-vl-30b-disagg-ep-d-analysis -n ${NAMESPACE} --timeout=600s
kubectl logs job/qwen3-vl-30b-disagg-ep-d-analysis -n ${NAMESPACE}

# E/PD
kubectl delete job qwen3-vl-30b-disagg-e-pd-analysis -n ${NAMESPACE} --ignore-not-found=true
kubectl apply -f vllm/disagg-e-pd/analysis.yaml -n ${NAMESPACE}
kubectl wait --for=condition=Complete job/qwen3-vl-30b-disagg-e-pd-analysis -n ${NAMESPACE} --timeout=600s
kubectl logs job/qwen3-vl-30b-disagg-e-pd-analysis -n ${NAMESPACE}
```

Analysis CSV outputs:

- `/perf-cache/artifacts/qwen3_vl_30b_encoder_cache/agg/analysis/cache_on_vs_off_summary.csv`
- `/perf-cache/artifacts/qwen3_vl_30b_encoder_cache/disagg_ep_d/analysis/cache_on_vs_off_summary.csv`
- `/perf-cache/artifacts/qwen3_vl_30b_encoder_cache/disagg_e_pd/analysis/cache_on_vs_off_summary.csv`