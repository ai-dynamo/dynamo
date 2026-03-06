# Qwen3-VL-30B-A3B: Encoder Cache in Disaggregated E/PD

This recipe demonstrates disaggregation configs and how to enable encoder cache for a multi-modal LLM in vLLM. It also includes dataset generation and perf scripts to evaluate the deployment against varying levels of image re-use.

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

## Available Configurations

| Configuration | GPUs | Mode | Description |
|--------------|------|------|-------------|
| [**vllm/agg**](vllm/agg/) | 8x GPU | Aggregated | agg TP=1 x8 replicas |
| [**vllm/disagg-e-pd**](vllm/disagg-e-pd/) | 8x GPU | Disaggregated E/PD | disagg E/PD x1 encode worker, x7 pd workers |
| [**vllm/disagg-ep-d**](vllm/disagg-ep-d/) | 8x GPU | Disaggregated EP/D | disagg EP/D x4 EP workers, x4 decode workers |

## Dataset Generation

`data-gen/generate-datasets-job.yaml` creates 5 datasets of synthetic text + image data each with varying levels of image per request overlap. The script does this by manipulating the "total slots" and "image pool".

Total number of slots is calculated as `num_requests*images/request`, representing how many total images the benchmark will iterate through. The image pool is how many images the benchmark can choose from to attach to a request.

Each dataset is tagged as `r_{##}` representing the ratio of total slots to image slots. For example, r_{50} refers to a dataset where the image pool is half the size of the total slots. In other words, you would need to traverse half of the entire dataset to have seen every image. Refer to jsonl [documention](https://github.com/ai-dynamo/dynamo/tree/main/benchmarks/multimodal/jsonl) for more details on data generation.

Each dataset is hardcoded to have 500 requests and 320 tokens of user-input text. AIPerf configures the system prompt via `--shared-system-prompt-length` and the `perf.yaml` scripts have this set to 160 tokens.

## Notes

1. Exact cache hit rates cannot be explicitly controlled via dataset due potential LRU encoder cache eviction policies; however, decreasing the image pool relative to the number of requests allows for proportionally higher probabilities of seeing duplicate images and cache hits.

**2. Agg encoder cache requires `ec_both` ECConnector role in vLLM, but that functionality was merged post 1.0.0 release. If you see an error such as `Input should be 'ec_producer' or 'ec_consumer' [type=literal_error, input_value='ec_both', input_type=str]`, you can use the `patches/patch_vllm_agg_encoder_cache.sh` script to re-tag your dynamo image with the patch applied. See [mulitmodal-vllm.md](https://github.com/ai-dynamo/dynamo/blob/main/docs/features/multimodal/multimodal-vllm.md#embedding-cache) for more details.**

3. Replace placeholders in `*.yaml` before running:
   - `storageClassName: "your-storage-class-name"` in `model-cache/model-cache.yaml`
   - `image: <your-dynamo-image>` in all `vllm/*/deploy.yaml` files
   - `NAMESPACE=your-namespace` and `HF_TOKEN="your-token"` in the setup commands

## Directory setup

Each `vllm/<config>` directory contains `deploy.yaml` and `perf.yaml` that set up Dynamo and benchmark with [AIPerf](https://github.com/ai-dynamo/aiperf), respectively. A shared `vllm/analysis.yaml` performs post-run analysis for any configuration.

```text
qwen3-vl-30b/
├── model-cache/
│   ├── model-cache.yaml
│   └── model-download.yaml
├── data-gen/
│   └── generate-datasets-job.yaml
└── vllm/
    ├── analysis.yaml
    ├── agg/
    │   ├── deploy.yaml
    │   └── perf.yaml
    ├── disagg-e-pd/
    │   ├── deploy.yaml
    │   └── perf.yaml
    └── disagg-ep-d/
        ├── deploy.yaml
        └── perf.yaml
```

The `deploy.yaml` scripts have `MM_EMBEDDING_CACHE_GB=0` by default, which represents an embedding cache **off** configuration. To toggle it on, set the env variable to a non-zero value. 

Similarly, each `perf.yaml` exposes a `CACHE_MODE` env variable to control where AIPerf dumps its results. Set it to either `cache_on` or `cache_off` depending on your deployment.

The shared `vllm/analysis.yaml` exposes a `TOPOLOGY` env variable. Set it to `agg`, `disagg_ep_d`, or `disagg_e_pd` before running analysis to point it to the correct AIPerf benchmarking results.

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
# Cache OFF by default, set MM_EMBEDDING_CACHE_GB to enable
# Requires patched image
kubectl apply -f vllm/agg/deploy.yaml -n ${NAMESPACE}
kubectl wait --for=condition=Ready dynamographdeployment/qwen3-vl-agg -n ${NAMESPACE} --timeout=900s

kubectl apply -f vllm/agg/perf.yaml -n ${NAMESPACE}
kubectl wait --for=condition=Ready pod/qwen3-vl-agg-benchmark -n ${NAMESPACE} --timeout=300s
```

**Option B: Disaggregated EP/D (`disagg_ep_d`)**

```bash
# Cache OFF by default, set MM_EMBEDDING_CACHE_GB to enable
# Requires patched image
kubectl apply -f vllm/disagg-ep-d/deploy.yaml -n ${NAMESPACE}
kubectl wait --for=condition=Ready dynamographdeployment/qwen3-vl-disagg-ep-d -n ${NAMESPACE} --timeout=900s

kubectl apply -f vllm/disagg-ep-d/perf.yaml -n ${NAMESPACE}
kubectl wait --for=condition=Ready pod/qwen3-vl-disagg-ep-d-benchmark -n ${NAMESPACE} --timeout=300s
```

**Option C: Disaggregated E/PD (`disagg_e_pd`)**

```bash
# Cache OFF by default, set MM_EMBEDDING_CACHE_GB to enable
kubectl apply -f vllm/disagg-e-pd/deploy.yaml -n ${NAMESPACE}
kubectl wait --for=condition=Ready dynamographdeployment/qwen3-vl-disagg-e-pd -n ${NAMESPACE} --timeout=900s

kubectl apply -f vllm/disagg-e-pd/perf.yaml -n ${NAMESPACE}
kubectl wait --for=condition=Ready pod/qwen3-vl-disagg-e-pd-benchmark -n ${NAMESPACE} --timeout=300s
```

### 4. Monitor Benchmark Progress

```bash
kubectl get pods -n ${NAMESPACE} -l app=benchmark

# Follow one benchmark pod to see AIPerf in realtime
# {config} = agg, disagg-ep-d, or disagg-e-pd
kubectl logs -f qwen3-vl-{config}-benchmark -n ${NAMESPACE}

# Wait for completion
kubectl wait --for=jsonpath='{.status.phase}'=Succeeded pod/qwen3-vl-{config}-benchmark -n ${NAMESPACE} --timeout=7200s
```

Wait for `All runs complete. Artifacts in /perf-cache/artifacts/qwen3_vl_30b_encoder_cache/<config>`.

### 5. Run Analysis and View Results

```bash
# Set TOPOLOGY: {config} from above in vllm/analysis.yaml 
kubectl apply -f vllm/analysis.yaml -n ${NAMESPACE}
kubectl wait --for=condition=Complete job/qwen3-vl-analysis -n ${NAMESPACE} --timeout=600s
kubectl logs job/qwen3-vl-analysis -n ${NAMESPACE}
```

Analysis CSV outputs:

- `/perf-cache/artifacts/qwen3_vl_30b_encoder_cache/agg/analysis/cache_on_vs_off_summary.csv`
- `/perf-cache/artifacts/qwen3_vl_30b_encoder_cache/disagg_ep_d/analysis/cache_on_vs_off_summary.csv`
- `/perf-cache/artifacts/qwen3_vl_30b_encoder_cache/disagg_e_pd/analysis/cache_on_vs_off_summary.csv`