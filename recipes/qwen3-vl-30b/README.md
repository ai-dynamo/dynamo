# Qwen3-VL-30B-A3B-Instruct-FP8: Aggregated Embedding Cache On vs Off Comparison

This recipe demonstrates the performance difference when embedding cache is enabled for multi-modal payloads. It includes guidance on creating an artificial dataset with user-defined image re-use, and production-ready deployments for `Qwen/Qwen3-VL-30B-A3B-Instruct-FP8`.

## Results

| Metric               | Cache ON | Cache OFF | Delta  |
|----------------------|---------:|----------:|-------:|
| Output TPS (tok/s)           |   3575.6 |    3072.3 | +16.4% |
| TTFT avg (ms)        |    526.0 |     727.5 | -27.7% |
| TTFT p50 (ms)        |    356.8 |     510.8 | -30.1% |
| ITL avg (ms)         |     14.1 |      15.5 |  -8.8% |
| Req Latency avg (ms) |   2630.0 |    3035.7 | -13.4% |

**Enabling embedding cache on `Qwen3-VL-30B-A3B-Instruct-FP8` shows an average improvement of +16% throughput, -28% TTFT, and -13% request latency on a single aggregated replica of GB200 using the vLLM backend**

## Pre-requisites

To reproduce the results in the table, the following is required:

1. **Dynamo Platform installed** - See [Kubernetes Deployment Guide](../../docs/kubernetes/README.md)
2. **GB200**
3. **HuggingFace token** configured:
   ```bash
   export NAMESPACE=your-namespace
   kubectl create secret generic hf-token-secret \
     --from-literal=HF_TOKEN="your-token" \
     -n ${NAMESPACE}
   ```

## Dataset Generation

`data-gen/generate-datasets-job.yaml` creates a dataset of synthetic text + image data with 80% image overlap. The script does this by manipulating the "total slots" and "image pool".

Total number of slots is calculated as `num_requests*images/request`, representing how many total images the benchmark will iterate through. The image pool is how many images the benchmark can choose from to attach to a request.

The `data-gen/generate-datasets-job.yaml` script creates a dataset of 1000 requests, 1 image per request, and an image pool of 200. Each request will pick an image from this pool without replacement, and loop back through the image pool after it has been exhausted. Thus, the first 200 out of 1000 requests will contain unique images, while the remaining 800 out of 1000 requests will have been seen already by the inference engine. Refer to jsonl [documentation](https://github.com/ai-dynamo/dynamo/tree/main/benchmarks/multimodal/jsonl) for more details on data generation.

Each dataset is hardcoded to have 400 tokens of user-input text.

To generate the dataset, run:

```bash
kubectl apply -f data-gen/generate-datasets-job.yaml -n ${NAMESPACE}
```

## Notes

1. Exact cache hit rates cannot be explicitly controlled via dataset due to potential LRU embedding cache eviction policies; however, decreasing the image pool relative to the number of requests allows for proportionally higher probabilities of seeing duplicate images and cache hits. Increasing the embedding cache capacity also allows for higher cache hit rate because it will evict less.

**2. Agg embedding cache uses vLLM's native `ec_both` ECConnector role, supported in vLLM 0.17+. No patches required. See [multimodal-vllm.md](https://github.com/ai-dynamo/dynamo/blob/main/docs/features/multimodal/multimodal-vllm.md#embedding-cache) for more details.**

3. Replace placeholders in `*.yaml` before running:
   - `storageClassName: "your-storage-class-name"` in `model-cache/model-cache.yaml`
   - `image: <your-dynamo-image>` in all `vllm/*/deploy.yaml` files
   - `NAMESPACE=your-namespace` and `HF_TOKEN="your-token"` in the setup commands

## Directory setup

This recipe has three top-level components: `model-cache/` for PVC/model prep, `data-gen/` for dataset creation, and `vllm/agg-embedding-cache/` for deployment and benchmarking with [AIPerf](https://github.com/ai-dynamo/aiperf).

```text
qwen3-vl-30b/
├── data-gen/
│   └── generate-datasets-job.yaml
├── model-cache/
│   ├── model-cache.yaml
│   └── model-download.yaml
└── vllm/
    └── agg-embedding-cache/
        ├── deploy.yaml
        ├── perf.yaml
        └── run-benchmark.sh
```

The `deploy.yaml` script has `DYN_MULTIMODAL_EMBEDDING_CACHE_GB=10` by default, which represents an embedding cache **on** configuration. To toggle it off, set the env variable to 0.

Similarly, each `perf.yaml` exposes a `CACHE_MODE` env variable to control where AIPerf dumps its results. Set it to either `cache_on` or `cache_off` depending on your deployment.

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

### 3. Deploy and Benchmark (`agg-embedding-cache`)

```bash
# deploy.yaml defaults to cache ON (DYN_MULTIMODAL_EMBEDDING_CACHE_GB=10)
kubectl apply -f vllm/agg-embedding-cache/deploy.yaml -n ${NAMESPACE}
kubectl wait --for=condition=Ready dynamographdeployment/qwen3-vl-agg -n ${NAMESPACE} --timeout=900s

kubectl apply -f vllm/agg-embedding-cache/perf.yaml -n ${NAMESPACE}
kubectl wait --for=condition=Ready pod/qwen3-vl-agg-benchmark -n ${NAMESPACE} --timeout=300s
```

Optional: to run cache OFF, change `DYN_MULTIMODAL_EMBEDDING_CACHE_GB` to `0` in `vllm/agg-embedding-cache/deploy.yaml` and set `CACHE_MODE=cache_off` in `vllm/agg-embedding-cache/perf.yaml` before applying.

### 4. Monitor Benchmark Progress

```bash
kubectl get pods -n ${NAMESPACE} -l app=benchmark

# Follow benchmark logs in real time
kubectl logs -f qwen3-vl-agg-benchmark -n ${NAMESPACE}

# Wait for completion
kubectl wait --for=jsonpath='{.status.phase}'=Succeeded pod/qwen3-vl-agg-benchmark -n ${NAMESPACE} --timeout=7200s
```

Wait for `Run complete. Artifacts in /perf-cache/artifacts/qwen3_vl_30b_embedding_cache/agg/<cache_mode>`.

`vllm/agg-embedding-cache/run-benchmark.sh` is also provided as a helper to launch cache-on/cache-off runs.