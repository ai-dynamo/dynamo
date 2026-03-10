# Qwen3-VL-30B-A3B-FP8: Aggregated Encoder Cache On vs Off Comparison

This recipe demonstrates the performance difference when encoder cache is enabled for multi-modal payloads. It includes guidance on creating an artificial dataset with user-defined image re-use, and production-ready deployments for `Qwen3-VL-30B-A3B`.

## Results

| Metric                         | Cache OFF  | Cache ON (4GB) | Delta  |
|-------------------------------|------------|----------------|--------|
| TTFT (avg)                    | 369 ms     | 302 ms         | -18.2% |
| ITL (avg)                     | 7.56 ms    | 7.61 ms        | ~same  |
| Prefill Throughput/user (avg) | 1633 tok/s | 2158 tok/s     | +32.1% |

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

Each dataset is hardcoded to have 400 tokens of user-input text. AIPerf configures the system prompt via `--shared-system-prompt-length` and the `perf.yaml` scripts have this set to 160 tokens.

To generate the dataset, run:

```bash
kubectl apply -f data-gen/generate-datasets-job.yaml -n ${NAMESPACE}
```

## Notes

1. Exact cache hit rates cannot be explicitly controlled via dataset due to potential LRU encoder cache eviction policies; however, decreasing the image pool relative to the number of requests allows for proportionally higher probabilities of seeing duplicate images and cache hits. Increasing the encoder cache capacity also allows for higher cache hit rate because it will evict less.

**2. Agg encoder cache requires `ec_both` ECConnector role in vLLM, but that functionality was merged post 1.0.0 release. If you see an error such as `Input should be 'ec_producer' or 'ec_consumer' [type=literal_error, input_value='ec_both', input_type=str]`, you can use the `patch_vllm_agg_encoder_cache.sh` script to re-tag your dynamo image with the patch applied. See [multimodal-vllm.md](https://github.com/ai-dynamo/dynamo/blob/main/docs/features/multimodal/multimodal-vllm.md#embedding-cache) for more details.**

3. Replace placeholders in `*.yaml` before running:
   - `storageClassName: "your-storage-class-name"` in `model-cache/model-cache.yaml`
   - `image: <your-dynamo-image>` in all `vllm/*/deploy.yaml` files
   - `NAMESPACE=your-namespace` and `HF_TOKEN="your-token"` in the setup commands

## Directory setup

This recipe has three top-level components: `model-cache/` for PVC/model prep, `data-gen/` for dataset creation, and `vllm/agg-encoder-cache/` for deployment and benchmarking with [AIPerf](https://github.com/ai-dynamo/aiperf).

```text
qwen3-vl-30b/
├── data-gen/
│   └── generate-datasets-job.yaml
├── model-cache/
│   ├── model-cache.yaml
│   └── model-download.yaml
└── vllm/
    └── agg-encoder-cache/
        ├── deploy.yaml
        ├── patch_vllm_agg_encoder_cache.sh
        ├── perf.yaml
        └── run-benchmark.sh
```

The `deploy.yaml` scripts have `MM_EMBEDDING_CACHE_GB=4` by default, which represents an embedding cache **on** configuration. To toggle it off, set the env variable to 0.

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

### 3. Build Patched Image

```bash
# Build a patched runtime image outside the cluster.
# This applies the required vLLM diffs and produces a new image tag.
./vllm/agg-encoder-cache/patch_vllm_agg_encoder_cache.sh \
  --base-image <your-dynamo-image>:<tag> \
  --output-image <your-dynamo-image>:<tag>-agg-ec-patched
```

Then set `image: <your-dynamo-image>` in `vllm/agg-encoder-cache/deploy.yaml` to your patched output image tag.

### 4. Deploy and Benchmark (`agg-encoder-cache`)

```bash
# deploy.yaml defaults to cache ON (MM_EMBEDDING_CACHE_GB=4)
kubectl apply -f vllm/agg-encoder-cache/deploy.yaml -n ${NAMESPACE}
kubectl wait --for=condition=Ready dynamographdeployment/qwen3-vl-agg -n ${NAMESPACE} --timeout=900s

kubectl apply -f vllm/agg-encoder-cache/perf.yaml -n ${NAMESPACE}
kubectl wait --for=condition=Ready pod/qwen3-vl-agg-benchmark -n ${NAMESPACE} --timeout=300s
```

Optional: to run cache OFF, change `MM_EMBEDDING_CACHE_GB` to `0` in `vllm/agg-encoder-cache/deploy.yaml` and set `CACHE_MODE=cache_off` in `vllm/agg-encoder-cache/perf.yaml` before applying.

### 5. Monitor Benchmark Progress

```bash
kubectl get pods -n ${NAMESPACE} -l app=benchmark

# Follow benchmark logs in real time
kubectl logs -f qwen3-vl-agg-benchmark -n ${NAMESPACE}

# Wait for completion
kubectl wait --for=jsonpath='{.status.phase}'=Succeeded pod/qwen3-vl-agg-benchmark -n ${NAMESPACE} --timeout=7200s
```

Wait for `Run complete. Artifacts in /perf-cache/artifacts/qwen3_vl_30b_encoder_cache/agg/<cache_mode>`.

`vllm/agg-encoder-cache/run-benchmark.sh` is also provided as a helper to launch cache-on/cache-off runs.