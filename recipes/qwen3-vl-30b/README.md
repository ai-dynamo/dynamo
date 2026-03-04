# Qwen3-VL-30B-A3B: Encoder Cache in Disaggregated E/PD

This recipe demonstrates throughput/latency gains from enabling multimodal encoder cache in disaggregated serving for `Qwen/Qwen3-VL-30B-A3B-Instruct`.

The experiment uses:
- E/PD topology: 1 encode worker + 7 PD workers
- 5000 requests per run
- 1000-token user text target
- image reuse sweep: duplicate probability `{0.00, 0.10, 0.25, 0.50, 0.75}`
- cache comparison: embedding cache OFF vs ON
- embedding transfer mode: `DYN_VLLM_EMBEDDING_TRANSFER_MODE=nixl-write`
- request plane: `DYN_REQUEST_PLANE=tcp`

## Why TCP Request Plane Is Used

Multimodal requests can include large image payloads (base64 or expanded metadata). The TCP request plane avoids message-size constraints that can appear on the default messaging plane and is commonly used in E/PD multimodal launches for stability with large payloads.

For this recipe:
- `DYN_REQUEST_PLANE=tcp` is strongly recommended for robustness in multimodal runs.
- It is not an image-encoder optimization by itself; it is a transport/reliability choice for request delivery.

## Hardware Requirements

- 8 GPUs on one node (validated target: 1 GPU for encoder + 7 GPUs for PD workers)
- Kubernetes cluster with Dynamo CRDs installed
- RWX-capable StorageClass for shared PVCs

## Repository Layout

```text
qwen3-vl-30b/
  README.md
  patches/
    patch_vllm_agg_encoder_cache.sh
    patch_vllm_mm_router.sh
  model-cache/
    model-cache.yaml
    model-download.yaml
  vllm/
    agg/
      deploy-cache-off.yaml
      deploy-cache-on.yaml
    disagg-ep-d/
      deploy-cache-off.yaml
      deploy-cache-on.yaml
    disagg-e-pd/
      deploy.yaml
      perf.yaml
      generate_datasets.sh
```

## Deployment Variants

- `vllm/agg/deploy-cache-off.yaml`: aggregated DP8xTP1 workers (8 replicas, TP1 each), cache disabled.
- `vllm/agg/deploy-cache-on.yaml`: aggregated DP8xTP1 workers (8 replicas, TP1 each), cache enabled.
- `vllm/disagg-e-pd/deploy.yaml`: E/PD (1E:7PD), cache enabled by default.
- `vllm/disagg-ep-d/deploy-cache-off.yaml`: EP/D (4EP:4D), cache disabled.
- `vllm/disagg-ep-d/deploy-cache-on.yaml`: EP/D (4EP:4D), cache enabled (`--multimodal-embedding-cache-capacity-gb 10` on EP workers).

For your “no dedicated encoder GPU” goal, EP/D is the relevant topology: encoding is handled in the EP workers (prefill stage), so there is no standalone encode worker service consuming its own GPU.

## Offline Patch Scripts (Build New Images)

Use these scripts to patch vLLM at image build time (offline from runtime pod startup).

### Aggregated Encoder Cache Patch Image

```bash
cd patches
./patch_vllm_agg_encoder_cache.sh \
  --base-image nvcr.io/nvidia/ai-dynamo/vllm-runtime:0.8.0 \
  --output-image <registry>/vllm-runtime:0.8.0-agg-ec-patched
```

### MM-Aware Router Patch Image

```bash
cd patches
./patch_vllm_mm_router.sh \
  --base-image nvcr.io/nvidia/ai-dynamo/vllm-runtime:0.8.0 \
  --output-image <registry>/vllm-runtime:0.8.0-mm-router-patched
```

After building, push with your normal registry workflow if needed, then set deployment `image` fields to your patched tags.

## Experiment Matrix

`generate_datasets.sh` maps duplicate probabilities to `--images-pool` values (for 5000 requests x 3 images/request = 15000 total image slots):

| Tag | Duplicate Probability | Images Pool |
|-----|------------------------|-------------|
| `r00` | 0.00 | 15000 |
| `r10` | 0.10 | 13500 |
| `r25` | 0.25 | 11250 |
| `r50` | 0.50 | 7500 |
| `r75` | 0.75 | 3750 |

Run each reuse level twice:
- cache OFF (`MM_EMBEDDING_CACHE_GB=0`)
- cache ON (`MM_EMBEDDING_CACHE_GB>0`, recommended start: `10`)

Total runs: 10

## Prerequisites

1. Dynamo platform is installed.
2. HuggingFace token is configured:

```bash
export NAMESPACE=your-namespace
kubectl create secret generic hf-token-secret \
  --from-literal=HF_TOKEN="your-token" \
  -n ${NAMESPACE}
```

3. Update `model-cache/model-cache.yaml` with your `storageClassName`.

## Quick Start

### 1) Create storage

```bash
kubectl apply -f model-cache/model-cache.yaml -n ${NAMESPACE}
```

### 2) Download model into shared cache

```bash
kubectl apply -f model-cache/model-download.yaml -n ${NAMESPACE}
kubectl wait --for=condition=Complete job/model-download -n ${NAMESPACE} --timeout=3600s
```

### 3) Deploy graph

Pick one deployment manifest from the variants above. For E/PD (cache on by default):

```bash
kubectl apply -f vllm/disagg-e-pd/deploy.yaml -n ${NAMESPACE}
kubectl wait --for=condition=ready pod \
  -l nvidia.com/dynamo-graph-deployment-name=qwen3-vl-30b-disagg-e-pd-1e-7pd \
  -n ${NAMESPACE} --timeout=1800s
```

### 4) Generate datasets (local machine)

From this recipe directory:

```bash
cd vllm/disagg-e-pd
chmod +x generate_datasets.sh
./generate_datasets.sh
```

This generates five JSONL files under `vllm/disagg-e-pd/datasets/`.

### 5) Upload datasets to perf cache

Create a temporary helper pod that mounts `perf-cache`, copy datasets into it, then remove the pod:

```bash
kubectl run perf-cache-helper -n ${NAMESPACE} --image=python:3.11 --restart=Never \
  --overrides='
{
  "apiVersion": "v1",
  "spec": {
    "containers": [{
      "name": "helper",
      "image": "python:3.11",
      "command": ["sleep","3600"],
      "volumeMounts": [{"name":"perf-cache","mountPath":"/perf-cache"}]
    }],
    "volumes": [{
      "name":"perf-cache",
      "persistentVolumeClaim":{"claimName":"perf-cache"}
    }]
  }
}'

kubectl cp ./datasets/. ${NAMESPACE}/perf-cache-helper:/perf-cache/datasets
kubectl delete pod perf-cache-helper -n ${NAMESPACE}
```

### 6) Run benchmark sweep

`perf.yaml` runs tags `r00,r10,r25,r50,r75` and writes artifacts under `/perf-cache/artifacts/qwen3_vl_30b_encoder_cache/<CACHE_MODE>_*`.
Set `FRONTEND` in `perf.yaml` to match your selected deployment:
- `qwen3-vl-30b-disagg-e-pd-1e-7pd-frontend`
- `qwen3-vl-30b-disagg-ep-d-cache-on-frontend` or `qwen3-vl-30b-disagg-ep-d-cache-off-frontend`
- `qwen3-vl-30b-agg-cache-on-frontend` or `qwen3-vl-30b-agg-cache-off-frontend`

```bash
kubectl apply -f vllm/disagg-e-pd/perf.yaml -n ${NAMESPACE}
```

### 7) Cache mode handling

- For cache-off runs, apply a `*-cache-off.yaml` manifest and set `CACHE_MODE=cache_off` in `perf.yaml`.
- For cache-on runs, apply a `*-cache-on.yaml` manifest (or E/PD default deploy) and set `CACHE_MODE=cache_on` in `perf.yaml`.

```bash
# Re-apply after editing deploy.yaml MM_EMBEDDING_CACHE_GB
kubectl apply -f vllm/disagg-e-pd/deploy.yaml -n ${NAMESPACE}

# Re-apply after editing perf.yaml CACHE_MODE=cache_on
kubectl apply -f vllm/disagg-e-pd/perf.yaml -n ${NAMESPACE}
```

### 8) Collect artifacts

```bash
kubectl get pods -n ${NAMESPACE} -l app=benchmark
kubectl exec -it -n ${NAMESPACE} qwen3-vl-30b-disagg-e-pd-benchmark -- \
  ls -la /perf-cache/artifacts/qwen3_vl_30b_encoder_cache
kubectl cp ${NAMESPACE}/qwen3-vl-30b-disagg-e-pd-benchmark:/perf-cache/artifacts/qwen3_vl_30b_encoder_cache \
  ./qwen3_vl_30b_encoder_cache_results
```

## Expected Results

As duplicate-image probability increases, cache-hit opportunity increases. With cache ON, expect:
- lower TTFT relative to cache OFF
- improved tail latency at higher reuse levels
- higher effective throughput under the same request shape

The largest gains should appear at `r50` and `r75`.

## Cleanup

```bash
kubectl delete pod -l app=benchmark -n ${NAMESPACE}
kubectl delete dynamographdeployment qwen3-vl-30b-disagg-e-pd-1e-7pd -n ${NAMESPACE}
kubectl delete job model-download -n ${NAMESPACE}
```
