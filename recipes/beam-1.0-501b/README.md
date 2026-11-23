<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Beam-1.0-501B-A23B Recipes

Aggregated and disaggregated serving for **Beam-1.0-501B-A23B** with NVIDIA
Dynamo on H200, B200, B300, GB200, and GB300. Select a target below for KV-aware routing
with SGLang or vLLM and NIXL transfer between disaggregated prefill/decode workers.

## Configurations

All profiles use DFlash2 block length 8, FP8 KV cache, 64-token KV pages,
and up to 1,048,576 context tokens.

| Target | GPUs | Workers | Weight precision | KV cache |
| --- | --- | --- | --- | --- |
| [SGLang H200 aggregated](sglang/agg-h200/deploy.yaml) | 8 | 1 × TP8 | FP8 | Unified GPU pool |
| [SGLang H200 disaggregated](sglang/disagg-h200/deploy.yaml) | 16 | 1P1D, TP8 per role | FP8 | GPU KV pools |
| [SGLang B200 aggregated](sglang/agg-b200-nvfp4/deploy.yaml) | 8 | 2 × TP4 | NVFP4 | Unified GPU pool |
| [SGLang B200 disaggregated](sglang/disagg-b200-nvfp4/deploy.yaml) | 8 | 1P1D, TP4 per role | NVFP4 | GPU KV pools |
| [SGLang B300 aggregated](sglang/agg-b300-nvfp4/deploy.yaml) | 8 | 2 × TP4 | NVFP4 | Unified GPU pool |
| [SGLang B300 disaggregated](sglang/disagg-b300-nvfp4/deploy.yaml) | 8 | 1P1D, TP4 per role | NVFP4 | GPU KV pools |
| [SGLang GB200 aggregated](sglang/agg-gb200-nvfp4/deploy.yaml) | 8 | 2 × TP4 | NVFP4 | Unified GPU pool |
| [SGLang GB200 disaggregated](sglang/disagg-gb200-nvfp4/deploy.yaml) | 8 | 1P1D, TP4 per role | NVFP4 | GPU KV pools |
| [SGLang GB300 aggregated](sglang/agg-gb300-nvfp4/deploy.yaml) | 8 | 2 × TP4 | NVFP4 | Unified GPU pool |
| [SGLang GB300 disaggregated](sglang/disagg-gb300-nvfp4/deploy.yaml) | 8 | 1P1D, TP4 per role | NVFP4 | GPU KV pools |
| [vLLM H200 aggregated](vllm/agg-h200/deploy.yaml) | 8 | 1 × TP8 | FP8 | Native host offload |
| [vLLM H200 disaggregated](vllm/disagg-h200/deploy.yaml) | 16 | 1P1D, TP8 per role | FP8 | NIXL + native host offload |
| [vLLM B200 aggregated](vllm/agg-b200-nvfp4/deploy.yaml) | 8 | 2 × TP4 | NVFP4 | Native host offload |
| [vLLM B200 disaggregated](vllm/disagg-b200-nvfp4/deploy.yaml) | 8 | 1P1D, TP4 per role | NVFP4 | NIXL + native host offload |
| [vLLM B300 aggregated](vllm/agg-b300-nvfp4/deploy.yaml) | 8 | 2 × TP4 | NVFP4 | Native host offload |
| [vLLM B300 disaggregated](vllm/disagg-b300-nvfp4/deploy.yaml) | 8 | 1P1D, TP4 per role | NVFP4 | NIXL + native host offload |
| [vLLM GB200 aggregated](vllm/agg-gb200-nvfp4/deploy.yaml) | 8 | 2 × TP4 | NVFP4 | Native host offload |
| [vLLM GB200 disaggregated](vllm/disagg-gb200-nvfp4/deploy.yaml) | 8 | 1P1D, TP4 per role | NVFP4 | NIXL + native host offload |
| [vLLM GB300 aggregated](vllm/agg-gb300-nvfp4/deploy.yaml) | 8 | 2 × TP4 | NVFP4 | Native host offload |
| [vLLM GB300 disaggregated](vllm/disagg-gb300-nvfp4/deploy.yaml) | 8 | 1P1D, TP4 per role | NVFP4 | NIXL + native host offload |

`1P1D` means one prefill worker and one decode worker. SGLang publishes KV
events; vLLM uses prediction-based KV routing. Each `deploy.yaml` contains the
engine, cache, and memory settings for that target.

## Prerequisites

- [Dynamo Kubernetes Platform](../../docs/fern/pages/kubernetes/getting-started/quickstart.mdx)
  with the `nvidia.com/v1beta1` CRD and the GPUs listed above.
- The manifests use Dynamo 1.6.0 with these NVIDIA runtime images:
  `nvcr.io/nvidia/ai-dynamo/sglang-runtime:1.6.0-beam-1.0-501b-dev.1` and
  `nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.6.0-beam-1.0-501b-dev.1`.
- Access to the Hugging Face checkpoints below, including DFlash2 weights
  for every target.
- A `shared-model-cache` ReadWriteMany PVC with at least 2Ti capacity. Configure
  your storage class, GPU placement, and registry access using your cluster's
  settings; the [Kustomization starter](../templates/kustomize/README.md) can help.
  Keep each TP group on one NVLink-connected host.
- For disaggregated targets, provide RDMA device access and NIXL/UCX connectivity
  between both roles (see [Network for disaggregated targets](#network-for-disaggregated-targets)).
  SGLang uses bootstrap port `8998`; vLLM uses NIXL side-channel port `5600` at
  each worker's pod IP.

### Network for disaggregated targets

KV blocks move between prefill and decode through NIXL over UCX. Wire the
fabric in your cluster overlay:

- **RDMA.** Hand GPUDirect-capable NICs to both worker pods (device-plugin
  resource or a DRA `ResourceClaimTemplate`). SGLang pins
  `UCX_TLS=cuda_ipc,cuda_copy,rc` on GB200/GB300 and `UCX_TLS=cuda_copy,rc`
  on H200/B200/B300; vLLM lets UCX choose.
- **vLLM registers its whole KV pool as one memory region** (about 172 GB per
  rank on GB300). If the NIC rejects it (`ibv_reg_dmabuf_mr ... Resource
  temporarily unavailable`, `NIXL_ERR_BACKEND`), lower
  `--gpu-memory-utilization` (the GB300 and B300 disaggregated profiles ship
  with 0.5 for this reason) or use TCP.
- **TCP fallback** (no GPUDirect RDMA): set on both roles
  `UCX_TLS=tcp,cuda_copy,cuda_ipc` and `UCX_NET_DEVICES`,
  `NCCL_SOCKET_IFNAME`, `GLOO_SOCKET_IFNAME` to the pod interface (`eth0`).
  Functional, but KV transfer goes through host memory and the pod network.

## Quick Start

### 1. Choose a profile

Run from the repository root:

```bash
cd recipes/beam-1.0-501b
export NAMESPACE=beam
export PROFILE=sglang/agg-h200 # choose a target from the table above
PROFILE_TARGET=${PROFILE#*/}
PROFILE_TARGET=${PROFILE_TARGET%-nvfp4}
export DEPLOYMENT="beam-${PROFILE%%/*}-${PROFILE_TARGET#*-}-${PROFILE_TARGET%%-*}"
kubectl create namespace "$NAMESPACE"
```

### 2. Stage the model and drafter

Uncomment `storageClassName` in `model-cache/model-cache.yaml` and set it to
an RWX-capable storage class. The commands below set the download Job's
`MODEL_REPO` from your selected profile:

| Checkpoint | Hugging Face repository | Recipe use |
| --- | --- | --- |
| NVFP4 | [reflection/Beam-1.0-501B-A23B-NVFP4](https://huggingface.co/reflection/Beam-1.0-501B-A23B-NVFP4) | B200/B300/GB200/GB300 |
| FP8 | [reflection/Beam-1.0-501B-A23B-FP8](https://huggingface.co/reflection/Beam-1.0-501B-A23B-FP8) | H200 |
| DFlash2 | [reflection/Beam-1.0-501B-A23B-DFlash2](https://huggingface.co/reflection/Beam-1.0-501B-A23B-DFlash2) | `DRAFT_REPO` for every target |

Keep `MODEL_REVISION` and `DRAFT_REVISION` at `main` for these offline recipes.
Dynamo and the benchmark resolve repository IDs through the cache's `refs/main`.
For gated repositories, add an `HF_TOKEN` Secret reference to the download Job.
The Job populates the Hugging Face cache under `HF_HOME=/shared-model-cache`.
Workers resolve repository IDs from that cache while offline. Re-run the download
Job if the PVC contains only checkpoints staged with `--local-dir`.

```bash
case "$PROFILE" in
  *-h200)
    MODEL_REPO=reflection/Beam-1.0-501B-A23B-FP8
    ;;
  *-nvfp4)
    MODEL_REPO=reflection/Beam-1.0-501B-A23B-NVFP4
    ;;
  *) printf 'Unknown profile: %s\n' "$PROFILE" >&2; exit 1 ;;
esac

kubectl apply -n "$NAMESPACE" -f model-cache/model-cache.yaml
kubectl patch --local -f model-cache/model-download.yaml --type=strategic \
  -p "{\"spec\":{\"template\":{\"spec\":{\"containers\":[{
    \"name\":\"download\",\"env\":[
      {\"name\":\"MODEL_REPO\",\"value\":\"$MODEL_REPO\"}
    ]}]}}}}" -o yaml \
  | kubectl apply -n "$NAMESPACE" -f -
kubectl logs -n "$NAMESPACE" -f job/beam-model-download
kubectl wait -n "$NAMESPACE" --for=condition=Complete job/beam-model-download --timeout=7200s
```

### 3. Deploy

Use your cluster overlay for GPU and CPU-architecture placement when nodes differ.
On a cluster with matching hardware, apply the portable manifest below.
SGLang engine settings live in the ConfigMap included in each manifest.

```bash
kubectl apply -n "$NAMESPACE" -f "$PROFILE/deploy.yaml"
kubectl wait -n "$NAMESPACE" --for=condition=Ready "dgd/$DEPLOYMENT" --timeout=7200s
kubectl port-forward -n "$NAMESPACE" "svc/$DEPLOYMENT-frontend" 8000:8000
```

### 4. Send a request

In another terminal:

```bash
curl --fail-with-body http://localhost:8000/v1/models
curl --fail-with-body -N http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"Beam-1.0-501B-A23B","messages":[{"role":"user","content":"What is 2 + 2?"}],"max_tokens":256,"temperature":0,"stream":true}'
```
