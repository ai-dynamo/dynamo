<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Qwen3-235B-A22B-FP8 Recipes

Production-ready deployments for **Qwen3-235B-A22B** (MoE model with 22B active parameters) using TensorRT-LLM.

## Available Configurations

| Configuration | GPUs | Hardware | Mode | Description |
|--------------|------|----------|------|-------------|
| [**trtllm/agg/hopper**](trtllm/agg/hopper/) | 16x GPU | H100/H200 | Aggregated | TP4, EP4, KV-aware routing |
| [**trtllm/agg/blackwell**](trtllm/agg/blackwell/) | 16x GPU | B100/B200 | Aggregated | TP4, EP4, KV-aware routing, DEEPGEMM |
| [**trtllm/disagg/hopper**](trtllm/disagg/hopper/) | 16x GPU | H100/H200 | Disaggregated | Prefill/decode separation |
| [**trtllm/disagg/blackwell**](trtllm/disagg/blackwell/) | 16x GPU | B100/B200 | Disaggregated | Prefill/decode separation, DEEPGEMM |

## Prerequisites

1. **Dynamo Platform installed** — See [Kubernetes Deployment Guide](../../docs/fern/pages/kubernetes/getting-started/quickstart.mdx)
2. **GPU cluster** with H100/H200 (Hopper) or B100/B200 (Blackwell) GPUs — see [Hardware Requirements](#hardware-requirements)
3. **HuggingFace token** with access to Qwen models

## Quick Start

```bash
# Set namespace
export NAMESPACE=dynamo-demo
kubectl create namespace ${NAMESPACE}

# Create HuggingFace token secret
kubectl create secret generic hf-token-secret \
  --from-literal=HF_TOKEN="your-token-here" \
  -n ${NAMESPACE}

# Download model (update storageClassName in model-cache.yaml first!)
kubectl apply -f model-cache/ -n ${NAMESPACE}
kubectl wait --for=condition=Complete job/model-download -n ${NAMESPACE} --timeout=3600s

# Deploy — choose the variant matching your hardware:
kubectl apply -f trtllm/agg/hopper/deploy.yaml -n ${NAMESPACE}       # H100/H200
# OR: kubectl apply -f trtllm/agg/blackwell/deploy.yaml -n ${NAMESPACE}   # B100/B200
# OR: kubectl apply -f trtllm/disagg/hopper/deploy.yaml -n ${NAMESPACE}   # H100/H200

# For Blackwell disaggregated, choose the provider-specific manifest matching your cluster interconnect:
# OR: kubectl apply -f trtllm/disagg/blackwell/deploy-aws-p6-b200.48xlarge.yaml -n ${NAMESPACE}
# OR: kubectl apply -f trtllm/disagg/blackwell/deploy-gcp-roce.yaml -n ${NAMESPACE}
# OR: kubectl apply -f trtllm/disagg/blackwell/deploy-nscale-ib.yaml -n ${NAMESPACE}
```

## Cloud Provider Overlays

The Blackwell disaggregated recipe keeps the shared deployment in `trtllm/disagg/blackwell/kustomize/base/deploy.yaml`.
Provider-specific deltas live in Kustomize Components and instance-type templates
and are selected by `trtllm/disagg/blackwell/.kustomize-matrix.yaml`.
Shared Kustomize building blocks belong under `recipes/kustomize/components/`;
the disaggregated provider Components use the
backend-neutral `PrefillWorker` and `DecodeWorker` service keys.
Kustomize is both the authoring model and documentation for the variants: the
base and Components explain provider settings, and each public overlay documents
the selected composition. Cluster users can apply a checked-in `deploy-*.yaml`
manifest directly or apply its public overlay with `kubectl apply -k`; neither
path requires regeneration.
For recipe contributors, the source of truth is the matrix, `kustomize/base/`,
recipe-local Components, shared template sources, plus any referenced shared
Components under `recipes/kustomize/components/`. Public overlay
`kustomization.yaml` files, generated template Components, and `deploy-*.yaml`
files are generated artifacts: commit them for review, but do not edit them by
hand.
Kustomize drops comments while rendering Kubernetes objects, so the renderer re-inserts non-SPDX comments from source YAML before matching rendered fields.
Comments inside literal block scalars already render in place.

| Rendered manifest | Provider fabric | Patch source |
|-------------------|-----------------|--------------|
| `trtllm/disagg/blackwell/deploy-generic.yaml` | Provider-neutral baseline | `trtllm/disagg/blackwell/kustomize/overlays/generic/` |
| `trtllm/disagg/blackwell/deploy-aws-p6-b200.48xlarge.yaml` | AWS EFA on `p6-b200.48xlarge`, derived from each worker's GPU count | `recipes/kustomize/templates/aws-efa/p6-b200.48xlarge/` |
| `trtllm/disagg/blackwell/deploy-gcp-roce.yaml` | GKE RoCE | `recipes/kustomize/components/disagg-workers/gke-roce/` |
| `trtllm/disagg/blackwell/deploy-nscale-ib.yaml` | Nscale InfiniBand | `recipes/kustomize/components/disagg-workers/nscale-ib/` |

For example, apply the GKE RoCE composition from its checked-in Kustomization:

```bash
kubectl apply -k trtllm/disagg/blackwell/kustomize/overlays/gcp-roce -n ${NAMESPACE}
```

To make a local, uncommitted composition, create your own `kustomization.yaml` in
the repository checkout or run `compose` from the repository root:

```bash
scripts/kustomize-matrix.py compose \
  recipes/qwen3-235b-a22b-fp8/trtllm/disagg/blackwell/kustomize/base \
  recipes/kustomize/components/disagg-workers/gke-roce \
  | kubectl apply -f - -n ${NAMESPACE}
```

Only recipe contributors update checked-in derived artifacts. After editing the
matrix, base, or Components, regenerate the public overlays and apply-able
manifests from the repository root:

```bash
scripts/kustomize-matrix.py unfold trtllm/disagg/blackwell/.kustomize-matrix.yaml
scripts/kustomize-matrix.py render trtllm/disagg/blackwell/.kustomize-matrix.yaml
```

## Test the Deployment

```bash
# Port-forward the frontend
kubectl port-forward svc/qwen3-235b-a22b-agg-frontend 8000:8000 -n ${NAMESPACE}

# Send a test request
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-235B-A22B-FP8",
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 50
  }'
```

## Model Details

- **Model**: `Qwen/Qwen3-235B-A22B-FP8`
- **Architecture**: 235B parameter Mixture-of-Experts (MoE)
- **Active parameters**: ~22B per token
- **Backend**: TensorRT-LLM (PyTorch backend)
- **Parallelism**: TP4 × EP4 (Expert Parallel)

## Hardware Requirements

This recipe has separate variants for Hopper and Blackwell because the two architectures require different MoE backend configurations with TRT-LLM 1.3.x:

- **Hopper (H100/H200, SM90)**: uses the default MoE backend — `trtllm/{agg,disagg}/hopper/`
- **Blackwell (B100/B200, SM100+)**: requires `moe_config.backend: DEEPGEMM` — `trtllm/{agg,disagg}/blackwell/`

The difference: the default CUTLASS MoE backend in TRT-LLM 1.3.x falls through to a Hopper-specific JIT path when running on SM100, causing a crash. DEEPGEMM is the required workaround on Blackwell for this version. DEEPGEMM in turn crashes on Hopper due to a scale-factor dtype mismatch. Hence two separate variants.

| Configuration | GPUs | Min GPU VRAM (Total) |
|--------------|------|----------------------|
| Aggregated (Hopper) | 16x H100/H200 | ~1.3TB |
| Aggregated (Blackwell) | 16x B100/B200 | ~1.3TB |
| Disaggregated (Hopper) | 16x H100/H200 | ~1.3TB |
| Disaggregated (Blackwell) | 16x B100/B200 | ~1.3TB |

## Notes

- Update `storageClassName` in `model-cache/model-cache.yaml` before deploying
- Model download may take 30-60 minutes
- Uses KV-aware routing for efficient cache utilization
- Chunked prefill enabled for aggregated mode (disabled for disaggregated)
