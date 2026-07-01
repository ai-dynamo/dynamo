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

1. **Dynamo Platform installed** — See [Kubernetes Deployment Guide](../../docs/kubernetes/README.md)
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
# OR: kubectl apply -f trtllm/disagg/blackwell/deploy-aws-efa.yaml -n ${NAMESPACE}
# OR: kubectl apply -f trtllm/disagg/blackwell/deploy-gcp-roce.yaml -n ${NAMESPACE}
# OR: kubectl apply -f trtllm/disagg/blackwell/deploy-nscale-ib.yaml -n ${NAMESPACE}
```

## Cloud Provider Overlays

The Blackwell disaggregated recipe keeps the shared deployment in `trtllm/disagg/blackwell/kustomize/base/deploy.yaml`.
Provider-specific deltas live in Kustomize merge patches under `trtllm/disagg/blackwell/kustomize/overlays/`.
Overlay directories starting with `_` are intermediate and are not rendered. Shared Kustomize building blocks belong under
`trtllm/disagg/blackwell/kustomize/components/`.
The checked-in `deploy-*.yaml` files are rendered artifacts so you can review and apply the final manifests from GitHub.
Kustomize drops comments while rendering Kubernetes objects, so the renderer re-inserts non-SPDX comments from the base and
overlay YAML before matching rendered fields. Comments inside literal block scalars already render in place.

| Rendered manifest | Provider fabric | Patch source |
|-------------------|-----------------|--------------|
| `trtllm/disagg/blackwell/deploy-generic.yaml` | Provider-neutral baseline | `trtllm/disagg/blackwell/kustomize/overlays/generic/` |
| `trtllm/disagg/blackwell/deploy-aws-efa.yaml` | AWS EFA | `trtllm/disagg/blackwell/kustomize/overlays/aws-efa/patch-dgd.yaml` |
| `trtllm/disagg/blackwell/deploy-gcp-roce.yaml` | GKE RoCE | `trtllm/disagg/blackwell/kustomize/overlays/gcp-roce/patch-dgd.yaml` |
| `trtllm/disagg/blackwell/deploy-nscale-ib.yaml` | Nscale InfiniBand | `trtllm/disagg/blackwell/kustomize/overlays/nscale-ib/patch-dgd.yaml` |

After editing the base or overlays, render the apply-able manifests from the repo root:

```bash
python3 scripts/render_recipe_kustomize.py
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
