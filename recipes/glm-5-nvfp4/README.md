<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# GLM-5 NVFP4 Recipes

Deployment recipes for [nvidia/GLM-5-NVFP4](https://huggingface.co/nvidia/GLM-5-NVFP4)
using both SGLang and TensorRT-LLM.

## Available Configurations

| Configuration | Framework | GPUs | Mode | Description |
|--------------|-----------|------|------|-------------|
| [**sglang/disagg**](sglang/disagg/) | SGLang | 20x GB200 | Disaggregated | Prefill/decode split with EAGLE speculative decoding |
| [**trtllm/agg/deploy.yaml**](trtllm/agg/deploy.yaml) | TensorRT-LLM | 8x B200 | Aggregated | NVFP4 baseline |
| [**trtllm/agg/deploy-specdec.yaml**](trtllm/agg/deploy-specdec.yaml) | TensorRT-LLM | 8x B200 | Aggregated | NVFP4 with MTP speculative decoding |

## Prerequisites

- A Kubernetes cluster with the [Dynamo Operator](../../docs/kubernetes/README.md) installed
- Shared PVC for model weights
- HuggingFace token with access to `nvidia/GLM-5-NVFP4`
- Update `storageClassName` in `model-cache/model-cache.yaml` before deploying

## Model Download

Create the PVC, HuggingFace token secret, and download the model weights:

```bash
kubectl apply -f recipes/glm-5-nvfp4/model-cache/model-cache.yaml

kubectl create secret generic hf-token-secret \
  --from-literal=HF_TOKEN=<your-hf-token>

kubectl apply -f recipes/glm-5-nvfp4/model-cache/model-download.yaml
kubectl wait --for=condition=complete job/model-download --timeout=3600s
```

The TensorRT-LLM recipes use the pinned model revision:

- `nvidia/GLM-5-NVFP4@dc54ff55a7e9e71b85db953d8bc22eca894b44c6`

## SGLang Disaggregated Recipe

Serves `nvidia/GLM-5-NVFP4` using SGLang with disaggregated prefill/decode and
EAGLE speculative decoding via Dynamo on GB200 nodes.

## Topology

| Role    | Nodes | GPUs/node | Total GPUs | Parallelism        |
|---------|-------|-----------|------------|--------------------|
| Decode  | 4     | 4         | 16         | TP16 / DP16 / EP16 |
| Prefill | 1     | 4         | 4          | TP4                |

### Prerequisites

- 5 4xGB200 nodes in an NVL36 or NVL72 domain

## Step 1: Build the Container

This recipe currently requires using unreleased versions of SGLang and Dynamo.
Use the command below to build and push a container with the necessary dependencies.

```bash
docker buildx build \
  --platform linux/arm64 \
  --build-arg ARCH=arm64 \
  -t <your-registry>/sglang-dynamo-glm5:latest \
  -f recipes/glm-5-nvfp4/Dockerfile \
  --push .
```

## Step 2: Download the Model

Use the model download flow from the section above.

## Step 3: Deploy

Edit `sglang/disagg/deploy.yaml` and replace all `<placeholder>` values:

- `<your-namespace>` — your Kubernetes namespace
- `<your-registry>/sglang-dynamo-glm5:latest` — your built container image

```bash
kubectl apply -f recipes/glm-5-nvfp4/sglang/disagg/deploy.yaml
```

Monitor startup (decode takes ~15 min to load and capture CUDA graphs):

```bash
kubectl get pods -n <your-namespace> -l app.kubernetes.io/part-of=glm5-sglang -w
```

## Step 4: Test

```bash
kubectl port-forward svc/glm5-sglang-frontend 8000:8000 -n <your-namespace> &
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"nvidia/GLM-5-NVFP4","messages":[{"role":"user","content":"Hello!"}],"max_tokens":128}'
```

## Step 5: Benchmark (optional)

Edit `sglang/disagg/perf.yaml` to set your namespace and PVC, then run:

```bash
kubectl apply -f recipes/glm-5-nvfp4/sglang/disagg/perf.yaml
kubectl logs -f -l job-name=glm5-disagg-bench -n <your-namespace>
```

Default benchmark: ISL=1000, OSL=8192, concurrency=512 (32/GPU).

## Key Configuration Notes

### Speculative Decoding (EAGLE MTP)
Two env vars enable working speculative decoding (~85-95% accept rate):

- `SGLANG_ENABLE_SPEC_V2=1` — uses EAGLEWorkerV2 with overlap scheduler
- `SGLANG_NVFP4_CKPT_FP8_NEXTN_MOE=1` — quantizes BF16 MTP layer to FP8 at load
  time, matching the base model's compute path

The MTP layer weights in `nvidia/GLM-5-NVFP4` are BF16 (split across shards 271-274)
and are fully indexed in the checkpoint's `model.safetensors.index.json`.

### KV Cache
Uses `--kv-cache-dtype fp8_e4m3` (NSA backend auto-selects this on SM100/GB200).
Saves ~50% KV memory vs BF16.

### Discovery
Uses Kubernetes service discovery. Worker registration is tied to pod lifetime
via Kubernetes EndpointSlices, preventing TTL expiry issues under high load.

## Performance (ISL=1k, OSL=8k, concurrency=512)

| Metric | Value |
|--------|-------|
| Output throughput | ~19,000 tokens/sec |
| TTFT p50 | ~850ms |
| ITL avg | ~24ms/token |
| Tokens/user/sec | ~41 |

## TensorRT-LLM Aggregated Recipes

These recipes serve `nvidia/GLM-5-NVFP4` on a single 8x B200 worker pod using
TensorRT-LLM.

### TensorRT-LLM Deployment Options

- `trtllm/agg/deploy.yaml`: recommended NVFP4 baseline
- `trtllm/agg/deploy-specdec.yaml`: NVFP4 with MTP speculative decoding enabled

### TRTLLM Quick Start

```bash
# Deploy baseline
kubectl apply -f recipes/glm-5-nvfp4/trtllm/agg/deploy.yaml

# OR deploy the MTP specdec variant
# kubectl apply -f recipes/glm-5-nvfp4/trtllm/agg/deploy-specdec.yaml
```

### TRTLLM Test

```bash
# Baseline
kubectl port-forward svc/glm5-nvfp4-trtllm-agg-frontend 8000:8000 -n <your-namespace>

# MTP specdec variant
# kubectl port-forward svc/glm5-nvfp4-specdec-trtllm-agg-frontend 8000:8000 -n <your-namespace>

curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "nvidia/GLM-5-NVFP4",
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 50
  }'
```

### TRTLLM Notes

- Runtime image is controlled by `trtllm/agg/deploy*.yaml`; update the tag to your release.
- These manifests use the GLM custom tokenizer flow:
  - `--trtllm.custom_tokenizer glm_moe_dsa`
  - `--trtllm.tokenizer /opt/models/.../snapshots/dc54ff55a7e9e71b85db953d8bc22eca894b44c6`
