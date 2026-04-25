<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Nemotron Nano Omni NVFP4 - Aggregated vLLM

Serves [nvidia/Nemotron-Nano-V3-Omni-GA0420-NVFP4](https://huggingface.co/nvidia/Nemotron-Nano-V3-Omni-GA0420-NVFP4)
using vLLM with an aggregated Dynamo deployment.

This recipe requires a custom container because it layers Dynamo from this
source tree onto a vLLM nightly base image.

## Topology

| Role | Replicas | GPUs/replica | Notes |
|------|----------|--------------|-------|
| Frontend | 1 | 0 | Dynamo frontend with prefix-hash KV routing |
| vLLM worker | 1 | 1 | Text, image, video, and audio inputs |

## Prerequisites

- A Kubernetes cluster with the [Dynamo Operator](../../docs/kubernetes/README.md) installed
- One NVIDIA GPU per worker replica
- Access to the vLLM nightly base image configured in `Dockerfile`
- Shared PVC storage for the Hugging Face model cache
- Hugging Face access to `nvidia/Nemotron-Nano-V3-Omni-GA0420-NVFP4`

## Step 1: Build the Container

Use the command below to build a container with the necessary dependencies:

```bash
docker build \
  -t <your-registry>/nemotron-omni-vllm:latest \
  -f recipes/nemotron-omni/Dockerfile .
docker push <your-registry>/nemotron-omni-vllm:latest
```

Set `BASE_IMAGE=<image>` if you need to build from a different compatible
vLLM base image.

## Step 2: Download the Model

Create the PVC, Hugging Face token secret, and download the model weights:

```bash
export NAMESPACE=<your-namespace>

# Create the namespace if it does not already exist.
kubectl create namespace ${NAMESPACE} --dry-run=client -o yaml | kubectl apply -f -

# First edit storageClassName in model-cache.yaml for your cluster.
kubectl apply -f recipes/nemotron-omni/model-cache/model-cache.yaml -n ${NAMESPACE}

kubectl create secret generic hf-token-secret \
  --from-literal=HF_TOKEN=<your-hf-token> \
  -n ${NAMESPACE}

kubectl apply -f recipes/nemotron-omni/model-cache/model-download.yaml -n ${NAMESPACE}
kubectl wait --for=condition=complete job/model-download -n ${NAMESPACE} --timeout=3600s
```

## Step 3: Deploy

Edit `vllm/agg/deploy.yaml` and replace all `<placeholder>` values:

- `<your-registry>/nemotron-omni-vllm:latest` - your built container image

If your registry is private, add the appropriate `imagePullSecrets` to the
deployment.

```bash
kubectl apply -f recipes/nemotron-omni/vllm/agg/deploy.yaml -n ${NAMESPACE}
```

Monitor startup:

```bash
kubectl get pods -n ${NAMESPACE} -l nvidia.com/dynamo-graph-deployment-name=nemotron-omni-vllm-agg -w
```

## Step 4: Test

```bash
kubectl port-forward svc/nemotron-omni-vllm-agg-frontend 8000:8000 -n ${NAMESPACE}
```

In another terminal, send a minimal text request:

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "nvidia/Nemotron-Nano-V3-Omni-GA0420-NVFP4",
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 128
  }'
```

## Key Configuration Notes

- `--enable-multimodal` enables image, video, and audio inputs.
- `--media-io-kwargs '{"video": {"num_frames": 512, "fps": 1}}'` samples long
  videos at one frame per second, capped at 512 frames.
- `--dyn-tool-call-parser nemotron_nano` and
  `--dyn-reasoning-parser nemotron_nano` enable Nemotron Nano tool-call and
  reasoning parsing.
- The frontend uses `--router-mode kv --no-kv-events`, which approximates
  KV-aware routing with prefix hashing without requiring backend KV events.

## File Layout

```text
recipes/nemotron-omni/
  README.md
  Dockerfile
  model-cache/
    model-cache.yaml
    model-download.yaml
  vllm/
    agg/
      deploy.yaml
```
