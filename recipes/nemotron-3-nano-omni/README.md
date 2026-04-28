<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Nemotron 3 Nano Omni NVFP4

Serves [nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-NVFP4](https://huggingface.co/nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-NVFP4)
using vLLM with an aggregated Dynamo deployment.

This recipe builds a custom container that layers the `ai-dynamo` wheel
(from <https://pypi.nvidia.com/ai-dynamo/>) onto an upstream vLLM image — no
source build, no Rust toolchain.

## Topology

| Role | Replicas | GPUs/replica | Notes |
|------|----------|--------------|-------|
| Frontend | 1 | 0 | Dynamo frontend with prefix-hash KV routing |
| vLLM worker | 1 | 1 | Text, image, video, and audio inputs |

## Prerequisites

- A Kubernetes cluster with the [Dynamo Operator](../../docs/kubernetes/README.md) installed
- One NVIDIA GPU per worker replica
- Shared PVC storage for the Hugging Face model cache
- Hugging Face access to `nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-NVFP4`

## Step 1: Build the Container

```bash
docker build \
  -t <your-registry>/nemotron-omni-vllm:latest \
  -f recipes/nemotron-3-nano-omni/Dockerfile \
  recipes/nemotron-3-nano-omni
docker push <your-registry>/nemotron-omni-vllm:latest
```

Useful build args:

- `BASE_IMAGE=<image>` — pin to a different vLLM base (default `vllm/vllm-openai:v0.20.0`).
- `DYNAMO_VERSION=<version>` — pin to a specific `ai-dynamo` release or nightly from <https://pypi.nvidia.com/ai-dynamo/>. Default tracks the latest tested nightly. Make sure the chosen wheel's `vllm` dependency matches `BASE_IMAGE`.

## Step 2: Download the Model

Create the PVC, Hugging Face token secret, and download the model weights:

```bash
export NAMESPACE=<your-namespace>

# Create the namespace if it does not already exist.
kubectl create namespace ${NAMESPACE} --dry-run=client -o yaml | kubectl apply -f -

# First edit storageClassName in model-cache.yaml for your cluster.
kubectl apply -f recipes/nemotron-3-nano-omni/model-cache/model-cache.yaml -n ${NAMESPACE}

kubectl create secret generic hf-token-secret \
  --from-literal=HF_TOKEN=<your-hf-token> \
  -n ${NAMESPACE}

kubectl apply -f recipes/nemotron-3-nano-omni/model-cache/model-download.yaml -n ${NAMESPACE}
kubectl wait --for=condition=complete job/model-download -n ${NAMESPACE} --timeout=3600s
```

## Step 3: Deploy

Edit `vllm/agg/deploy.yaml` and replace all `<placeholder>` values:

- `<your-registry>/nemotron-omni-vllm:latest` - your built container image

If your registry is private, add the appropriate `imagePullSecrets` to the
deployment.

```bash
kubectl apply -f recipes/nemotron-3-nano-omni/vllm/agg/deploy.yaml -n ${NAMESPACE}
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
    "model": "nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-NVFP4",
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 128
  }'
```

To exercise the multimodal path, attach an image:

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-NVFP4",
    "messages": [{
      "role": "user",
      "content": [
        {"type": "image_url", "image_url": {"url": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/inpaint.png"}},
        {"type": "text", "text": "Describe what is in this image."}
      ]
    }],
    "max_tokens": 256
  }'
```

…or an audio clip:

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-NVFP4",
    "messages": [{
      "role": "user",
      "content": [
        {"type": "audio_url", "audio_url": {"url": "https://raw.githubusercontent.com/yuekaizhang/Triton-ASR-Client/main/datasets/mini_en/wav/1221-135766-0002.wav"}},
        {"type": "text", "text": "Transcribe this audio clip."}
      ]
    }],
    "max_tokens": 256
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

## Optional: Run without NATS

The Dynamo runtime defaults to NATS for the event plane and connects to a
NATS server if `NATS_SERVER` is set in the environment (the operator
auto-injects this on most clusters). On clusters without NATS — or where
you'd rather avoid the dependency — you can run on TCP request plane + ZMQ
event plane only. Add to both Frontend and VllmWorker:

```yaml
mainContainer:
  env:
    - name: DYN_EVENT_PLANE
      value: zmq
  command: ["/bin/bash", "-lc"]
  args:
    # Operator-injected NATS_SERVER takes effect even when set to ""; we have
    # to actually unset it before the runtime reads env.
    - >-
      unset NATS_SERVER &&
      exec python3 -m dynamo.frontend ...   # or dynamo.vllm
```

The request plane defaults to TCP already, so no further flags are needed.

## File Layout

```text
recipes/nemotron-3-nano-omni/
  README.md
  Dockerfile
  model-cache/
    model-cache.yaml
    model-download.yaml
  vllm/
    agg/
      deploy.yaml
```
