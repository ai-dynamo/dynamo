<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# GLM-5 NVFP4 — Disaggregated Prefill/Decode on GB200

Serves [nvidia/GLM-5-NVFP4](https://huggingface.co/nvidia/GLM-5-NVFP4) using SGLang with
disaggregated prefill/decode and EAGLE speculative decoding via Dynamo on GB200 NVL4 nodes.

## Topology

| Role    | Nodes | GPUs/node | Total GPUs | Parallelism        |
|---------|-------|-----------|------------|--------------------|
| Decode  | 4     | 4         | 16         | TP16 / DP16 / EP16 |
| Prefill | 1     | 4         | 4          | TP4                |

## Prerequisites

- GB200 NVL4 nodes with RDMA networking
- Dynamo operator installed
- dynamo-platform HelmRelease deployed (NATS only — etcd not required)
- Kubernetes service discovery enabled
- Shared NFS PVC for model weights

## Step 1: Build the Container

The container requires custom flashinfer binaries and patches to the dynamo sglang
package for compatibility with sglang 0.5.10. Build and push before deploying.

```bash
docker buildx build \
  --platform linux/arm64 \
  --build-arg ARCH=arm64 \
  -t <your-registry>/sglang-dynamo-glm5:latest \
  -f recipes/glm-5-nvfp4/Dockerfile \
  --push .
```

## Step 2: Download the Model

Create a HuggingFace token secret and download the model weights:

```bash
kubectl create secret generic hf-token-secret \
  --from-literal=HF_TOKEN=<your-hf-token> -n <your-namespace>

# Edit model-cache/model-download.yaml to set your namespace and PVC name
kubectl apply -f recipes/glm-5-nvfp4/model-cache/model-download.yaml
kubectl wait --for=condition=complete job/glm5-nvfp4-download -n <your-namespace> --timeout=3600s
```

## Step 3: Deploy

Edit `sglang/disagg/deploy.yaml` and replace all `<placeholder>` values:

- `<your-namespace>` — your Kubernetes namespace
- `<your-registry>/sglang-dynamo-glm5:latest` — your built container image
- `<your-model-pvc>` — PVC name containing model weights at `/models/nvidia-GLM-5-NVFP4`

Then apply to create the ComputeDomain and get its UID:

```bash
kubectl apply -f recipes/glm-5-nvfp4/sglang/disagg/deploy.yaml

# Get the auto-generated ComputeDomain UID
kubectl get computedomain glm5-compute-domain -n <your-namespace> \
  -o jsonpath='{.metadata.uid}'
```

Update `<compute-domain-uid>` in `deploy.yaml` with the UID from above, then re-apply:

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

### NIXL Disaggregation
KV cache is transferred from prefill to decode over RoCE RDMA using NIXL.
The GKE multi-network annotations inject `rdma0-3` interfaces (one per GPU).

### Discovery
Uses `--discovery-backend kubernetes` (no etcd required). Worker registration
is tied to pod lifetime via Kubernetes EndpointSlices, preventing TTL expiry
issues under high load.

## Performance (ISL=1k, OSL=8k, concurrency=512)

| Metric | Value |
|--------|-------|
| Output throughput | ~19,000 tokens/sec |
| TTFT p50 | ~850ms |
| ITL avg | ~24ms/token |
| Tokens/user/sec | ~41 |
