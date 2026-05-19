<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# GLM-5 NVFP4 — Disaggregated Prefill/Decode on GB200

Serves [nvidia/GLM-5-NVFP4](https://huggingface.co/nvidia/GLM-5-NVFP4) using SGLang with
disaggregated prefill/decode and EAGLE speculative decoding via Dynamo on GB200 nodes.

## Topology

| Role    | Nodes | GPUs/node | Total GPUs | Parallelism        |
|---------|-------|-----------|------------|--------------------|
| Decode  | 4     | 4         | 16         | TP16 / DP16 / EP16 |
| Prefill | 1     | 4         | 4          | TP4                |

## Prerequisites

- 5 4xGB200 nodes in an NVL36 or NVL72 domain
- A Kubernetes cluster with the [Dynamo Operator](../../docs/kubernetes/README.md) installed
- DRA / ComputeDomain support for MNNVL placement
- Access to `nvcr.io/nvidia/ai-dynamo/sglang-runtime:1.1.1-cuda13`
- Shared RWX PVC for model weights and FlashInfer JIT artifacts

The manifest uses standard NVIDIA GPU Feature Discovery labels to select GB200
nodes and includes common GPU/ARM tolerations. If your cluster uses different
labels, taints, or storage classes, update `nodeSelector`, `tolerations`, and
`storageClassName` before deploying.

## Step 1: Use the Published Runtime Image

This recipe uses the stable Dynamo SGLang runtime image for GB200:
`nvcr.io/nvidia/ai-dynamo/sglang-runtime:1.1.1-cuda13`.
The worker containers run as root because FlashInfer's bundled cubin package creates
TRTLLM MoE symlinks inside its installed package directory during startup.

## Step 2: Download the Model

Create the PVC, HuggingFace token secret, and download the model weights:

```bash
export NAMESPACE=<your-namespace>
kubectl create namespace ${NAMESPACE}

# Edit model-cache.yaml first and set storageClassName to a RWX storage class.
kubectl apply -f recipes/glm-5-nvfp4/model-cache/model-cache.yaml -n ${NAMESPACE}

kubectl create secret generic hf-token-secret \
  --from-literal=HF_TOKEN=<your-hf-token> \
  -n ${NAMESPACE}

kubectl apply -f recipes/glm-5-nvfp4/model-cache/model-download.yaml -n ${NAMESPACE}
kubectl wait --for=condition=complete job/model-download -n ${NAMESPACE} --timeout=3600s
```

If your cluster already provides a shared RWX cache PVC, skip
`model-cache/model-cache.yaml` and update `claimName: model-cache` in
`model-cache/model-download.yaml`, `sglang/disagg/deploy.yaml`, and
`sglang/disagg/perf.yaml` to that PVC's name. Keep the mount path as
`/model-store` so the workers, model download job, and benchmark job share the
same HuggingFace cache and JIT artifacts.

## Step 3: Deploy

Edit `sglang/disagg/deploy.yaml` and replace the namespace placeholder:

- `<your-namespace>` — the value of `${NAMESPACE}`

```bash
kubectl apply -f recipes/glm-5-nvfp4/sglang/disagg/deploy.yaml
kubectl wait --for=condition=Ready pod \
  -l nvidia.com/dynamo-graph-deployment-name=glm5-sglang \
  -n ${NAMESPACE} --timeout=7200s
```

Monitor startup. First cold starts can take up to about an hour while the stable
runtime loads weights and JIT-compiles FlashInfer/DeepGEMM kernels:

```bash
kubectl get pods -n ${NAMESPACE} -l app.kubernetes.io/part-of=glm5-sglang -w
```

## Step 4: Test

```bash
kubectl port-forward svc/glm5-sglang-frontend 8000:8000 -n ${NAMESPACE} &
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"nvidia/GLM-5-NVFP4","messages":[{"role":"user","content":"Hello!"}],"max_tokens":128}'
```

## Step 5: Benchmark (optional)

Edit `sglang/disagg/perf.yaml` to replace the namespace placeholder with
`${NAMESPACE}`, then run:

```bash
kubectl apply -f recipes/glm-5-nvfp4/sglang/disagg/perf.yaml
kubectl logs -f -l job-name=glm5-disagg-bench -n ${NAMESPACE}
```

Default benchmark: ISL=1000, OSL=8192, concurrency=512 (32/GPU).

The benchmark pod runs as a non-root user and installs AIPerf into `/tmp/.local`.
It pins Transformers v5 because `nvidia/GLM-5-NVFP4` declares
`tokenizer_class=TokenizersBackend`, which older Transformers v4 releases do not
load.

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

### FlashInfer JIT Cache
The stable runtime includes FlashInfer but not a prebuilt `flashinfer-jit-cache`
wheel. The recipe sets `FLASHINFER_WORKSPACE_BASE=/model-store` so first-run JIT
artifacts are written to the shared model PVC and reused by later pod starts.

### NIXL / UCX
NIXL uses UCX for disaggregated KV transfer. The recipe sets
`UCX_TLS=cuda_copy,cuda_ipc,tcp` so CUDA IPC/copy can handle GPU memory movement
while TCP provides UCX active-message control traffic.

### Discovery
Uses Kubernetes service discovery. Worker registration is tied to pod lifetime
via Kubernetes EndpointSlices, preventing TTL expiry issues under high load.

### Recovery and Rollouts
The decode side is one TP16 rank group spread across four GB200 nodes. Treat
single decode-pod replacement as disruptive, and validate full-group recovery in
your cluster before relying on individual decode pod restarts for production
rollouts. In validation, cold deploy plus frontend and prefill replacement
recovered successfully, while deleting one decode worker restarted the decode
leader and the graph stayed NotReady during repeated rank-group reinitialization
attempts.

## Performance (ISL=1k, OSL=8k, concurrency=512)

Reference AIPerf run: 1,536 requests, 0 errors, 747.87s benchmark duration.
This is a concurrency-burst benchmark, so TTFT includes queueing under 512
concurrent users.

| Metric | Value |
|--------|-------|
| Output throughput | 16,824 tokens/sec |
| Request throughput | 2.05 requests/sec |
| TTFT p50 | 15,423ms |
| ITL avg | 23.31ms/token |
| Tokens/user/sec avg | 43.39 |
| Request errors | 0 |
