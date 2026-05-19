<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# GLM-5.1-FP8 — Disaggregated Prefill/Decode on GB200

Serves [zai-org/GLM-5.1-FP8](https://huggingface.co/zai-org/GLM-5.1-FP8) using vLLM with
disaggregated prefill/decode and NIXL KV transfer via Dynamo on GB200 nodes.

GLM-5.1-FP8 is a 256-expert MoE model with 8 active experts per token, 78 layers, and a
202K-token context window, quantized in FP8 E4M3 (~704 GiB weights across 142 safetensors shards).
It supports chain-of-thought reasoning (exposed via `reasoning_content`) and OpenAI-compatible
tool calling.

## Topology

| Role    | Nodes | GPUs/node | Total GPUs | Parallelism |
|---------|-------|-----------|------------|-------------|
| Prefill | 2     | 8         | 16         | TP8         |
| Decode  | 2     | 8         | 16         | TP8         |

Each prefill and decode worker is a 2-pod multinode group (leader + worker) running TP8 across
8 GB200 GPUs. All four nodes must reside within the same MNNVL ComputeDomain (NVL36 or NVL72).

## Prerequisites

- 4 x 8xGB200 nodes in a shared MNNVL NVLink domain
- A Kubernetes cluster with the [Dynamo Operator](../../docs/kubernetes/README.md) installed
- A HuggingFace account with access to `zai-org/GLM-5.1-FP8`
- A shared ReadWriteMany PVC for model weights (800 GiB minimum)

## Step 1: Create Storage

Apply the PVC manifests and update `storageClassName` to match your cluster:

```bash
# Check available storage classes
kubectl get storageclass

# Edit storageClassName in model-cache.yaml, then apply
kubectl apply -f recipes/glm-5.1-fp8/model-cache/model-cache.yaml
```

## Step 2: Download the Model

Create the HuggingFace token secret and run the download job.
The model is ~760 GiB; expect 20-60 minutes depending on network bandwidth.

```bash
kubectl create secret generic hf-token-secret \
  --from-literal=HF_TOKEN=<your-hf-token> \
  -n <your-namespace>

kubectl apply -f recipes/glm-5.1-fp8/model-cache/model-download.yaml -n <your-namespace>
kubectl wait --for=condition=complete job/model-download \
  --timeout=7200s -n <your-namespace>
```

## Step 3: Deploy

Edit `vllm/disagg/deploy.yaml` and replace all `<placeholder>` values:

- `<your-namespace>` — your Kubernetes namespace

You may also need to adjust:

- `NCCL_IB_HCA` — set to your cluster's InfiniBand HCA device name
  (`ibdev2netdev` on a node, or check `UCX_TLS=ib` with `ucx_info -d`)
- `nvidia.com/gpu.product: NVIDIA-GB200` — verify the label matches your nodes
  (`kubectl get nodes --show-labels | grep gpu.product`)
- `storageClassName` in `model-cache/model-cache.yaml`

```bash
kubectl apply -f recipes/glm-5.1-fp8/vllm/disagg/chat-template.yaml -n <your-namespace>
kubectl apply -f recipes/glm-5.1-fp8/vllm/disagg/deploy.yaml -n <your-namespace>
```

Monitor startup. Prefill and decode each need to load 142 shards (~700 GiB) and
JIT-compile DeepGEMM/FlashInfer kernels; expect 15-40 minutes on a cold start.

```bash
kubectl get pods -n <your-namespace> -w
# Expect: 1 frontend + 2 prefill leaders + 2 prefill workers + 2 decode leaders + 2 decode workers
```

## Step 4: Test

```bash
kubectl port-forward svc/glm51-fp8-frontend 3000:3000 -n <your-namespace> &

# List models
curl http://localhost:3000/v1/models

# Basic chat
curl http://localhost:3000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "zai-org/GLM-5.1-FP8",
    "messages": [{"role": "user", "content": "What is 3 + 4?"}],
    "max_tokens": 512
  }'
```

Note: GLM-5.1 is a reasoning model. The thinking trace is returned in
`choices[0].message.reasoning_content`; the final answer appears in
`choices[0].message.content`. Use `max_tokens >= 512` to leave room for thinking.

## Step 5: Benchmark (optional)

Edit `vllm/disagg/perf.yaml` to set `<your-namespace>`, then run:

```bash
kubectl apply -f recipes/glm-5.1-fp8/vllm/disagg/perf.yaml -n <your-namespace>
kubectl logs -f -l job-name=glm51-fp8-disagg-bench -n <your-namespace>
```

Default benchmark: ISL=1024, OSL=512, concurrency=128 (8 req/GPU x 16 decode GPUs).
Adjust `CONCURRENCY` and `OSL` to match your workload profile.

## Key Configuration Notes

### ComputeDomain (MNNVL)

All four worker nodes (2 prefill + 2 decode) must be in the same NVLink clique.
The `ComputeDomain` resource pins every pod to that clique via the DRA `ResourceClaimTemplate`:

```yaml
apiVersion: resource.nvidia.com/v1beta1
kind: ComputeDomain
metadata:
  name: glm51-fp8-compute-domain
spec:
  type: MultiNodeNVLink
  numNodes: 4
```

Update `numNodes` only if your topology differs. The `ResourceClaimTemplate` named
`glm51-fp8-compute-domain-channel` is referenced by both prefill and decode workers.

### FP8 KV Cache

Decode workers use `--kv-cache-dtype fp8` to halve KV memory usage relative to BF16.
This is required at TP8 scale to fit 202K-context sequences without OOM.

### DeepGEMM

`VLLM_USE_DEEP_GEMM=1` enables fused FP8 GEMM kernels optimized for GB200.
`VLLM_DEEP_GEMM_WARMUP=skip` skips the full warmup sweep at startup (saves ~10 min)
at the cost of cold first-request latency on shapes not yet JIT-compiled.
Remove `VLLM_DEEP_GEMM_WARMUP=skip` once the deployment is stable and the
`compilation-cache` PVC is populated.

### Reasoning Parser

`--dyn-reasoning-parser glm45` extracts `<think>...</think>` tokens from the model
output and populates `message.reasoning_content` in the response. Always set
`max_tokens` large enough to accommodate reasoning tokens (typically 200-2000) plus
the final answer.

### Tool Calling

`--dyn-tool-call-parser glm47` parses GLM's native XML `<tool_call>` format and
converts it to the standard OpenAI `tool_calls` JSON schema.

### Multinode Startup (TCPStore)

Each 2-node worker group uses PyTorch `init_process_group` with a TCPStore rendezvous
on the leader pod (port 29500). The worker's init container waits for the leader to
be `Running` and TCP-reachable before launching.

If a leader pod restarts, delete the corresponding stale worker pod so it re-joins
the new TCPStore:

```bash
kubectl delete pod <worker-pod-name> -n <your-namespace>
```

### NIXL KV Transfer

Prefill and decode exchange KV blocks via NIXL (UCX backend). The topology exchange
is lazy — it triggers on the first KV transfer, not at pod startup. If both prefill
and decode pods restart simultaneously, send one request to prime the side-channel
before benchmarking.
