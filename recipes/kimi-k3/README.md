<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Kimi-K3 Recipes

Recipes for [Kimi-K3](https://huggingface.co/moonshotai/Kimi-K3) on Dynamo + vLLM, targeting GB300 and GB200.

K3 is a hybrid-attention MoE model: 93 layers of Kimi Delta Attention (KDA, linear attention with a
short convolution state) with full MLA attention on 24 of them (every fourth layer), 896 routed
experts (16 active per token) plus 2 shared experts, a vision tower for image input, and 1,048,576
max positions. Routed-expert weights are MXFP4-packed (`compressed-tensors`, group size 32);
attention, shared experts, `lm_head`, and the vision tower stay BF16.

Every profile spans multiple 4-GPU nodes over MNNVL — **TP8 across two nodes on GB300, TP16 across four on GB200** — so the tensor-parallel group crosses the node boundary on NVLink, which is why every worker pod set claims a DRA `ComputeDomain` channel.

## Configurations

Dynamo + vLLM deployment profiles for the GB300 and GB200 agentic workload:

|                          | [GB300 aggregated agentic](vllm/agg-gb300-agentic/deploy.yaml) | [GB300 disaggregated agentic](vllm/disagg-gb300-agentic/deploy.yaml) | [GB200 aggregated agentic](vllm/agg-gb200-agentic/deploy.yaml) | [GB200 disaggregated agentic](vllm/disagg-gb200-agentic/deploy.yaml) |
| ------------------------ | ------------------------------------------------- | ------------------------------------------------------------- | ------------------------------------------------- | ------------------------------------------------------------- |
| **GPU** (per worker)     | 8x GB300 (2 nodes x 4)                            | 8x GB300 prefill + 8x GB300 decode (2 nodes x 4 each)          | 16x GB200 (4 nodes x 4)                           | 16x GB200 prefill + 16x GB200 decode (4 nodes x 4 each)       |
| **Replicas**             | 2 decode workers — 16 GPUs / 4 nodes              | 1P2D — 24 GPUs / 6 nodes                                       | 1 decode worker — 16 GPUs / 4 nodes               | 1P1D — 32 GPUs / 8 nodes                                      |
| **Mode**                 | Aggregated                                        | Prefill/decode disaggregated                                   | Aggregated                                        | Prefill/decode disaggregated                                   |
| **Framework**            | vLLM                                              | vLLM                                                           | vLLM                                              | vLLM                                                           |
| **Precision**            | MXFP4 experts + BF16 dense, FP8 KV                | MXFP4 experts + BF16 dense, FP8 KV                             | MXFP4 experts + BF16 dense, FP8 KV               | MXFP4 experts + BF16 dense, FP8 KV                            |
| **Parallelism**          | TP8 over MNNVL                                    | TP8 over MNNVL on both roles                                   | TP16 over MNNVL                                   | TP16 over MNNVL on both roles                                  |
| **Attention backend**    | `FLASHINFER_MLA`, `TRTLLM_RAGGED` MLA prefill with prefill query quantization | Same as aggregated, on both roles                   | `FLASHINFER_MLA`, `TRTLLM_RAGGED` MLA prefill with prefill query quantization | `FLASHINFER_MLA`; `TRTLLM_RAGGED` on prefill, `FLASHINFER` on decode |
| **MoE backend**          | trtllm-gen cubins + K3 latent-MoE tail fusion (CuTeDSL) | trtllm-gen cubins; tail fusion on decode only             | `flashinfer_trtllm`                               | `flashinfer_trtllm`                                            |
| **AllReduce backend**    | FlashInfer MNNVL                                  | FlashInfer MNNVL                                               | FlashInfer MNNVL                                  | NCCL (MNNVL + NVLS)                                            |
| **CUDA graphs**          | `FULL_AND_PIECEWISE`, capture up to 8192          | `FULL_DECODE_ONLY`, capture up to 2048 on decode; prefill runs eager | `FULL_AND_PIECEWISE`, capture up to 8192     | `FULL_AND_PIECEWISE`, capture up to 2048 on decode             |
| **Scheduler**            | vLLM `AsyncScheduler`                             | vLLM `AsyncScheduler`                                          | vLLM `AsyncScheduler`                             | vLLM `AsyncScheduler`                                          |
| **Batching**             | 8192 batched tokens / 256 seqs                    | 32768 tokens / 4 seqs prefill; 2048 tokens / 256 seqs decode   | 16384 batched tokens / 256 seqs                   | 16384 tokens / 4 seqs prefill; 2048 tokens / 256 seqs decode   |
| **GPU memory util**      | 0.85                                              | 0.90                                                           | 0.92                                              | 0.90                                                           |
| **Routing**              | KV-aware                                          | KV-aware                                                       | KV-aware                                          | KV-aware                                                       |
| **Prefix caching**       | Enabled                                           | Enabled on both roles                                          | Enabled                                           | Enabled on both roles                                          |
| **KV transfer**          | N/A                                               | NIXL (`NixlConnector`, `kv_both`) over MNNVL; UCX left at defaults | N/A                                          | NIXL (`NixlConnector`, `kv_both`) over RDMA; `UCX_TLS=^cuda_ipc` |
| **Context length**       | 1,048,576 (explicit `--max-model-len`)            | 1,048,576 (model default)                                      | 1,048,576 (model default)                         | 1,048,576 (model default)                                      |


## Supported features

- Modalities: Text and image (`--enable-multimodal`; K3 ships a vision tower)
- Reasoning (`kimi_k3` reasoning parser, engine and frontend side)
- Tool calling (`kimi_k3` tool-call parser)
- KV-aware routing and prefix caching (both profiles)
- Disaggregated serving

## Prerequisites

1. **Dynamo Platform installed** — see [Kubernetes Deployment Guide](../../docs/fern/pages/kubernetes/getting-started/quickstart.mdx).
2. **DRA / ComputeDomain controller** for the cross-node NVLink channel:
   ```bash
   kubectl get crd | grep computedomain
   ```
   Each manifest creates its own `ComputeDomain` and the workers claim its channel. Names vary by profile — check the `metadata.name` fields in each `deploy.yaml`.
3. **Hugging Face token** with access to `moonshotai/Kimi-K3`. The workers read the weights from the
   `model-cache` PVC — see [Download the model](#3-download-the-model).

## Cluster assumptions

These manifests are written against specific cluster shapes per SKU, with a ReadWriteMany storage class. Every value below is expected to differ elsewhere. Nothing outside this section is cluster-specific.

| Assumption | Where | Shipped value |
| ---------- | ----- | ------------- |
| GPU nodes carry a product label | `nodeAffinity` on every worker | `NVIDIA-GB300` (GB300) / `NVIDIA-GB200` (GB200) |
| Pool taint | `tolerations` | none (GB300) / `kubernetes.io/arch=arm64:NoSchedule` (GB200) |
| GPUs per node × nodes per TP group | `resources` and `multinode.nodeCount` | 4 GPUs × 2 nodes = TP8 (GB300) / 4 GPUs × 4 nodes = TP16 (GB200) |
| A ReadWriteMany storage class exists for the weights | `model-cache/model-cache.yaml` | `your-storage-class-name` placeholder |

### Reading the values off your cluster

```bash
# Which product label identifies the GPUs?
kubectl get nodes -o custom-columns='NODE:.metadata.name,PRODUCT:.metadata.labels.nvidia\.com/gpu\.product'

# Set this to whatever the command above reports, then reuse it below.
export GPU_PRODUCT=NVIDIA-GB300   # NVIDIA-GB200 on a GB200 cluster

# Allocatable GPUs per node, then any taints as key=value:effect.
kubectl get nodes -l nvidia.com/gpu.product=${GPU_PRODUCT} \
  -o go-template='{{range .items}}{{.metadata.name}}{{"\t"}}{{index .status.allocatable "nvidia.com/gpu"}}{{"\t"}}{{range .spec.taints}}{{.key}}={{.value}}:{{.effect}} {{end}}{{"\n"}}{{end}}'

# Available storage classes — pick one whose provisioner supports ReadWriteMany
# (a network filesystem, not block storage).
kubectl get storageclass
```

If the product label differs from the shipped value, update the `nodeAffinity` `values` list in each
`deploy.yaml`. If the first command reports no GPU nodes at all, the recipe has nothing to land on.

### Tainted GPU pools

The GB300 manifests ship with no tolerations, which assumes an unreserved GPU pool. The GB200 manifests include a `kubernetes.io/arch=arm64:NoSchedule` toleration for the arm64 node pool. If your GPU pool carries additional taints, add matching tolerations to **every** pod template in the `deploy.yaml` you are using.

```yaml
# In each podTemplate.spec
tolerations:
  - key: dedicated          # from the key=value:effect output above
    operator: Equal
    value: your-pool-name
    effect: NoSchedule
```

To cover several pools at once, use `operator: Exists` with the key and drop `value`. Keep
`effect: NoSchedule`: a bare `operator: Exists` with no key also tolerates `NoExecute` and
`node.kubernetes.io/unschedulable`, which would schedule pods onto cordoned nodes and ignore
eviction taints.

> [!NOTE]
> A toleration only permits placement, it does not attract it — node selection stays with the
> `nodeAffinity` above. Adding one widens the candidate set to include reserved pools, so check
> that the pool is yours to consume.

## Quick Start

### 1. Create namespace and secret

```bash
export NAMESPACE=your-namespace
kubectl create namespace ${NAMESPACE}
kubectl create secret generic hf-token-secret \
  --from-literal=HF_TOKEN="your-token" \
  -n ${NAMESPACE}
```

### 2. Create storage

> [!NOTE]
> Edit `model-cache/model-cache.yaml` and set `storageClassName` to a ReadWriteMany storage class
> available on the target cluster — `kubectl get storageclass` lists the candidates. The claim
> requests 3000Gi; the checkpoint is ~1.5 TB.

```bash
kubectl apply -f model-cache/model-cache.yaml -n ${NAMESPACE}
```

### 3. Download the model

```bash
kubectl apply -f model-cache/model-download.yaml -n ${NAMESPACE}
kubectl wait --for=condition=Complete job/model-download -n ${NAMESPACE} --timeout=14400s
```

The Job sets `HF_HOME=/model-cache`, so the checkpoint lands in the PVC's Hugging Face cache.

**This flow applies to the GB300 profiles.** Their worker pods mount the PVC at `/model-cache` with
`HF_HOME=/model-cache` and pass the repo id (`moonshotai/Kimi-K3`) to `--model`, so the weights
resolve straight out of the cache. The frontend does not mount the PVC — see
[Configuration notes](#configuration-notes).

**The GB200 profiles do not use the PVC.** They mount the host path
`/mnt/stateful_partition/kube-ephemeral-ssd/models` at `/models` and serve
`--model /models/model_weight` with `HF_HUB_OFFLINE=1`, so steps 2-3 do not make them deployable.
For GB200, either pre-stage the checkpoint at `<host-path>/model_weight` on every node in the
deployment and adjust the `models` `hostPath` to match your cluster, or convert the manifest to the
PVC flow: replace the `models` `hostPath` volume with `claimName: model-cache` mounted at
`/model-cache`, add `HF_HOME=/model-cache`, and set `MODEL_PATH` to `moonshotai/Kimi-K3`.

> [!NOTE]
> The containers run as `runAsUser: 0` because the cached weight files are root-owned while the
> image defaults to uid 1000.

> [!WARNING]
> Serving ~1.5 TB of weights off a shared filesystem is the slow path. Every worker pod pulls the
> full checkpoint over the PVC on each cold start — 4 pods for the aggregated profile, 6 for the
> disaggregated one — so first-load time is bounded by the storage backend, not by the GPUs. The
> startup probes budget 120 minutes per worker (`failureThreshold: 720`); raise it if your storage
> class is slower. Keeping the pods on nodes that already hold a warm page cache, or staging the
> checkpoint on node-local NVMe and mounting it as a `hostPath` instead, both cut this substantially.

### 4. Deploy the DGD

```bash
SKU=gb300 # or gb200
MODE=agg  # or disagg
kubectl apply -f vllm/${MODE}-${SKU}-agentic/deploy.yaml -n ${NAMESPACE}

DGD=kimi-k3-${MODE}
kubectl wait --for=condition=Ready pod \
  -l nvidia.com/dynamo-graph-deployment-name=${DGD} \
  -n ${NAMESPACE} --timeout=7200s
```

First launch loads ~1.5 TB of weights across the TP ranks and warms CuTeDSL kernels and CUDA graphs.
The startup probes allow a 120-minute budget per worker; the multinode pod set only reports ready
once all nodes have joined the TP group.

### 5. Smoke test

```bash
kubectl port-forward svc/${DGD}-frontend 8000:8000 -n ${NAMESPACE} &
```

#### Text

```bash
curl -s http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "moonshotai/Kimi-K3",
    "messages": [{"role": "user", "content": "Hello, who are you?"}],
    "max_tokens": 128
  }'
```

Chain-of-thought lands in `choices[0].message.reasoning_content` and the answer in
`choices[0].message.content`. If `reasoning_content` is `null` and raw `</think>` markers show up in
`content`, the parsers are not wired — check `--reasoning-parser kimi_k3` and
`--dyn-reasoning-parser kimi_k3` on the worker command.

#### Image

Images are sent as `image_url` content parts. Use a public HTTP(S) URL the deployment can fetch, or
a base64 `data:` URI on air-gapped clusters.

```bash
curl -s http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "moonshotai/Kimi-K3",
    "messages": [{
      "role": "user",
      "content": [
        {"type": "image_url", "image_url": {"url": "https://huggingface.co/datasets/Xenova/transformers.js-docs/resolve/main/cats.jpg"}},
        {"type": "text", "text": "Describe what is in this image."}
      ]
    }],
    "max_tokens": 512
  }'
```

#### Tool calling

```bash
curl -s http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "moonshotai/Kimi-K3",
    "messages": [{"role": "user", "content": "What is the weather in San Francisco?"}],
    "tools": [{
      "type": "function",
      "function": {
        "name": "get_weather",
        "description": "Get the current weather for a location",
        "parameters": {
          "type": "object",
          "properties": {"location": {"type": "string", "description": "City name"}},
          "required": ["location"]
        }
      }
    }],
    "max_tokens": 300
  }' | python3 -m json.tool
```

Expected: `choices[0].message.tool_calls[0].function.name` is `get_weather` with JSON arguments
naming San Francisco, and `finish_reason` is `tool_calls`.

## Configuration notes

Non-obvious knobs, all already set in the manifests:

- **Model resolution.** Each worker pod mounts the `model-cache` PVC at `/model-cache` with
  `HF_HOME=/model-cache`, and the workers pass the repo id (`MODEL_ID=moonshotai/Kimi-K3`) to
  `--model`, so vLLM loads the checkpoint out of the PVC's Hugging Face cache instead of
  downloading it. `envFrom: hf-token-secret` on the workers covers the hub lookup at startup.
- **MNNVL all-reduce.** The aggregated profiles (GB300 and GB200) and the GB300 disaggregated profile use `VLLM_ALLREDUCE_USE_FLASHINFER=1` with `VLLM_FLASHINFER_ALLREDUCE_BACKEND=mnnvl`. Do not enable the NCCL symmetric-memory knobs alongside it — `VLLM_USE_NCCL_SYMM_MEM=0` is required because symmetric memory breaks CUDA-graph capture on this build. The GB200 disaggregated profile uses NCCL directly (MNNVL + NVLS) for all-reduce.
- **Pod networking.** DGD pods use CNI networking, so `NCCL_SOCKET_IFNAME` and `GLOO_SOCKET_IFNAME`
  are pinned to `eth0`.
- **UCX.** The GB300 disaggregated profile leaves UCX at its defaults for the NIXL transport. The GB200 disaggregated profile sets `UCX_TLS=^cuda_ipc` to disable NVLink IPC and route NIXL KV transfer over RDMA IB (allocated via `networking.gke.io.networks/rdma-*` resources).
- **KDA state transfer.** The disaggregated profile sets `VLLM_SSM_CONV_STATE_LAYOUT=DS` — the
  Mamba-style conv state needs the DS layout to move across NIXL.
- **Asymmetric worker config (disaggregated).** On GB300, prefill runs `--enforce-eager` (no CUDA graphs) while decode captures graphs up to 2048. On GB200, both prefill and decode use `FULL_AND_PIECEWISE` graphs (up to 2048 on decode), with separate attention configs: `TRTLLM_RAGGED` on prefill and `FLASHINFER` on decode.
- **Scheduler override.** Both profiles force
  `--scheduler-cls vllm.v1.core.sched.async_scheduler.AsyncScheduler`. See
  [Known issues](#known-issues).
- **Cubins.** `FLASHINFER_PRIVATE_CUBIN_DIR` points at the trtllm-gen MoE cubins bundled in the
  image, with `CUDA_HOME=/usr/local/cuda` for JIT.

## Known issues

1. `dynamo.vllm`'s `InstrumentedScheduler.schedule(self)` is stale against this vLLM build's
   `schedule(self, throttle_prefills=False)` and raises a `TypeError` on the first request. Both
   profiles work around it by forcing vLLM's native `AsyncScheduler` via `--scheduler-cls`, which
   gives up the Dynamo scheduler instrumentation. Drop the override once the image carries a fix.
2. Every worker replica loads its own copy of the ~1.5 TB checkpoint from the PVC on every cold
   start — 4 worker pods for the aggregated profile, 6 for the disaggregated one, with no shared or
   streamed loading between them. This makes pod restarts expensive, which matters most for
   benchmark sweeps that restart workers between concurrency points to reset prefix-cache state.
