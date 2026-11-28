<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# DeepSeek-V4.1-Flash on DeepInfer

**Experimental.** Serves
[deepseek-ai/DeepSeek-V4.1-Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash) with
the [DeepInfer backend](../../../examples/backends/deepinfer/README.md) on GB200. Each worker
runs four DeepInfer data-parallel ranks with EP4, one per GPU, on one node.

| Manifest | Workers | GPUs | Routing |
|---|---|---|---|
| [`agg-gb200/deploy.yaml`](agg-gb200/deploy.yaml) | 1 aggregated (DP4, EP4) | 4 | KV router, prefix cache with KV events |
| [`disagg-gb200/deploy.yaml`](disagg-gb200/deploy.yaml) | 1 prefill + 1 decode (DP4, EP4 each) | 8 | KV router on prefill, KV and state read over NIXL |

The manifests target GKE GB200 node pools with RDMA networks. On other clusters, change the
node selector, the network annotation and the RDMA resources (see
[Pod Requirements](#pod-requirements)).

## Prerequisites

- Dynamo operator with the `nvidia.com/v1beta1` CRDs, and the NVIDIA DRA driver for
  `ComputeDomain`.
- A GB200 node pool whose nodes expose the RDMA networks `rdma-0` to `rdma-3`.
- The `shared-model-cache` ReadWriteMany PVC holding the checkpoint.
- An image built from [`examples/backends/deepinfer/Dockerfile`](../../../examples/backends/deepinfer/Dockerfile).
  Set it in every component, together with the `nvcr-imagepullsecret` pull secret or your own.

## Deploy

1. Download the weights, unless the PVC already has them. This is the same Job as the other
   DeepSeek-V4.1-Flash recipes, so the checkpoint can be shared:

   ```bash
   export NAMESPACE=<namespace>
   # Set storageClassName in model-cache.yaml first.
   kubectl apply -n ${NAMESPACE} -f recipes/deepseek-v4.1-flash/model-cache/model-cache.yaml
   kubectl apply -n ${NAMESPACE} -f recipes/deepseek-v4.1-flash/model-cache/model-download.yaml
   ```

2. Create the ConfigMap with the CPU-pinning shim:

   ```bash
   kubectl create configmap deepinfer-sitecustomize -n ${NAMESPACE} \
     --from-file=examples/backends/deepinfer/cpu_affinity/sitecustomize.py
   ```

3. Apply one manifest. Each also creates its `ComputeDomain`:

   ```bash
   kubectl apply -n ${NAMESPACE} -f recipes/deepseek-v4.1-flash/deepinfer/agg-gb200/deploy.yaml
   ```

4. Wait for the worker to become ready. Loading the weights from a Lustre PVC took about
   7 minutes per rank, and JIT compilation follows; the startup probe allows two hours. Then
   send a request:

   ```bash
   kubectl port-forward -n ${NAMESPACE} svc/dsv41-flash-deepinfer-gb200-agg-frontend 8000:8000 &
   curl -s localhost:8000/v1/chat/completions -H 'Content-Type: application/json' -d '{
     "model": "deepseek-ai/DeepSeek-V4.1-Flash",
     "messages": [{"role": "user", "content": "Hello"}],
     "max_tokens": 64
   }'
   ```

### Checkpoint

The workers read `DEEPINFER_HF_MODEL_DIR`, which points at the revision that `model-download.yaml`
pins (`dba1be0a`). DeepInfer's offline example runs on it. DeepInfer's reference checkpoint is
[DeepSeek-V4.1-Flash-1017](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017). It has the
same `config.json` and tokenizer, but all 48 weight shards differ. To serve it, download it
(another 510 GB on the PVC) and set `DEEPINFER_HF_MODEL_DIR` to
`/shared-model-cache/deepseek-ai/DeepSeek-V4.1-Flash-1017` in every worker:

```yaml
apiVersion: batch/v1
kind: Job
metadata:
  name: model-download-1017
spec:
  backoffLimit: 3
  template:
    spec:
      restartPolicy: Never
      containers:
      - name: model-download
        image: python:3.10-slim
        command: ["sh", "-c"]
        args:
        - |
          set -eux
          pip install --no-cache-dir huggingface_hub==1.16.4
          hf download deepseek-ai/DeepSeek-V4.1-Flash-1017 --max-workers 16 \
            --local-dir /shared-model-cache/deepseek-ai/DeepSeek-V4.1-Flash-1017
        envFrom:
        - secretRef:
            name: hf-token-secret
        env:
        # The hf_xet writer fails on Lustre; plain HTTP downloads work.
        - name: HF_HUB_DISABLE_XET
          value: "1"
        volumeMounts:
        - name: shared-model-cache
          mountPath: /shared-model-cache
      volumes:
      - name: shared-model-cache
        persistentVolumeClaim:
          claimName: shared-model-cache
```

## Pod Requirements

DeepInfer assumes a whole node with its RDMA NICs. Each worker pod needs the following; the
last column is what happens without it.

| Requirement | Why | Without it |
|---|---|---|
| `ComputeDomain` claim (IMEX channel) | DeepEP allocates FABRIC memory for the Engram tables | `cuMemCreate` fails with `CUDA_ERROR_OUT_OF_MEMORY` |
| The four RDMA networks: the `networking.gke.io/interfaces` annotation and the `rdma-N` and `rdma-N.IP` resources | DeepEP fetches the host-memory Engram tables over NCCL GIN | `CUDA_ERROR_MISALIGNED_ADDRESS` in `EngramBuffer.fetch`. `EP_DISABLE_GIN=1` with `DEEPINFER_ENGRAM_FORCE_GPU_STORAGE=1` runs out of GPU memory at EP4 |
| `NCCL_CROSS_NIC=0` | GKE's `rdma-0` to `rdma-3` are isolated rails (a multi-plane network); each GIN communicator must stay on its own rail | `IBV_WC_RETRY_EXC_ERR` during GIN setup. Pinning every rank to one NIC (`NCCL_IB_HCA==mlx5_0`) works but costs ~30% prefill throughput |
| `NCCL_SOCKET_IFNAME=eth0`, `GLOO_SOCKET_IFNAME=eth0` | Keeps bootstrap traffic on the pod network | NCCL or Gloo may bootstrap over an RDMA interface |
| `sharedMemorySize: 300Gi` | DeepInfer's shared-memory buffers; a crashed run leaves segments behind, and a container restart keeps them | `No space left on device` in `/dev/shm`, seen at 64Gi |
| `IPC_LOCK`, `SYS_RESOURCE`, `SYS_PTRACE`, user 0 | RDMA memory registration, CUDA IPC between ranks, and DeepEP hybrid mode sharing Engram storage via `pidfd_getfd` (GKE nodes run yama `ptrace_scope=1`) | Registration, IPC or Engram-sharing failures |
| `sitecustomize.py` first on `PYTHONPATH`, `DEEPINFER_LOCAL_NGPUS=4` | DeepInfer pins each rank to a slice of all host cores | `CPU number N is not eligible` when the pod's CPU set is smaller |
| `DEEPINFER_JIT_CACHE_DIR` on a writable volume | JIT-compiled kernels | Kernels recompile on every start |

The worker command checks for the checkpoint and for `/dev/infiniband/uverbs*` before it starts.
`/sys/class/infiniband` is not a usable check: it lists the host's NICs even when none are
attached. DeepInfer aborts a checkpoint load that takes more than 30 minutes, so the model storage
must read about 510 GB within that time.

## Disaggregation

The prefill worker keeps the prefix cache. It holds each request's KV blocks and state until
the decode worker has read them over NIXL, or until the 120-second lease expires. In
[`disagg-gb200/deploy.yaml`](disagg-gb200/deploy.yaml):

- KV moves over UCX `rc` (RoCE). MNNVL `cuda_ipc` cannot carry DeepInfer's `cuMemAlloc` KV pools
  between nodes, and UCX drops it without an error.
- `UCX_NET_DEVICES=mlx5_0:1` puts every rank on rail 0. Any prefill rank can hand off to any
  decode rank, and the GKE rails are isolated.
- Both workers set `DEEPINFER_YOCO_PREFILL_TRUNCATION_MODE=compact` and
  `DEEPINFER_YOCO_PREFILL_KEEP_TOKENS=140`. These values are part of the KV layout hash, and a
  mismatch fails every handoff with `KV layout mismatch`.
- Leave conditional disaggregation (`DYN_ROUTER_CONDITIONAL_DISAGG`) off. A decode worker rejects
  requests that have not been prefilled.

## Session Affinity

For agentic traffic, keep a session on one worker and DP rank, where its prefix is cached. The
router's binding includes the DP rank, and the worker honors it. Add to the frontend:

```yaml
          args:
          - -m
          - dynamo.frontend
          - --http-port
          - "8000"
          - --router-mode
          - kv
          - --router-session-affinity-ttl-secs
          - "14400"
          env:
          - name: DYN_ROUTER_SESSION_AFFINITY_MODE
            value: soft
```

Clients send the session in the `X-Dynamo-Session-ID` header. With `soft`, worker selection
treats the binding as a preference; the default, `hard`, dispatches to the bound worker exactly.
See
[Session Affinity](https://github.com/ai-dynamo/dynamo/blob/main/docs/fern/pages/developer-guide/knowledge-base/modular-components/router/configuration-and-tuning.md#session-affinity).
