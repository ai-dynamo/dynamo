<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# DeepSeek-V4-Pro — Disaggregated Prefill/Decode on B200

Serves [deepseek-ai/DeepSeek-V4-Pro](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro) using SGLang with disaggregated prefill/decode via Dynamo on B200 nodes with InfiniBand RDMA.

## Topology

| Role    | Nodes | GPUs/node | Total GPUs | Parallelism |
|---------|-------|-----------|------------|-------------|
| Decode  | 1     | 8         | 8          | TP8         |
| Prefill | 1     | 8         | 8          | TP8         |

## Prerequisites

- 2× B200 nodes with InfiniBand and `rdma/ib` device plugin
- Dynamo operator + dynamo-platform HelmRelease installed
- Shared RWX PVC for model weights (`shared-model-cache`)
- Hugging Face token secret:
  ```bash
  kubectl create secret generic hf-token-secret \
    --from-literal=HF_TOKEN=<your-token> -n <namespace>
  ```

## Quick Start

```bash
# 0. Replace <your-namespace> in deploy.yaml with your namespace

# 1. Deploy
kubectl apply -f deploy.yaml

# 2. Test
kubectl port-forward svc/dsv4-pro-disagg-frontend 8000:8000 &
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"deepseek-ai/DeepSeek-V4-Pro","messages":[{"role":"user","content":"Hello!"}],"max_tokens":128}'
```

## InfiniBand RDMA Configuration

The `deploy.yaml` includes InfiniBand-specific UCX configuration validated on nscale B200 (ConnectX-8):

| Variable | Value | Purpose |
|----------|-------|---------|
| `UCX_TLS` | `rc_x,rc,cuda_copy,cuda_ipc` | Accelerated RC transport for IB RDMA |
| `UCX_NET_DEVICES` | `mlx5_0:1` | Single IB device (avoids bonded device issues) |
| `UCX_IB_ADDR_TYPE` | `eth` | Ethernet addressing — required for cross-pod IB on K8s |
| `UCX_RNDV_SCHEME` | `get_zcopy` | Zero-copy RDMA GET for KV transfers |
| `UCX_RNDV_THRESH` | `0` | Use rendezvous for all message sizes |
| `rdma/ib` | `8` (1 per GPU) | RDMA device plugin injects IB devices |

### Why `UCX_IB_ADDR_TYPE=eth`?

Without this setting, UCX uses LID-based InfiniBand addressing which does not route correctly between Kubernetes pods on different nodes. Setting `eth` switches to GID/Ethernet-style addressing that works across the pod network. This is the most common issue when bringing up NIXL disagg on InfiniBand clusters.

### Bonded IB Device Gotcha

Some clusters expose `mlx5_bond_0` with LID=0. Setting `UCX_NET_DEVICES=mlx5_0:1` avoids this device. If your cluster has a different IB device naming scheme, check with `ibv_devinfo`.

## Key Configuration Notes

- **EAGLE disabled** — Causes OOM on TP8 (4.23 GB free with EAGLE vs 7.75 without)
- **`mem-fraction-static=0.82`** — Higher than GB200 recipe (0.75) since B200 single-node TP avoids multi-node weight shuffle overhead
- **Security** — `IPC_LOCK` + `SYS_RESOURCE` capabilities required for RDMA memory pinning
- **Cold start ~20 min** — FP4 weight shuffle + CUDA graph capture

## Related

- [Disagg Communication Guide](../../../../../docs/kubernetes/disagg-communication-guide.md) — Full RDMA transport reference
- [GB200 Disagg Recipe](../disagg-gb200/) — Multi-node disagg on GB200 with ComputeDomain
