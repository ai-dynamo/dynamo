<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# DeepSeek-V4-Pro — Disaggregated Prefill/Decode on GB200

Serves [deepseek-ai/DeepSeek-V4-Pro](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro) using SGLang with disaggregated prefill/decode via Dynamo on GB200 nodes.

## Topology

| Role    | Nodes | GPUs/node | Total GPUs | Parallelism |
|---------|-------|-----------|------------|-------------|
| Decode  | 2     | 4         | 8          | TP8         |
| Prefill | 2     | 4         | 8          | TP8         |

## Prerequisites

- 4× GB200 nodes with RDMA networking
- Dynamo operator + dynamo-platform HelmRelease installed
- ComputeDomain DRA driver (`compute-domain-default-channel.nvidia.com` device class)
- Shared RWX PVC for model weights

## Quick Start

```bash
# 1. Deploy
kubectl apply -f deploy.yaml

# 2. Test
kubectl port-forward svc/dsv4-pro-disagg-frontend 8000:8000 &
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"deepseek-ai/DeepSeek-V4-Pro","messages":[{"role":"user","content":"Hello!"}],"max_tokens":128}'
```

## Cluster-Specific Configuration

The base `deploy.yaml` is generic. Each cluster type requires additional configuration for RDMA networking (needed for NIXL KV cache transfer between prefill and decode).

### GKE (GCP GB200)

**RDMA annotations** — Add to both decode and prefill services:

```yaml
# In spec.services.decode and spec.services.prefill:
annotations:
  networking.gke.io/default-interface: eth0
  networking.gke.io/interfaces: |
    [
      {"interfaceName":"eth0","network":"default"},
      {"interfaceName":"rdma0","network":"rdma-0"},
      {"interfaceName":"rdma1","network":"rdma-1"},
      {"interfaceName":"rdma2","network":"rdma-2"},
      {"interfaceName":"rdma3","network":"rdma-3"}
    ]
resources:
  limits:
    custom:
      networking.gke.io.networks/rdma-0: "1"
      networking.gke.io.networks/rdma-1: "1"
      networking.gke.io.networks/rdma-2: "1"
      networking.gke.io.networks/rdma-3: "1"
```

**NATS workaround** — GCP clusters may isolate `system-cpu` and `customer-gpu` pod networks. If the dynamo-system NATS is unreachable, deploy a namespace-local NATS on a `customer-cpu` node and override `NATS_SERVER` in `spec.envs`.

**Tolerations** — GCP GPU nodes have taints. Add to all services:

```yaml
tolerations:
  - key: dedicated
    operator: Equal
    value: user-workload
    effect: NoExecute
  - key: nvidia.com/gpu
    operator: Exists
    effect: NoSchedule
  - key: kubernetes.io/arch
    operator: Equal
    value: arm64
    effect: NoSchedule
```

### AWS EFA

See [Disagg Communication Guide](../../../../../docs/kubernetes/disagg-communication-guide.md#aws-efa-configuration) for EFA-specific NIXL configuration with libfabric.

### InfiniBand

For clusters with native IB (e.g., DGX), add `rdma/ib` resource limits and ensure UCX UD transport is supported on your ConnectX firmware.

## Key Configuration Notes

- **EAGLE disabled** — Causes OOM on TP8 (4.23 GB free with EAGLE vs 7.75 without)
- **`mem-fraction-static=0.75`** — Lower than default to prevent OOM during FP4 MxFP4 weight shuffle
- **`cuda-graph-max-bs=128`** — Limits CUDA graph capture memory
- **Cold start ~45 min** — FP4 weight shuffle dominates startup time
