---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: SGLang Multi-Node Tensor Parallelism
---

# SGLang Multi-Node Tensor Parallelism

SGLang supports multi-node TP via the native `--dist-init-addr`, `--node-rank`, `--nnodes` flags. This guide mirrors the existing vLLM multi-node TP setup for parity.

## When to use

- Model exceeds a single node's aggregate GPU memory (e.g. Llama-3.3-70B on 2 × H100-80GB is comfortable, but larger FP16 models or long-context workloads push past).
- You want TP across nodes rather than PP for latency-sensitive workloads.

## Prerequisites

- Two or more nodes on the same low-latency network (IB or 100GbE+).
- NCCL reachable between all nodes (firewall / security-group allows NCCL's port range).
- Same container image on all nodes (`nvcr.io/nvidia/ai-dynamo/sglang-runtime:1.0.1` or pinned variant).
- HF_TOKEN set on **every** node if the model is gated.
- Shared `HF_HOME` recommended to avoid per-node download of the same weights.

## Flags

| Flag | Meaning |
|---|---|
| `--dist-init-addr <rank0-host>:<port>` | Rendezvous address. All nodes must use the same value. |
| `--node-rank <N>` | `0` for rank-0 (rendezvous host), `1..N-1` for the rest. |
| `--nnodes <N>` | Total node count. Must match across all nodes. |
| `--tp <TP>` | Tensor-parallel degree across all GPUs on all nodes (e.g. 16 for 2 × 8-GPU nodes). |

## Example: 2 × 8-GPU (TP = 16)

On **rank 0** (`host0`):

```bash
python -m sglang.launch_server \
  --model meta-llama/Llama-3.3-70B-Instruct \
  --tp 16 \
  --nnodes 2 \
  --node-rank 0 \
  --dist-init-addr host0:29500
```

On **rank 1** (`host1`):

```bash
python -m sglang.launch_server \
  --model meta-llama/Llama-3.3-70B-Instruct \
  --tp 16 \
  --nnodes 2 \
  --node-rank 1 \
  --dist-init-addr host0:29500
```

Dynamo's frontend is launched once, typically co-located with rank 0. Workers on rank 1 register via etcd.

## Troubleshooting

- **Hang at startup** → NCCL port blocked. Confirm `firewall-cmd --list-ports` / security-group rules on both nodes.
- **`Address already in use` on `29500`** → pick a free port; the value is arbitrary as long as both nodes agree.
- **Ranks hang at different points** → time skew / clock drift. Sync via NTP.
- **One rank exits `NCCL timeout`** → one node's HF download is still running. Pre-download to shared `HF_HOME` before launch.

## See also

- [vLLM multi-node TP](../vllm/README.md) (parallel structure).
- [Kubernetes Deployment Guide](../../kubernetes/README.md#2-choose-your-backend) — backend selection and deployment reference.
