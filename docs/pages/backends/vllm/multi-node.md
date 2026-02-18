---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Multi-Node
---

# Multi-node Examples

This guide covers deploying vLLM across multiple nodes using Dynamo's distributed capabilities.

## Prerequisites

Multi-node deployments require:
- Multiple nodes with GPU resources
- Network connectivity between nodes (faster the better)
- Firewall rules allowing NATS/ETCD communication

## Infrastructure Setup

### Step 1: Start NATS/ETCD on Head Node

Start the required services on your head node. These endpoints must be accessible from all worker nodes:

```bash
# On head node (node-1)
docker compose -f deploy/docker-compose.yml up -d
```

Default ports:
- NATS: 4222
- ETCD: 2379

### Step 2: Configure Environment Variables

Set the head node IP address and service endpoints. **Set this on all nodes** for easy copy-paste:

```bash
# Set this on ALL nodes - replace with your actual head node IP
export HEAD_NODE_IP="<your-head-node-ip>"

# Service endpoints (set on all nodes)
export NATS_SERVER="nats://${HEAD_NODE_IP}:4222"
export ETCD_ENDPOINTS="${HEAD_NODE_IP}:2379"
```

## Deployment Patterns

### Multi-node Aggregated Serving

Deploy vLLM workers across multiple nodes for horizontal scaling:

**Node 1 (Head Node)**: Run ingress and first worker
```bash
# Start ingress
python -m dynamo.frontend --router-mode kv

# Start vLLM worker
python -m dynamo.vllm \
  --model meta-llama/Llama-3.3-70B-Instruct \
  --tensor-parallel-size 8 \
  --enforce-eager
```

**Node 2**: Run additional worker
```bash
# Start vLLM worker
python -m dynamo.vllm \
  --model meta-llama/Llama-3.3-70B-Instruct \
  --tensor-parallel-size 8 \
  --enforce-eager
```

### Multi-node Disaggregated Serving

Deploy prefill and decode workers on separate nodes for optimized resource utilization:

**Node 1**: Run ingress and decode worker
```bash
# Start ingress
python -m dynamo.frontend --router-mode kv &

# Start decode worker
python -m dynamo.vllm \
  --model meta-llama/Llama-3.3-70B-Instruct \
  --tensor-parallel-size 8 \
  --enforce-eager
```

**Node 2**: Run prefill worker
```bash
# Start prefill worker
python -m dynamo.vllm \
  --model meta-llama/Llama-3.3-70B-Instruct \
  --tensor-parallel-size 8 \
  --enforce-eager \
  --is-prefill-worker
```

### Multi-node Tensor/Pipeline Parallelism

When the total parallelism (TP × PP) exceeds the number of GPUs on a single node,
you need multiple nodes to host a **single** model instance. In this mode, one node
runs the full `dynamo.vllm` process (head node) while additional nodes run in
`--headless` mode — spawning only vLLM workers without Dynamo endpoints, NATS, or etcd.

**How it works:**

- **Head node** (`node_rank_within_dp == 0`): runs the full Dynamo stack — engine core,
  scheduler, and Dynamo endpoints (NATS/etcd required).
- **Worker nodes** (`--headless`): run `vLLM` workers only via `run_headless()`.
  No engine core, no scheduler, no Dynamo endpoints. Only `torch.distributed`
  connectivity to the head node is required.

**Infrastructure requirements:**

| Node | NATS/etcd | torch.distributed | Dynamo endpoints |
|------|-----------|-------------------|------------------|
| Head | Yes | Yes | Yes |
| Worker (`--headless`) | No | Yes | No |

**Example: TP=16 across 2× 8-GPU nodes**

```bash
# On both nodes
export HEAD_NODE_IP="<your-head-node-ip>"
export NATS_SERVER="nats://${HEAD_NODE_IP}:4222"
export ETCD_ENDPOINTS="${HEAD_NODE_IP}:2379"
```

Node 1 (head):
```bash
python -m dynamo.frontend --router-mode kv &

python -m dynamo.vllm \
  --model meta-llama/Llama-3.1-405B-Instruct \
  --tensor-parallel-size 16 \
  --enforce-eager
```

Node 2 (worker):
```bash
python -m dynamo.vllm \
  --model meta-llama/Llama-3.1-405B-Instruct \
  --tensor-parallel-size 16 \
  --enforce-eager \
  --headless
```

**Example: TP=8, PP=2 across 2× 8-GPU nodes**

Node 1 (head):
```bash
python -m dynamo.frontend --router-mode kv &

python -m dynamo.vllm \
  --model meta-llama/Llama-3.1-405B-Instruct \
  --tensor-parallel-size 8 \
  --pipeline-parallel-size 2 \
  --enforce-eager
```

Node 2 (worker):
```bash
python -m dynamo.vllm \
  --model meta-llama/Llama-3.1-405B-Instruct \
  --tensor-parallel-size 8 \
  --pipeline-parallel-size 2 \
  --enforce-eager \
  --headless
```
