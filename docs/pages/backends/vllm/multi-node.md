---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
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

# Start prefill worker
python -m dynamo.vllm \
  --model meta-llama/Llama-3.3-70B-Instruct \
  --tensor-parallel-size 8 \
  --enforce-eager
```

**Node 2**: Run prefill worker
```bash
# Start decode worker
python -m dynamo.vllm \
  --model meta-llama/Llama-3.3-70B-Instruct \
  --tensor-parallel-size 8 \
  --enforce-eager \
  --is-prefill-worker
```
