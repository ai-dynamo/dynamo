<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Parallelism & Replication

## Parallelism Concepts

### Tensor Parallelism (TP)

Splits model weight matrices (attention, FFN) across GPUs. Each GPU holds 1/TP of the weights and performs 1/TP of the
compute. Requires AllReduce communication after each layer.

- Must be a power of 2 for most models
- Higher TP = lower per-GPU memory, but more inter-GPU communication
- For single-node DGX: max is 8

### Pipeline Parallelism (PP)

Splits model layers across GPU groups. Each stage holds a subset of layers. Introduces pipeline bubbles (wasted cycles
between micro-batches).

- Use only when model is too large for TP alone
- Single node (<=8 GPUs): usually PP=1 (TP is preferred)
- Multi-node (>8 GPUs): PP of 2 or 4 becomes useful

### Expert Parallelism (EP)

For MoE models only. Shards expert weights across GPUs. Higher EP reduces expert memory per GPU but raises All-to-All
communication cost.

### Context Parallelism (CP)

Splits long sequences across GPUs for prefill (ring attention). Only beneficial for sequences > 32K tokens.

---

## Model Replication vs. TP Scaling

**CRITICAL: prefer lower TP over higher TP for throughput workloads.**

When a model fits in fewer GPUs than are available, the default strategy should be to keep TP at the **minimum required
to fit the model**. The remaining GPUs can then host additional independent replicas behind a load balancer. Each
benchmark trial tests a **single replica only** — aggregate throughput for N replicas is simply
`single_replica_throughput × N`, so there is no need to benchmark multiple replicas.

**Why lower TP is better for throughput:**
- Higher TP adds AllReduce overhead per layer, per forward pass, which grows with TP degree and becomes the bottleneck
  long before compute is saturated.
- Lower TP leaves more GPUs available for replicas, which scale linearly with zero cross-replica communication.

**Example:** a 70B BF16 model on 8x H100-80GB needs TP=2 to fit. Benchmarking **1 replica at TP=2** and projecting to 4
replicas (`throughput × 4`) will typically yield higher aggregate throughput than benchmarking **1 replica at TP=8**,
because the TP=8 instance spends more time on AllReduce.

**For throughput workloads, keep TP at `min_tp`, and consider at most `2 × min_tp`, unless the user explicitly requests
higher TP values in the deployment description.** Going beyond `2 × min_tp` wastes GPUs on AllReduce overhead that
replicas would avoid. For example, if `min_tp = 4`, use TP=4 (or TP=8 with a stated reason) — not TP=16.

**Only justify increasing TP above `2 × min_tp` when at least one of these conditions is true:**
1. The model does not fit in memory at lower TP (weights + activations + KV cache exceed GPU memory).
2. The user explicitly requests exploring higher TP (e.g., for latency optimization).
3. The workload is latency-sensitive with very low concurrency (<4 concurrent requests). In this regime replicas from
   lower TP go unused, so spreading compute across more GPUs via higher TP reduces per-request TTFT/TPOT.

**When proposing TP > `2 × min_tp`, always state the justification explicitly.**
