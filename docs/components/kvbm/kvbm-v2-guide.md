---
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: KVBM v2 Guide
---

> [!NOTE]
> **KVBM v2 is currently under active development on the [`ryan/kvbm-bindings`](https://github.com/ai-dynamo/dynamo/tree/ryan/kvbm-bindings) branch.** The instructions below require building from that branch.

## Run KVBM v2 in Dynamo with vLLM

### Docker Setup

Unlike v1, KVBM v2 does **not** require etcd or NATS. No `docker compose` step is needed.

```bash
# Build from the ryan/kvbm-bindings branch
git checkout ryan/kvbm-bindings

# Build a dynamo vLLM container (KVBM v2 is built in)
python container/render.py --framework vllm --target runtime --output-short-filename
docker build -t dynamo:latest-vllm-v2-runtime -f container/rendered.Dockerfile .

# Launch the container
container/run.sh --image dynamo:latest-vllm-v2-runtime -it --mount-workspace --use-nixl-gds
```

### Aggregated Serving

```bash
# run ingress
python -m dynamo.frontend &

# run worker with KVBM v2 enabled
# NOTE: remove --enforce-eager for production use
# NOTE: DYN_KVBM_NIXL_BACKEND_UCX=true is required to enable NIXL for RDMA transfers
DYN_KVBM_NIXL_BACKEND_UCX=true \
DYN_KVBM_CPU_CACHE_GB=20 \
  python -m dynamo.vllm \
    --model Qwen/Qwen3-0.6B \
    --kv-transfer-config '{"kv_connector":"DynamoConnector","kv_connector_module_path":"kvbm.v2.vllm.connector","kv_role":"kv_both"}' \
    --enforce-eager
```

#### Verify Deployment

```bash
curl localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-0.6B",
    "messages": [{"role": "user", "content": "Hello, how are you?"}],
    "stream": false,
    "max_tokens": 10
  }'
```

#### Alternative: Using Direct vllm serve

```bash
DYN_KVBM_NIXL_BACKEND_UCX=true \
DYN_KVBM_CPU_CACHE_GB=20 \
vllm serve --kv-transfer-config '{"kv_connector":"DynamoConnector","kv_role":"kv_both","kv_connector_module_path":"kvbm.v2.vllm.connector"}' Qwen/Qwen3-0.6B
```

## Configuration

### Onboard Mode

KVBM v2 supports two onboarding modes that control how KV cache blocks are loaded from host (G2) to device (G1) memory:

```bash
# Inter-pass onboarding (default)
# Blocks are loaded asynchronously between scheduler passes via Nova messages.
# The forward pass does not block on transfers.
export KVBM_ONBOARD_MODE=inter

# Intra-pass onboarding
# Blocks are loaded synchronously, layer-by-layer during the forward pass.
# Each layer waits for its KV cache load to complete before proceeding.
export KVBM_ONBOARD_MODE=intra
```

Full example with intra mode:

```bash
KVBM_ONBOARD_MODE=intra \
DYN_KVBM_NIXL_BACKEND_UCX=true \
DYN_KVBM_CPU_CACHE_GB=20 \
  python -m dynamo.vllm \
    --model Qwen/Qwen3-0.6B \
    --kv-transfer-config '{"kv_connector":"DynamoConnector","kv_connector_module_path":"kvbm.v2.vllm.connector","kv_role":"kv_both"}' \
    --enforce-eager
```



## Benchmarking

Use [LMBenchmark](https://github.com/LMCache/LMBenchmark) to evaluate KVBM v2 performance, same as v1:

```bash
git clone https://github.com/LMCache/LMBenchmark.git
cd LMBenchmark/synthetic-multi-round-qa

./long_input_short_output_run.sh \
    "Qwen/Qwen3-0.6B" \
    "http://localhost:8000" \
    "benchmark_kvbm_v2" \
    1
```
