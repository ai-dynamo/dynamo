#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Hierarchical Planner Example
# Run each command in a separate terminal, in order from bottom to top.
# Wait a few seconds between starting each component.

# ============================================================================
# frontend + global_router
# ============================================================================
# need to specify a namespace so that mockers are not registered to frontend
# and cannot use "dynamo" because that is reserved for all namespaces
python -m dynamo.frontend \
  --router-mode round-robin \
  --namespace hierarchical &

python -m dynamo.global_router \
  --config examples/hierarchical_planner/global_router_config.json \
  --model-name Qwen/Qwen3-0.6B \
  --default-ttft-target 100 \
  --default-itl-target 10 \
  --namespace hierarchical &

# ============================================================================
# prefill_pool_0 - local router + mocker worker (prefill)
# ============================================================================
DYN_NAMESPACE=prefill_pool_0 python -m dynamo.router \
  --endpoint prefill_pool_0.worker.generate \
  --block-size 16 \
  --no-track-active-blocks &  # prefill router does not need to track active blocks

python -m dynamo.mocker \
  --model-path Qwen/Qwen3-0.6B \
  --endpoint dyn://prefill_pool_0.worker.generate \
  --is-prefill-worker \
  --block-size 16 &

# ============================================================================
# prefill_pool_1 - local router + mocker worker (prefill)
# ============================================================================
DYN_NAMESPACE=prefill_pool_1 python -m dynamo.router \
  --endpoint prefill_pool_1.worker.generate \
  --block-size 16 \
  --no-track-active-blocks &  # prefill router does not need to track active blocks

python -m dynamo.mocker \
  --model-path Qwen/Qwen3-0.6B \
  --endpoint dyn://prefill_pool_1.worker.generate \
  --is-prefill-worker \
  --block-size 16 &

# ============================================================================
# decode_pool_0 - local router + mocker worker (decode)
# ============================================================================
DYN_NAMESPACE=decode_pool_0 python -m dynamo.router \
  --endpoint decode_pool_0.worker.generate \
  --block-size 16 \
  --kv-overlap-score-weight 0 &

python -m dynamo.mocker \
  --model-path Qwen/Qwen3-0.6B \
  --endpoint dyn://decode_pool_0.worker.generate \
  --is-decode-worker \
  --block-size 16 &

# ============================================================================
# test request
# ============================================================================

# wait for all components to start
# curl -X POST http://localhost:8000/v1/chat/completions \
#   -H "Content-Type: application/json" \
#   -d '{
#     "model": "Qwen/Qwen3-0.6B",
#     "messages": [{"role": "user", "content": "Hello!"}],
#     "max_tokens": 50,
#     "stream": true
#   }'
