#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ============================================================================
# Model Router Example (RouteLLM-powered Hierarchical Routing)
#
# This example demonstrates RouteLLM integration with Dynamo's hierarchical
# planner. A Model Router classifies incoming requests by complexity and
# routes them to either a "strong" or "weak" model pool.
#
# Architecture:
#
#   Client (HTTP)
#     │
#     ▼
#   Frontend (namespace=hierarchical)
#     │  preprocesses + tokenizes, then routes via dyn://
#     ▼
#   Model Router (RouteLLM classification)
#     │  detokenizes → classifies → forwards to pool
#     ├───────────────────────────┐
#     ▼                           ▼
#   Strong Pool                 Weak Pool
#   ┌─────────────────┐        ┌─────────────────┐
#   │ Local KV Router │        │ Local KV Router │
#   │       │         │        │       │         │
#   │  Mocker Worker  │        │  Mocker Worker  │
#   │  (strong model) │        │  (weak model)   │
#   └─────────────────┘        └─────────────────┘
#
# Run: bash examples/nemo_switchyard/run_example.sh
# Wait ~15 seconds for all components to start, then test with the curl
# command at the bottom.
# ============================================================================

set -e
trap 'echo "Cleaning up..."; kill 0' EXIT

# Set deterministic hash for KV event IDs
export PYTHONHASHSEED=0

# RouteLLM imports openai.OpenAI() at module level (for sw_ranking router).
# Set a dummy key to avoid import errors when using non-OpenAI routers (bert, mf).
export OPENAI_API_KEY="${OPENAI_API_KEY:-not-needed}"

# Model used for tokenizer (shared by both pools in this example)
MODEL=${MODEL:-"Qwen/Qwen3-0.6B"}

# RouteLLM configuration
ROUTER_TYPE=${ROUTER_TYPE:-"mf"}
THRESHOLD=${THRESHOLD:-"0.5"}

echo "=== Dynamo Model Router Example ==="
echo "Model:         $MODEL"
echo "Router type:   $ROUTER_TYPE"
echo "Threshold:     $THRESHOLD"
echo ""

# ============================================================================
# Strong Pool: local KV router + mocker worker
# ============================================================================
echo "[1/7] Starting strong pool KV router..."
DYN_NAMESPACE=strong_pool python3 -m dynamo.router \
  --endpoint strong_pool.worker.generate \
  --block-size 16 \
  --no-track-active-blocks &
sleep 2

echo "[2/7] Starting strong pool mocker worker..."
python3 -m dynamo.mocker \
  --model-path "$MODEL" \
  --model-name "${MODEL}-strong" \
  --endpoint "dyn://strong_pool.worker.generate" \
  --block-size 16 &
sleep 2

# ============================================================================
# Weak Pool: local KV router + mocker worker
# ============================================================================
echo "[3/7] Starting weak pool KV router..."
DYN_NAMESPACE=weak_pool python3 -m dynamo.router \
  --endpoint weak_pool.worker.generate \
  --block-size 16 \
  --no-track-active-blocks &
sleep 2

echo "[4/7] Starting weak pool mocker worker..."
python3 -m dynamo.mocker \
  --model-path "$MODEL" \
  --model-name "${MODEL}-weak" \
  --endpoint "dyn://weak_pool.worker.generate" \
  --block-size 16 &
sleep 2

# ============================================================================
# Model Router: RouteLLM classification → routes to strong/weak pool
# ============================================================================
echo "[5/7] Starting Model Router (RouteLLM)..."
python3 -m dynamo.nemo_switchyard \
  --model-path "$MODEL" \
  --model-name "$MODEL" \
  --strong-pool-endpoint strong_pool.router.generate \
  --weak-pool-endpoint weak_pool.router.generate \
  --router-type "$ROUTER_TYPE" \
  --threshold "$THRESHOLD" \
  --namespace hierarchical &
sleep 5

# ============================================================================
# Frontend: HTTP entry point, discovers nemo_switchyard in hierarchical namespace
# ============================================================================
echo "[6/7] Starting Frontend..."
python3 -m dynamo.frontend \
  --router-mode round-robin \
  --namespace hierarchical &
sleep 5

echo ""
echo "[7/7] All components started!"
echo ""

# ============================================================================
# Test request
# ============================================================================
echo "=== Test: Send a request ==="
echo "Try these commands:"
echo ""
echo "  # Simple question (should route to weak pool):"
echo "  curl -s http://localhost:8000/v1/chat/completions \\"
echo "    -H 'Content-Type: application/json' \\"
echo "    -d '{\"model\": \"$MODEL\", \"messages\": [{\"role\": \"user\", \"content\": \"What is 2+2?\"}], \"max_tokens\": 50, \"stream\": true}'"
echo ""
echo "  # Complex question (should route to strong pool):"
echo "  curl -s http://localhost:8000/v1/chat/completions \\"
echo "    -H 'Content-Type: application/json' \\"
echo "    -d '{\"model\": \"$MODEL\", \"messages\": [{\"role\": \"user\", \"content\": \"Derive the Navier-Stokes equations from first principles and explain the millennium prize problem associated with them.\"}], \"max_tokens\": 200, \"stream\": true}'"
echo ""

# Keep running until interrupted
wait

