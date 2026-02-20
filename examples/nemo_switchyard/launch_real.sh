#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ============================================================================
# NeMo Switchyard — Real Model Test (RouteLLM BERT Router)
#
# GPU layout (GPU 0 reserved):
#   GPUs 1-4: gpt-oss-120b (strong model, TP=4)
#   GPU 5:    gpt-oss-20b  (weak model, TP=1)
#   GPUs 6-7: unused
#
# Architecture:
#   Client (HTTP :8080)
#     → Frontend (namespace=hierarchical)
#       → NeMo Switchyard (RouteLLM BERT classifier)
#         → Strong Pool (KV router → vLLM gpt-oss-120b)
#         → Weak Pool   (KV router → vLLM gpt-oss-20b)
#
# Usage:
#   bash examples/nemo_switchyard/launch_real.sh
#
# Override defaults via env vars:
#   STRONG_MODEL=/data/models/gpt-oss-120b \
#   WEAK_MODEL=/data/models/gpt-oss-20b \
#   STRONG_GPUS=1,2,3,4 STRONG_TP=4 \
#   WEAK_GPUS=5 \
#   THRESHOLD=0.5 \
#   bash examples/nemo_switchyard/launch_real.sh
# ============================================================================

set -e
trap 'echo ""; echo "Cleaning up all background processes..."; kill 0' EXIT

# Set deterministic hash for KV event IDs
export PYTHONHASHSEED=0

# RouteLLM imports openai.OpenAI() at module level (for sw_ranking router).
# Set a dummy key to avoid import errors when using non-OpenAI routers (bert, mf).
export OPENAI_API_KEY="${OPENAI_API_KEY:-not-needed}"

# ── Configuration ──
STRONG_MODEL="${STRONG_MODEL:-/data/models/gpt-oss-120b}"
WEAK_MODEL="${WEAK_MODEL:-/data/models/gpt-oss-20b}"
STRONG_GPUS="${STRONG_GPUS:-1,2,3,4}"
STRONG_TP="${STRONG_TP:-4}"
WEAK_GPUS="${WEAK_GPUS:-5}"
ROUTER_TYPE="${ROUTER_TYPE:-bert}"
THRESHOLD="${THRESHOLD:-0.5}"

echo "============================================"
echo "  NeMo Switchyard — Real Model Test"
echo "============================================"
echo "  Strong model: $STRONG_MODEL (GPUs $STRONG_GPUS, TP=$STRONG_TP)"
echo "  Weak model:   $WEAK_MODEL (GPU $WEAK_GPUS)"
echo "  RouteLLM:     $ROUTER_TYPE (threshold=$THRESHOLD)"
echo "============================================"
echo ""

# ============================================================================
# 1. Strong pool: vLLM worker (GPUs 1-4, TP=4)
# ============================================================================
echo "[1/6] Starting strong pool vLLM worker ($STRONG_MODEL on GPUs $STRONG_GPUS)..."
DYN_NAMESPACE=strong_pool \
CUDA_VISIBLE_DEVICES=$STRONG_GPUS \
python3 -m dynamo.vllm \
  --model "$STRONG_MODEL" \
  --tensor-parallel-size "$STRONG_TP" \
  --enforce-eager \
  --connector none &
STRONG_PID=$!
sleep 5

# ============================================================================
# 2. Weak pool: vLLM worker (GPU 5)
# ============================================================================
echo "[2/6] Starting weak pool vLLM worker ($WEAK_MODEL on GPU $WEAK_GPUS)..."
DYN_NAMESPACE=weak_pool \
CUDA_VISIBLE_DEVICES=$WEAK_GPUS \
python3 -m dynamo.vllm \
  --model "$WEAK_MODEL" \
  --enforce-eager \
  --connector none &
WEAK_PID=$!
sleep 5

# ============================================================================
# 3. Strong pool: local KV router
# ============================================================================
echo "[3/6] Starting strong pool KV router..."
DYN_NAMESPACE=strong_pool \
python3 -m dynamo.router \
  --endpoint strong_pool.backend.generate \
  --block-size 16 \
  --no-kv-events \
  --no-track-active-blocks &
sleep 3

# ============================================================================
# 4. Weak pool: local KV router
# ============================================================================
echo "[4/6] Starting weak pool KV router..."
DYN_NAMESPACE=weak_pool \
python3 -m dynamo.router \
  --endpoint weak_pool.backend.generate \
  --block-size 16 \
  --no-kv-events \
  --no-track-active-blocks &
sleep 3

# ============================================================================
# 5. NeMo Switchyard (RouteLLM BERT router)
# ============================================================================
echo "[5/6] Starting NeMo Switchyard (RouteLLM $ROUTER_TYPE router)..."
python3 -m dynamo.nemo_switchyard \
  --model-path "$WEAK_MODEL" \
  --model-name "gpt-oss" \
  --strong-pool-endpoint strong_pool.router.generate \
  --weak-pool-endpoint weak_pool.router.generate \
  --router-type "$ROUTER_TYPE" \
  --threshold "$THRESHOLD" \
  --namespace hierarchical &
sleep 5

# ============================================================================
# 6. Frontend (HTTP entry point)
# ============================================================================
FRONTEND_PORT="${FRONTEND_PORT:-8080}"
echo "[6/6] Starting Frontend on port $FRONTEND_PORT..."
python3 -m dynamo.frontend \
  --router-mode round-robin \
  --namespace hierarchical \
  --http-port "$FRONTEND_PORT" &
sleep 5

echo ""
echo "============================================"
echo "  All components started!"
echo "============================================"
echo ""
echo "Test requests:"
echo ""
echo "  # Simple question (→ weak pool / gpt-oss-20b):"
echo "  curl -s http://localhost:$FRONTEND_PORT/v1/chat/completions \\"
echo "    -H 'Content-Type: application/json' \\"
echo "    -d '{\"model\":\"gpt-oss\",\"messages\":[{\"role\":\"user\",\"content\":\"What is 2+2?\"}],\"max_tokens\":50,\"stream\":true}'"
echo ""
echo "  # Complex question (→ strong pool / gpt-oss-120b):"
echo "  curl -s http://localhost:$FRONTEND_PORT/v1/chat/completions \\"
echo "    -H 'Content-Type: application/json' \\"
echo "    -d '{\"model\":\"gpt-oss\",\"messages\":[{\"role\":\"user\",\"content\":\"Derive the Navier-Stokes equations from first principles.\"}],\"max_tokens\":200,\"stream\":true}'"
echo ""
echo "  # Non-streaming:"
echo "  curl -s http://localhost:$FRONTEND_PORT/v1/chat/completions \\"
echo "    -H 'Content-Type: application/json' \\"
echo "    -d '{\"model\":\"gpt-oss\",\"messages\":[{\"role\":\"user\",\"content\":\"Hello!\"}],\"max_tokens\":30}'"
echo ""
echo "Watch the NeMo Switchyard logs for 'Routed to strong' / 'Routed to weak'."
echo "Press Ctrl+C to stop all components."
echo ""

# Keep running until interrupted
wait

