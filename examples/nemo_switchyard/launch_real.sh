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
#   Client (HTTP :8888)
#     → Frontend (namespace=hierarchical)
#       → NeMo Switchyard (RouteLLM BERT classifier)
#         → Strong Pool (KV router → vLLM gpt-oss-120b)
#         → Weak Pool   (KV router → vLLM gpt-oss-20b)

# Override defaults via env vars:
#   STRONG_MODEL=/data/models/gpt-oss-120b \
#   WEAK_MODEL=/data/models/gpt-oss-20b \
#   STRONG_GPUS=1,2,3,4 STRONG_TP=5 \
#   WEAK_GPUS=4 \
#   THRESHOLD=0.5 \
#   bash examples/nemo_switchyard/launch_real.sh --server
# ============================================================================

set -e

# ── Configuration ──
STRONG_MODEL="${STRONG_MODEL:-/data/models/gpt-oss-120b}"
WEAK_MODEL="${WEAK_MODEL:-/data/models/gpt-oss-20b}"
STRONG_GPUS="${STRONG_GPUS:-1,2,3,4}"
STRONG_TP="${STRONG_TP:-4}"
WEAK_GPUS="${WEAK_GPUS:-5}"
ROUTER_TYPE="${ROUTER_TYPE:-bert}"
THRESHOLD="${THRESHOLD:-0.5}"
FRONTEND_PORT="${FRONTEND_PORT:-8888}"

# ── Helper functions ──

show_help() {
    echo "NeMo Switchyard — RouteLLM Model Router Demo"
    echo ""
    echo "Usage:"
    echo "  bash $0 --server     Launch Docker container and start all components (left terminal)"
    echo "  bash $0 --query      Send demo requests (right terminal)"
    echo "  bash $0 --cleanup    Kill containers, clean etcd, prepare fresh start"
    echo "  bash $0 --help       Show this help"
    echo ""
    echo "Environment variables:"
    echo "  STRONG_MODEL   Strong model path      (default: /data/models/gpt-oss-120b)"
    echo "  WEAK_MODEL     Weak model path        (default: /data/models/gpt-oss-20b)"
    echo "  STRONG_GPUS    GPUs for strong model  (default: 1,2,3,4)"
    echo "  STRONG_TP      Tensor parallel size   (default: 4)"
    echo "  WEAK_GPUS      GPU for weak model     (default: 5)"
    echo "  ROUTER_TYPE    RouteLLM router        (default: bert)"
    echo "  THRESHOLD      Routing threshold 0-1  (default: 0.5)"
    echo "  FRONTEND_PORT  HTTP port              (default: 8888)"
    echo "  CONTAINER_IMAGE  Docker image          (default: dynamo:latest-vllm-local-dev)"
    echo "  CONTAINER_STRONG_GPUS  GPUs inside container for strong model (default: 0,1,2,3)"
    echo "  CONTAINER_WEAK_GPUS    GPU inside container for weak model    (default: 4)"
}

do_cleanup() {
    echo "=== Cleaning up ==="

    echo "[1/4] Killing dynamo containers..."
    docker ps --filter "ancestor=dynamo:latest-vllm-local-dev" -q | xargs -r docker kill 2>/dev/null || true

    echo "[2/4] Killing bare-metal dynamo processes..."
    pkill -f "dynamo\.(vllm|router|nemo_switchyard|frontend)" 2>/dev/null || true

    echo "[3/4] Clearing etcd state..."
    docker exec deploy-etcd-server-1 etcdctl del "" --prefix 2>/dev/null || true

    echo "[4/4] Cleaning torchinductor cache..."
    rm -rf /tmp/torchinductor_* 2>/dev/null || true

    sleep 2
    echo ""
    echo "✅ Cleanup complete. Ready for fresh start with: $0 --server"
}

do_server() {
    # If not already inside the container, launch one and re-exec this script inside it.
    if [ -z "${IN_DYNAMO_CONTAINER:-}" ]; then
        SCRIPT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
        CONTAINER_IMAGE="${CONTAINER_IMAGE:-dynamo:latest-vllm-local-dev}"

        echo "=== Launching inside Docker container ==="
        echo "  Image:  $CONTAINER_IMAGE"
        echo "  GPUs:   device=${STRONG_GPUS},${WEAK_GPUS}"
        echo ""

        exec docker run \
          --gpus "\"device=${STRONG_GPUS},${WEAK_GPUS}\"" \
          --rm --network host --runtime nvidia \
          --shm-size=10G \
          --ulimit memlock=-1 --ulimit stack=67108864 --ulimit nofile=65536:65536 \
          -v /data:/data \
          -v "${SCRIPT_DIR}":/workspace \
          --cap-add CAP_SYS_PTRACE --ipc host \
          -w /workspace \
          -e IN_DYNAMO_CONTAINER=1 \
          -e OPENAI_API_KEY="${OPENAI_API_KEY:-not-needed}" \
          -e PYTHONHASHSEED=0 \
          -e STRONG_GPUS="${CONTAINER_STRONG_GPUS:-0,1,2,3}" \
          -e WEAK_GPUS="${CONTAINER_WEAK_GPUS:-4}" \
          -e STRONG_TP="$STRONG_TP" \
          -e FRONTEND_PORT="$FRONTEND_PORT" \
          -e ROUTER_TYPE="$ROUTER_TYPE" \
          -e THRESHOLD="$THRESHOLD" \
          -e DYN_LOG="${DYN_LOG:-info}" \
          "$CONTAINER_IMAGE" \
          bash -c "pip install routellm transformers -q && bash examples/nemo_switchyard/launch_real.sh --server"
    fi

    # ── Inside the container from here ──
    trap 'echo ""; echo "Cleaning up all background processes..."; kill 0' EXIT

    # Set deterministic hash for KV event IDs
    export PYTHONHASHSEED=0

    # RouteLLM imports openai.OpenAI() at module level (for sw_ranking router).
    # Set a dummy key to avoid import errors when using non-OpenAI routers (bert, mf).
    export OPENAI_API_KEY="${OPENAI_API_KEY:-not-needed}"

    echo "============================================"
    echo "  NeMo Switchyard — Real Model Test"
    echo "============================================"
    echo "  Strong model: $STRONG_MODEL (GPUs $STRONG_GPUS, TP=$STRONG_TP)"
    echo "  Weak model:   $WEAK_MODEL (GPU $WEAK_GPUS)"
    echo "  RouteLLM:     $ROUTER_TYPE (threshold=$THRESHOLD)"
    echo "  Frontend:     http://localhost:$FRONTEND_PORT"
    echo "============================================"
    echo ""

    # ==================================================================
    # 1. Strong pool: vLLM worker
    # ==================================================================
    echo "[1/6] Starting strong pool vLLM worker ($STRONG_MODEL on GPUs $STRONG_GPUS)..."
    DYN_NAMESPACE=strong_pool \
    CUDA_VISIBLE_DEVICES=$STRONG_GPUS \
    python3 -m dynamo.vllm \
      --model "$STRONG_MODEL" \
      --tensor-parallel-size "$STRONG_TP" \
      --enforce-eager \
      --connector none &
    sleep 5

    # ==================================================================
    # 2. Weak pool: vLLM worker
    # ==================================================================
    echo "[2/6] Starting weak pool vLLM worker ($WEAK_MODEL on GPU $WEAK_GPUS)..."
    DYN_NAMESPACE=weak_pool \
    CUDA_VISIBLE_DEVICES=$WEAK_GPUS \
    python3 -m dynamo.vllm \
      --model "$WEAK_MODEL" \
      --enforce-eager \
      --connector none &
    sleep 5

    # ==================================================================
    # 3. Strong pool: local KV router
    # ==================================================================
    echo "[3/6] Starting strong pool KV router..."
    DYN_NAMESPACE=strong_pool \
    python3 -m dynamo.router \
      --endpoint strong_pool.backend.generate \
      --block-size 16 \
      --no-kv-events \
      --no-track-active-blocks &
    sleep 3

    # ==================================================================
    # 4. Weak pool: local KV router
    # ==================================================================
    echo "[4/6] Starting weak pool KV router..."
    DYN_NAMESPACE=weak_pool \
    python3 -m dynamo.router \
      --endpoint weak_pool.backend.generate \
      --block-size 16 \
      --no-kv-events \
      --no-track-active-blocks &
    sleep 3

    # ==================================================================
    # 5. NeMo Switchyard (RouteLLM BERT router)
    # ==================================================================
    echo "[5/6] Starting NeMo Switchyard (RouteLLM $ROUTER_TYPE router)..."
    python3 -m dynamo.nemo_switchyard \
      --model-path "$WEAK_MODEL" \
      --model-name "gpt-oss" \
      --pool strong=strong_pool.router.generate \
      --pool weak=weak_pool.router.generate \
      --router-type routellm \
      --routellm-algorithm "$ROUTER_TYPE" \
      --threshold "$THRESHOLD" \
      --namespace hierarchical &
    sleep 5

    # ==================================================================
    # 6. Frontend (HTTP entry point)
    # ==================================================================
    echo "[6/6] Starting Frontend on port $FRONTEND_PORT..."
    python3 -m dynamo.frontend \
      --router-mode round-robin \
      --namespace hierarchical \
      --http-port "$FRONTEND_PORT" &
    sleep 5

    echo ""
    echo "============================================"
    echo "  ✅ All components started!"
    echo "============================================"
    echo ""
    echo "  In the other terminal, run:"
    echo "    bash examples/nemo_switchyard/launch_real.sh --query"
    echo ""
    echo "  Watch the logs below for routing decisions."
    echo "  Look for 'total blocks: ~74K' (weak) vs '~184K' (strong)."
    echo "  Press Ctrl+C to stop all components."
    echo ""

    wait
}

do_query() {
    BASE_URL="http://localhost:$FRONTEND_PORT"

    echo "============================================"
    echo "  NeMo Switchyard — RouteLLM Demo Queries"
    echo "============================================"
    echo "  Target: $BASE_URL"
    echo ""

    # Check if server is up
    if ! curl -s "$BASE_URL/health" > /dev/null 2>&1; then
        echo "❌ Server not reachable at $BASE_URL"
        echo "   Start the server first: bash $0 --server"
        exit 1
    fi
    echo "✅ Server is reachable"
    echo ""

    # ── Test 1: Simple question ──
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  TEST 1: Simple question → expect WEAK pool (gpt-oss-20b)"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
    echo "  Request: What is 2+2?"
    echo ""
    echo "  Response:"
    curl -s "$BASE_URL/v1/chat/completions" \
      -H "Content-Type: application/json" \
      -d '{
        "model": "gpt-oss",
        "messages": [{"role": "user", "content": "What is 2+2?"}],
        "max_tokens": 50
      }' | python3 -m json.tool
    echo ""
    echo "  👈 Check left terminal: 'total_blocks: ~74K' = weak pool ✅"
    echo ""

    read -p "  Press Enter to send the next request..."
    echo ""

    # ── Test 2: Complex question ──
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  TEST 2: Complex question → expect STRONG pool (gpt-oss-120b)"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
    echo "  Request: Derive the Navier-Stokes equations from first principles"
    echo "           and explain the mathematical formulation of turbulence."
    echo ""
    echo "  Response:"
    curl -s "$BASE_URL/v1/chat/completions" \
      -H "Content-Type: application/json" \
      -d '{
        "model": "gpt-oss",
        "messages": [{"role": "user", "content": "Derive the Navier-Stokes equations from first principles and explain the mathematical formulation of turbulence."}],
        "max_tokens": 100
      }' | python3 -m json.tool
    echo ""
    echo "  👈 Check left terminal: 'total_blocks: ~184K' = strong pool ✅"
    echo ""

    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  ✅ Demo complete!"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
    echo "  RouteLLM BERT classifier routed:"
    echo "    Simple  → weak pool  (gpt-oss-20b,  ~74K blocks)"
    echo "    Complex → strong pool (gpt-oss-120b, ~184K blocks)"
    echo ""
    echo "  To stop: Ctrl+C in the left terminal"
    echo "  To restart fresh: bash $0 --cleanup"
}

# ── Main ──

case "${1:-}" in
    --server)
        do_server
        ;;
    --query)
        do_query
        ;;
    --cleanup)
        do_cleanup
        ;;
    --help|-h)
        show_help
        ;;
    *)
        show_help
        exit 1
        ;;
esac
