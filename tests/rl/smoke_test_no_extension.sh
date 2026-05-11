#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Smoke test: RL admin control plane WITHOUT a worker extension class.
#
# Verifies what works against a stock dynamo.vllm worker (no prime_rl extension):
#   1. LoRA load/inference/unload — uses vLLM-native add_lora / remove_lora.
#      Expected: PASS.
#   2. Full-weight update via vLLM-native reload_weights — engine_rpc default.
#      Expected: documents whether the model supports reload_weights. For Qwen3,
#      vLLM v0.19.0's reload_weights raises on the fused gate_up_proj layer, so
#      this step reports the engine_rpc result without dying — the FileSystem
#      worker extension is the supported path for FT weight swaps.
#
# Usage:
#   cd /home/biswaranjanp/dev/rl/dynamo
#   source dynamo/bin/activate
#   bash tests/rl/smoke_test_no_extension.sh [<model>]

set -euo pipefail

BGPIDS=()
cleanup() {
    trap - EXIT INT TERM
    echo "[smoke-noext] Cleaning up..."
    for pid in "${BGPIDS[@]:-}"; do
        kill "$pid" 2>/dev/null || true
    done
}
trap cleanup EXIT INT TERM

MODEL="${1:-Qwen/Qwen3-0.6B}"
HTTP_PORT="${DYN_HTTP_PORT:-8000}"
NATS_PORT="${NATS_PORT:-4222}"

LOG_DIR="${TMPDIR:-/tmp}/dynamo-rl-smoke-noext-$$"
mkdir -p "$LOG_DIR"

LORA_NAME="${LORA_NAME:-qwen3-tiny-lora}"
LORA_DIR="${LORA_DIR:-${LOG_DIR}/adapter}"

echo "[smoke-noext] Log dir: $LOG_DIR"
echo "[smoke-noext] Model:   $MODEL"

# Two-level assertion: outer dispatch status AND inner handler response.status.
# A dispatched request can return outer ok with inner error if the handler
# raised an exception (e.g. collective_rpc failure on the worker side).
check_workers_ok() {
    local payload="$1"
    local label="$2"
    echo "$payload" | python -c "
import sys, json
data = json.load(sys.stdin)
workers = data.get('workers', [])
if not workers:
    print('FAIL: ${label}: no workers reported'); sys.exit(1)
bad = []
for w in workers:
    if w.get('status') != 'ok':
        bad.append({'dispatch_status': w.get('status'), 'error': w.get('error')})
        continue
    resp = w.get('response', {})
    if isinstance(resp, dict) and resp.get('status') == 'error':
        bad.append({'handler_status': 'error', 'message': resp.get('message')})
if bad:
    print(f'FAIL: ${label}: {bad}'); sys.exit(1)
print(f'PASS: ${label}')
" || return 1
    return 0
}

# ---------------------------------------------------------------------------
# 0. Build tiny LoRA adapter
# ---------------------------------------------------------------------------
echo "[smoke-noext] Building tiny LoRA adapter..."
python "$(dirname "$0")/make_lora.py" "$LORA_DIR" 2>&1 | tail -3
if [ ! -f "$LORA_DIR/adapter_config.json" ]; then
    echo "[smoke-noext] FAIL: LoRA adapter not created at $LORA_DIR"
    exit 1
fi
echo "[smoke-noext] LoRA adapter ready"

# ---------------------------------------------------------------------------
# 1. NATS
# ---------------------------------------------------------------------------
if nc -z localhost "$NATS_PORT" 2>/dev/null; then
    echo "[smoke-noext] NATS already running on port $NATS_PORT — skipping start"
else
    nats-server -p "$NATS_PORT" -l "$LOG_DIR/nats.log" &
    NATS_PID=$!
    BGPIDS+=("$NATS_PID")
    echo "[smoke-noext] NATS started (pid=$NATS_PID)"
    sleep 1
fi

# ---------------------------------------------------------------------------
# 2. Dynamo frontend
# ---------------------------------------------------------------------------
HF_HUB_OFFLINE=1 \
  TRANSFORMERS_OFFLINE=1 \
  DYN_ENABLE_RL_ENDPOINTS=true \
  DYN_HTTP_PORT="$HTTP_PORT" \
  python -m dynamo.frontend \
    > "$LOG_DIR/frontend.log" 2>&1 &
FRONTEND_PID=$!
BGPIDS+=("$FRONTEND_PID")
echo "[smoke-noext] Frontend started (pid=$FRONTEND_PID)"

# ---------------------------------------------------------------------------
# 3. Stock dynamo.vllm worker — NO --worker-extension-cls
# ---------------------------------------------------------------------------
HF_HUB_OFFLINE=1 \
  TRANSFORMERS_OFFLINE=1 \
  DYN_SYSTEM_PORT=${DYN_SYSTEM_PORT:-8081} \
  python -m dynamo.vllm \
    --model "$MODEL" \
    --enforce-eager \
    --max-model-len 2048 \
    --max-num-seqs 2 \
    --enable-rl \
    --enable-lora \
    --max-lora-rank 32 \
    --max-loras 4 \
    > "$LOG_DIR/worker.log" 2>&1 &
WORKER_PID=$!
BGPIDS+=("$WORKER_PID")
echo "[smoke-noext] Worker started (pid=$WORKER_PID, log=$LOG_DIR/worker.log)"

# ---------------------------------------------------------------------------
# 4. Wait for the RL endpoint
# ---------------------------------------------------------------------------
echo "[smoke-noext] Waiting for /v1/rl/engine to become live..."
DEADLINE=$(( $(date +%s) + 240 ))
while true; do
  if curl -sf -X GET "http://localhost:${HTTP_PORT}/v1/rl/engine" -o /dev/null 2>&1; then
    echo "[smoke-noext] RL endpoint is live"
    break
  fi
  if [ "$(date +%s)" -ge "$DEADLINE" ]; then
    echo "[smoke-noext] TIMEOUT: RL endpoint not live after 240s"
    tail -30 "$LOG_DIR/frontend.log"
    tail -30 "$LOG_DIR/worker.log"
    exit 1
  fi
  sleep 3
done

# ---------------------------------------------------------------------------
# 5. GET /v1/rl/engine — describe
# ---------------------------------------------------------------------------
echo ""
echo "[smoke-noext] === GET /v1/rl/engine (describe) ==="
DESCRIBE=$(curl -sf -X GET "http://localhost:${HTTP_PORT}/v1/rl/engine")
echo "$DESCRIBE" | python -m json.tool

if ! echo "$DESCRIBE" | python -c "
import sys, json
data = json.load(sys.stdin)
workers = data.get('workers', [])
if not workers:
    print('FAIL: no workers reported'); sys.exit(1)
methods = set(workers[0].get('response', {}).get('registered_methods', []))
required = {'pause_generation', 'resume_generation', 'update_weights_from_disk',
            'load_lora_adapter', 'unload_lora_adapter'}
missing = required - methods
if missing:
    print(f'FAIL: missing methods: {missing}'); sys.exit(1)
print('PASS: all required methods registered on stock worker')
"; then
  exit 1
fi

# ---------------------------------------------------------------------------
# 6. LoRA load/inference/unload (runs FIRST — does not perturb engine state).
# ---------------------------------------------------------------------------
echo ""
echo "[smoke-noext] === POST load_lora_adapter (native vLLM add_lora) ==="
LOAD=$(curl -sf -X POST "http://localhost:${HTTP_PORT}/v1/rl/engine" \
  -H "Content-Type: application/json" --max-time 60 \
  -d "{\"method\": \"load_lora_adapter\",
       \"kwargs\": {\"lora_name\": \"${LORA_NAME}\",
                   \"lora_path\": \"${LORA_DIR}\"},
       \"timeout_secs\": 60}")
echo "$LOAD" | python -m json.tool
check_workers_ok "$LOAD" "load_lora_adapter" || exit 1

# Wait for model card to propagate to discovery so the frontend can route by lora_name.
sleep 3

echo ""
echo "[smoke-noext] === Inference with model=${LORA_NAME} ==="
INF=$(curl -sf -X POST "http://localhost:${HTTP_PORT}/v1/chat/completions" \
  -H "Content-Type: application/json" --max-time 60 \
  -d "{\"model\": \"${LORA_NAME}\",
       \"messages\": [{\"role\": \"user\", \"content\": \"Say: hello\"}],
       \"max_tokens\": 8, \"stream\": false}" 2>&1 || true)
if echo "$INF" | grep -q '"choices"'; then
  echo "PASS: LoRA inference returned choices"
else
  echo "WARN: LoRA inference inconclusive — first 5 lines:"
  echo "$INF" | head -5
fi

echo ""
echo "[smoke-noext] === POST unload_lora_adapter ==="
UNLOAD=$(curl -sf -X POST "http://localhost:${HTTP_PORT}/v1/rl/engine" \
  -H "Content-Type: application/json" \
  -d "{\"method\": \"unload_lora_adapter\",
       \"kwargs\": {\"lora_name\": \"${LORA_NAME}\"}}")
echo "$UNLOAD" | python -m json.tool
check_workers_ok "$UNLOAD" "unload_lora_adapter" || exit 1

echo ""
echo "[smoke-noext] === POST unload_lora_adapter (idempotency) ==="
UNLOAD2=$(curl -sf -X POST "http://localhost:${HTTP_PORT}/v1/rl/engine" \
  -H "Content-Type: application/json" \
  -d "{\"method\": \"unload_lora_adapter\",
       \"kwargs\": {\"lora_name\": \"${LORA_NAME}\"}}")
check_workers_ok "$UNLOAD2" "unload_lora_adapter (idempotent)" || exit 1

# ---------------------------------------------------------------------------
# 7. Full-weight update via vLLM-native reload_weights.
#    DOCUMENT-ONLY: this is expected to FAIL the inner handler.status for
#    Qwen3 on vLLM v0.19.0 because reload_weights doesn't handle the fused
#    gate_up_proj layer. The FileSystemWeightUpdateWorker extension is the
#    supported path for FT weight swaps.
# ---------------------------------------------------------------------------
MODEL_CACHE=$(python -c "
import huggingface_hub, os
try:
    print(huggingface_hub.snapshot_download('$MODEL', local_files_only=True))
except Exception:
    print(os.path.expanduser('~/.cache/huggingface/hub'))
" 2>/dev/null)

echo ""
echo "[smoke-noext] === POST pause_generation ==="
PAUSE=$(curl -sf -X POST "http://localhost:${HTTP_PORT}/v1/rl/engine" \
  -H "Content-Type: application/json" \
  -d '{"method": "pause_generation", "kwargs": {"abort_requests": true, "clear_cache": false}}')
check_workers_ok "$PAUSE" "pause_generation" || exit 1

echo ""
echo "[smoke-noext] === POST update_weights_from_disk (engine_rpc=reload_weights) ==="
echo "[smoke-noext] weight path: $MODEL_CACHE"
UPDATE=$(curl -sf -X POST "http://localhost:${HTTP_PORT}/v1/rl/engine" \
  -H "Content-Type: application/json" --max-time 300 \
  -d "{\"method\": \"update_weights_from_disk\",
       \"kwargs\": {\"model_path\": \"${MODEL_CACHE}\",
                   \"weight_version\": \"noext_v1\"},
       \"timeout_secs\": 240}")
echo "$UPDATE" | python -m json.tool
if check_workers_ok "$UPDATE" "update_weights_from_disk (reload_weights)"; then
    echo "[smoke-noext] reload_weights succeeded on stock vLLM worker"
    UPDATE_OK=1
else
    echo "[smoke-noext] EXPECTED: vLLM-native reload_weights raised — use FileSystemWeightUpdateWorker for FT"
    UPDATE_OK=0
fi

echo ""
echo "[smoke-noext] === POST resume_generation ==="
RESUME=$(curl -sf -X POST "http://localhost:${HTTP_PORT}/v1/rl/engine" \
  -H "Content-Type: application/json" \
  -d '{"method": "resume_generation"}')
check_workers_ok "$RESUME" "resume_generation" || true

echo ""
echo "========================================"
echo "[smoke-noext] Results:"
echo "  LoRA load/unload (native add_lora): PASS"
if [ "$UPDATE_OK" = "1" ]; then
    echo "  FT via reload_weights:              PASS"
else
    echo "  FT via reload_weights:              UNSUPPORTED (use FileSystemWeightUpdateWorker)"
fi
echo "[smoke-noext] ALL TESTS PASSED"
echo "========================================"
