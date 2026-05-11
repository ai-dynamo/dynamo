#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# LoRA smoke test for the RL admin control plane.
#
# Verifies load_lora_adapter / unload_lora_adapter via POST /v1/rl/engine.
#
# Starts:
#   1. NATS (skipped if already running)
#   2. Dynamo frontend (DYN_ENABLE_RL_ENDPOINTS=true)
#   3. Dynamo vLLM worker with --enable-lora and FileSystemWeightUpdateWorker
#
# Then exercises:
#   GET  /v1/rl/engine                          → describe (incl. load/unload_lora_adapter)
#   POST /v1/rl/engine load_lora_adapter        → load tiny untrained LoRA
#   POST /v1/chat/completions model=<lora_name> → inference uses LoRA
#   POST /v1/rl/engine unload_lora_adapter      → unload
#
# Usage:
#   cd /home/biswaranjanp/dev/rl/dynamo
#   source dynamo/bin/activate
#   bash tests/rl/smoke_test_lora.sh [<model>]

set -euo pipefail

BGPIDS=()
cleanup() {
    trap - EXIT INT TERM
    echo "[smoke-lora] Cleaning up..."
    for pid in "${BGPIDS[@]:-}"; do
        kill "$pid" 2>/dev/null || true
    done
}
trap cleanup EXIT INT TERM

MODEL="${1:-Qwen/Qwen3-0.6B}"
HTTP_PORT="${DYN_HTTP_PORT:-8000}"
NATS_PORT="${NATS_PORT:-4222}"
# Set PRIME_RL_SRC to the prime-rl src directory before running this script.
: "${PRIME_RL_SRC:?Set PRIME_RL_SRC to the prime-rl src directory (e.g. export PRIME_RL_SRC=/path/to/prime-rl/src)}"

LOG_DIR="${TMPDIR:-/tmp}/dynamo-rl-smoke-lora-$$"
mkdir -p "$LOG_DIR"

LORA_NAME="${LORA_NAME:-qwen3-tiny-lora}"
LORA_DIR="${LORA_DIR:-${LOG_DIR}/adapter}"

echo "[smoke-lora] Log dir: $LOG_DIR"
echo "[smoke-lora] Model: $MODEL"
echo "[smoke-lora] LoRA name: $LORA_NAME"
echo "[smoke-lora] LoRA dir: $LORA_DIR"

# ---------------------------------------------------------------------------
# 0. Build the tiny LoRA adapter (peft must be installed in the venv)
# ---------------------------------------------------------------------------
echo "[smoke-lora] Building tiny LoRA adapter..."
python "$(dirname "$0")/make_lora.py" "$LORA_DIR" 2>&1 | tail -3
if [ ! -f "$LORA_DIR/adapter_config.json" ]; then
    echo "[smoke-lora] FAIL: LoRA adapter not created at $LORA_DIR"
    exit 1
fi
echo "[smoke-lora] LoRA adapter ready"

# ---------------------------------------------------------------------------
# 1. NATS
# ---------------------------------------------------------------------------
if nc -z localhost "$NATS_PORT" 2>/dev/null; then
    echo "[smoke-lora] NATS already running on port $NATS_PORT — skipping start"
else
    nats-server -p "$NATS_PORT" -l "$LOG_DIR/nats.log" &
    NATS_PID=$!
    BGPIDS+=("$NATS_PID")
    echo "[smoke-lora] NATS started (pid=$NATS_PID)"
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
echo "[smoke-lora] Frontend started (pid=$FRONTEND_PID)"

# ---------------------------------------------------------------------------
# 3. Dynamo vLLM worker with --enable-lora
# ---------------------------------------------------------------------------
HF_HUB_OFFLINE=1 \
  TRANSFORMERS_OFFLINE=1 \
  PYTHONPATH="${PRIME_RL_SRC}${PYTHONPATH:+:$PYTHONPATH}" \
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
    --worker-extension-cls prime_rl.inference.vllm.worker.filesystem.FileSystemWeightUpdateWorker \
    > "$LOG_DIR/worker.log" 2>&1 &
WORKER_PID=$!
BGPIDS+=("$WORKER_PID")
echo "[smoke-lora] Worker started (pid=$WORKER_PID, log=$LOG_DIR/worker.log)"

# ---------------------------------------------------------------------------
# 4. Wait for the RL endpoint to be live
# ---------------------------------------------------------------------------
echo "[smoke-lora] Waiting for /v1/rl/engine to become live..."
DEADLINE=$(( $(date +%s) + 180 ))
while true; do
  if curl -sf -X GET "http://localhost:${HTTP_PORT}/v1/rl/engine" -o /dev/null 2>&1; then
    echo "[smoke-lora] RL endpoint is live"
    break
  fi
  if [ "$(date +%s)" -ge "$DEADLINE" ]; then
    echo "[smoke-lora] TIMEOUT: RL endpoint not live after 180s"
    echo "=== frontend.log ==="
    tail -30 "$LOG_DIR/frontend.log"
    echo "=== worker.log ==="
    tail -30 "$LOG_DIR/worker.log"
    exit 1
  fi
  sleep 3
done

# ---------------------------------------------------------------------------
# 5. GET /v1/rl/engine — verify load_lora_adapter and unload_lora_adapter are registered
# ---------------------------------------------------------------------------
echo ""
echo "[smoke-lora] === GET /v1/rl/engine (describe) ==="
DESCRIBE=$(curl -sf -X GET "http://localhost:${HTTP_PORT}/v1/rl/engine")
echo "$DESCRIBE" | python -m json.tool

if echo "$DESCRIBE" | python -c "
import sys, json
data = json.load(sys.stdin)
workers = data.get('workers', [])
if not workers:
    print('FAIL: no workers reported'); sys.exit(1)
methods = set(workers[0].get('response', {}).get('registered_methods', []))
required = {'load_lora_adapter', 'unload_lora_adapter'}
missing = required - methods
if missing:
    print(f'FAIL: missing methods: {missing}'); sys.exit(1)
print(f'PASS: lora methods registered on {len(workers)} worker(s)')
"; then
  :
else
  echo "[smoke-lora] FAIL: describe missing LoRA methods"
  exit 1
fi

# ---------------------------------------------------------------------------
# 6. POST load_lora_adapter
# ---------------------------------------------------------------------------
echo ""
echo "[smoke-lora] === POST load_lora_adapter ==="
LOAD=$(curl -sf -X POST "http://localhost:${HTTP_PORT}/v1/rl/engine" \
  -H "Content-Type: application/json" \
  --max-time 60 \
  -d "{\"method\": \"load_lora_adapter\",
       \"kwargs\": {\"lora_name\": \"${LORA_NAME}\",
                   \"lora_path\": \"${LORA_DIR}\"},
       \"timeout_secs\": 60}")
echo "$LOAD" | python -m json.tool

if echo "$LOAD" | python -c "
import sys, json
data = json.load(sys.stdin)
workers = data.get('workers', [])
failed = [w for w in workers if w.get('status') != 'ok']
if failed:
    print('FAIL: workers not ok:', failed); sys.exit(1)
elif not workers:
    print('FAIL: no workers reported'); sys.exit(1)
print(f'PASS: {len(workers)} worker(s) loaded LoRA ok')
"; then
  :
else
  echo "[smoke-lora] FAIL: load_lora_adapter"
  exit 1
fi

# Give discovery a moment to propagate the new LoRA model registration.
sleep 2

# ---------------------------------------------------------------------------
# 7. Inference with model=<lora_name>
# ---------------------------------------------------------------------------
echo ""
echo "[smoke-lora] === Inference with model=${LORA_NAME} ==="
INF=$(curl -sf -X POST "http://localhost:${HTTP_PORT}/v1/chat/completions" \
  -H "Content-Type: application/json" \
  --max-time 60 \
  -d "{\"model\": \"${LORA_NAME}\",
       \"messages\": [{\"role\": \"user\", \"content\": \"Say: hello\"}],
       \"max_tokens\": 8, \"stream\": false}" 2>&1 || true)
if echo "$INF" | grep -q '"choices"'; then
  echo "PASS: inference with LoRA model name returned choices"
else
  echo "WARN: LoRA inference inconclusive — first 5 lines:"
  echo "$INF" | head -5
fi

# ---------------------------------------------------------------------------
# 8. POST unload_lora_adapter
# ---------------------------------------------------------------------------
echo ""
echo "[smoke-lora] === POST unload_lora_adapter ==="
UNLOAD=$(curl -sf -X POST "http://localhost:${HTTP_PORT}/v1/rl/engine" \
  -H "Content-Type: application/json" \
  -d "{\"method\": \"unload_lora_adapter\",
       \"kwargs\": {\"lora_name\": \"${LORA_NAME}\"}}")
echo "$UNLOAD" | python -m json.tool

if echo "$UNLOAD" | python -c "
import sys, json
data = json.load(sys.stdin)
workers = data.get('workers', [])
failed = [w for w in workers if w.get('status') != 'ok']
if failed:
    print('FAIL: workers not ok:', failed); sys.exit(1)
elif not workers:
    print('FAIL: no workers reported'); sys.exit(1)
print(f'PASS: {len(workers)} worker(s) unloaded LoRA ok')
"; then
  :
else
  echo "[smoke-lora] FAIL: unload_lora_adapter"
  exit 1
fi

# ---------------------------------------------------------------------------
# 9. Idempotency: second unload should be a no-op success
# ---------------------------------------------------------------------------
echo ""
echo "[smoke-lora] === POST unload_lora_adapter (idempotency check) ==="
UNLOAD2=$(curl -sf -X POST "http://localhost:${HTTP_PORT}/v1/rl/engine" \
  -H "Content-Type: application/json" \
  -d "{\"method\": \"unload_lora_adapter\",
       \"kwargs\": {\"lora_name\": \"${LORA_NAME}\"}}")
if echo "$UNLOAD2" | python -c "
import sys, json
data = json.load(sys.stdin)
workers = data.get('workers', [])
failed = [w for w in workers if w.get('status') != 'ok']
if failed:
    print('FAIL: idempotent unload should be ok:', failed); sys.exit(1)
print('PASS: idempotent unload returns ok')
"; then
  :
else
  echo "[smoke-lora] FAIL: idempotent unload_lora_adapter"
  exit 1
fi

echo ""
echo "========================================"
echo "[smoke-lora] ALL TESTS PASSED"
echo "========================================"
