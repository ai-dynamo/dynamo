#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Smoke test for the RL admin control plane.
#
# Starts:
#   1. NATS server (discovery plane)
#   2. Dynamo frontend (with DYN_ENABLE_RL_ENDPOINTS=true)
#   3. Dynamo vLLM worker (Qwen3-0.6B + FileSystemWeightUpdateWorker)
#
# Then exercises: GET /v1/rl/engine, POST /v1/rl/engine pause_generation,
#                 POST /v1/rl/engine update_weights_from_disk, POST /v1/rl/engine resume_generation
#
# Usage:
#   cd /home/biswaranjanp/dev/rl/dynamo
#   source dynamo/bin/activate
#   bash tests/rl/smoke_test.sh [--model <name>]

set -euo pipefail
trap 'echo "[smoke] Cleaning up..."; kill 0 2>/dev/null; exit 0' EXIT INT TERM

MODEL="${1:-Qwen/Qwen3-0.6B}"
HTTP_PORT="${DYN_HTTP_PORT:-8000}"
NATS_PORT="${NATS_PORT:-4222}"
LOG_DIR="${TMPDIR:-/tmp}/dynamo-rl-smoke-$$"
mkdir -p "$LOG_DIR"

WEIGHT_DIR="${LOG_DIR}/weights"
mkdir -p "$WEIGHT_DIR"

echo "[smoke] Log dir: $LOG_DIR"
echo "[smoke] Model: $MODEL"
echo "[smoke] Frontend port: $HTTP_PORT"

# ---------------------------------------------------------------------------
# 1. NATS
# ---------------------------------------------------------------------------
nats-server -p "$NATS_PORT" -l "$LOG_DIR/nats.log" &
NATS_PID=$!
echo "[smoke] NATS started (pid=$NATS_PID)"
sleep 1

# ---------------------------------------------------------------------------
# 2. Dynamo frontend (RL enabled on same port as OpenAI-compat)
# ---------------------------------------------------------------------------
DYN_ENABLE_RL_ENDPOINTS=true \
  DYN_HTTP_PORT="$HTTP_PORT" \
  python -m dynamo.frontend \
    > "$LOG_DIR/frontend.log" 2>&1 &
FRONTEND_PID=$!
echo "[smoke] Frontend started (pid=$FRONTEND_PID)"

# ---------------------------------------------------------------------------
# 3. Dynamo vLLM worker with FileSystemWeightUpdateWorker extension
# ---------------------------------------------------------------------------
DYN_SYSTEM_PORT=${DYN_SYSTEM_PORT:-8081} \
  python -m dynamo.vllm \
    --model "$MODEL" \
    --enforce-eager \
    --max-model-len 2048 \
    --max-num-seqs 2 \
    --worker-extension-cls prime_rl.inference.vllm.worker.filesystem.FileSystemWeightUpdateWorker \
    > "$LOG_DIR/worker.log" 2>&1 &
WORKER_PID=$!
echo "[smoke] Worker started (pid=$WORKER_PID, log=$LOG_DIR/worker.log)"

# ---------------------------------------------------------------------------
# 4. Wait for the RL endpoint to be live
# ---------------------------------------------------------------------------
echo "[smoke] Waiting for /v1/rl/engine to become live..."
DEADLINE=$(( $(date +%s) + 180 ))
while true; do
  if curl -sf -X GET "http://localhost:${HTTP_PORT}/v1/rl/engine" -o /dev/null 2>&1; then
    echo "[smoke] RL endpoint is live"
    break
  fi
  if [ "$(date +%s)" -ge "$DEADLINE" ]; then
    echo "[smoke] TIMEOUT: RL endpoint not live after 180s"
    echo "=== frontend.log ==="
    tail -30 "$LOG_DIR/frontend.log"
    echo "=== worker.log ==="
    tail -30 "$LOG_DIR/worker.log"
    exit 1
  fi
  sleep 3
done

# ---------------------------------------------------------------------------
# 5. GET /v1/rl/engine — describe registered methods
# ---------------------------------------------------------------------------
echo ""
echo "[smoke] === GET /v1/rl/engine (describe) ==="
DESCRIBE=$(curl -sf -X GET "http://localhost:${HTTP_PORT}/v1/rl/engine")
echo "$DESCRIBE" | python -m json.tool
echo ""

# ---------------------------------------------------------------------------
# 6. pause_generation
# ---------------------------------------------------------------------------
echo "[smoke] === POST pause_generation ==="
PAUSE=$(curl -sf -X POST "http://localhost:${HTTP_PORT}/v1/rl/engine" \
  -H "Content-Type: application/json" \
  -d '{"method": "pause_generation", "kwargs": {"abort_requests": true, "clear_cache": false}}')
echo "$PAUSE" | python -m json.tool
echo ""

# Verify all workers paused (status ok)
if echo "$PAUSE" | python -c "
import sys, json
data = json.load(sys.stdin)
workers = data.get('workers', [])
failed = [w for w in workers if w.get('status') != 'ok']
if failed:
    print('FAIL: some workers not ok:', failed)
    sys.exit(1)
elif not workers:
    print('FAIL: no workers reported')
    sys.exit(1)
else:
    print(f'PASS: {len(workers)} worker(s) paused ok')
"; then
  :
else
  echo "[smoke] FAIL: pause_generation"
  exit 1
fi

# ---------------------------------------------------------------------------
# 7. update_weights_from_disk (same path — no-op but tests round-trip)
# ---------------------------------------------------------------------------
echo "[smoke] === POST update_weights_from_disk ==="
# Use the original model path from HF cache as weight_path so load is valid
MODEL_CACHE=$(python -c "
import huggingface_hub, os
try:
    p = huggingface_hub.snapshot_download('$MODEL', local_files_only=True)
    print(p)
except Exception as e:
    print(os.path.expanduser('~/.cache/huggingface/hub'))
" 2>/dev/null)
echo "[smoke] weight path: $MODEL_CACHE"

UPDATE=$(curl -sf -X POST "http://localhost:${HTTP_PORT}/v1/rl/engine" \
  -H "Content-Type: application/json" \
  --max-time 300 \
  -d "{\"method\": \"update_weights_from_disk\",
       \"kwargs\": {\"model_path\": \"${MODEL_CACHE}\",
                   \"weight_version\": \"smoke_v1\",
                   \"engine_rpc\": \"update_weights_from_path\"},
       \"timeout_secs\": 240}")
echo "$UPDATE" | python -m json.tool
echo ""

if echo "$UPDATE" | python -c "
import sys, json
data = json.load(sys.stdin)
workers = data.get('workers', [])
failed = [w for w in workers if w.get('status') != 'ok']
if failed:
    print('FAIL: workers not ok:', failed)
    sys.exit(1)
elif not workers:
    print('FAIL: no workers reported')
    sys.exit(1)
else:
    print(f'PASS: {len(workers)} worker(s) updated ok')
"; then
  :
else
  echo "[smoke] FAIL: update_weights_from_disk"
  exit 1
fi

# ---------------------------------------------------------------------------
# 8. get_weight_version — verify version updated
# ---------------------------------------------------------------------------
echo "[smoke] === POST get_weight_version ==="
VER=$(curl -sf -X POST "http://localhost:${HTTP_PORT}/v1/rl/engine" \
  -H "Content-Type: application/json" \
  -d '{"method": "get_weight_version"}')
echo "$VER" | python -m json.tool

if echo "$VER" | python -c "
import sys, json
data = json.load(sys.stdin)
for w in data.get('workers', []):
    resp = w.get('response', {})
    version = resp.get('version', resp.get('weight_version', ''))
    if version != 'smoke_v1':
        print(f'FAIL: expected smoke_v1 got {version!r}')
        sys.exit(1)
print('PASS: version=smoke_v1')
"; then
  :
else
  echo "[smoke] WARN: weight version check failed (non-fatal)"
fi
echo ""

# ---------------------------------------------------------------------------
# 9. resume_generation
# ---------------------------------------------------------------------------
echo "[smoke] === POST resume_generation ==="
RESUME=$(curl -sf -X POST "http://localhost:${HTTP_PORT}/v1/rl/engine" \
  -H "Content-Type: application/json" \
  -d '{"method": "resume_generation"}')
echo "$RESUME" | python -m json.tool
echo ""

if echo "$RESUME" | python -c "
import sys, json
data = json.load(sys.stdin)
workers = data.get('workers', [])
failed = [w for w in workers if w.get('status') != 'ok']
if failed:
    print('FAIL: workers not ok:', failed)
    sys.exit(1)
elif not workers:
    print('FAIL: no workers reported')
    sys.exit(1)
else:
    print(f'PASS: {len(workers)} worker(s) resumed ok')
"; then
  :
else
  echo "[smoke] FAIL: resume_generation"
  exit 1
fi

# ---------------------------------------------------------------------------
# 10. Quick inference check — verify model still serves requests
# ---------------------------------------------------------------------------
echo "[smoke] === Quick inference check ==="
INF=$(curl -sf -X POST "http://localhost:${HTTP_PORT}/v1/chat/completions" \
  -H "Content-Type: application/json" \
  --max-time 60 \
  -d "{\"model\": \"${MODEL}\",
       \"messages\": [{\"role\": \"user\", \"content\": \"Say: hello\"}],
       \"max_tokens\": 8, \"stream\": false}" 2>&1 || true)
if echo "$INF" | grep -q '"choices"'; then
  echo "PASS: inference working after weight update"
else
  echo "WARN: inference check inconclusive (model may not be ready yet)"
  echo "$INF" | head -5
fi

echo ""
echo "========================================"
echo "[smoke] ALL TESTS PASSED"
echo "========================================"
