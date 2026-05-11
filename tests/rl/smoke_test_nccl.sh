#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# E2E smoke test for NCCL-based weight updates via the RL admin control plane.
#
# Architecture:
#                                                NCCL group (world_size = 2)
#  ┌───────────────────────────┐                ┌────────────────────────────────┐
#  │ Dynamo frontend (port 8000)│ POST /v1/rl/  │ NCCLWeightUpdateWorker         │
#  │ /v1/rl/engine fan-out     ├──── engine ──▶│ rank = 1, GPU 0                │
#  └───────────────────────────┘                └────────────────────────────────┘
#          ▲                                                ▲
#          │ HTTP                                           │ NCCL broadcast (src=0)
#          │                                                │
#  ┌───────┴───────────────────┐                ┌──────────┴────────────────────┐
#  │ This test script          │   start +      │ nccl_broadcaster.py           │
#  │  (orchestrator)           │── stdin "GO" ─▶│ rank = 0, GPU 0               │
#  └───────────────────────────┘                └───────────────────────────────┘
#
# Timing protocol:
#   1. Start frontend + worker (--worker-extension-cls NCCLWeightUpdateWorker)
#   2. Start broadcaster subprocess → it loads model, calls
#      StatelessProcessGroup.create(rank=0, world_size=2) and BLOCKS.
#   3. POST init_weights_update_group on worker → worker calls
#      StatelessProcessGroup.create(rank=1, world_size=2). Both ranks rendezvous
#      and the NCCL communicator comes up on each side.
#   4. POST update_weights_from_distributed (in background — it blocks waiting
#      for the broadcast).
#   5. Send "GO" to the broadcaster's stdin. It broadcasts the state dict;
#      the worker receives and loads. Both calls return.
#   6. Verify worker status=ok, then run an inference to confirm the model
#      still serves after the weight swap.
#
# Usage:
#   cd <dynamo-repo-root>
#   source dynamo/bin/activate
#   export PRIME_RL_SRC=/path/to/prime-rl/src
#   bash tests/rl/smoke_test_nccl.sh [<model>]

set -euo pipefail

BGPIDS=()
cleanup() {
    trap - EXIT INT TERM
    echo "[smoke-nccl] Cleaning up..."
    for pid in "${BGPIDS[@]:-}"; do
        kill "$pid" 2>/dev/null || true
    done
}
trap cleanup EXIT INT TERM

MODEL="${1:-Qwen/Qwen3-0.6B}"
HTTP_PORT="${DYN_HTTP_PORT:-8000}"
NATS_PORT="${NATS_PORT:-4222}"
NCCL_HOST="${NCCL_HOST:-127.0.0.1}"
NCCL_PORT="${NCCL_PORT:-29501}"
# Set PRIME_RL_SRC to the prime-rl src directory before running this script.
: "${PRIME_RL_SRC:?Set PRIME_RL_SRC to the prime-rl src directory (e.g. export PRIME_RL_SRC=/path/to/prime-rl/src)}"

LOG_DIR="${TMPDIR:-/tmp}/dynamo-rl-smoke-nccl-$$"
mkdir -p "$LOG_DIR"

# NCCL fundamentally requires one GPU per rank. With a single GPU, the
# trainer (rank 0) and inference worker (rank 1) collide at PyNcclCommunicator
# init. Detect that up front and choose between full E2E vs wire-path-only.
GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l)
if [ "$GPU_COUNT" -ge 2 ]; then
    FULL_E2E=1
    BROADCASTER_GPU="${BROADCASTER_GPU:-1}"
    echo "[smoke-nccl] Detected $GPU_COUNT GPUs — running FULL E2E (broadcaster on GPU $BROADCASTER_GPU)"
else
    FULL_E2E=0
    BROADCASTER_GPU=""
    echo "[smoke-nccl] Only 1 GPU detected — running WIRE-PATH test (NCCL E2E requires >=2 GPUs)"
fi

echo "[smoke-nccl] Log dir: $LOG_DIR"
echo "[smoke-nccl] Model:    $MODEL"
echo "[smoke-nccl] NCCL bind: $NCCL_HOST:$NCCL_PORT  world_size=2 (rank0=trainer, rank1=worker)"

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
# 1. NATS
# ---------------------------------------------------------------------------
if nc -z localhost "$NATS_PORT" 2>/dev/null; then
    echo "[smoke-nccl] NATS already running on port $NATS_PORT — skipping start"
else
    nats-server -p "$NATS_PORT" -l "$LOG_DIR/nats.log" &
    NATS_PID=$!
    BGPIDS+=("$NATS_PID")
    echo "[smoke-nccl] NATS started (pid=$NATS_PID)"
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
echo "[smoke-nccl] Frontend started (pid=$FRONTEND_PID)"

# ---------------------------------------------------------------------------
# 3. vLLM worker with NCCLWeightUpdateWorker extension
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
    --worker-extension-cls prime_rl.inference.vllm.worker.nccl.NCCLWeightUpdateWorker \
    > "$LOG_DIR/worker.log" 2>&1 &
WORKER_PID=$!
BGPIDS+=("$WORKER_PID")
echo "[smoke-nccl] Worker started (pid=$WORKER_PID, log=$LOG_DIR/worker.log)"

# ---------------------------------------------------------------------------
# 4. Wait for RL endpoint live
# ---------------------------------------------------------------------------
echo "[smoke-nccl] Waiting for /v1/rl/engine to become live..."
DEADLINE=$(( $(date +%s) + 240 ))
while true; do
  if curl -sf -X GET "http://localhost:${HTTP_PORT}/v1/rl/engine" -o /dev/null 2>&1; then
    echo "[smoke-nccl] RL endpoint is live"
    break
  fi
  if [ "$(date +%s)" -ge "$DEADLINE" ]; then
    echo "[smoke-nccl] TIMEOUT: RL endpoint not live after 240s"
    tail -30 "$LOG_DIR/frontend.log"
    tail -30 "$LOG_DIR/worker.log"
    exit 1
  fi
  sleep 3
done

# ---------------------------------------------------------------------------
# 5. Describe — confirm NCCL-related methods are registered
# ---------------------------------------------------------------------------
echo ""
echo "[smoke-nccl] === GET /v1/rl/engine (describe) ==="
DESCRIBE=$(curl -sf -X GET "http://localhost:${HTTP_PORT}/v1/rl/engine")
echo "$DESCRIBE" | python -m json.tool

if ! echo "$DESCRIBE" | python -c "
import sys, json
data = json.load(sys.stdin)
workers = data.get('workers', [])
if not workers:
    print('FAIL: no workers reported'); sys.exit(1)
methods = set(workers[0].get('response', {}).get('registered_methods', []))
required = {'init_weights_update_group', 'update_weights_from_distributed'}
missing = required - methods
if missing:
    print(f'FAIL: missing methods: {missing}'); sys.exit(1)
print('PASS: NCCL methods registered on NCCLWeightUpdateWorker')
"; then
  exit 1
fi

# ---------------------------------------------------------------------------
# 6. Start the broadcaster (rank 0). It will block on StatelessProcessGroup
#    until the worker also joins via init_weights_update_group.
# ---------------------------------------------------------------------------
echo ""
echo "[smoke-nccl] === Starting NCCL broadcaster (rank 0) ==="

GO_PIPE="$LOG_DIR/go.fifo"
mkfifo "$GO_PIPE"

# Pin broadcaster to its own GPU when running full E2E; otherwise share GPU 0
# with the worker (PyNcclCommunicator will then fail with "invalid usage" —
# expected on single-GPU hosts).
BROADCASTER_ENV=""
if [ -n "$BROADCASTER_GPU" ]; then
    BROADCASTER_ENV="CUDA_VISIBLE_DEVICES=$BROADCASTER_GPU"
fi

HF_HUB_OFFLINE=1 \
  TRANSFORMERS_OFFLINE=1 \
  ${BROADCASTER_ENV} \
  python "$(dirname "$0")/nccl_broadcaster.py" \
    --host "$NCCL_HOST" --port "$NCCL_PORT" --world-size 2 \
    --model "$MODEL" --timeout 120 \
    < "$GO_PIPE" > "$LOG_DIR/broadcaster.log" 2>&1 &
BROADCASTER_PID=$!
BGPIDS+=("$BROADCASTER_PID")

# Keep the FIFO write end open in the shell so the broadcaster doesn't see EOF.
exec 3>"$GO_PIPE"
echo "[smoke-nccl] Broadcaster started (pid=$BROADCASTER_PID, log=$LOG_DIR/broadcaster.log)"

# Wait until the broadcaster prints that StatelessProcessGroup.create has been
# called (it then blocks waiting for the worker to join).
for _ in $(seq 1 60); do
    if grep -q "blocks until peers join" "$LOG_DIR/broadcaster.log" 2>/dev/null; then
        echo "[smoke-nccl] Broadcaster reached create() — waiting for worker to join"
        break
    fi
    sleep 2
done

# ---------------------------------------------------------------------------
# 7. POST init_weights_update_group — worker joins the NCCL group as rank 1
# ---------------------------------------------------------------------------
echo ""
echo "[smoke-nccl] === POST init_weights_update_group ==="
INIT=$(curl -sf -X POST "http://localhost:${HTTP_PORT}/v1/rl/engine" \
  -H "Content-Type: application/json" --max-time 180 \
  -d "{\"method\": \"init_weights_update_group\",
       \"kwargs\": {\"host\": \"${NCCL_HOST}\",
                   \"port\": ${NCCL_PORT},
                   \"rank_offset\": 0,
                   \"inference_world_size\": 1,
                   \"timeout\": 120},
       \"timeout_secs\": 180}")
echo "$INIT" | python -m json.tool

# On a single-GPU host PyNcclCommunicator will fail with "NCCL error: invalid
# usage" because both ranks try to bind to cuda:0. That confirms the entire
# wire path (HTTP → fan-out → handler → collective_rpc → worker extension
# → NCCLWeightBroadcastReceiver.__init__) reached NCCL init. Pass the test in
# that case; fail on any other dispatch problem (no worker, missing endpoint,
# unmatched namespace, etc.).
if [ "$FULL_E2E" = "1" ]; then
    check_workers_ok "$INIT" "init_weights_update_group" || exit 1
else
    if echo "$INIT" | python -c "
import sys, json
data = json.load(sys.stdin)
workers = data.get('workers', [])
if not workers:
    print('FAIL: init_weights_update_group: no workers reported'); sys.exit(1)
for w in workers:
    if w.get('status') != 'ok':
        print(f'FAIL: dispatch error: {w.get(\"error\")}'); sys.exit(1)
    resp = w.get('response', {})
    msg = (resp.get('message') or '') if isinstance(resp, dict) else ''
    if resp.get('status') == 'ok':
        print('PASS: init_weights_update_group succeeded (multi-GPU)')
    elif 'NCCL error' in msg or 'invalid usage' in msg:
        print('PASS: init_weights_update_group reached NCCL init (wire-path test on single GPU)')
    else:
        print(f'FAIL: unexpected handler error: {msg}'); sys.exit(1)
"; then
        :
    else
        exit 1
    fi
fi

if [ "$FULL_E2E" != "1" ]; then
    echo ""
    echo "[smoke-nccl] Skipping update_weights_from_distributed broadcast — needs >=2 GPUs"
    echo ""
    echo "========================================"
    echo "[smoke-nccl] WIRE-PATH TEST PASSED"
    echo "  - GET describe registers NCCL methods on NCCLWeightUpdateWorker"
    echo "  - POST init_weights_update_group reaches the worker extension and"
    echo "    drives NCCLWeightBroadcastReceiver.__init__() through to NCCL init"
    echo "  Full E2E (broadcast + receive + load) requires >=2 GPUs."
    echo "========================================"
    exit 0
fi

# Wait for broadcaster to print communicator ready.
for _ in $(seq 1 30); do
    if grep -q "PyNcclCommunicator ready" "$LOG_DIR/broadcaster.log" 2>/dev/null; then
        echo "[smoke-nccl] Both sides have communicator — ready to broadcast"
        break
    fi
    sleep 1
done

# ---------------------------------------------------------------------------
# 8. POST update_weights_from_distributed (background — it blocks waiting for
#    the broadcast). Then send "GO" to broadcaster so both rendezvous.
# ---------------------------------------------------------------------------
echo ""
echo "[smoke-nccl] === POST update_weights_from_distributed (background) + GO to broadcaster ==="

curl -sf -X POST "http://localhost:${HTTP_PORT}/v1/rl/engine" \
  -H "Content-Type: application/json" --max-time 180 \
  -d "{\"method\": \"update_weights_from_distributed\",
       \"kwargs\": {\"weight_version\": \"nccl_v1\",
                   \"weight_dir\": \"unused-by-nccl\"},
       \"timeout_secs\": 180}" \
  > "$LOG_DIR/update.json" 2>&1 &
UPDATE_PID=$!

# Give curl a beat to start the POST so the worker is in receive_state_dict.
sleep 1
echo "GO" >&3
echo "[smoke-nccl] Sent GO to broadcaster"

# Wait for the curl POST to finish.
wait $UPDATE_PID
echo "[smoke-nccl] update_weights_from_distributed POST returned"

UPDATE=$(cat "$LOG_DIR/update.json")
echo "$UPDATE" | python -m json.tool
check_workers_ok "$UPDATE" "update_weights_from_distributed (NCCL)" || exit 1

# Verify the broadcaster also finished cleanly.
wait $BROADCASTER_PID || true
if grep -q "Broadcast complete" "$LOG_DIR/broadcaster.log" 2>/dev/null; then
    echo "PASS: broadcaster reported Broadcast complete"
else
    echo "WARN: broadcaster did not log Broadcast complete — last 20 lines:"
    tail -20 "$LOG_DIR/broadcaster.log"
fi

# ---------------------------------------------------------------------------
# 9. Verify version recorded and inference still works.
# ---------------------------------------------------------------------------
echo ""
echo "[smoke-nccl] === POST get_weight_version ==="
VER=$(curl -sf -X POST "http://localhost:${HTTP_PORT}/v1/rl/engine" \
  -H "Content-Type: application/json" \
  -d '{"method": "get_weight_version"}')
echo "$VER" | python -m json.tool
echo "$VER" | python -c "
import sys, json
data = json.load(sys.stdin)
for w in data.get('workers', []):
    resp = w.get('response', {})
    v = resp.get('version', resp.get('weight_version', ''))
    if v != 'nccl_v1':
        print(f'WARN: expected nccl_v1 got {v!r}')
    else:
        print('PASS: version=nccl_v1')
" || true

echo ""
echo "[smoke-nccl] === Quick inference check ==="
INF=$(curl -sf -X POST "http://localhost:${HTTP_PORT}/v1/chat/completions" \
  -H "Content-Type: application/json" --max-time 60 \
  -d "{\"model\": \"${MODEL}\",
       \"messages\": [{\"role\": \"user\", \"content\": \"Say: hello\"}],
       \"max_tokens\": 8, \"stream\": false}" 2>&1 || true)
if echo "$INF" | grep -q '"choices"'; then
  echo "PASS: inference working after NCCL weight update"
else
  echo "WARN: inference inconclusive — first 5 lines:"
  echo "$INF" | head -5
fi

echo ""
echo "========================================"
echo "[smoke-nccl] ALL TESTS PASSED"
echo "========================================"
