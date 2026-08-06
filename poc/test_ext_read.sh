#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Does the engine's KV persist in GMS memory + read back by an INDEPENDENT external
# client? Single shadow-mode engine (winner write path) writes KV, sleeps, external
# minimal client imports the same allocation and fingerprints it. If MISMATCH, wake the
# engine and check whether IT re-reads its own KV correctly (physical intact vs corrupt).
set -u
MODEL_NAME="${1:-Qwen/Qwen3-0.6B}"
LOG_DIR="/tmp/extread_$$"; mkdir -p "$LOG_DIR"
LOCK_PATH="$LOG_DIR/failover.lock"
ENGINE_LOG="$LOG_DIR/engine.log"; FRONTEND_LOG="$LOG_DIR/frontend.log"
GMS_W_LOG="$LOG_DIR/gms_w.log"; GMS_KV_LOG="$LOG_DIR/gms_kv.log"
SYS_PORT=8100
export PYTHONHASHSEED=0
export GMS_KV_INDEX_PATH="$LOG_DIR/kv_index.log"
export GMS_KV_TARGET_FILE="$LOG_DIR/kv_target.json"
GMS_PIDS=(); ENGINE_PID=""; FRONTEND_PID=""
pass_count=0; fail_count=0
pass(){ pass_count=$((pass_count+1)); echo "  PASS: $1"; }
fail(){ fail_count=$((fail_count+1)); echo "  FAIL: $1"; }
strip_ansi(){ sed 's/\x1b\[[0-9;]*m//g'; }
have(){ cat "$1" 2>/dev/null | strip_ansi | grep -q "$2"; }
cleanup(){
  echo ""; echo "=== Cleaning up ==="
  for pid in "$FRONTEND_PID" "$ENGINE_PID" "${GMS_PIDS[@]}"; do
    [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null && { pkill -9 -P "$pid" 2>/dev/null; kill -9 "$pid" 2>/dev/null; }
  done
  pkill -9 -f "dynamo[.]vllm" 2>/dev/null; pkill -9 -f "dynamo[.]frontend" 2>/dev/null
  pkill -9 -f "[g]pu_memory_service" 2>/dev/null; pkill -9 -f "[E]ngineCore" 2>/dev/null
  sleep 2
  for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do kill -9 "$p" 2>/dev/null; done
  rm -f /tmp/gms_*.sock 2>/dev/null
  echo "  Results: $pass_count passed, $fail_count failed"
}
trap cleanup EXIT
# preflight sweep
pkill -9 -f "dynamo[.]vllm" 2>/dev/null; pkill -9 -f "[E]ngineCore" 2>/dev/null; pkill -9 -f "[g]pu_memory_service" 2>/dev/null
for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do kill -9 "$p" 2>/dev/null; done
rm -f /tmp/gms_*.sock 2>/dev/null; sleep 2
echo "preflight gpu: $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits) MiB"
wait_for(){ local log="$1" pat="$2" to="$3" pid="$4" desc="$5"
  for i in $(seq 1 "$to"); do have "$log" "$pat" && return 0
    if [ -n "$pid" ] && ! kill -0 "$pid" 2>/dev/null; then echo "ERR died: $desc"; tail -30 "$log"|strip_ansi; return 1; fi
    sleep 1; done; echo "ERR timeout: $desc"; tail -30 "$log"|strip_ansi; return 1; }
REUSE_PROMPT=""; build_reuse_prompt(){ local i s=""; for i in $(seq 1 240); do s+="Section $i: the quick brown fox jumps over the lazy dog near the riverbank. "; done; REUSE_PROMPT="$s"; }
infer(){ curl -s -X POST http://localhost:8000/v1/completions -H "Content-Type: application/json" \
  -d "$(python3 -c 'import json,sys;print(json.dumps({"model":sys.argv[1],"prompt":sys.argv[2],"max_tokens":4,"temperature":0}))' "$MODEL_NAME" "$REUSE_PROMPT")"; }

echo "=== Phase 0: GMS servers (weights + kv_cache/persist-on-abort) ==="
python3 -m gpu_memory_service --device 0 --tag weights  > "$GMS_W_LOG"  2>&1 & GMS_PIDS+=($!)
python3 -m gpu_memory_service --device 0 --tag kv_cache --persist-on-abort > "$GMS_KV_LOG" 2>&1 & GMS_PIDS+=($!)
wait_for "$GMS_W_LOG"  "Server started" 30 "" "weights gms"  || exit 1
wait_for "$GMS_KV_LOG" "Server started" 30 "" "kv gms"       || exit 1

echo "=== Phase 1: single SHADOW-mode engine (winner write path) ==="
ENGINE_ID=0 FAILOVER_LOCK_PATH="$LOCK_PATH" DYN_SYSTEM_PORT=$SYS_PORT \
VLLM_NIXL_SIDE_CHANNEL_PORT=5600 DYN_VLLM_KV_EVENT_PORT=20080 \
python3 -m dynamo.vllm --model "$MODEL_NAME" -tp 1 \
  --gpu-memory-utilization 0.4 --num-gpu-blocks-override 8192 \
  --load-format gms --gms-shadow-mode > "$ENGINE_LOG" 2>&1 &
ENGINE_PID=$!
wait_for "$ENGINE_LOG" "Registered endpoint 'dynamo.backend.generate'" 300 "$ENGINE_PID" "engine wake+register" || exit 1
pass "shadow engine woke + registered"

echo "=== Phase 2: warm long prefix (model writes KV) ==="
python3 -m dynamo.frontend > "$FRONTEND_LOG" 2>&1 & FRONTEND_PID=$!
wait_for "$FRONTEND_LOG" "Completions is ready" 60 "$FRONTEND_PID" "frontend" || exit 1
build_reuse_prompt
infer >/dev/null; infer >/dev/null; sleep 2
pass "warmed prefix (2 infers)"

echo "=== Phase 3: sleep (dumps STABLE KV fingerprint pre-unmap, then unmaps; persist keeps it) ==="
curl -s -X POST "http://localhost:$SYS_PORT/engine/control/sleep" -H "Content-Type: application/json" -d '{"level":1}'; echo
for i in $(seq 1 20); do have "$ENGINE_LOG" "Sleep freed" && break; sleep 1; done
have "$ENGINE_LOG" "Sleep freed" && pass "engine slept" || fail "sleep not observed"
echo "  in-process cross-check (winner fresh-import vs model tensor):"
grep -iE "WINNER fresh-import" "$ENGINE_LOG" | strip_ansi | tail -1
[ -f "$GMS_KV_TARGET_FILE" ] && { pass "KV target dumped"; echo "  target: $(cat "$GMS_KV_TARGET_FILE")"; } || { fail "no KV target dumped"; grep -iE "fresh-import|dump|pre-sleep" "$ENGINE_LOG"|strip_ansi|tail; exit 1; }

echo "=== Phase 4: INDEPENDENT external client imports + fingerprints the KV allocation ==="
PYTHONPATH=/tmp/gmsoverride python3 /tmp/gms_ext_read.py "$GMS_KV_TARGET_FILE" 2>&1 | tee "$LOG_DIR/ext.log"
if grep -q "RESULT MATCH" "$LOG_DIR/ext.log"; then
  pass "external client read SAME fingerprint -> KV persisted + externally readable (points reader-side)"
else
  fail "external client MISMATCH -> KV not externally visible"
  echo "=== Phase 5: does the SLEPT engine itself re-read its own KV on wake? ==="
  curl -s -X POST "http://localhost:$SYS_PORT/engine/control/wake_up" -H "Content-Type: application/json" -d '{}'; echo
  sleep 6
  echo "  engine reattach fingerprint (83k=intact, 17M=corrupt):"
  grep -iE "reattached KV:" "$ENGINE_LOG" | strip_ansi | tail -1
fi
echo ""; echo "  TEST COMPLETE"
