#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
#
# Test 1 (e2e, single engine): does the KV cache survive sleep/wake on ONE
# non-shadow engine, end to end through vLLM?
#
# This establishes the "byte floor" for KV reuse WITHOUT scratch, failover, or a
# second process: the engine allocates real KV (--enable-sleep-mode, not shadow),
# warms a long prefix, we HTTP-sleep it (unmap + GMS abort), HTTP-wake it (remap),
# and re-send the same prefix. Because the KV server runs --persist-on-abort, the
# bytes survive the sleep-disconnect and the wake reattaches them via the
# authentic-slot keyed remap (was_scratch=False -> no prepare_scratch flatten).
#
# Pass = the re-sent prefix HITs (prefix_hits bumps) with byte-identical output,
# proving persist + remap + the vLLM tensor rebinding are all sound in the simplest
# possible setting. Sleep/wake are driven via the engine system-port control routes:
#   POST /engine/control/sleep   {"level":1}
#   POST /engine/control/wake_up {}
set -u

MODEL_NAME="${1:-Qwen/Qwen3-0.6B}"
LOG_DIR="/tmp/test1_$$"; mkdir -p "$LOG_DIR"
ENGINE_LOG="$LOG_DIR/engine.log"; FRONTEND_LOG="$LOG_DIR/frontend.log"
GMS_W_LOG="$LOG_DIR/gms_w.log"; GMS_KV_LOG="$LOG_DIR/gms_kv.log"
SYS_PORT=8100

# Determinism + a shared index path (not strictly needed since it's one process that
# keeps its index, but harmless and mirrors the failover harness).
export PYTHONHASHSEED="${PYTHONHASHSEED:-0}"
export GMS_KV_INDEX_PATH="$LOG_DIR/kv_index.log"

GMS_PIDS=(); ENGINE_PID=""; FRONTEND_PID=""

pass_count=0; fail_count=0
pass() { pass_count=$((pass_count+1)); echo "  PASS: $1"; }
fail() { fail_count=$((fail_count+1)); echo "  FAIL: $1"; }
strip_ansi() { sed 's/\x1b\[[0-9;]*m//g'; }
have() { cat "$1" 2>/dev/null | strip_ansi | grep -q "$2"; }

cleanup() {
    echo ""; echo "=== Cleaning up ==="
    for pid in "$FRONTEND_PID" "$ENGINE_PID" "${GMS_PIDS[@]}"; do
        [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null && { pkill -9 -P "$pid" 2>/dev/null; kill -9 "$pid" 2>/dev/null; }
    done
    pkill -9 -f "dynamo[.]vllm" 2>/dev/null; pkill -9 -f "dynamo[.]frontend" 2>/dev/null
    pkill -9 -f "[g]pu_memory_service" 2>/dev/null; pkill -9 -f "[E]ngineCore" 2>/dev/null
    sleep 2
    for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do kill -9 "$p" 2>/dev/null; done
    rm -f /tmp/gms_*.sock 2>/dev/null
    echo "Logs: $LOG_DIR"
    echo "  Results: $pass_count passed, $fail_count failed"
    [ "$fail_count" -gt 0 ] && exit 1 || exit 0
}
trap cleanup EXIT

# Preflight: shared dev pod, sweep leftovers so we start clean.
pkill -9 -f "dynamo[.]vllm" 2>/dev/null; pkill -9 -f "dynamo[.]frontend" 2>/dev/null
pkill -9 -f "[g]pu_memory_service" 2>/dev/null; pkill -9 -f "[E]ngineCore" 2>/dev/null
for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do kill -9 "$p" 2>/dev/null; done
rm -f /tmp/gms_*.sock 2>/dev/null; sleep 2
echo "Preflight GPU used: $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits) MiB"

wait_for() {  # wait_for <log> <pattern> <timeout_s> <pid> <desc>
    local log="$1" pat="$2" to="$3" pid="$4" desc="$5"
    for i in $(seq 1 "$to"); do
        have "$log" "$pat" && return 0
        if [ -n "$pid" ] && ! kill -0 "$pid" 2>/dev/null; then
            echo "ERROR: process died waiting for: $desc"; tail -40 "$log" | strip_ansi; return 1
        fi
        sleep 1
    done
    echo "ERROR: timeout ${to}s waiting for: $desc"; tail -40 "$log" | strip_ansi; return 1
}

# ---- KV-reuse probe helpers (same as the failover harness) ----
REUSE_PROMPT=""
build_reuse_prompt() {
    local i s=""
    for i in $(seq 1 240); do
        s+="Section $i: the quick brown fox jumps over the lazy dog near the riverbank. "
    done
    REUSE_PROMPT="$s"
}
# echo "<time_total_ms>|<text>|<http_code>"; max_tokens=8, greedy.
timed_completion() {
    local prompt="$1" out last t code text ms
    local body; body=$(python3 -c 'import json,sys; print(json.dumps({"model":sys.argv[1],"prompt":sys.argv[2],"max_tokens":8,"temperature":0}))' "$MODEL_NAME" "$prompt")
    out=$(curl -s -w '\n%{time_total} %{http_code}' -X POST http://localhost:8000/v1/completions \
        -H "Content-Type: application/json" -d "$body")
    last=$(echo "$out" | tail -1); t=${last% *}; code=${last##* }
    text=$(echo "$out" | sed '$d' | python3 -c 'import sys,json
try: print(json.load(sys.stdin)["choices"][0]["text"].strip().replace("\n"," "))
except Exception: print("")' 2>/dev/null)
    ms=$(python3 -c "print(int(float('$t')*1000))" 2>/dev/null || echo -1)
    echo "${ms}|${text}|${code}"
}
prefix_hits() {
    curl -s "http://localhost:$1/metrics" 2>/dev/null | strip_ansi \
      | grep -iE 'prefix_cache_hits' | grep -vE '^#|_created|_sum|_bucket' \
      | awk '{s+=$NF} END{if(NR>0) printf "%d", s; else print "NA"}'
}

echo "=============================================="
echo "  Test 1: KV survives sleep/wake (single non-shadow engine)"
echo "  Model: $MODEL_NAME | Logs: $LOG_DIR"
echo "=============================================="

# ---- Phase 0: GMS servers (weights + kv_cache with --persist-on-abort) ----
echo ""; echo "=== Phase 0: GPU Memory Service (weights + kv_cache/persist-on-abort) ==="
python3 -m gpu_memory_service --device 0 --tag weights  > "$GMS_W_LOG"  2>&1 & GMS_PIDS+=($!)
python3 -m gpu_memory_service --device 0 --tag kv_cache --persist-on-abort > "$GMS_KV_LOG" 2>&1 & GMS_PIDS+=($!)
wait_for "$GMS_W_LOG"  "Server started" 30 "" "GMS weights ready"  || exit 1
wait_for "$GMS_KV_LOG" "Server started" 30 "" "GMS kv_cache ready" || exit 1
echo "GMS PIDs: ${GMS_PIDS[*]}"

# ---- Phase 1: Start ONE non-shadow engine (real KV, sleep-enabled) ----
echo ""; echo "=== Phase 1: Start engine (non-shadow, --enable-sleep-mode, real KV) ==="
ENGINE_ID=0 DYN_SYSTEM_PORT=$SYS_PORT \
VLLM_NIXL_SIDE_CHANNEL_PORT=5600 DYN_VLLM_KV_EVENT_PORT=20080 \
python3 -m dynamo.vllm --model "$MODEL_NAME" -tp 1 \
    --gpu-memory-utilization "${GPU_UTIL:-0.4}" \
    --num-gpu-blocks-override "${KV_BLOCKS:-8192}" \
    --enable-sleep-mode \
    --load-format gms > "$ENGINE_LOG" 2>&1 &
ENGINE_PID=$!
echo "Engine PID: $ENGINE_PID"
wait_for "$ENGINE_LOG" "Registered endpoint 'dynamo.backend.generate'" 300 "$ENGINE_PID" "engine registers generate" || exit 1
pass "Engine started and registered"

# ---- Phase 2: Frontend + warm long prefix + confirm in-engine HIT ----
echo ""; echo "=== Phase 2: Warm long prefix, confirm in-engine hit ==="
python3 -m dynamo.frontend > "$FRONTEND_LOG" 2>&1 &
FRONTEND_PID=$!
wait_for "$FRONTEND_LOG" "Completions is ready" 60 "$FRONTEND_PID" "frontend ready" || exit 1

build_reuse_prompt
echo "  long prefix ~${#REUSE_PROMPT} chars"
COLD=$(timed_completion "$REUSE_PROMPT")   # first send: full prefill
HOT=$(timed_completion "$REUSE_PROMPT")    # second send: should HIT in-engine
HITS_WARM=$(prefix_hits "$SYS_PORT")
REF_REST=${COLD#*|}; REF_OUT=${REF_REST%|*}
HOT_REST=${HOT#*|};  HOT_OUT=${HOT_REST%|*}
echo "  cold TTFT=${COLD%%|*}ms out='${REF_OUT}' | hot TTFT=${HOT%%|*}ms out='${HOT_OUT}' | prefix_hits=$HITS_WARM"
[ "${HOT%%|*}" -lt "${COLD%%|*}" ] 2>/dev/null && [ "$HITS_WARM" != "0" ] \
    && pass "In-engine prefix caching works (hot<cold, prefix_hits>0)" \
    || fail "In-engine prefix caching not observed (cold=${COLD%%|*} hot=${HOT%%|*} hits=$HITS_WARM)"

# ---- Phase 3: Sleep via control route ----
echo ""; echo "=== Phase 3: Sleep (POST /engine/control/sleep) ==="
SR=$(curl -s -X POST "http://localhost:$SYS_PORT/engine/control/sleep" -H "Content-Type: application/json" -d '{"level":1}')
echo "  sleep response: $SR"
echo "$SR" | grep -qiE '"status"\s*:\s*"ok"|sleeping|asleep' && pass "Engine slept" || fail "Sleep response unexpected: $SR"
# Observe whether the prefix cache got reset on sleep (informational — if it did, the
# re-send below will MISS and we'll need to suppress that reset for the same-process case).
if have "$ENGINE_LOG" "reset prefix cache"; then echo "  NOTE: prefix cache was reset during this run (may need suppression)"; fi

# ---- Phase 4: Wake via control route ----
echo ""; echo "=== Phase 4: Wake (POST /engine/control/wake_up) ==="
WR=$(curl -s -X POST "http://localhost:$SYS_PORT/engine/control/wake_up" -H "Content-Type: application/json" -d '{}')
echo "  wake response: $WR"
# Give discovery re-registration a moment.
sleep 5
grep -iE "KV reuse: reattaching|reattached KV:|Reallocated .* handles|Remap complete|StaleMemoryLayout" "$ENGINE_LOG" 2>/dev/null | strip_ansi | tail -6

# ---- Phase 5: Re-send prefix, confirm HIT + correctness ----
echo ""; echo "=== Phase 5: Re-send prefix after wake ==="
HITS_BEFORE=$(prefix_hits "$SYS_PORT")
FO=""; for attempt in $(seq 1 10); do
    FO=$(timed_completion "$REUSE_PROMPT")
    [ "${FO##*|}" = "200" ] && break
    echo "  post-wake attempt $attempt not ready; retry 2s..."; sleep 2
done
HITS_AFTER=$(prefix_hits "$SYS_PORT")
FO_MS=${FO%%|*}; FO_REST=${FO#*|}; FO_OUT=${FO_REST%|*}
echo "=========================================="
echo "  SLEEP/WAKE KV-REUSE"
echo "  cold TTFT (fresh prefill): ${COLD%%|*} ms"
echo "  hot  TTFT (in-engine hit): ${HOT%%|*} ms"
echo "  post-wake TTFT:            ${FO_MS} ms"
echo "  prefix_hits: before=${HITS_BEFORE} after=${HITS_AFTER}"
echo "  cold out='${REF_OUT}' | post-wake out='${FO_OUT}'"
# HIT if the counter bumped on the re-send.
if [ "$HITS_AFTER" != "NA" ] && [ "$HITS_BEFORE" != "NA" ] && [ "$HITS_AFTER" -gt "$HITS_BEFORE" ] 2>/dev/null; then
    pass "Post-wake prefix HIT (counter bumped $HITS_BEFORE -> $HITS_AFTER)"
else
    fail "Post-wake prefix MISS (counter $HITS_BEFORE -> $HITS_AFTER)"
fi
if [ -n "$REF_OUT" ] && [ "$FO_OUT" = "$REF_OUT" ]; then
    pass "Post-wake output byte-identical to fresh-prefill reference"
else
    fail "Post-wake output differs (cold='$REF_OUT' post-wake='$FO_OUT')"
fi
echo "=========================================="
echo ""; echo "  TEST COMPLETE"
