#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Does the vLLM prefix cache survive a real GMS engine failover?
#
# Two `dynamo.vllm --gms-shadow-mode` engines share one GPU through GMS. One
# holds a POSIX flock and serves; the other sleeps in STANDBY polling the lock.
# We warm the active engine's prefix cache, SIGKILL its whole process group, and
# check whether the standby that takes over reports a cache HIT on the same
# prompt.
#
# This crosses a process boundary: the mirror must be readable by a different
# process that computed its own identity digest, and the kill is a crash rather
# than a graceful sleep.
#
# DESTRUCTIVE -- run it on a dedicated box or dev pod, never on a shared host.
# cleanup() runs on every exit and kills *every* dynamo, GMS and CUDA process on
# the machine (it sweeps `nvidia-smi --query-compute-apps`), not only the ones
# it started.
#
# Control vs treatment is exactly one env var:
#   GMS_KV_INDEX_PATH unset -> control    (expect 0 cached tokens after failover)
#   GMS_KV_INDEX_PATH set   -> treatment  (expect a hit)
#
# Usage: ./kv_index_failover_check.sh [MODEL] [NUM_GPU_BLOCKS]
set -u

MODEL_NAME="${1:-Qwen/Qwen3-0.6B}"
# Pinned, not profiled: both engines size their pool independently and GMS's
# remap hard-fails if the geometry differs. Pinning is also what makes the
# identity digest match across the two processes.
NUM_BLOCKS="${2:-4096}"

LOG_DIR="${KVIDX_LOG_DIR:-/tmp/kv_index_failover_$$}"
mkdir -p "$LOG_DIR"
LOCK_PATH="$LOG_DIR/failover.lock"
ENGINE_A_LOG="$LOG_DIR/engine_a.log"; ENGINE_B_LOG="$LOG_DIR/engine_b.log"
FRONTEND_LOG="$LOG_DIR/frontend.log"
A_PORT=8100; B_PORT=8101

GMS_PIDS=(); LOAD_PIDS=(); A_PID=""; B_PID=""; FRONTEND_PID=""
pass_count=0; fail_count=0
pass() { pass_count=$((pass_count+1)); echo "  PASS: $1"; }
fail() { fail_count=$((fail_count+1)); echo "  FAIL: $1"; }
strip_ansi() { sed 's/\x1b\[[0-9;]*m//g'; }
have() { cat "$1" 2>/dev/null | strip_ansi | grep -q "$2"; }

# A long prompt so a hit is unmistakable; block_size is 16.
PROMPT=$(python3 -c 'print("The quick brown fox jumps over the lazy dog. " * 90, end="")')

cleanup() {
    echo ""; echo "=== Cleaning up ==="
    for pid in "$FRONTEND_PID" "$A_PID" "$B_PID" "${LOAD_PIDS[@]:-}" "${GMS_PIDS[@]:-}"; do
        [ -n "${pid:-}" ] && kill -0 "$pid" 2>/dev/null && kill "$pid" 2>/dev/null
    done
    pkill -9 -f "dynamo[.]vllm"        2>/dev/null
    pkill -9 -f "dynamo[.]frontend"    2>/dev/null
    pkill -9 -f "[E]ngineCore"         2>/dev/null
    pkill -9 -f "[g]pu_memory_service" 2>/dev/null
    sleep 2
    for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do
        kill -9 "$p" 2>/dev/null
    done
    echo "Logs: $LOG_DIR"
    echo "=============================================="
    echo "  Results: $pass_count passed, $fail_count failed"
    echo "=============================================="
    [ "$fail_count" -gt 0 ] && exit 1 || exit 0
}
trap cleanup EXIT

wait_for() {  # <log> <pattern> <timeout_s> <pid> <desc>
    local log="$1" pat="$2" to="$3" pid="$4" desc="$5"
    for _ in $(seq 1 "$to"); do
        have "$log" "$pat" && return 0
        if [ -n "$pid" ] && ! kill -0 "$pid" 2>/dev/null; then
            echo "ERROR: process died waiting for: $desc"; tail -30 "$log"; return 1
        fi
        sleep 1
    done
    echo "ERROR: timeout (${to}s) waiting for: $desc"; tail -30 "$log"; return 1
}

# Returns "<cached_tokens> <sha of text>" for one completion through the frontend.
infer() {
    python3 - "$MODEL_NAME" "$PROMPT" <<'PY'
import hashlib, json, sys, urllib.request
model, prompt = sys.argv[1], sys.argv[2]
body = json.dumps({"model": model, "prompt": prompt, "max_tokens": 24,
                   "temperature": 0, "seed": 1234}).encode()
req = urllib.request.Request("http://localhost:8000/v1/completions", data=body,
                             headers={"Content-Type": "application/json"})
try:
    d = json.load(urllib.request.urlopen(req, timeout=180))
except Exception as e:
    print(f"ERR {e}"); sys.exit(0)
text = d["choices"][0]["text"]
cached = (d.get("usage", {}).get("prompt_tokens_details") or {}).get("cached_tokens", -1)
print(f"{cached} {hashlib.sha256(text.encode()).hexdigest()[:16]}")
PY
}

MODE="control"; [ -n "${GMS_KV_INDEX_PATH:-}" ] && MODE="treatment"
echo "=============================================="
echo "   prefix cache across a REAL failover"
echo "=============================================="
echo "Model=$MODEL_NAME blocks=$NUM_BLOCKS mode=$MODE"
echo "persist_kv=${DYN_GMS_PERSIST_KV:-unset} index=${GMS_KV_INDEX_PATH:-unset}"
echo "Logs: $LOG_DIR"
echo ""

# ---- Phase 0: GMS (weights + kv_cache) on device 0 ----
echo "=== Phase 0: GMS servers ==="
for tag in weights kv_cache; do
    setsid nohup python3 -m gpu_memory_service --device 0 --tag "$tag" \
        > "$LOG_DIR/gms_$tag.log" 2>&1 < /dev/null &
    GMS_PIDS+=($!)
done
for tag in weights kv_cache; do
    wait_for "$LOG_DIR/gms_$tag.log" "waiting for connections\|Server started" 60 "" "GMS $tag" || exit 1
done
echo "GMS ready (weights + kv_cache)"

# ---- Phase 1: both engines, shadow mode ----
# setsid so each engine owns its process group: the SIGKILL below must take the
# vLLM EngineCore and worker children with it, or the dead engine's GMS session
# stays open and the standby can never adopt.
echo ""; echo "=== Phase 1: Engines ==="
start_engine() {  # <engine_id> <system_port> <log>
    ENGINE_ID="$1" FAILOVER_LOCK_PATH="$LOCK_PATH" DYN_SYSTEM_PORT="$2" \
    VLLM_NIXL_SIDE_CHANNEL_PORT=$((5600+$1)) DYN_VLLM_KV_EVENT_PORT=$((20080+$1)) \
    setsid nohup python3 -m dynamo.vllm --model "$MODEL_NAME" -tp 1 \
        --load-format gms --gms-shadow-mode --enforce-eager \
        --num-gpu-blocks-override "$NUM_BLOCKS" --max-model-len 4096 \
        > "$3" 2>&1 < /dev/null &
    echo $!
}
B_PID=$(start_engine 1 $B_PORT "$ENGINE_B_LOG")   # RO, waits for A to publish weights
sleep 15
A_PID=$(start_engine 0 $A_PORT "$ENGINE_A_LOG")   # RW, loads + commits weights
echo "Engine A pid=$A_PID  Engine B pid=$B_PID"

for n in A B; do
    eval "log=\$ENGINE_${n}_LOG; pid=\$${n}_PID"
    wait_for "$log" "waiting for lock" 600 "$pid" "Engine $n -> STANDBY" || exit 1
    echo "Engine $n reached STANDBY"
done

# ---- Phase 2: whoever won the lock is the active engine ----
for _ in $(seq 1 120); do
    A_WOKE=$(cat "$ENGINE_A_LOG" | strip_ansi | grep -c "Lock acquired, waking engine")
    B_WOKE=$(cat "$ENGINE_B_LOG" | strip_ansi | grep -c "Lock acquired, waking engine")
    { [ "$A_WOKE" -gt 0 ] || [ "$B_WOKE" -gt 0 ]; } && break
    sleep 1
done
if [ "${A_WOKE:-0}" -gt 0 ]; then
    WIN_PID=$A_PID WIN_LOG=$ENGINE_A_LOG LOSE_PID=$B_PID LOSE_LOG=$ENGINE_B_LOG WIN=A
else
    WIN_PID=$B_PID WIN_LOG=$ENGINE_B_LOG LOSE_PID=$A_PID LOSE_LOG=$ENGINE_A_LOG WIN=B
fi
echo "Engine $WIN won the lock (pid $WIN_PID)"
wait_for "$WIN_LOG" "Engine awake, registering with discovery" 300 "$WIN_PID" "winner register" || exit 1

# ---- Phase 3: frontend ----
echo ""; echo "=== Phase 3: Frontend ==="
setsid nohup python3 -m dynamo.frontend > "$FRONTEND_LOG" 2>&1 < /dev/null &
FRONTEND_PID=$!
wait_for "$FRONTEND_LOG" "Completions is ready" 120 "$FRONTEND_PID" "frontend ready" || exit 1

# ---- Phase 4: warm the active engine's prefix cache ----
echo ""; echo "=== Phase 4: Warm the active engine ==="
read -r COLD_CACHED COLD_HASH <<< "$(infer)"
echo "  cold : cached=$COLD_CACHED hash=$COLD_HASH"
read -r WARM_CACHED WARM_HASH <<< "$(infer)"
echo "  warm : cached=$WARM_CACHED hash=$WARM_HASH"
[ "${WARM_CACHED:-0}" -gt 0 ] 2>/dev/null \
    && pass "prefix caching works before the failover ($WARM_CACHED cached)" \
    || { fail "no prefix cache hit before failover (cached=$WARM_CACHED) -- nothing to preserve"; exit 1; }

# ---- Phase 5: SIGKILL the whole active process group ----
echo ""; echo "=== Phase 5: Failover (SIGKILL) ==="
# KVIDX_KILL_UNDER_LOAD=1 crashes the engine with batches in flight rather than
# idle. That is the case the publication fence exists for: labels are attached
# during schedule(), before their KV is computed, so an engine killed mid-batch
# must not hand those labels to its successor.
if [ "${KVIDX_KILL_UNDER_LOAD:-0}" = "1" ]; then
    echo "Firing concurrent traffic so the kill lands mid-batch..."
    for i in $(seq 1 8); do
        python3 - "$MODEL_NAME" "$i" >/dev/null 2>&1 <<'PY' &
import json, sys, urllib.request
model, i = sys.argv[1], sys.argv[2]
prompt = f"Distinct prefix {i}. " + "Tell me a long story about the sea. " * 120
body = json.dumps({"model": model, "prompt": prompt, "max_tokens": 256,
                   "temperature": 0}).encode()
req = urllib.request.Request("http://localhost:8000/v1/completions", data=body,
                             headers={"Content-Type": "application/json"})
try: urllib.request.urlopen(req, timeout=60)
except Exception: pass
PY
        LOAD_PIDS+=($!)
    done
    sleep 2   # let them reach the scheduler
fi
KILL_MS=$(date +%s%3N)
WIN_PGID=$(ps -o pgid= -p "$WIN_PID" | tr -d ' ')
echo "Killing engine $WIN process group $WIN_PGID"
kill -9 -- "-$WIN_PGID" 2>/dev/null
[ "$WIN_PID" = "$A_PID" ] && A_PID="" || B_PID=""

wait_for "$LOSE_LOG" "Lock acquired, waking engine" 180 "$LOSE_PID" "standby auto-wake" || exit 1
pass "standby acquired the flock after the crash"
wait_for "$LOSE_LOG" "Registered endpoint 'dynamo.backend.generate'" 300 "$LOSE_PID" "standby register" || exit 1
REG_MS=$(date +%s%3N)
echo "  kill -> serving: $((REG_MS-KILL_MS)) ms"

# ---- Phase 6: did the prefix cache come with it? ----
echo ""; echo "=== Phase 6: Prefix cache after failover ==="
sleep 5
read -r POST_CACHED POST_HASH <<< "$(infer)"
echo "  after failover: cached=$POST_CACHED hash=$POST_HASH"

[ "$POST_HASH" = "$WARM_HASH" ] \
    && pass "output byte-identical across the failover" \
    || fail "output changed across failover ($WARM_HASH -> $POST_HASH)"

if [ "$MODE" = "treatment" ]; then
    [ "${POST_CACHED:-0}" -gt 0 ] 2>/dev/null \
        && pass "prefix cache SURVIVED the failover ($POST_CACHED cached tokens)" \
        || fail "prefix cache lost across failover (cached=$POST_CACHED)"
    if [ -f "${GMS_KV_INDEX_PATH}.status.jsonl" ]; then
        echo "  takeover record:"; sed 's/^/    /' "${GMS_KV_INDEX_PATH}.status.jsonl"
    fi
else
    [ "${POST_CACHED:-0}" -eq 0 ] 2>/dev/null \
        && pass "control: prefix cache lost as expected (cached=0)" \
        || fail "control: unexpected hit (cached=$POST_CACHED) -- control is not a control"
fi

echo ""; echo "  COMPLETE"
