#!/bin/bash
# Asymmetric-TP smoke: TP=2 decode + TP=1 prefill (cross-GPU).
#
# Sibling of two-request-smoke.sh. Same R1+R2 protocol but spawns
# launch-decode-tp2.sh (TP=2 on GPUs 0,1) and launch-prefill-gpu2.sh
# (TP=1 on GPU 2). Validates:
#
#  - asymmetric vLLM end-to-end (cargo asymmetric_tp_session_round_trip
#    already validates engine-level; this is the missing vLLM-level
#    coverage)
#  - SpmdParallelWorkers::dispatch_asymmetric_pull fires (only triggers
#    when local_tp != remote_tp; symmetric TP=1↔TP=1 will NOT emit this)
#  - Per-bug-fix (commit 62adabe8ca): no "TCP streaming: peer X not
#    registered" errors on decode.log
#
# Honors all KVBM_* env vars from two-request-smoke.sh PLUS the GPU/TP
# overrides on the two new launchers (KVBM_DECODE_GPUS, KVBM_DECODE_TP,
# KVBM_DECODE_MEMUTIL, KVBM_PREFILL_GPU, KVBM_PREFILL_MEMUTIL).
#
# Usage: bash tp2-decode-smoke.sh [logs_dir]
set -eu

DYNAMO=${KVBM_REPO:-/home/ryan/repos/dynamo}
SKILL_BRINGUP=$DYNAMO/.claude/skills/disagg-bringup
SKILL_TRACE=$DYNAMO/.claude/skills/disagg-trace
LABEL=${KVBM_EXPERIMENT_LABEL:-tp2-decode-tp1-prefill}

ROOT=${1:-$(bash $SKILL_BRINGUP/new-experiment.sh "$LABEL")}
echo "EXP=$ROOT"
echo "$ROOT" > /tmp/cd-trace-current-exp

# Teardown anything stale.
pkill -f "vllm.entrypoints.openai" 2>/dev/null || true
sleep 3
PIDS=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | tr -d ' ' | grep -v '^$' || true)
[ -n "$PIDS" ] && echo "$PIDS" | xargs -r kill -9 2>/dev/null || true
sleep 2
pkill -9 -f kvbm_hub 2>/dev/null || true
sleep 1

# Start hub + vLLMs.
bash $SKILL_BRINGUP/start-hub.sh "$ROOT/hub.log" &
disown

RUST_LOG=${RUST_LOG:-info,kvbm_connector=debug,kvbm_engine=debug,kvbm_audit=info} \
  bash $SKILL_BRINGUP/launch-prefill-gpu2.sh > "$ROOT/prefill.log" 2>&1 &
disown
RUST_LOG=${RUST_LOG:-info,kvbm_connector=debug,kvbm_engine=debug,kvbm_audit=info} \
  bash $SKILL_BRINGUP/launch-decode-tp2.sh > "$ROOT/decode.log" 2>&1 &
disown

echo "waiting for both vLLMs (TP=2 decode boots slower)..."
# Bump timeout: TP=2 NCCL init + 2× CUDA graph capture takes ~2-3x longer
START_TIME=$(date +%s)
while true; do
  if curl -fsS http://127.0.0.1:8000/v1/models >/dev/null 2>&1 \
     && curl -fsS http://127.0.0.1:8001/v1/models >/dev/null 2>&1; then
    break
  fi
  ELAPSED=$(( $(date +%s) - START_TIME ))
  if [ $ELAPSED -gt 600 ]; then
    echo "TIMEOUT: vLLMs did not come up within 10 min"
    tail -30 "$ROOT/decode.log" "$ROOT/prefill.log"
    exit 28
  fi
  sleep 5
done
echo "BOTH UP $(date) (took ${ELAPSED}s)"

INSTS=$(curl -sS http://127.0.0.1:8337/v1/features/conditional-disagg/instances)
PREFILL_ID=$(echo "$INSTS" | python3 -c 'import json,sys; print(json.load(sys.stdin)["prefill"][0])')
DECODE_ID=$(echo "$INSTS" | python3 -c 'import json,sys; print(json.load(sys.stdin)["decode"][0])')
echo "PREFILL_ID=$PREFILL_ID DECODE_ID=$DECODE_ID"

# Clear both caches before R1.
curl -sS -X PUT http://127.0.0.1:8337/v1/instances/$PREFILL_ID/reset \
  -H 'content-type: application/json' -d '{}' >/dev/null
curl -sS -X PUT http://127.0.0.1:8337/v1/instances/$DECODE_ID/reset \
  -H 'content-type: application/json' -d '{}' >/dev/null

# ---- R1 (cold cache) ----
P1='The quick brown fox jumps over the lazy dog and then keeps on running through the meadow until it reaches the river where it finally stops to drink some water and rest for a while before continuing its journey home. After resting, the fox decides to take a different path back'

echo === R1 SMOKE ===
R1=$(P="$P1" python3 -c 'import json,os; print(json.dumps({"model":"Qwen/Qwen3-0.6B","prompt":os.environ["P"],"max_tokens":16,"temperature":0}))' \
  | curl -m 120 -sS -X POST http://127.0.0.1:8001/v1/completions \
      -H "Content-Type: application/json" -d @-)
echo "$R1" | head -c 400; echo

sleep 2

echo === RESETTING prefill G2 ONLY ===
curl -sS -X PUT http://127.0.0.1:8337/v1/instances/$PREFILL_ID/reset \
  -H 'content-type: application/json' -d '{}' ; echo

# ---- R2 (warm decode, cleared prefill) ----
P2="$P1 The river flowed gently between the green hills and bright wildflowers swaying in the spring breeze, while birds sang above."

echo === R2 SMOKE ===
R2=$(P="$P2" python3 -c 'import json,os; print(json.dumps({"model":"Qwen/Qwen3-0.6B","prompt":os.environ["P"],"max_tokens":16,"temperature":0}))' \
  | curl -m 120 -sS -X POST http://127.0.0.1:8001/v1/completions \
      -H "Content-Type: application/json" -d @-)
echo "$R2" | head -c 400; echo

sleep 2

# ---- Validation report ----
echo
echo "================================================================"
echo "  Validation report  (TP=2 decode + TP=1 prefill — asymmetric)"
echo "================================================================"

echo
echo "-- Asymmetric-pull dispatch (TP=2↔TP=1 specific — empty on symmetric) --"
grep -aE "dispatch_asymmetric_pull|asymmetric.*pull|stamped.*pull" "$ROOT/decode.log" "$ROOT/prefill.log" 2>/dev/null \
  | sed 's/\x1b\[[0-9;]*m//g' | head -10

echo
echo "-- R1 decode: full pull pipeline --"
grep -aE "kvbm_audit.*event=\"(worker_pull_chunk_start|worker_session_pull_call|session_pull_rdma_done|worker_session_pull_returned|worker_g2_to_g1_done|cd_payload_drop)\"" "$ROOT/decode.log" \
  | sed 's/\x1b\[[0-9;]*m//g' | head -20

echo
echo "-- R2 decode policy_decision (expect matched_tokens >= 48 on warm) --"
grep -aE "kvbm_audit.*event=\"policy_decision\"" "$ROOT/decode.log" | sed 's/\x1b\[[0-9;]*m//g'

echo
echo "-- R2 prefill: gnmt path (expect ensure_started_async_onboard) --"
grep -aE "kvbm_audit.*event=\"(cd_bound_ensure_started|ensure_started_async_onboard|ensure_started_zero_passthrough)\"" "$ROOT/prefill.log" \
  | sed 's/\x1b\[[0-9;]*m//g' | head -5

echo
echo "-- ANY ERRORs across all logs (post-fix: should be 0 on decode.log) --"
for s in hub prefill decode; do
  cnt=$(grep -aE "ERROR" "$ROOT/$s.log" | sed 's/\x1b\[[0-9;]*m//g' | grep -v "kvbm_audit\|UCX\|invalid configuration\|kernel_config" | wc -l)
  echo "  $s.log: $cnt error lines"
done

echo
echo "-- 'peer X not registered' regression check (must be empty) --"
grep -aE "TCP streaming: peer .* not registered" "$ROOT/decode.log" "$ROOT/prefill.log" 2>/dev/null \
  | sed 's/\x1b\[[0-9;]*m//g' | head -5 || echo "  (no matches — fix holds)"

if [ -x "$SKILL_TRACE/cd-trace.py" ]; then
    python3 "$SKILL_TRACE/cd-trace.py" "$ROOT"
    echo
    echo "Open: file://$ROOT/trace.html"
fi
