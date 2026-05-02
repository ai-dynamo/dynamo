#!/bin/bash
# Two-request CD smoke (R1 cold-cache + R2 warm-decode/cleared-prefill).
# Self-contained — references /home/ryan/repos/dynamo/.claude/skills/
# scripts only.  Wraps disagg-bringup's launchers + start-hub.
#
# Honors:
#   KVBM_DISAGG_LEADER  (legacy | unified ; default legacy)  → exported into launches
#   KVBM_EXPERIMENT_LABEL (default: two-request)             → folded into experiment dir name
#
# R1: prompt P1 (~54 tokens / 3 full blocks). Cold caches on both sides.
#     Prefill computes 3 blocks; decode pulls them back.
# (between) Reset prefill G2 only — decode keeps its cache.
# R2: prompt P2 = P1 + extra → ~80 tokens / 5 full blocks. Decode sees
#     3-block local match in G2, forwards 3 hashes; prefill pulls those,
#     forward-passes the net-new blocks, observer publishes back.
#
# Usage: bash two-request-smoke.sh [logs_dir]
#   If logs_dir not given, mints /tmp/kvbm-experiments/<ts>-<label>/.
set -eu

DYNAMO=/home/ryan/repos/dynamo
SKILL_BRINGUP=$DYNAMO/.claude/skills/disagg-bringup
SKILL_TRACE=$DYNAMO/.claude/skills/disagg-trace
LABEL=${KVBM_EXPERIMENT_LABEL:-two-request}

ROOT=${1:-$(bash $SKILL_BRINGUP/new-experiment.sh "$LABEL")}
echo "EXP=$ROOT"
echo "$ROOT" > /tmp/cd-trace-current-exp

# Tear down anything stale (best-effort).
pkill -f "vllm.entrypoints.openai" 2>/dev/null || true
sleep 3
PIDS=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | tr -d ' ' | grep -v '^$' || true)
[ -n "$PIDS" ] && echo "$PIDS" | xargs -r kill -9 2>/dev/null || true
sleep 2
pkill -9 -f kvbm_hub 2>/dev/null || true
sleep 1

# Start hub + vLLMs.  start-hub.sh bakes in --prefill-vllm-url +
# --prefill-vllm-model so the CD prefill dispatcher worker is enabled.
bash $SKILL_BRINGUP/start-hub.sh "$ROOT/hub.log" &
disown

RUST_LOG=${RUST_LOG:-info,kvbm_connector=debug,kvbm_audit=info} \
  bash $SKILL_BRINGUP/launch-prefill.sh > "$ROOT/prefill.log" 2>&1 &
disown
RUST_LOG=${RUST_LOG:-info,kvbm_connector=debug,kvbm_audit=info} \
  bash $SKILL_BRINGUP/launch-decode.sh > "$ROOT/decode.log" 2>&1 &
disown

echo "waiting for both vLLMs..."
until curl -fsS http://127.0.0.1:8000/v1/models >/dev/null 2>&1 \
   && curl -fsS http://127.0.0.1:8001/v1/models >/dev/null 2>&1; do sleep 5; done
echo "BOTH UP $(date)"

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
  | curl -m 90 -sS -X POST http://127.0.0.1:8001/v1/completions \
      -H "Content-Type: application/json" -d @-)
echo "$R1" | head -c 400; echo

sleep 2  # let any tail-end audit events flush

# Reset prefill G2 ONLY — decode keeps its 3-block cache.
echo === RESETTING prefill G2 ONLY ===
curl -sS -X PUT http://127.0.0.1:8337/v1/instances/$PREFILL_ID/reset \
  -H 'content-type: application/json' -d '{}' ; echo

# ---- R2 (warm decode, cleared prefill) ----
# Same prefix as P1 + ~30 extra tokens to push past 4 full blocks.
P2="$P1 The river flowed gently between the green hills and bright wildflowers swaying in the spring breeze, while birds sang above."

echo === R2 SMOKE ===
R2=$(P="$P2" python3 -c 'import json,os; print(json.dumps({"model":"Qwen/Qwen3-0.6B","prompt":os.environ["P"],"max_tokens":16,"temperature":0}))' \
  | curl -m 90 -sS -X POST http://127.0.0.1:8001/v1/completions \
      -H "Content-Type: application/json" -d @-)
echo "$R2" | head -c 400; echo

sleep 2

# ---- Validation report ----
echo
echo "================================================================"
echo "  Validation report  (full trace at $ROOT/trace.html)"
echo "================================================================"

echo
echo "-- R1 prefill: cache-hit rates (expect 0/N then forward-pass fills) --"
grep -a "Cache Hit Rates" "$ROOT/prefill.log" | head -3
echo
echo "-- R1 decode: full pull pipeline --"
grep -aE "kvbm_audit.*event=\"(worker_pull_chunk_start|worker_session_pull_call|session_pull_rdma_done|worker_session_pull_returned|worker_g2_to_g1_done|cd_payload_drop)\"" "$ROOT/decode.log" \
  | sed 's/\x1b\[[0-9;]*m//g' | head -20

echo
echo "-- R1 prefill: G1->G2 register events (proves G2 cache populated) --"
grep -aE "register_blocks|G1→G2|register_g2" "$ROOT/prefill.log" | sed 's/\x1b\[[0-9;]*m//g' | head -10

echo
echo "-- R2 decode policy_decision (expect matched_tokens >= 48) --"
grep -aE "kvbm_audit.*event=\"policy_decision\"" "$ROOT/decode.log" | sed 's/\x1b\[[0-9;]*m//g'

echo
echo "-- R2 prefill: gnmt path (expect ensure_started_async_onboard, NOT zero_passthrough) --"
grep -aE "kvbm_audit.*event=\"(cd_bound_ensure_started|ensure_started_async_onboard|ensure_started_zero_passthrough)\"" "$ROOT/prefill.log" \
  | sed 's/\x1b\[[0-9;]*m//g' | head -5

echo
echo "-- R2 prefill: pull-from-decode events (n>0 path) --"
grep -aE "kvbm_audit.*event=\"(session_pull_request|session_pull_send|session_pull_rdma_start|session_pull_rdma_done)\"" "$ROOT/prefill.log" \
  | sed 's/\x1b\[[0-9;]*m//g' | head -10

echo
echo "-- R2 decode: pull-from-prefill events (the net-new blocks) --"
grep -aE "kvbm_audit.*event=\"(worker_pull_chunk_start|worker_session_pull_call|session_pull_rdma_done|worker_g2_to_g1_done)\"" "$ROOT/decode.log" \
  | sed 's/\x1b\[[0-9;]*m//g' | tail -15

echo
echo "-- ANY ERRORs across all logs --"
for s in hub prefill decode; do
  cnt=$(grep -aE "ERROR" "$ROOT/$s.log" | sed 's/\x1b\[[0-9;]*m//g' | grep -v "kvbm_audit\|UCX\|invalid configuration\|kernel_config" | wc -l)
  echo "  $s.log: $cnt error lines"
done

# Render HTML.
if [ -x "$SKILL_TRACE/cd-trace.py" ]; then
    python3 "$SKILL_TRACE/cd-trace.py" "$ROOT"
    echo
    echo "Open: file://$ROOT/trace.html"
fi
