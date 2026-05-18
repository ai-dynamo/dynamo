#!/bin/bash
# Two-request CD smoke (R1 cold-cache + R2 warm-decode/cleared-prefill).
# Self-contained — references the four bringup helpers in disagg-bringup/.
# Wraps disagg-bringup's launchers + start-hub.
#
# Honors:
#   KVBM_REPO           (default: /home/ryan/repos/dynamo)  → repo root
#   KVBM_DISAGG_LEADER  (legacy | unified ; default legacy)  → exported into launches
#   KVBM_BLOCK_LAYOUT   (operational | universal ; default operational)
#                         → injected into kv_connector_extra_config JSON; verified
#                           post-bringup against the hub describe endpoint
#   KVBM_ONBOARD_MODE   (inter | intra ; default inter)
#                         → injected into kv_connector_extra_config.leader.onboard.mode.
#                           Note: the disagg CD wrapper short-circuits the
#                           inner connector's intra-pass-onboard branch, so
#                           setting this to `intra` here is a no-op in
#                           practice. Use `intra-pass-onboard-smoke.sh`
#                           (single-instance, no disagg) to actually exercise
#                           the per-layer G2→G1 path. Threading the var
#                           through is kept here so audit-equiv runs match
#                           production config exactly.
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
#   If logs_dir not given, mints $KVBM_EXPERIMENTS_DIR/<ts>-<label>/
#   (KVBM_EXPERIMENTS_DIR defaults to /tmp/kvbm-experiments — see new-experiment.sh).
set -eu

DYNAMO=${KVBM_REPO:-/home/ryan/repos/dynamo}
SKILL_BRINGUP=$DYNAMO/.claude/skills/disagg-bringup
SKILL_TRACE=$DYNAMO/.claude/skills/disagg-trace
LABEL=${KVBM_EXPERIMENT_LABEL:-two-request}
KVBM_BLOCK_LAYOUT=${KVBM_BLOCK_LAYOUT:-operational}
KVBM_ONBOARD_MODE=${KVBM_ONBOARD_MODE:-inter}
export KVBM_BLOCK_LAYOUT KVBM_ONBOARD_MODE

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

# Verify block layout mode matches requested mode.  This assertion fails
# before the JSON-injection fix: the env-var is stripped by vLLM's EngineCore
# subprocess spawn, so the connector falls back to Operational regardless of
# what KVBM_BLOCK_LAYOUT was set to.
echo "--- block layout verification (expected: $KVBM_BLOCK_LAYOUT) ---"
bash "$SKILL_BRINGUP/verify-block-layout.sh" "$PREFILL_ID" "$KVBM_BLOCK_LAYOUT"
bash "$SKILL_BRINGUP/verify-block-layout.sh" "$DECODE_ID"  "$KVBM_BLOCK_LAYOUT"
echo "--- block layout OK ---"

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

# ---- Intra-pass-onboard counts (informational under disagg) ---------------
#
# Heads-up: under P/D disagg the conditional-disagg coordinator wrapper
# short-circuits `ConnectorLeader::update_state_after_alloc` and never
# calls `prepare_intra_pass_onboarding`, so `metadata.intra_pass_load`
# stays `None` regardless of `onboard.mode`. That means *this* smoke can
# never trigger `execute_local_layerwise_onboard` even with
# `KVBM_ONBOARD_MODE=intra` — see `intra-pass-onboard-smoke.sh` for the
# aggregated (single-instance) variant that actually exercises that
# code path. The counts below are kept as a regression tripwire: if any
# of them ever flip non-zero in disagg mode, something on the CD path
# has changed and is worth investigating.
#
# Intra-pass *offload* (G1→G2 per-layer during forward pass) is not yet
# wired up to drive from vLLM.
#
# `grep_count` swallows grep's exit-1-on-no-match and outputs a single
# clean integer so `[ "$x" -lt 1 ]` arithmetic is robust under `set -eu`.
grep_count() {
  # Args: pattern, file(s) ...
  local pattern=$1
  shift
  grep -aEc "$pattern" "$@" 2>/dev/null | awk -F: '{ sum += $NF } END { print sum+0 }'
}

echo
echo "-- intra-pass onboard (G2→G1 per-layer; mode=$KVBM_ONBOARD_MODE) --"

# 1. Engine-side start/complete pair (INFO; carries num_layers + num_blocks).
IPO_START_DECODE=$(grep_count  "Starting layer-wise onboard from G2 to G1" "$ROOT/decode.log")
IPO_START_PREFILL=$(grep_count "Starting layer-wise onboard from G2 to G1" "$ROOT/prefill.log")
IPO_DONE_DECODE=$(grep_count   "Layer-wise onboard complete - events recorded" "$ROOT/decode.log")
IPO_DONE_PREFILL=$(grep_count  "Layer-wise onboard complete - events recorded" "$ROOT/prefill.log")
echo "  engine start  (G2→G1 layer-wise) decode=$IPO_START_DECODE prefill=$IPO_START_PREFILL"
echo "  engine complete                   decode=$IPO_DONE_DECODE prefill=$IPO_DONE_PREFILL"

# 2. Connector-side breadcrumb (only visible because kvbm_connector=debug).
IPO_CONN_DECODE=$(grep_count  "Starting intra-pass layer-wise onboard" "$ROOT/decode.log")
IPO_CONN_PREFILL=$(grep_count "Starting intra-pass layer-wise onboard" "$ROOT/prefill.log")
echo "  connector breadcrumb (kvbm_connector=debug)"
echo "    decode.log : $IPO_CONN_DECODE"
echo "    prefill.log: $IPO_CONN_PREFILL"

# 3. Surface the start line so the reader can eyeball num_layers + num_blocks.
#    Block layout was already verified upstream via verify-block-layout.sh
#    against the hub describe endpoint, so we don't re-grep for it here.
echo "  decode start lines:"
grep -a "Starting layer-wise onboard from G2 to G1" "$ROOT/decode.log" 2>/dev/null \
  | sed 's/\x1b\[[0-9;]*m//g' | head -5 | sed 's/^/    /' || true

# 4. Phase-4b regression guard: under Universal mode the per-layer onboard
#    must route through the kernel catalog (`dispatch_transform_kernel`).
#    Any kernel-launch / FFI failure surfaces as an ERROR line — count
#    those so a future stride regression turns into a loud failure, not
#    a silent `xfer_ms=0` no-op.
IPO_TRANSFORM_ERRS=$(grep -aE "dispatch_transform_kernel.*(fail|launch failed|out of bounds)" \
  "$ROOT/decode.log" "$ROOT/prefill.log" 2>/dev/null | wc -l | tr -d ' ')
echo "  permute-kernel error lines (decode+prefill): $IPO_TRANSFORM_ERRS"

# 5. Regression tripwire only — disagg can't reach the intra-pass-onboard
#    code path today (see header note). Non-zero counts mean something on
#    the CD wrapper changed and intra_pass_load is now flowing through;
#    that's not a failure, just worth a flag so the next person knows to
#    look. The strict pass/fail validation for intra-pass onboard lives in
#    `intra-pass-onboard-smoke.sh` (single-instance non-disagg topology).
if [ "$IPO_START_DECODE" -gt 0 ] || [ "$IPO_START_PREFILL" -gt 0 ]; then
  echo "  NOTE: layer-wise onboard fired under disagg ($IPO_START_DECODE/$IPO_START_PREFILL)"
  echo "        — CD wrapper may have grown intra_pass_load support; verify intent."
fi
if [ "$IPO_TRANSFORM_ERRS" -gt 0 ]; then
  echo "  WARN: kernel transform errors during onboard ($IPO_TRANSFORM_ERRS)"
fi

echo
echo "-- ANY ERRORs across all logs --"
TOTAL_ERRS=0
for s in hub prefill decode; do
  cnt=$(grep -aE "ERROR" "$ROOT/$s.log" | sed 's/\x1b\[[0-9;]*m//g' | grep -v "kvbm_audit\|UCX\|invalid configuration\|kernel_config" | wc -l)
  echo "  $s.log: $cnt error lines"
  TOTAL_ERRS=$((TOTAL_ERRS + cnt))
done

# Render HTML.
if [ -x "$SKILL_TRACE/cd-trace.py" ]; then
    python3 "$SKILL_TRACE/cd-trace.py" "$ROOT"
    echo
    echo "Open: file://$ROOT/trace.html"
fi

echo
echo "smoke: done (intra-pass onboard not validated under disagg — use intra-pass-onboard-smoke.sh)"
