#!/usr/bin/env bash
# onboarding-cancel.sh — S4 Onboarding cancel regression check (anchor: 0cc11281d9)
#
# Inputs (env vars):
#   EXPERIMENT_DIR — /scratch/kvbm-experiments/<ts>-<scenario>/
#   PREFILL_LOG    — $EXPERIMENT_DIR/prefill.log
#   DECODE_LOG     — $EXPERIMENT_DIR/decode.log
#   HUB_LOG        — $EXPERIMENT_DIR/hub.log     (unused here)
#   HUB_API        — host:port for kvbm_hub control API (default localhost:8337)
#   PREFILL_API    — vLLM prefill HTTP endpoint  (unused here)
#   DECODE_API     — vLLM decode HTTP endpoint   (default localhost:8080)
#   MODEL          — model name to send (default Qwen/Qwen3-0.6B)
#   CANCEL_DELAY_S — seconds before SIGINT'ing the curl (default 0.2)
#
# Outputs:
#   stdout: one line "PASS|FAIL|INCONCLUSIVE: onboarding-cancel | cleanup_pending_usaa=<bool> invalid_transition_lines=<n> states_after=<n> ..."
#   exit:   0=PASS, 1=FAIL, 2=INCONCLUSIVE
#
# Side effects:
#   - Issues a streaming POST to $DECODE_API/v1/completions and SIGINTs it ~CANCEL_DELAY_S
#     after launch. This forces a finish_request while the request is still in
#     `Onboarding` state on the prefill side.
#   - Reads $PREFILL_LOG; HTTP GET http://$HUB_API/v1/coordinator/states.
#   - Must NOT modify EXPERIMENT_DIR or restart any service.
#
# Anchor: 0cc11281d9 — handle Onboarding state in request_finished. Without that
# fix, mid-onboarding cancels surface as `Invalid transition from Onboarding to
# Inactive` errors and leak state.

set -euo pipefail
# Best-effort source of shared helpers; absent file is fine.
# shellcheck disable=SC1091
source "$(dirname "$0")/../scripts/_assert.sh" 2>/dev/null || true

NAME="onboarding-cancel"
HUB_API="${HUB_API:-localhost:8337}"
DECODE_API="${DECODE_API:-localhost:8080}"
PREFILL_LOG="${PREFILL_LOG:-${EXPERIMENT_DIR:-}/prefill.log}"
MODEL="${MODEL:-Qwen/Qwen3-0.6B}"
CANCEL_DELAY_S="${CANCEL_DELAY_S:-0.2}"

# ---- Robustness: prefill log must exist ----
if [[ ! -f "$PREFILL_LOG" ]]; then
  echo "INCONCLUSIVE: $NAME | error=missing_log path=$PREFILL_LOG"
  exit 2
fi

# ---- Snapshot baseline counts BEFORE firing the curl ----
# We use the line count to scope new cleanup events to lines added after this point.
BASELINE_LINES="$(wc -l < "$PREFILL_LOG" 2>/dev/null | tr -d ' ' || echo 0)"
BASELINE_LINES="${BASELINE_LINES:-0}"
BASELINE_INVALID="$(grep -cF 'Invalid transition from Onboarding to Inactive' "$PREFILL_LOG" 2>/dev/null || true)"
BASELINE_INVALID="${BASELINE_INVALID:-0}"
BASELINE_BAD_TAKE="$(grep -cF 'Failed to take Onboarding state for request ID' "$PREFILL_LOG" 2>/dev/null || true)"
BASELINE_BAD_TAKE="${BASELINE_BAD_TAKE:-0}"

# ---- Fire and cancel: stream a completion, SIGINT the curl mid-stream ----
# Long enough prompt + max_tokens so the request can't possibly complete before we cancel.
PROMPT='Tell me a long, slow-paced bedtime story about a family of foxes who live in a meadow by the river. Describe their morning routine, the sounds of birds, the smell of wildflowers, and the gentle rustle of leaves overhead.'
PAYLOAD="$(MODEL_VAL="$MODEL" PROMPT_VAL="$PROMPT" python3 -c '
import json, os
print(json.dumps({
    "model": os.environ["MODEL_VAL"],
    "prompt": os.environ["PROMPT_VAL"],
    "max_tokens": 256,
    "temperature": 0,
    "stream": True,
}))
' 2>/dev/null || echo "")"
if [[ -z "$PAYLOAD" ]]; then
  echo "INCONCLUSIVE: $NAME | error=python3_payload_build_failed"
  exit 2
fi

set +e
curl -sS -N -X POST "http://${DECODE_API}/v1/completions" \
  -H "Content-Type: application/json" \
  -d "$PAYLOAD" >/dev/null 2>&1 &
CURL_PID=$!
# Sleep CANCEL_DELAY_S, then SIGINT the curl. -N disables curl's output buffer
# so the connection is open and streaming when we kill it.
sleep "$CANCEL_DELAY_S"
kill -INT "$CURL_PID" 2>/dev/null || true
# Reap; don't care about exit code (will be non-zero by design).
wait "$CURL_PID" 2>/dev/null
CURL_EXIT=$?
set -e

# ---- Wait for the prefill side to log the cleanup event ----
sleep 3

# ---- ASSERT 1: a NEW prefill_cleanup_pending_usaa event appeared after our cancel ----
# Scope to lines beyond BASELINE_LINES so we don't pick up cleanup events from prior requests.
NEW_CLEANUP_LINES="$(awk -v b="$BASELINE_LINES" 'NR > b && /prefill_cleanup_pending_usaa/' "$PREFILL_LOG" 2>/dev/null || true)"
NEW_CLEANUP_COUNT=0
if [[ -n "$NEW_CLEANUP_LINES" ]]; then
  NEW_CLEANUP_COUNT="$(printf '%s\n' "$NEW_CLEANUP_LINES" | grep -c . || true)"
  NEW_CLEANUP_COUNT="${NEW_CLEANUP_COUNT:-0}"
fi

# Extract the rid for the cancelled request from the first new cleanup line.
RID=""
if (( NEW_CLEANUP_COUNT > 0 )); then
  RID="$(printf '%s\n' "$NEW_CLEANUP_LINES" | head -1 \
         | grep -oE 'request_id=[^[:space:]"]+' | head -1 | sed 's/^request_id=//' || true)"
fi

# ---- ASSERT 2: zero NEW `Invalid transition from Onboarding to Inactive` lines ----
NOW_INVALID="$(grep -cF 'Invalid transition from Onboarding to Inactive' "$PREFILL_LOG" 2>/dev/null || true)"
NOW_INVALID="${NOW_INVALID:-0}"
NEW_INVALID=$(( NOW_INVALID - BASELINE_INVALID ))
if (( NEW_INVALID < 0 )); then
  NEW_INVALID=0  # log was rotated/truncated — treat as 0 new
fi

# ---- Negative: zero NEW `Failed to take Onboarding state for request ID` lines ----
NOW_BAD_TAKE="$(grep -cF 'Failed to take Onboarding state for request ID' "$PREFILL_LOG" 2>/dev/null || true)"
NOW_BAD_TAKE="${NOW_BAD_TAKE:-0}"
NEW_BAD_TAKE=$(( NOW_BAD_TAKE - BASELINE_BAD_TAKE ))
if (( NEW_BAD_TAKE < 0 )); then
  NEW_BAD_TAKE=0
fi

# ---- Settle the additional 2s so total elapsed since cancel is ~5s ----
sleep 2

# ---- ASSERT 3: states API has no row for the cancelled rid ----
STATES_AFTER="0"
if [[ -n "$RID" ]]; then
  STATES_JSON="$(curl -fsS -m 5 "http://${HUB_API}/v1/coordinator/states" 2>/dev/null || true)"
  if [[ -z "$STATES_JSON" ]]; then
    echo "INCONCLUSIVE: $NAME | error=states_api_unreachable url=http://${HUB_API}/v1/coordinator/states cleanup_count=${NEW_CLEANUP_COUNT} new_invalid=${NEW_INVALID} request_id=${RID}"
    exit 2
  fi
  STATES_AFTER="$(echo "$STATES_JSON" \
                  | jq -r --arg r "$RID" '[.[] | select((.request_id // "") == $r)] | length' 2>/dev/null \
                  || echo "")"
  if [[ -z "$STATES_AFTER" || ! "$STATES_AFTER" =~ ^[0-9]+$ ]]; then
    echo "INCONCLUSIVE: $NAME | error=states_api_unparseable cleanup_count=${NEW_CLEANUP_COUNT} new_invalid=${NEW_INVALID} request_id=${RID}"
    exit 2
  fi
fi

# ---- Verdict ----
CLEANUP_BOOL="false"
if (( NEW_CLEANUP_COUNT > 0 )); then
  CLEANUP_BOOL="true"
fi

DIGEST="cleanup_pending_usaa=${CLEANUP_BOOL} invalid_transition_lines=${NEW_INVALID} states_after=${STATES_AFTER} cleanup_count=${NEW_CLEANUP_COUNT} new_bad_take=${NEW_BAD_TAKE} curl_exit=${CURL_EXIT} request_id=${RID:-unknown}"

# If we never observed a cleanup event, the curl probably finished too fast or
# never reached the prefill onboarding path — INCONCLUSIVE rather than FAIL.
if (( NEW_CLEANUP_COUNT == 0 )); then
  echo "INCONCLUSIVE: $NAME | error=no_new_cleanup_event hint=request_may_have_completed_before_cancel $DIGEST"
  exit 2
fi

# PASS requires: cleanup observed, zero new invalid-transition lines, zero new
# bad-take lines, and no row in the states API for the rid.
if [[ "$CLEANUP_BOOL" == "true" ]] \
   && (( NEW_INVALID == 0 )) \
   && (( NEW_BAD_TAKE == 0 )) \
   && [[ "$STATES_AFTER" == "0" ]]; then
  echo "PASS: $NAME | $DIGEST"
  exit 0
fi

echo "FAIL: $NAME | $DIGEST"
exit 1
