#!/usr/bin/env bash
# release-evict.sh — S3 release-evict regression check (anchor: cd20043ca1)
#
# Inputs (env vars):
#   EXPERIMENT_DIR  — /scratch/kvbm-experiments/<ts>-<scenario>/
#   PREFILL_LOG     — $EXPERIMENT_DIR/prefill.log
#   DECODE_LOG     — $EXPERIMENT_DIR/decode.log     (unused here)
#   HUB_LOG        — $EXPERIMENT_DIR/hub.log        (unused here)
#   HUB_API         — host:port for kvbm_hub control API (default localhost:8337)
#   PREFILL_API    — vLLM prefill HTTP endpoint     (unused here)
#   DECODE_API     — vLLM decode HTTP endpoint      (unused here)
#   MOST_RECENT_RID — request_id of the most recent request (set by chain-runner;
#                     comes from R1/S1's prior r1). Required.
#
# Outputs:
#   stdout: one line "PASS|FAIL|INCONCLUSIVE: release-evict | rid_in_states=<bool> drop_count=<n> reinstall_after_drop=<n> ..."
#   exit:   0=PASS, 1=FAIL, 2=INCONCLUSIVE
#
# Side effects: read $PREFILL_LOG; HTTP GET http://$HUB_API/v1/coordinator/states.
#               Must NOT modify EXPERIMENT_DIR or restart any service.
#
# Anchor: cd20043ca1 — atomic coordinator.release() remove. Without that fix, a
# get-then-remove race under kv_load_failure_policy=recompute can leave the rid
# in the states map after `prefill_cd_payload_drop`, or worse, allow a SECOND
# `prefill_cd_payload_installed` event for the same rid after the drop.

set -euo pipefail
# Best-effort source of shared helpers; absent file is fine.
# shellcheck disable=SC1091
source "$(dirname "$0")/../scripts/_assert.sh" 2>/dev/null || true

NAME="release-evict"
HUB_API="${HUB_API:-localhost:8337}"
PREFILL_LOG="${PREFILL_LOG:-${EXPERIMENT_DIR:-}/prefill.log}"
RID="${MOST_RECENT_RID:-}"

# Brief settle so any tail-end audit events for the prior request flush.
sleep 2

# ---- Robustness: required inputs ----
if [[ -z "$RID" ]]; then
  echo "INCONCLUSIVE: $NAME | error=missing_most_recent_rid"
  exit 2
fi
if [[ ! -f "$PREFILL_LOG" ]]; then
  echo "INCONCLUSIVE: $NAME | error=missing_log path=$PREFILL_LOG request_id=${RID}"
  exit 2
fi
if [[ ! -s "$PREFILL_LOG" ]]; then
  echo "INCONCLUSIVE: $NAME | error=empty_log path=$PREFILL_LOG request_id=${RID}"
  exit 2
fi

# ---- ASSERT 1: prefill log shows request_finished_exit role="prefill" for this rid ----
# Audit lines look like: kvbm_audit ... event="request_finished_exit" role="prefill" ... request_id=<uuid> ...
FINISHED_EXIT="$(grep -cE "request_finished_exit.*role=\"prefill\".*request_id=${RID}" "$PREFILL_LOG" 2>/dev/null || true)"
FINISHED_EXIT="${FINISHED_EXIT:-0}"
# Order may vary in the format string; try the reverse too if zero.
if (( FINISHED_EXIT == 0 )); then
  FINISHED_EXIT="$(grep -cE "request_id=${RID}.*request_finished_exit.*role=\"prefill\"" "$PREFILL_LOG" 2>/dev/null || true)"
  FINISHED_EXIT="${FINISHED_EXIT:-0}"
fi

# ---- ASSERT 2: prefill log shows prefill_cd_payload_drop for this rid ----
DROP_COUNT="$(grep -cE "prefill_cd_payload_drop.*request_id=${RID}" "$PREFILL_LOG" 2>/dev/null || true)"
DROP_COUNT="${DROP_COUNT:-0}"

# ---- Negative scan: any prefill_cd_payload_installed AFTER the first drop, same rid ----
# Find the line number of the first drop event; count installed events that occur later.
REINSTALL_AFTER_DROP=0
DROP_LINE="$(grep -nE "prefill_cd_payload_drop.*request_id=${RID}" "$PREFILL_LOG" 2>/dev/null \
             | head -1 | cut -d: -f1 || true)"
if [[ -n "$DROP_LINE" ]]; then
  REINSTALL_AFTER_DROP="$(grep -nE "prefill_cd_payload_installed.*request_id=${RID}" "$PREFILL_LOG" 2>/dev/null \
                          | awk -F: -v d="$DROP_LINE" '$1 > d' \
                          | wc -l | tr -d ' ' || true)"
  REINSTALL_AFTER_DROP="${REINSTALL_AFTER_DROP:-0}"
fi

# ---- ASSERT 3: states API does NOT list this rid ----
STATES_JSON="$(curl -fsS -m 5 "http://${HUB_API}/v1/coordinator/states" 2>/dev/null || true)"
if [[ -z "$STATES_JSON" ]]; then
  echo "INCONCLUSIVE: $NAME | error=states_api_unreachable url=http://${HUB_API}/v1/coordinator/states finished_exit=${FINISHED_EXIT} drop_count=${DROP_COUNT} reinstall_after_drop=${REINSTALL_AFTER_DROP} request_id=${RID}"
  exit 2
fi
RID_IN_STATES="$(echo "$STATES_JSON" \
                 | jq -r --arg r "$RID" '[.[] | select((.request_id // "") == $r)] | length' 2>/dev/null \
                 || echo "")"
if [[ -z "$RID_IN_STATES" || ! "$RID_IN_STATES" =~ ^[0-9]+$ ]]; then
  echo "INCONCLUSIVE: $NAME | error=states_api_unparseable finished_exit=${FINISHED_EXIT} drop_count=${DROP_COUNT} reinstall_after_drop=${REINSTALL_AFTER_DROP} request_id=${RID}"
  exit 2
fi
RID_IN_STATES_BOOL="false"
if (( RID_IN_STATES > 0 )); then
  RID_IN_STATES_BOOL="true"
fi

DIGEST="rid_in_states=${RID_IN_STATES_BOOL} drop_count=${DROP_COUNT} reinstall_after_drop=${REINSTALL_AFTER_DROP} finished_exit=${FINISHED_EXIT} request_id=${RID}"

# ---- Verdict ----
# PASS: finished_exit >= 1, drop_count >= 1, no rid in states, no reinstall after drop.
if (( FINISHED_EXIT >= 1 )) && (( DROP_COUNT >= 1 )) \
   && [[ "$RID_IN_STATES_BOOL" == "false" ]] \
   && (( REINSTALL_AFTER_DROP == 0 )); then
  echo "PASS: $NAME | $DIGEST"
  exit 0
fi

echo "FAIL: $NAME | $DIGEST"
exit 1
