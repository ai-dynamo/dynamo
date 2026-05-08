#!/usr/bin/env bash
# cleanup-audit.sh — S9 session-cleanup audit (Category A)
#
# Inputs (env vars):
#   EXPERIMENT_DIR — /scratch/kvbm-experiments/<ts>-<scenario>/
#   PREFILL_LOG    — $EXPERIMENT_DIR/prefill.log
#   DECODE_LOG     — $EXPERIMENT_DIR/decode.log     (unused here)
#   HUB_LOG        — $EXPERIMENT_DIR/hub.log        (unused here)
#   HUB_API        — host:port for kvbm_hub control API (default localhost:8337)
#   PREFILL_API    — vLLM prefill HTTP endpoint     (unused here)
#   DECODE_API     — vLLM decode HTTP endpoint      (unused here)
#
# Outputs:
#   stdout: one line "PASS|FAIL|INCONCLUSIVE: cleanup-audit | installed=<n> drop=<n> states_size=<n> ..."
#   exit:   0=PASS, 1=FAIL, 2=INCONCLUSIVE
#
# Side effects: read $PREFILL_LOG; HTTP GET http://$HUB_API/v1/coordinator/states.
#               Must NOT modify EXPERIMENT_DIR or restart any service.

set -euo pipefail
# Best-effort source of shared helpers; absent file is fine.
# shellcheck disable=SC1091
source "$(dirname "$0")/../scripts/_assert.sh" 2>/dev/null || true

NAME="cleanup-audit"
HUB_API="${HUB_API:-localhost:8337}"
PREFILL_LOG="${PREFILL_LOG:-${EXPERIMENT_DIR:-}/prefill.log}"

# Give the connector's release path time to flush before we read the log / hit the API.
sleep 5

# ---- Robustness: log must exist and be non-empty ----
if [[ ! -f "$PREFILL_LOG" ]]; then
  echo "INCONCLUSIVE: $NAME | error=missing_log path=$PREFILL_LOG"
  exit 2
fi
if [[ ! -s "$PREFILL_LOG" ]]; then
  echo "INCONCLUSIVE: $NAME | error=empty_log path=$PREFILL_LOG"
  exit 2
fi

# ---- Identify the most recent rid touched by an install/drop event ----
RID="$( { grep -E 'prefill_cd_payload_(installed|drop)' "$PREFILL_LOG" 2>/dev/null \
          | grep -oE 'request_id=[^[:space:]"]+' \
          | tail -1 \
          | sed 's/^request_id=//'; } || true )"

if [[ -z "$RID" ]]; then
  echo "INCONCLUSIVE: $NAME | error=no_rid_found path=$PREFILL_LOG"
  exit 2
fi

# ---- ASSERT 1: install-count == drop-count for the most recent rid ----
INSTALLED="$(grep -cE "prefill_cd_payload_installed.*request_id=${RID}" "$PREFILL_LOG" 2>/dev/null || true)"
DROP="$(grep -cE "prefill_cd_payload_drop.*request_id=${RID}" "$PREFILL_LOG" 2>/dev/null || true)"
INSTALLED="${INSTALLED:-0}"
DROP="${DROP:-0}"

# ---- Negative scan: any rid with installed > drop (orphan install) ----
ORPHANS=0
while IFS= read -r r; do
  [[ -z "$r" ]] && continue
  inst="$(grep -cE "prefill_cd_payload_installed.*request_id=${r}" "$PREFILL_LOG" 2>/dev/null || true)"
  drp="$(grep -cE "prefill_cd_payload_drop.*request_id=${r}" "$PREFILL_LOG" 2>/dev/null || true)"
  inst="${inst:-0}"; drp="${drp:-0}"
  if (( inst > drp )); then
    ORPHANS=$((ORPHANS + 1))
  fi
done < <( { grep -oE 'request_id=[^[:space:]"]+' "$PREFILL_LOG" 2>/dev/null \
            | sed 's/^request_id=//' \
            | sort -u; } || true )

# ---- ASSERT 2: states API returns 0 entries ----
STATES_JSON="$(curl -fsS -m 5 "http://${HUB_API}/v1/coordinator/states" 2>/dev/null || true)"
if [[ -z "$STATES_JSON" ]]; then
  echo "INCONCLUSIVE: $NAME | error=states_api_unreachable url=http://${HUB_API}/v1/coordinator/states installed=${INSTALLED} drop=${DROP}"
  exit 2
fi
STATES_SIZE="$(echo "$STATES_JSON" | jq '. | length' 2>/dev/null || echo "")"
if [[ -z "$STATES_SIZE" || ! "$STATES_SIZE" =~ ^[0-9]+$ ]]; then
  echo "INCONCLUSIVE: $NAME | error=states_api_unparseable installed=${INSTALLED} drop=${DROP}"
  exit 2
fi

DIGEST="installed=${INSTALLED} drop=${DROP} states_size=${STATES_SIZE} orphans=${ORPHANS} request_id=${RID}"

if [[ "$INSTALLED" == "$DROP" && "$STATES_SIZE" == "0" && "$ORPHANS" == "0" ]]; then
  echo "PASS: $NAME | $DIGEST"
  exit 0
else
  echo "FAIL: $NAME | $DIGEST"
  exit 1
fi
