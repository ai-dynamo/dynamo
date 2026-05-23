#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0
# orphan-probe.sh — S10 orphan-state probe (Category A)
#
# Inputs (env vars):
#   EXPERIMENT_DIR — /scratch/kvbm-experiments/<ts>-<scenario>/
#   PREFILL_LOG    — $EXPERIMENT_DIR/prefill.log
#   DECODE_LOG     — $EXPERIMENT_DIR/decode.log
#   HUB_LOG        — $EXPERIMENT_DIR/hub.log     (unused here)
#   HUB_API        — host:port for kvbm_hub control API (default localhost:8337)
#   PREFILL_API    — vLLM prefill HTTP endpoint  (unused here)
#   DECODE_API     — vLLM decode HTTP endpoint   (unused here)
#
# Outputs:
#   stdout: one line "PASS|FAIL|INCONCLUSIVE: orphan-probe | local_onboard_complete=<n> mark_onboarding_complete=<n> orphan_onboarding_rows=<n> watchdog_fires=<n> ..."
#   exit:   0=PASS, 1=FAIL, 2=INCONCLUSIVE
#
# Side effects: read $PREFILL_LOG, $DECODE_LOG; HTTP GET http://$HUB_API/v1/coordinator/states.
#               Must NOT modify EXPERIMENT_DIR or restart any service.

set -euo pipefail
# Best-effort source of shared helpers; absent file is fine.
# shellcheck disable=SC1091
source "$(dirname "$0")/../scripts/_assert.sh" 2>/dev/null || true

NAME="orphan-probe"
HUB_API="${HUB_API:-localhost:8337}"
PREFILL_LOG="${PREFILL_LOG:-${EXPERIMENT_DIR:-}/prefill.log}"
DECODE_LOG="${DECODE_LOG:-${EXPERIMENT_DIR:-}/decode.log}"

# Give the Onboarding observer up to 10 s to settle (longer than S9 because the
# observer has its own timer; mid-quiescence reads can still see the transient).
sleep 10

# ---- Robustness: prefill log must exist and be non-empty ----
if [[ ! -f "$PREFILL_LOG" ]]; then
  echo "INCONCLUSIVE: $NAME | error=missing_log path=$PREFILL_LOG"
  exit 2
fi
if [[ ! -s "$PREFILL_LOG" ]]; then
  echo "INCONCLUSIVE: $NAME | error=empty_log path=$PREFILL_LOG"
  exit 2
fi

# ---- ASSERT 1: local_onboard_complete_set count == mark_onboarding_complete count (prefill log) ----
LOCAL_ONBOARD="$(grep -cE 'local_onboard_complete_set' "$PREFILL_LOG" 2>/dev/null || true)"
MARK_ONBOARD="$(grep -cE 'mark_onboarding_complete' "$PREFILL_LOG" 2>/dev/null || true)"
LOCAL_ONBOARD="${LOCAL_ONBOARD:-0}"
MARK_ONBOARD="${MARK_ONBOARD:-0}"

# ---- Negative scans ----
WATCHDOG="$(grep -cE 'prefill_finalize_observer_watchdog' "$PREFILL_LOG" 2>/dev/null || true)"
WATCHDOG="${WATCHDOG:-0}"

INVALID_TAKE_PREFILL="$(grep -cF 'Failed to take Onboarding state' "$PREFILL_LOG" 2>/dev/null || true)"
INVALID_TAKE_PREFILL="${INVALID_TAKE_PREFILL:-0}"
INVALID_TAKE_DECODE=0
if [[ -f "$DECODE_LOG" && -s "$DECODE_LOG" ]]; then
  INVALID_TAKE_DECODE="$(grep -cF 'Failed to take Onboarding state' "$DECODE_LOG" 2>/dev/null || true)"
  INVALID_TAKE_DECODE="${INVALID_TAKE_DECODE:-0}"
fi
INVALID_TAKE=$((INVALID_TAKE_PREFILL + INVALID_TAKE_DECODE))

# ---- ASSERT 2: states API returns no row with status=="Onboarding" ----
STATES_JSON="$(curl -fsS -m 5 "http://${HUB_API}/v1/coordinator/states" 2>/dev/null || true)"
if [[ -z "$STATES_JSON" ]]; then
  echo "INCONCLUSIVE: $NAME | error=states_api_unreachable url=http://${HUB_API}/v1/coordinator/states local_onboard_complete=${LOCAL_ONBOARD} mark_onboarding_complete=${MARK_ONBOARD}"
  exit 2
fi
ORPHAN_ONBOARDING="$(echo "$STATES_JSON" | jq '[.[] | select((.status // "") == "Onboarding")] | length' 2>/dev/null || echo "")"
if [[ -z "$ORPHAN_ONBOARDING" || ! "$ORPHAN_ONBOARDING" =~ ^[0-9]+$ ]]; then
  echo "INCONCLUSIVE: $NAME | error=states_api_unparseable local_onboard_complete=${LOCAL_ONBOARD} mark_onboarding_complete=${MARK_ONBOARD}"
  exit 2
fi

DIGEST="local_onboard_complete=${LOCAL_ONBOARD} mark_onboarding_complete=${MARK_ONBOARD} orphan_onboarding_rows=${ORPHAN_ONBOARDING} watchdog_fires=${WATCHDOG} invalid_take=${INVALID_TAKE}"

if [[ "$LOCAL_ONBOARD" == "$MARK_ONBOARD" \
   && "$ORPHAN_ONBOARDING" == "0" \
   && "$WATCHDOG" == "0" \
   && "$INVALID_TAKE" == "0" ]]; then
  echo "PASS: $NAME | $DIGEST"
  exit 0
else
  echo "FAIL: $NAME | $DIGEST"
  exit 1
fi
