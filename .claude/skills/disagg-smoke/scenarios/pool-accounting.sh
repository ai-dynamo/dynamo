#!/usr/bin/env bash
# pool-accounting.sh — S12 pool accounting via proxy events (Category B: resource accounting)
#
# Inputs (env vars):
#   EXPERIMENT_DIR — /scratch/kvbm-experiments/<ts>-<scenario>/
#   PREFILL_LOG    — $EXPERIMENT_DIR/prefill.log
#   DECODE_LOG     — $EXPERIMENT_DIR/decode.log
#   HUB_LOG        — $EXPERIMENT_DIR/hub.log     (unused here)
#   HUB_API        — host:port for kvbm_hub control API (unused here)
#   PREFILL_API    — vLLM prefill HTTP endpoint  (unused here)
#   DECODE_API     — vLLM decode HTTP endpoint   (unused here)
#
# Outputs:
#   stdout: one line "PASS|FAIL|INCONCLUSIVE: pool-accounting | prefill_install=<n> prefill_drop=<n> pull_start=<n> g2_to_g1_done=<n> orphan_pulls=<n>"
#   exit:   0=PASS, 1=FAIL, 2=INCONCLUSIVE
#
# Side effects: read $PREFILL_LOG and $DECODE_LOG only.
#               Must NOT modify EXPERIMENT_DIR or restart any service.

set -euo pipefail
# Best-effort source of shared helpers; absent file is fine.
# shellcheck disable=SC1091
source "$(dirname "$0")/../scripts/_assert.sh" 2>/dev/null || true

NAME="pool-accounting"
PREFILL_LOG="${PREFILL_LOG:-${EXPERIMENT_DIR:-}/prefill.log}"
DECODE_LOG="${DECODE_LOG:-${EXPERIMENT_DIR:-}/decode.log}"

# ---- Robustness: both logs must exist and be non-empty ----
for entry in "PREFILL_LOG=${PREFILL_LOG}" "DECODE_LOG=${DECODE_LOG}"; do
  label="${entry%%=*}"
  path="${entry#*=}"
  if [[ ! -f "$path" ]]; then
    echo "INCONCLUSIVE: $NAME | error=missing_log which=${label} path=${path}"
    exit 2
  fi
  if [[ ! -s "$path" ]]; then
    echo "INCONCLUSIVE: $NAME | error=empty_log which=${label} path=${path}"
    exit 2
  fi
done

# ---- ASSERT 1 (prefill side): cd_payload_installed count == cd_payload_drop count ----
PREFILL_INSTALL="$(grep -cE 'cd_payload_installed' "$PREFILL_LOG" 2>/dev/null || true)"
PREFILL_DROP="$(grep -cE 'cd_payload_drop' "$PREFILL_LOG" 2>/dev/null || true)"
PREFILL_INSTALL="${PREFILL_INSTALL:-0}"
PREFILL_DROP="${PREFILL_DROP:-0}"

# ---- ASSERT 2 (decode side): worker_pull_chunk_start count == worker_g2_to_g1_done count ----
PULL_START="$(grep -cE 'worker_pull_chunk_start' "$DECODE_LOG" 2>/dev/null || true)"
G2_DONE="$(grep -cE 'worker_g2_to_g1_done' "$DECODE_LOG" 2>/dev/null || true)"
PULL_START="${PULL_START:-0}"
G2_DONE="${G2_DONE:-0}"

# ---- Negative: any pull_start request_id without a matching done event ----
# Build sorted-unique sets and diff them. comm requires sorted input.
START_RIDS_FILE="$(mktemp -t poolacct_start.XXXXXX)"
DONE_RIDS_FILE="$(mktemp -t poolacct_done.XXXXXX)"
trap 'rm -f "$START_RIDS_FILE" "$DONE_RIDS_FILE"' EXIT

{ grep -E 'worker_pull_chunk_start' "$DECODE_LOG" 2>/dev/null \
   | grep -oE 'request_id=[^[:space:]"]+' \
   | sort -u; } > "$START_RIDS_FILE" || true

{ grep -E 'worker_g2_to_g1_done' "$DECODE_LOG" 2>/dev/null \
   | grep -oE 'request_id=[^[:space:]"]+' \
   | sort -u; } > "$DONE_RIDS_FILE" || true

# request_ids in START but not in DONE = orphan pulls
ORPHAN_PULLS="$(comm -23 "$START_RIDS_FILE" "$DONE_RIDS_FILE" 2>/dev/null | grep -c . || true)"
ORPHAN_PULLS="${ORPHAN_PULLS:-0}"

DIGEST="prefill_install=${PREFILL_INSTALL} prefill_drop=${PREFILL_DROP} pull_start=${PULL_START} g2_to_g1_done=${G2_DONE} orphan_pulls=${ORPHAN_PULLS}"

if [[ "$PREFILL_INSTALL" == "$PREFILL_DROP" \
   && "$PULL_START" == "$G2_DONE" \
   && "$ORPHAN_PULLS" == "0" ]]; then
  echo "PASS: $NAME | $DIGEST"
  exit 0
else
  echo "FAIL: $NAME | $DIGEST"
  exit 1
fi
