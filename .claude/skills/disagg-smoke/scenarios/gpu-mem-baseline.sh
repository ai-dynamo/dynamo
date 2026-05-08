#!/usr/bin/env bash
# gpu-mem-baseline.sh — S11 GPU memory baseline (Category B: resource accounting)
#
# Inputs (env vars):
#   EXPERIMENT_DIR — /scratch/kvbm-experiments/<ts>-<scenario>/
#   PREFILL_LOG    — $EXPERIMENT_DIR/prefill.log
#   DECODE_LOG     — $EXPERIMENT_DIR/decode.log
#   HUB_LOG        — $EXPERIMENT_DIR/hub.log     (unused here)
#   HUB_API        — host:port for kvbm_hub control API (unused here)
#   PREFILL_API    — vLLM prefill HTTP endpoint  (unused here)
#   DECODE_API     — vLLM decode HTTP endpoint   (unused here)
#   MEM_BASELINE   — pre-S1 nvidia-smi memory.used in MB (REQUIRED — set by chain-runner)
#
# Outputs:
#   stdout: one line "PASS|FAIL|INCONCLUSIVE: gpu-mem-baseline | mem_baseline=<n> mem_now=<n> delta_mb=<n> ..."
#   exit:   0=PASS, 1=FAIL, 2=INCONCLUSIVE
#
# Side effects: read $PREFILL_LOG, $DECODE_LOG; runs `nvidia-smi --query-gpu=memory.used`.
#               Must NOT modify EXPERIMENT_DIR or restart any service.

set -euo pipefail
# Best-effort source of shared helpers; absent file is fine.
# shellcheck disable=SC1091
source "$(dirname "$0")/../scripts/_assert.sh" 2>/dev/null || true

NAME="gpu-mem-baseline"
PREFILL_LOG="${PREFILL_LOG:-${EXPERIMENT_DIR:-}/prefill.log}"
DECODE_LOG="${DECODE_LOG:-${EXPERIMENT_DIR:-}/decode.log}"
MEM_BASELINE="${MEM_BASELINE:-}"

# ---- MEM_BASELINE is mandatory and numeric ----
if [[ -z "$MEM_BASELINE" ]]; then
  echo "INCONCLUSIVE: $NAME | error=missing_baseline hint=set_MEM_BASELINE_pre_S1"
  exit 2
fi
if ! [[ "$MEM_BASELINE" =~ ^[0-9]+$ ]]; then
  echo "INCONCLUSIVE: $NAME | error=non_numeric_baseline value=${MEM_BASELINE}"
  exit 2
fi

# ---- Negative scan: any OOM marker in prefill or decode logs → INCONCLUSIVE ----
# (OOM is a separate failure mode the chain-runner handles; we don't want to mask it as a leak.)
OOM_FOUND=0
for f in "$PREFILL_LOG" "$DECODE_LOG"; do
  if [[ -f "$f" && -s "$f" ]]; then
    if grep -qiE 'out of memory|OOM' "$f" 2>/dev/null; then
      OOM_FOUND=1
    fi
  fi
done

# ---- Sample current GPU memory.used (sum across all GPUs if multi-GPU) ----
if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "INCONCLUSIVE: $NAME | error=nvidia_smi_not_found"
  exit 2
fi

SMI_OUTPUT="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null || true)"
if [[ -z "$SMI_OUTPUT" ]]; then
  echo "INCONCLUSIVE: $NAME | error=nvidia_smi_no_output"
  exit 2
fi

# Sum non-empty integer rows. awk handles "single-GPU one row" and "multi-GPU N rows" identically.
MEM_NOW="$(echo "$SMI_OUTPUT" | awk 'BEGIN{s=0; n=0} /^[[:space:]]*[0-9]+[[:space:]]*$/ {s+=$1; n++} END{ if (n==0) print ""; else print s }')"
if [[ -z "$MEM_NOW" || ! "$MEM_NOW" =~ ^[0-9]+$ ]]; then
  echo "INCONCLUSIVE: $NAME | error=nvidia_smi_unparseable raw=$(echo "$SMI_OUTPUT" | tr '\n' ',')"
  exit 2
fi

DELTA=$((MEM_NOW - MEM_BASELINE))
DIGEST="mem_baseline=${MEM_BASELINE} mem_now=${MEM_NOW} delta_mb=${DELTA}"

# OOM marker → INCONCLUSIVE (a leak diagnosis is unsafe under OOM conditions).
if [[ "$OOM_FOUND" == "1" ]]; then
  echo "INCONCLUSIVE: $NAME | $DIGEST oom_found=1"
  exit 2
fi

# Negative delta (memory went down post-cleanup) is fine — it satisfies "< 100 MB".
if (( DELTA < 100 )); then
  echo "PASS: $NAME | $DIGEST"
  exit 0
else
  echo "FAIL: $NAME | $DIGEST"
  exit 1
fi
