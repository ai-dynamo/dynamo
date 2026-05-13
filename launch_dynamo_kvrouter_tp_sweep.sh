#!/bin/bash
# Launch the 16 TP-variant dynamo_kvrouter sbatch scripts × REPS reps.
#
# Sweep grid:
#   Prefill (TP1):  CTX_LIST  ∈ {1, 2, 3, 4}
#   Decode:         DECODE_LIST ∈ {1x2, 2x2, 3x2, 1x4}     (i.e. DxTP)
#
# Each sbatch script internally sweeps the full concurrency list, so the outer
# loop here is (script, rep) only.
#
# Each sbatch script writes to $REPO_DIR/bench/results/dynamo/${EXP_NAME}_${TIMESTAMP}_${SLURM_JOB_ID}
# so multiple reps land in distinct dirs.
#
# Submitted job IDs are appended to launch_tp_<timestamp>.log next to this script.
#
# Override defaults with env:
#   SCRIPTS="..."                  # space-separated absolute paths; default = all 16
#   CTX_LIST="1 2 3 4"             # filter prefill counts
#   DECODE_LIST="1x2 2x2 3x2 1x4"  # filter (decode-count x decode-TP) pairs
#   REPS=1
#   CONCURRENCY="48"               # override sbatch's internal sweep with a single value
#                                  # (empty = use the script's built-in sweep list)
#   HOSTCACHE=0
#   WORKER_METRICS=0               # 1 = enable --publish-events-and-metrics + capture_metrics sidecar
#
# Examples:
#   ./launch_dynamo_kvrouter_tp_sweep.sh                        # all 16
#   DECODE_LIST="1x4" ./launch_dynamo_kvrouter_tp_sweep.sh      # only the 4 TP4 configs
#   CTX_LIST="1 2" DECODE_LIST="2x2 3x2" REPS=2 ./launch_dynamo_kvrouter_tp_sweep.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ -n "${SCRIPTS:-}" ]; then
  read -r -a SCRIPTS_ARR <<<"$SCRIPTS"
else
  read -r -a CTX_ARR <<<"${CTX_LIST:-1 2 3 4}"
  read -r -a DEC_ARR <<<"${DECODE_LIST:-1x2 2x2 3x2 1x4}"
  SCRIPTS_ARR=()
  for n in "${CTX_ARR[@]}"; do
    for combo in "${DEC_ARR[@]}"; do
      d="${combo%x*}"
      tp="${combo#*x}"
      if [ -z "$d" ] || [ -z "$tp" ] || [ "$d" = "$combo" ]; then
        echo "ERROR: bad DECODE_LIST entry '$combo' (expected form DxTP, e.g. 2x2)" >&2
        exit 1
      fi
      SCRIPTS_ARR+=("$SCRIPT_DIR/run_benchx_${n}ctx${d}gen_tp${tp}_dynamo_kvrouter.sh")
    done
  done
fi

REPS="${REPS:-1}"
HOSTCACHE="${HOSTCACHE:-0}"
WORKER_METRICS="${WORKER_METRICS:-0}"
CONCURRENCY="${CONCURRENCY:-}"

for s in "${SCRIPTS_ARR[@]}"; do
  if [ ! -f "$s" ]; then
    echo "ERROR: script not found: $s" >&2
    exit 1
  fi
done

LOG="$SCRIPT_DIR/launch_tp_$(date +%Y%m%d_%H%M%S).log"
echo "Launch log: $LOG"

submit() {
  local script="$1" rep="$2"
  local out jid export_args
  export_args="ALL,HOSTCACHE=$HOSTCACHE,WORKER_METRICS=$WORKER_METRICS"
  if [ -n "$CONCURRENCY" ]; then
    export_args="$export_args,CONCURRENCY=$CONCURRENCY"
  fi
  out=$(sbatch --export="$export_args" "$script")
  jid="${out##* }"
  printf '%s\tjob=%s\tscript=%s\trep=%s\n' \
    "$(date +%Y-%m-%dT%H:%M:%S)" "$jid" "$(basename "$script")" "$rep" \
    | tee -a "$LOG"
}

total=$(( REPS * ${#SCRIPTS_ARR[@]} ))
echo "Submitting $total jobs: ${#SCRIPTS_ARR[@]} scripts × $REPS reps (HOSTCACHE=$HOSTCACHE WORKER_METRICS=$WORKER_METRICS CONCURRENCY=${CONCURRENCY:-<script-default>})"

for rep in $(seq 1 "$REPS"); do
  for script in "${SCRIPTS_ARR[@]}"; do
    submit "$script" "$rep"
  done
done

echo "Done. Submitted $total jobs. Log: $LOG"
