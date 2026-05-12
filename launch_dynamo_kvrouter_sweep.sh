#!/bin/bash
# Launch 2 dynamo_kvrouter sbatch scripts × 8 concurrencies × 3 reps = 48 jobs.
# Reps are interleaved so a complete (script,concurrency) sweep finishes before
# any duplicate runs start — useful for partial best-of-3.
#
# Each sbatch script writes to $REPO_DIR/bench/results/dynamo/${EXP_NAME}_${TIMESTAMP}_${SLURM_JOB_ID}
# so the 3 reps land in distinct dirs.
#
# Submitted job IDs are appended to launch_<timestamp>.log next to this script.
#
# Override defaults with env:
#   CONCURRENCIES="16 32 48 64 80 96 112 128"
#   REPS=3
#   HOSTCACHE=0
#   WORKER_METRICS=0   # 1 = enable --publish-events-and-metrics + capture_metrics sidecar

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

SCRIPTS=(
  "$SCRIPT_DIR/run_benchx_1ctx1gen_dynamo_kvrouter.sh"
  # "$SCRIPT_DIR/run_benchx_2ctx1gen_dyanmo_kvrouter.sh"   # skipped
  # "$SCRIPT_DIR/run_benchx_3ctx1gen_dynamo_kvrouter.sh"   # skipped
  "$SCRIPT_DIR/run_benchx_4ctx1gen_dynamo_kvrouter.sh"
)

read -r -a CONCURRENCIES <<<"${CONCURRENCIES:-16 32 48 64 80 96 112 128}"
REPS="${REPS:-3}"
HOSTCACHE="${HOSTCACHE:-0}"
WORKER_METRICS="${WORKER_METRICS:-0}"

LOG="$SCRIPT_DIR/launch_$(date +%Y%m%d_%H%M%S).log"
echo "Launch log: $LOG"

submit() {
  local script="$1" c="$2" rep="$3"
  local out jid
  out=$(sbatch --export=ALL,CONCURRENCY=$c,HOSTCACHE=$HOSTCACHE,WORKER_METRICS=$WORKER_METRICS "$script")
  jid="${out##* }"
  printf '%s\tjob=%s\tscript=%s\tc=%s\trep=%s\n' \
    "$(date +%Y-%m-%dT%H:%M:%S)" "$jid" "$(basename "$script")" "$c" "$rep" \
    | tee -a "$LOG"
}

total=$(( REPS * ${#SCRIPTS[@]} * ${#CONCURRENCIES[@]} ))
echo "Submitting $total jobs: ${#SCRIPTS[@]} scripts × ${#CONCURRENCIES[@]} concurrencies × $REPS reps (HOSTCACHE=$HOSTCACHE WORKER_METRICS=$WORKER_METRICS)"

for rep in $(seq 1 "$REPS"); do
  for script in "${SCRIPTS[@]}"; do
    for c in "${CONCURRENCIES[@]}"; do
      submit "$script" "$c" "$rep"
    done
  done
done

echo "Done. Submitted $total jobs. Log: $LOG"
