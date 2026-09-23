#!/usr/bin/env bash
set -euo pipefail
ROOT=/lustre/fsw/coreai_comparch_trtllm/jothomson/dynamo-velo-response-20260923
LABEL=${1:?condition}
export SLURM_JOB_ID=$(cat "$ROOT/control/job-id")
export DYNAMO_PROFILE_ROOT="$ROOT" PYTHONUNBUFFERED=1
export PATH="/lustre/fsw/coreai_comparch_trtllm/jothomson/dynamo-numa/env/services/bin:$PATH"
export SLURM_TMPDIR="/var/tmp/frontend-main-quic-$SLURM_JOB_ID-$LABEL"
unset RAYON_NUM_THREADS RAYON_RS_NUM_THREADS TOKENIZERS_PARALLELISM FASTOKENS_BPE_THREADS
unset DYN_ZMQ_IO_THREADS LD_PRELOAD
export OTEL_SDK_DISABLED=true DYN_LOG=warn
export TELEMETRY_DIR="$ROOT/artifacts/network-$LABEL"
test -f "$ROOT/control/BUILD_COMPLETE"
python3 "$ROOT/harness/freeze.py" --root "$ROOT" --verify
srun --jobid="$SLURM_JOB_ID" --overlap --input=none --nodes=5 --ntasks=5 --ntasks-per-node=1 --cpus-per-task=1 \
 --cpu-bind=none --kill-on-bad-exit=0 \
 bash -c 'exec python3 "$1" --output-dir "$TELEMETRY_DIR/rank-$SLURM_PROCID" --interface enP6p3s0f1np1' network-collector "$ROOT/harness/rootcause_telemetry.py" &
MONITOR_STEP=$!
cleanup_monitor() { kill "$MONITOR_STEP" 2>/dev/null || true; wait "$MONITOR_STEP" 2>/dev/null || true; }
trap cleanup_monitor EXIT INT TERM
srun --jobid="$SLURM_JOB_ID" --overlap --input=none --nodes=5 --ntasks=5 --ntasks-per-node=1 --cpus-per-task=144 \
 --cpu-bind=none --kill-on-bad-exit=1 --label \
 bash -c 'ulimit -Sn 131072; mkdir -p "$SLURM_TMPDIR"; exec python3 "$1" --config "$2"' campaign-rank "$ROOT/harness/saturation_agent.py" "$ROOT/configs/$LABEL.json"
test -f "$ROOT/results/$SLURM_JOB_ID-main-$LABEL/COMPLETE"
