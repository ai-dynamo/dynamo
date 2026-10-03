#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Slurm batch script (submitted by submit_eval.sh; runs on the Slurm CPU node).
#   node_eval.sh BUNDLE_NFS OUT_NFS SHARD MAX_WALL_S [extra lr-eval args...]
# Stages the bundle node-local, verifies trace SHA-256s, runs the bundle's lr-eval shard and syncs
# results back to OUT_NFS every LR_SYNC_S seconds and at exit, so a killed job still returns what
# finished. Slots: LR_SLOTS (default: physical cores).
set -euo pipefail
BUNDLE_NFS="$1"; OUT_NFS="$2"; SHARD="$3"; MAX_WALL="$4"; shift 4
LOCAL="${LR_NODE_LOCAL:-/tmp}/lr-$USER-${SLURM_JOB_ID:-nojob}-$(basename "$BUNDLE_NFS")"
mkdir -p "$OUT_NFS"
exec > >(tee -a "$OUT_NFS/node.log") 2>&1
# Slurm runs a spooled copy of this script, so locate the lane dir from the bundle path.
LANE_REMOTE="${LR_LANE_REMOTE:-$(dirname "$(dirname "$BUNDLE_NFS")")/lane}"
source "$LANE_REMOTE/node_common.sh"
echo "[node_eval] $(date -Is) job=$JOB host=$(hostname) shard=$SHARD local=$LOCAL"

lane_stage
lane_probe
lane_sampler_start
lane_periodic_sync_start

t2=$(date +%s.%N)
if [ "${LR_PREMAT:-1}" = 1 ]; then
  # Parallel CRN replicate materialization (the harness would do it serially while planning).
  read -r REPEATS OFFSET < <("$PY" -S -c 'import json,sys; m=json.load(open(sys.argv[1])); print(m["repeats"], m["repeat_offset"])' "$LOCAL/MANIFEST.json")
  PYTHONPATH="$LOCAL/site" "$PY" -S "$LANE_REMOTE/prematerialize.py" --root "$LOCAL" \
    --cells "$LOCAL/cells.jsonl" --k "$OFFSET-$((OFFSET + REPEATS - 1))" --shard "$SHARD" \
    --workers "$SLOTS" | tee "$OUT_NFS/prematerialize.json"
fi
t2b=$(date +%s.%N)
MAX_WALL=$(lane_clamp_wall "$MAX_WALL" "${LR_TAIL_MARGIN_S:-600}")
echo "[node_eval] lr-eval --max-wall-seconds $MAX_WALL (job ends at ${SLURM_JOB_END_TIME:-?})"
set +e
PYTHON="$PY" LR_SLOTS="$SLOTS" bash "$LOCAL/run.sh" --shard "$SHARD" --max-wall-seconds "$MAX_WALL" "$@"
rc=$?
set -e
t3=$(date +%s.%N)
echo "{\"rc\": $rc, \"stage_s\": $STAGE_S, \"prematerialize_s\": $(awk "BEGIN {print $t2b - $t2}"), \"eval_wall_s\": $(awk "BEGIN {print $t3 - $t2b}"), \"started\": $t2, \"finished\": $t3, \"slots\": $SLOTS}" > "$OUT_NFS/timing.json"
echo "[node_eval] $(date -Is) lr-eval exit $rc"
exit 0
