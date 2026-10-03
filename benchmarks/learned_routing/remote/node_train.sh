#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Slurm batch script (submitted by submit_train.sh; runs on the Slurm CPU node).
#   node_train.sh BUNDLE_NFS OUT_NFS RUN MAX_WALL_S
# Runs one whole lr-train job (entry RUN of the bundle's train_jobs.jsonl) on this node with its own
# cache, in --max-wall-seconds chunks of LR_CHUNK_S (default 540). After every chunk it syncs the run
# directory and the cache records back to OUT_NFS. If OUT_NFS already holds the run (an earlier
# allocation hit its limit), the checkpoint and cache are restored first, so resubmitting the same
# command resumes exactly where it stopped.
set -euo pipefail
BUNDLE_NFS="$1"; OUT_NFS="$2"; RUN="$3"; MAX_WALL="$4"
LOCAL="${LR_NODE_LOCAL:-/tmp}/lr-$USER-${SLURM_JOB_ID:-nojob}-$(basename "$BUNDLE_NFS")-$RUN"
CHUNK="${LR_CHUNK_S:-540}"
mkdir -p "$OUT_NFS"
exec > >(tee -a "$OUT_NFS/node.log") 2>&1
LANE_REMOTE="${LR_LANE_REMOTE:-$(dirname "$(dirname "$BUNDLE_NFS")")/lane}"
source "$LANE_REMOTE/node_common.sh"
echo "[node_train] $(date -Is) job=$JOB host=$(hostname) run=$RUN local=$LOCAL"

lane_stage
lane_probe
mkdir -p "$LOCAL/runs/cache" "$LOCAL/runs/train"
if [ -d "$OUT_NFS/runs/train/$RUN" ]; then
  echo "[node_train] resuming from $OUT_NFS/runs/train/$RUN"
  rsync -a "$OUT_NFS/runs/train/$RUN" "$LOCAL/runs/train/"
  [ -d "$OUT_NFS/runs/cache/results" ] && rsync -a "$OUT_NFS/runs/cache/results" "$LOCAL/runs/cache/"
fi
lane_sampler_start
lane_periodic_sync_start

mapfile -t ARGS < <("$PY" -S -c '
import json, sys
for line in open(sys.argv[1]):
    job = json.loads(line)
    if job["run"] == sys.argv[2]:
        print(job["space"])
        for a in job.get("args", []):
            print(a)
        break
else:
    sys.exit("run not in train_jobs.jsonl")
' "$LOCAL/train_jobs.jsonl" "$RUN")
SPACE="${ARGS[0]}"
ARGS=("${ARGS[@]:1}")
VAL=()
[ -s "$LOCAL/val.jsonl" ] && VAL=(--val "$LOCAL/val.jsonl")

POOL=8 VALR=3
for ((i = 0; i < ${#ARGS[@]}; i++)); do
  case "${ARGS[$i]}" in
    --replicate-pool) POOL="${ARGS[$((i + 1))]}" ;;
    --val-replicates) VALR="${ARGS[$((i + 1))]}" ;;
  esac
done
t1b=$(date +%s.%N)
if [ "${LR_PREMAT:-1}" = 1 ]; then
  # Parallel CRN replicate materialization of the train pool and the validation replicates.
  PYTHONPATH="$LOCAL/site" "$PY" -S "$LANE_REMOTE/prematerialize.py" --root "$LOCAL" \
    --cells "$LOCAL/train.jsonl" --k "0-$((POOL - 1))" --workers "$SLOTS" | tee "$OUT_NFS/prematerialize.json"
  [ -s "$LOCAL/val.jsonl" ] && PYTHONPATH="$LOCAL/site" "$PY" -S "$LANE_REMOTE/prematerialize.py" \
    --root "$LOCAL" --cells "$LOCAL/val.jsonl" --k "0-$((VALR - 1))" --workers "$SLOTS" \
    | tee -a "$OUT_NFS/prematerialize.json"
fi
PREMAT_S=$(awk "BEGIN {print $(date +%s.%N) - $t1b}")

export PYTHONPATH="$LOCAL/site:$LOCAL/site-train" PYTHONDONTWRITEBYTECODE=1 DYN_LOG="${DYN_LOG:-warn}" LR_ROOT="$LOCAL"
# Pin CMA-ES numerics (numpy + OpenBLAS kernels) to the AVX2 code paths the campaign workstation
# uses: on AVX-512 nodes (EPYC 9654P) the default dispatch changes the CMA-ES state in the last
# bits from generation 0, so a run would not be reproducible or resumable across hosts. With these
# pins, 60-generation trajectories are bit-identical on the workstation, EPYC 7702P and EPYC 9654P.
# Replays never use numpy; this affects only the lr-train parent. Set LR_PIN_NUMERICS=0 to opt out.
if [ "${LR_PIN_NUMERICS:-1}" = 1 ]; then
  export OPENBLAS_CORETYPE="${OPENBLAS_CORETYPE:-Haswell}" OPENBLAS_NUM_THREADS=1 \
    NPY_DISABLE_CPU_FEATURES="${NPY_DISABLE_CPU_FEATURES-X86_V4 AVX512_ICL AVX512_SPR}"
fi
t2=$(date +%s.%N)
MAX_WALL=$(lane_clamp_wall "$MAX_WALL" "${LR_TAIL_MARGIN_S:-600}")
echo "[node_train] lr-train budget $MAX_WALL s (job ends at ${SLURM_JOB_END_TIME:-?})"
deadline=$(( $(date +%s) + MAX_WALL ))
rc=3
chunks=0
while [ "$rc" -eq 3 ] && [ "$(date +%s)" -lt "$deadline" ]; do
  left=$(( deadline - $(date +%s) ))
  chunk=$(( left < CHUNK ? left : CHUNK ))
  set +e
  "$PY" -S -m learned_routing.train_cli --root "$LOCAL" --space "$LOCAL/$SPACE" \
    --cells "$LOCAL/train.jsonl" "${VAL[@]}" --run-dir "$LOCAL/runs/train/$RUN" \
    --slots "$SLOTS" --num-slots "$SLOTS" --max-wall-seconds "$chunk" "${ARGS[@]}"
  rc=$?
  set -e
  chunks=$((chunks + 1))
  lane_sync_back
  echo "[node_train] $(date -Is) chunk $chunks exit $rc"
done
t3=$(date +%s.%N)
echo "{\"rc\": $rc, \"chunks\": $chunks, \"stage_s\": $STAGE_S, \"prematerialize_s\": $PREMAT_S, \"train_wall_s\": $(awk "BEGIN {print $t3 - $t2}"), \"started\": $t2, \"finished\": $t3, \"slots\": $SLOTS}" > "$OUT_NFS/timing.json"
echo "[node_train] $(date -Is) lr-train exit $rc (3 = paused at the wall limit; resubmit to resume)"
exit 0
