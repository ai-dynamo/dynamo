#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Run whole lr-train (CMA-ES) jobs on Slurm CPU nodes: one batch job (one node) per training run,
# each with its own result cache; finished generations sync back every chunk.
#
#   submit_train.sh --name NAME --jobs JOBS.jsonl --cells TRAIN.jsonl... [--val VAL.jsonl...] \
#       [--only RUN[,RUN...]] [--time HH:MM:SS] [--mem 180G] [--slots S] [--partitions P,P] \
#       [--bundle-dir DIR]
#
# JOBS.jsonl, one training run per line (args are lr-train options other than --root, --space,
# --cells, --val, --run-dir, --slots, --num-slots and --max-wall-seconds, which the node sets):
#   {"run": "default-tuned-s1", "space": "spaces/default_cost_fn.yaml",
#    "args": ["--budget-evals", "400", "--popsize", "12", "--seed", "1"]}
#
# Resubmitting the same command (with --bundle-dir CR/runs/remote/bundles/NAME) resumes every run
# from the checkpoint synced back to the cluster's scratch. Bring results home with
# fetch_ingest.sh --name NAME; run directories land in CR/runs/remote/returned/NAME/<run>/runs/train/.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

NAME="" JOBS_JSONL="" BUNDLE="" SLOTS="" ONLY=""
TRAIN=() VAL=()
while [ $# -gt 0 ]; do
  case "$1" in
    --name) NAME="$2"; shift 2 ;;
    --jobs) JOBS_JSONL="$2"; shift 2 ;;
    --bundle-dir) BUNDLE="$2"; shift 2 ;;
    --time) TIME="$2"; shift 2 ;;
    --mem) MEM="$2"; shift 2 ;;
    --slots) SLOTS="$2"; shift 2 ;;
    --partitions) PARTITIONS="$2"; shift 2 ;;
    --only) ONLY="$2"; shift 2 ;;
    --cells) shift; while [ $# -gt 0 ] && [ "${1#--}" = "$1" ]; do TRAIN+=("$1"); shift; done ;;
    --val) shift; while [ $# -gt 0 ] && [ "${1#--}" = "$1" ]; do VAL+=("$1"); shift; done ;;
    *) die "unknown argument $1 (see the header of $0)" ;;
  esac
done
[ -n "$NAME" ] || die "--name is required"
[[ "$NAME" =~ ^[A-Za-z0-9._-]+$ ]] || die "--name must match [A-Za-z0-9._-]+"

if [ -z "$BUNDLE" ]; then
  [ -n "$JOBS_JSONL" ] && [ ${#TRAIN[@]} -gt 0 ] || die "--jobs and --cells are required without --bundle-dir"
  "$PY" - "$JOBS_JSONL" <<'EOF'
import json, sys
reserved = {"--root", "--space", "--cells", "--val", "--run-dir", "--slots", "--num-slots", "--max-wall-seconds"}
runs = set()
for n, line in enumerate(open(sys.argv[1]), 1):
    if not line.strip():
        continue
    job = json.loads(line)
    bad = reserved & set(job.get("args", []))
    if bad or not job.get("run") or not job.get("space") or job["run"] in runs:
        sys.exit(f"{sys.argv[1]}:{n}: needs a unique run, a space and no {sorted(reserved)} (got {sorted(bad)})")
    runs.add(job["run"])
EOF
  BUNDLE="$LOCAL_RUNS/bundles/$NAME"
  "$LR_EVAL" --root "$CR" --policy-spec default --cells "${TRAIN[@]}" "${VAL[@]}" --repeats 1 \
    --bundle-out "$BUNDLE"
  mapfile -t SPACES < <("$PY" -c 'import json,sys; print("\n".join(sorted({json.loads(l)["space"] for l in open(sys.argv[1]) if l.strip()})))' "$JOBS_JSONL")
  val_args=()
  [ ${#VAL[@]} -gt 0 ] && val_args=(--val "${VAL[@]}")
  "$PY" "$LANE_DIR/lane.py" train-bundle "$BUNDLE" --train "${TRAIN[@]}" "${val_args[@]}" \
    --space "${SPACES[@]}" --jobs "$JOBS_JSONL"
  "$PY" "$LANE_DIR/lane.py" trace-list "$BUNDLE" >/dev/null
  # lr-train's parent imports numpy and cma; prove they load from site/ + site-train/ alone.
  env -i PATH=/usr/bin:/bin PYTHONPATH="$BUNDLE/site:$BUNDLE/site-train" PYTHONDONTWRITEBYTECODE=1 \
    "$PY" -S -s -P -c 'import numpy, cma, learned_routing.train_cli; print("site-train ok", numpy.__version__, cma.__version__)'
fi
[ -f "$BUNDLE/train_jobs.jsonl" ] || die "$BUNDLE is not a train bundle (no train_jobs.jsonl)"
BUILD="$("$PY" -c 'import json,sys; print(json.load(open(sys.argv[1]))["build_id"])' "$BUNDLE/MANIFEST.json")"

rssh "mkdir -p '$REMOTE_ROOT/lane'"
rsync -a "$LANE_DIR/node_common.sh" "$LANE_DIR/node_eval.sh" "$LANE_DIR/node_train.sh" \
  "$LANE_DIR/prematerialize.py" "$SSH_ALIAS:$REMOTE_ROOT/lane/"
REMOTE_BUNDLE="$(push_bundle "$BUNDLE" "$NAME" "$BUILD")"
MAX_WALL=$(( $(hms_to_s "$TIME") - ${LR_MARGIN_S:-600} ))
[ "$MAX_WALL" -gt 60 ] || die "--time too short"
[ -n "$SLOTS" ] && export LR_SBATCH_EXTRA="--export=ALL,LR_SLOTS=$SLOTS ${LR_SBATCH_EXTRA:-}"

mapfile -t RUNS < <("$PY" -c 'import json,sys; print("\n".join(json.loads(l)["run"] for l in open(sys.argv[1]) if l.strip()))' "$BUNDLE/train_jobs.jsonl")
JOBS=()
for run in "${RUNS[@]}"; do
  if [ -n "$ONLY" ] && [[ ",$ONLY," != *",$run,"* ]]; then continue; fi
  out="$REMOTE_ROOT/returned/$NAME/$run"
  job="$(submit_job "lr-train-$NAME-$run" "lr-train $NAME run $run" \
    "$REMOTE_ROOT/lane/node_train.sh" "$REMOTE_BUNDLE" "$out" "$run" "$MAX_WALL")"
  "$PY" "$LANE_DIR/lane.py" alloc-record --job "$job" --set "bundle=$REMOTE_BUNDLE" \
    --set "returned=$out" --set "local_bundle=$BUNDLE" --set "run=$run" >/dev/null
  JOBS+=("$job")
  echo "run $run: job $job -> $out"
done
[ ${#JOBS[@]} -gt 0 ] || die "no runs selected"
cat > "$LOCAL_RUNS/$NAME.jobs.json" <<EOF
{"name": "$NAME", "kind": "train", "bundle": "$BUNDLE", "remote_bundle": "$REMOTE_BUNDLE",
 "jobs": [$(printf '"%s",' "${JOBS[@]}" | sed 's/,$//')], "runs": [$(printf '"%s",' "${RUNS[@]}" | sed 's/,$//')],
 "submitted": "$(date '+%Y-%m-%d %H:%M:%S %Z')"}
EOF
echo
echo "cancel commands (exact job IDs):"
for job in "${JOBS[@]}"; do echo "  ssh $SSH_ALIAS scancel $job"; done
echo "status:  ssh $SSH_ALIAS squeue -j $(IFS=,; echo "${JOBS[*]}")"
echo "results: $(dirname "$0")/fetch_ingest.sh --name $NAME"
echo "resume:  $0 --name $NAME --bundle-dir $BUNDLE [--time ...]"
