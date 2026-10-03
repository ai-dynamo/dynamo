#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Shard a cells x policies x replicates lr-eval batch over N Slurm CPU nodes (one batch job per
# shard; each job releases its node when its shard is done).
#
#   submit_eval.sh --name NAME --nodes N --repeats K [--repeat-offset O] \
#       --specs SPEC... --cells CELLS.jsonl... [--cell-filter RE] \
#       [--bundle-dir DIR] [--time HH:MM:SS] [--mem 180G] [--slots S] [--partitions P,P] \
#       [-- extra lr-eval args]
#
# Without --bundle-dir the bundle is built with `lr-eval --bundle-out CR/runs/remote/bundles/NAME`.
# Results come back with: fetch_ingest.sh --name NAME
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

NAME="" NODES=1 REPEATS="" OFFSET=0 BUNDLE="" SLOTS="" FILTER=""
SPECS=() CELLS=() EXTRA=()
while [ $# -gt 0 ]; do
  case "$1" in
    --name) NAME="$2"; shift 2 ;;
    --nodes) NODES="$2"; shift 2 ;;
    --repeats) REPEATS="$2"; shift 2 ;;
    --repeat-offset) OFFSET="$2"; shift 2 ;;
    --bundle-dir) BUNDLE="$2"; shift 2 ;;
    --time) TIME="$2"; shift 2 ;;
    --mem) MEM="$2"; shift 2 ;;
    --slots) SLOTS="$2"; shift 2 ;;
    --partitions) PARTITIONS="$2"; shift 2 ;;
    --cell-filter) FILTER="$2"; shift 2 ;;
    --specs) shift; while [ $# -gt 0 ] && [ "${1#--}" = "$1" ]; do SPECS+=("$1"); shift; done ;;
    --cells) shift; while [ $# -gt 0 ] && [ "${1#--}" = "$1" ]; do CELLS+=("$1"); shift; done ;;
    --) shift; EXTRA=("$@"); break ;;
    *) die "unknown argument $1 (see the header of $0)" ;;
  esac
done
[ -n "$NAME" ] || die "--name is required"
[[ "$NAME" =~ ^[A-Za-z0-9._-]+$ ]] || die "--name must match [A-Za-z0-9._-]+"

if [ -z "$BUNDLE" ]; then
  [ -n "$REPEATS" ] && [ ${#SPECS[@]} -gt 0 ] && [ ${#CELLS[@]} -gt 0 ] \
    || die "--repeats, --specs and --cells are required without --bundle-dir"
  BUNDLE="$LOCAL_RUNS/bundles/$NAME"
  args=(--policy-spec "${SPECS[@]}" --cells "${CELLS[@]}" --repeats "$REPEATS"
        --repeat-offset "$OFFSET" --bundle-out "$BUNDLE")
  [ -n "$FILTER" ] && args+=(--cell-filter "$FILTER")
  "$LR_EVAL" --root "$CR" "${args[@]}"
fi
[ -f "$BUNDLE/MANIFEST.json" ] || die "$BUNDLE is not an lr-eval bundle"
[ -f "$BUNDLE/traces.sha256" ] || "$PY" "$LANE_DIR/lane.py" trace-list "$BUNDLE" >/dev/null
BUILD="$("$PY" -c 'import json,sys; print(json.load(open(sys.argv[1]))["build_id"])' "$BUNDLE/MANIFEST.json")"
TASKS="$("$PY" -c 'import json,sys; print(json.load(open(sys.argv[1]))["tasks"])' "$BUNDLE/MANIFEST.json")"

rssh "mkdir -p '$REMOTE_ROOT/lane'"
rsync -a "$LANE_DIR/node_common.sh" "$LANE_DIR/node_eval.sh" "$LANE_DIR/node_train.sh" \
  "$LANE_DIR/prematerialize.py" "$SSH_ALIAS:$REMOTE_ROOT/lane/"
REMOTE_BUNDLE="$(push_bundle "$BUNDLE" "$NAME" "$BUILD")"

# lr-eval stops launching replays this long before the Slurm limit (staging + tail margin).
MAX_WALL=$(( $(hms_to_s "$TIME") - ${LR_MARGIN_S:-900} ))
[ "$MAX_WALL" -gt 60 ] || die "--time too short"
[ -n "$SLOTS" ] && export LR_SBATCH_EXTRA="--export=ALL,LR_SLOTS=$SLOTS ${LR_SBATCH_EXTRA:-}"

echo "bundle $BUNDLE -> $REMOTE_BUNDLE ($TASKS tasks, build ${BUILD:0:12})"
JOBS=()
for ((i = 0; i < NODES; i++)); do
  out="$REMOTE_ROOT/returned/$NAME/shard-$i-of-$NODES"
  job="$(submit_job "lr-eval-$NAME-$i" "lr-eval $NAME shard $i/$NODES ($TASKS tasks total)" \
    "$REMOTE_ROOT/lane/node_eval.sh" "$REMOTE_BUNDLE" "$out" "$i/$NODES" "$MAX_WALL" "${EXTRA[@]}")"
  "$PY" "$LANE_DIR/lane.py" alloc-record --job "$job" --set "bundle=$REMOTE_BUNDLE" \
    --set "returned=$out" --set "local_bundle=$BUNDLE" >/dev/null
  JOBS+=("$job")
  echo "shard $i/$NODES: job $job -> $out"
done
cat > "$LOCAL_RUNS/$NAME.jobs.json" <<EOF
{"name": "$NAME", "kind": "eval", "bundle": "$BUNDLE", "remote_bundle": "$REMOTE_BUNDLE",
 "jobs": [$(printf '"%s",' "${JOBS[@]}" | sed 's/,$//')], "nodes": $NODES, "tasks": $TASKS,
 "submitted": "$(date '+%Y-%m-%d %H:%M:%S %Z')"}
EOF
echo
echo "cancel commands (exact job IDs):"
for job in "${JOBS[@]}"; do echo "  ssh $SSH_ALIAS scancel $job"; done
echo "status:  ssh $SSH_ALIAS squeue -j $(IFS=,; echo "${JOBS[*]}")"
echo "results: $(dirname "$0")/fetch_ingest.sh --name $NAME"
