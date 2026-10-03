# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Shared settings for the Slurm CPU lane scripts (sourced, not executed). Site-specific values (the
# campaign root, the cluster's ssh alias, remote scratch, Slurm account, QoS and partitions) have no
# defaults: set them in site.env next to this file (copy site.env.example), in the file named by
# LR_SITE_ENV, or in the environment. Variables already set in the environment win over site.env.

LANE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Read KEY=VALUE lines (quotes and $VAR references allowed, # comments ignored) and export every key
# that is not already set.
lr_load_site_env() {
  local file="$1" line key value
  [ -f "$file" ] || return 0
  while IFS= read -r line || [ -n "$line" ]; do
    [[ "$line" =~ ^[[:space:]]*(export[[:space:]]+)?([A-Za-z_][A-Za-z0-9_]*)=(.*)$ ]] || continue
    key="${BASH_REMATCH[2]}"
    [ -z "${!key+x}" ] || continue
    eval "value=${BASH_REMATCH[3]}"
    export "$key=$value"
  done < "$file"
}
lr_load_site_env "${LR_SITE_ENV:-$LANE_DIR/site.env}"

lr_need() {
  [ -n "${!1:-}" ] || {
    echo "error: set $1 in site.env (see site.env.example) or the environment" >&2
    exit 1
  }
}
for _var in LR_CR LR_SSH_ALIAS LR_REMOTE_ROOT LR_ACCOUNT LR_QOS LR_PARTITIONS; do lr_need "$_var"; done
unset _var

WT="${LR_WT:-$(cd "$LANE_DIR/../../.." && pwd)}"
PY="${LR_PY:-$WT/.venv/bin/python}"
LR_EVAL="${LR_EVAL:-$WT/.venv/bin/lr-eval}"
CR="$LR_CR"
SSH_ALIAS="$LR_SSH_ALIAS"
REMOTE_ROOT="$LR_REMOTE_ROOT"
ACCOUNT="$LR_ACCOUNT"
QOS="$LR_QOS"
# Comma-separated zero-GPU partitions whose nodes can load the bundle (x86_64 bindings, glibc at
# least the bundle's floor). Slurm starts the job in the first listed partition that can run it.
PARTITIONS="$LR_PARTITIONS"
MEM="${LR_MEM:-180G}"
TIME="${LR_TIME:-04:00:00}"
LOCAL_RUNS="${LR_LOCAL_RUNS:-$CR/runs/remote}"
export PYTHONDONTWRITEBYTECODE=1

die() {
  echo "error: $*" >&2
  exit 1
}

rssh() {
  ssh -o BatchMode=yes -o ConnectTimeout=20 "$SSH_ALIAS" "$@"
}

# hms_to_s 04:00:00 -> 14400 ; also accepts D-HH:MM:SS and MM:SS
hms_to_s() {
  awk -v t="$1" 'BEGIN {
    d = 0; if (index(t, "-")) { split(t, a, "-"); d = a[1]; t = a[2] }
    n = split(t, p, ":"); s = 0; for (i = 1; i <= n; i++) s = s * 60 + p[i]
    print d * 86400 + s }'
}

# push_bundle LOCAL_DIR NAME BUILD_ID -> prints the remote path. Reuses an earlier bundle of the
# same bindings build as an rsync --link-dest base, so the 365 MB site/ tree is hard-linked on the
# remote scratch instead of copied again.
push_bundle() {
  local local_dir="$1" name="$2" build="$3" remote="$REMOTE_ROOT/bundles/$2" base
  base="$("$PY" "$LANE_DIR/lane.py" latest-bundle "$build")"
  rssh "mkdir -p '$REMOTE_ROOT/bundles' '$REMOTE_ROOT/returned' '$REMOTE_ROOT/jobs'"
  if [ -n "$base" ] && [ "$base" != "$remote" ]; then
    rsync -a --link-dest="$base/" "$local_dir/" "$SSH_ALIAS:$remote/" >&2
  else
    rsync -a "$local_dir/" "$SSH_ALIAS:$remote/" >&2
  fi
  "$PY" "$LANE_DIR/lane.py" note-bundle "$name" "$build" "$remote"
  echo "$remote"
}

# submit_job NAME SCRIPT ARGS... -> prints the job ID (sbatch --parsable) and records the job.
submit_job() {
  local name="$1" purpose="$2"
  shift 2
  local quoted="" arg
  for arg in "$@"; do quoted+=" $(printf '%q' "$arg")"; done
  local job
  job="$(rssh "cd '$REMOTE_ROOT/jobs' && sbatch --parsable \
    --account='$ACCOUNT' --qos='$QOS' --partition='$PARTITIONS' \
    --nodes=1 --ntasks=1 --cpus-per-task=1 --mem='$MEM' --time='$TIME' \
    --job-name='$name' --output='$REMOTE_ROOT/jobs/%j.out' \
    ${LR_SBATCH_EXTRA:-} $quoted")" || die "sbatch failed for $name"
  job="${job%%;*}"
  "$PY" "$LANE_DIR/lane.py" alloc-record --job "$job" \
    --set "name=$name" --set "purpose=$purpose" --set "account=$ACCOUNT" --set "qos=$QOS" \
    --set "partitions_requested=$PARTITIONS" --set "mem_requested=$MEM" \
    --set "time_limit_requested=$TIME" --set "submitted=$(date '+%Y-%m-%d %H:%M:%S %Z')" \
    --set "submitted_by=$(basename "$0")" --set "state=SUBMITTED" >/dev/null
  echo "$job"
}
