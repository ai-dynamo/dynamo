#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Submit one live deployment job to the GPU cluster from the workstation.
#
#   submit.sh --plan-id ID --commit SHA [--nodes 1|2] [--partition P] [--time 01:55:00]
#             [--policies "slug ..."] [--payload CLUSTER_PATH] [--payload-timeout S] [--tag TAG]
#             [--purpose TEXT] [--env KEY=VALUE]... [--test-only]
#
# Requires stage_source.sh to have staged COMMIT and the plan. --partition defaults to LR_PARTITION.
# Without --test-only it takes the cooperative GPU hold lock (hold_lock.sh; hard expiry = time limit + 15 min, at most 3 h), submits a
# whole-node exclusive job running job_node.sh, and records the job with its cancel command in
# CR/facts/live.json. finish.sh records the end state and releases the lock.
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$here/common.sh"
lr_need LR_SSH_ALIAS LR_ACCOUNT LR_PARTITION LR_LIVE_ROOT LR_CR
ssh_alias="$LR_SSH_ALIAS"
account="$LR_ACCOUNT"
nodes=1 partition="$LR_PARTITION" time_limit=01:55:00 policies="" payload="" payload_timeout=5400
tag="" purpose="live deployment" test_only=0 plan_id="" commit="" extra_env=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --plan-id) plan_id=$2; shift 2 ;;
    --commit) commit=$2; shift 2 ;;
    --nodes) nodes=$2; shift 2 ;;
    --partition) partition=$2; shift 2 ;;
    --time) time_limit=$2; shift 2 ;;
    --policies) policies=$2; shift 2 ;;
    --payload) payload=$2; shift 2 ;;
    --payload-timeout) payload_timeout=$2; shift 2 ;;
    --tag) tag=$2; shift 2 ;;
    --purpose) purpose=$2; shift 2 ;;
    --env) extra_env+=("$2"); shift 2 ;;
    --test-only) test_only=1; shift ;;
    *) die "unknown argument $1" ;;
  esac
done
[[ -n "$plan_id" && "$commit" =~ ^[0-9a-f]{40}$ ]] || die "need --plan-id and a full 40-hex --commit"
[[ "$nodes" == 1 || "$nodes" == 2 ]] || die "--nodes must be 1 or 2"
tag="${tag:-lr-live-n$((4 * nodes))-$(date -u +%Y%m%dT%H%M%SZ)}"
deploy="$LR_LIVE_ROOT/src/${commit:0:12}/benchmarks/learned_routing/live/deploy"
plan="$LR_LIVE_ROOT/plans/$plan_id"
ssh -o BatchMode=yes "$ssh_alias" "test -f '$deploy/job_node.sh' && test -f '$plan/PLAN.json' \
  && test -s '$LR_LIVE_ROOT/src/${commit:0:12}.source-manifest.json' && mkdir -p '$LR_LIVE_ROOT/jobs' '$LR_RUNS_ROOT'" \
  || die "commit or plan not staged on the cluster; run stage_source.sh"

env_file="$LR_LIVE_ROOT/jobs/$tag.env"
{
  printf '%s=%q\n' LR_COMMIT "$commit" LR_PLAN_DIR "$plan" LR_DEPLOY_DIR "$deploy" LR_RUN_TAG "$tag" \
    LR_POLICIES "$policies" LR_PAYLOAD "$payload" LR_PAYLOAD_TIMEOUT_S "$payload_timeout"
  for kv in "${extra_env[@]}"; do
    [[ "$kv" =~ ^LR_[A-Z0-9_]+= ]] || die "--env takes LR_*=VALUE, got $kv"
    printf '%s=%q\n' "${kv%%=*}" "${kv#*=}"
  done
} | ssh -o BatchMode=yes "$ssh_alias" "set -o noclobber; cat > '$env_file'" || die "could not write $env_file (tag reused?)"

sbatch_args=(-A "$account" -p "$partition" -N "$nodes" --exclusive --mem=0 --time "$time_limit"
  --job-name "$account-learnedrouting.$tag" --no-requeue --output "$LR_RUNS_ROOT/slurm-%j-$tag.out"
  --export "ALL,LR_JOB_ENV=$env_file")
if [[ $test_only -eq 1 ]]; then
  ssh -o BatchMode=yes "$ssh_alias" sbatch --test-only "${sbatch_args[@]}" "$deploy/job_node.sh"
  exit 0
fi

limit_s=$(awk -F: '{ if (NF == 3) print $1 * 3600 + $2 * 60 + $3; else print $1 * 60 + $2 }' <<< "${time_limit#*-}")
hours=$(awk -v s="$limit_s" 'BEGIN { h = (s + 900) / 3600; if (h > 3) h = 3; printf "%.3f", h }')
(( limit_s + 900 <= 10800 )) || die "time limit too long for the 3 h lock; shorten --time"
bash "$here/hold_lock.sh" acquire "learned-routing live ($purpose)" "$tag" "$hours"
if ! job_id=$(ssh -o BatchMode=yes "$ssh_alias" sbatch --parsable "${sbatch_args[@]}" "$deploy/job_node.sh"); then
  bash "$here/hold_lock.sh" release "$tag"
  die "sbatch failed; lock released"
fi
job_id="${job_id%%;*}"
python3 "$here/live_facts.py" add-job --job-id "$job_id" \
  --field cluster="${LR_CLUSTER:-gpu}" --field account="$account" --field partition="$partition" \
  --field nodes="$nodes" --field gpus="$((8 * nodes))" --field workers="$((4 * nodes))" \
  --field time_limit="$time_limit" --field tag="$tag" --field purpose="$purpose" \
  --field commit="$commit" --field plan_id="$plan_id" --field policies="$policies" \
  --field payload="$payload" --field env_file="$env_file" \
  --field run_dir="$LR_RUNS_ROOT/$job_id-$tag" --field slurm_output="$LR_RUNS_ROOT/slurm-$job_id-$tag.out" \
  --field lock_start="$tag" --field cancel="ssh $ssh_alias scancel $job_id" --field state=SUBMITTED \
  --field submitted_utc="$(date -u +%FT%TZ)"
echo "submitted job $job_id (tag $tag); cancel: ssh $ssh_alias scancel $job_id"
echo "when it ends: bash $here/finish.sh $job_id $tag"
