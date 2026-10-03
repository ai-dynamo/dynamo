#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Close out one live job from the workstation: record its Slurm end state in CR/facts/live.json and
# release the cooperative GPU hold lock it holds.
#   finish.sh JOB_ID TAG [--cancel]
# Without --cancel a job that is still pending or running is left alone (exit 1). With --cancel the
# exact job ID is cancelled first (never by name or pattern).
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$here/common.sh"
[[ $# -ge 2 ]] || { echo "usage: finish.sh JOB_ID TAG [--cancel]" >&2; exit 2; }
job_id=$1 tag=$2 cancel=${3:-}
lr_need LR_SSH_ALIAS LR_CR
ssh_alias="$LR_SSH_ALIAS"
[[ "$job_id" =~ ^[0-9]+$ ]] || { echo "error: JOB_ID must be numeric" >&2; exit 2; }
state() {
  ssh -o BatchMode=yes "$ssh_alias" "sacct -X -n -P -j $job_id -o State,Elapsed,Start,End,NodeList,ExitCode" | head -1
}
current="$(state)"
case "${current%%|*}" in
  PENDING | RUNNING | REQUEUED | SUSPENDED | CONFIGURING | COMPLETING)
    if [[ "$cancel" != --cancel ]]; then
      echo "job $job_id is ${current%%|*}; pass --cancel to cancel it" >&2
      exit 1
    fi
    ssh -o BatchMode=yes "$ssh_alias" "scancel $job_id"
    for _ in $(seq 60); do
      current="$(state)"
      case "${current%%|*}" in PENDING | RUNNING | COMPLETING | CONFIGURING) sleep 5 ;; *) break ;; esac
    done
    ;;
esac
IFS='|' read -r st elapsed start end nodelist exit_code <<< "$current"
python3 "$here/live_facts.py" set-job --job-id "$job_id" --field state="$st" --field elapsed="$elapsed" \
  --field start="$start" --field end="$end" --field nodelist="$nodelist" --field exit_code="$exit_code" \
  --field closed_utc="$(date -u +%FT%TZ)"
if [[ "$(bash "$here/hold_lock.sh" status | awk -F'\t' '$1 == "start" {print $2}')" == "$tag" ]]; then
  bash "$here/hold_lock.sh" release "$tag"
fi
