#!/usr/bin/env bash
set -euo pipefail
ROOT=/lustre/fsw/coreai_comparch_trtllm/jothomson/dynamo-velo-response-20260923
mkdir -p "$ROOT/manifests" "$ROOT/control" "$ROOT/logs"
scontrol show hostnames "$SLURM_JOB_NODELIST" > "$ROOT/manifests/nodes.txt"
printf '%s\n' "$SLURM_JOB_ID" > "$ROOT/control/job-id"
while [[ ! -f "$ROOT/control/RELEASE" ]]; do sleep 15; done
