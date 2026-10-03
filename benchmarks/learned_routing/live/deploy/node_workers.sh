#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# All vLLM workers of one node, as one long-running Slurm step inside the container.
# Usage: node_workers.sh PLAN_DIR LOG_DIR
# Starts LR_WORKERS_PER_NODE worker.sh children (local indices 0..n-1), exits when any child exits,
# and stops the rest with SIGTERM on exit (Slurm forwards SIGTERM to this step on teardown).
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
[[ $# -eq 2 ]] || die "usage: node_workers.sh PLAN_DIR LOG_DIR"
plan_dir=$1 log_dir=$2
mkdir -p "$log_dir"
children=()
cleanup() {
  trap - EXIT TERM INT
  local child
  for child in "${children[@]}"; do kill -TERM "$child" 2>/dev/null || true; done
  for child in "${children[@]}"; do wait "$child" 2>/dev/null || true; done
}
trap cleanup EXIT TERM INT
for ((i = 0; i < LR_WORKERS_PER_NODE; i++)); do
  bash "$LR_DEPLOY_DIR/worker.sh" "$i" "$plan_dir" "$log_dir" > "$log_dir/worker-$i.log" 2>&1 &
  children+=("$!")
done
printf '%s host=%s worker_pids=%s\n' "$(date -u +%FT%TZ)" "$(hostname)" "${children[*]}" \
  | tee "$log_dir/worker-pids.txt"
set +e
wait -n "${children[@]}"
status=$?
set -e
echo "a worker exited with status $status; stopping the others" >&2
exit "$status"
