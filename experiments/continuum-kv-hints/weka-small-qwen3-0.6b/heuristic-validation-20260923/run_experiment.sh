#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
experiment_root="$(cd "$script_dir/.." && pwd)"
output_root="$experiment_root/artifacts/heuristic-validation-20260923"
run_ablation="$experiment_root/common/run_ablation.sh"
generator="$script_dir/generate_schedules.py"

export CONTINUUM_IMAGE="${CONTINUUM_IMAGE:-continuum-weka-qwen-prioritized-retention:local}"
export RUN_ID="heuristic-validation-20260923"
export OUTPUT_ROOT="$output_root"
export NUM_TRACES=4
export MAX_CONTEXT_LENGTH=32768
export MAX_REQUESTS_PER_TRACE=120
export TRACE_SELECTION_MODE=highest-request-count
export MAX_SOURCE_SPAN_SECONDS=600
export TOKEN_SCALE_FACTOR=8
export TIME_SCALE_FACTOR=1
export MAX_THINK_TIME=1000000
export CONCURRENCY=4
export BENCHMARK_DURATION=600
export NUM_SESSIONS=4
export TRAJECTORY_START_MIN_RATIO=0
export TRAJECTORY_START_MAX_RATIO=0
export SYSTEM_IDLE_GAP_CAP_SECONDS=1000000
export GPU_BLOCKS=1024
export GPU_DEVICE="${GPU_DEVICE:-0}"
export RETENTION_PRIORITY=10
export RETENTION_MAX_FRACTION=0.25
export CAPTURE_RAW_KV_EVENTS=0
export REQUIRE_IDLE_GPU=1

run_case() {
  local case_name="$1"
  local policy_mode="$2"
  local schedule="${3:-}"
  if [[ -n "$schedule" ]]; then
    export RETENTION_ORACLE_SCHEDULE="$schedule"
  else
    unset RETENTION_ORACLE_SCHEDULE
  fi
  "$run_ablation" "$case_name" "$policy_mode"
  sleep 15
}

generate_round_schedules() {
  local round="$1"
  local baseline_dir="$output_root/round-${round}-baseline"
  local schedule_dir="$output_root/schedules/round-${round}"
  python3 "$generator" \
    --dataset "$output_root/generated-dataset/traces.jsonl" \
    --profile "$baseline_dir/aiperf/profile.jsonl" \
    --server-metrics "$baseline_dir/aiperf/profile_server_metrics.json" \
    --min-missing-blocks 24 \
    --min-removal-pressure-5s 4 \
    --fixed-ttl-seconds 10 \
    --output-dir "$schedule_dir"
}

run_case "round-1-baseline" none
generate_round_schedules 1
run_case \
  "round-1-heuristic-fixed-10s" \
  inferred-tool-retain \
  "$output_root/schedules/round-1/fixed-10s.json"
run_case \
  "round-1-heuristic-future-ttl" \
  inferred-tool-retain \
  "$output_root/schedules/round-1/future-derived-ttl.json"

python3 "$script_dir/summarize.py" \
  --run-root "$output_root" \
  --output-dir "$output_root/summary"
