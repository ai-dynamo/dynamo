#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
experiment_dir="$(cd "$script_dir/.." && pwd)"

export RUN_ID="${RUN_ID:-oracle-retention-20260921-$(date -u +%Y%m%dT%H%M%SZ)}"
export TRACE_SELECTION_MODE=highest-request-count
export MAX_SOURCE_SPAN_SECONDS=600
export TOKEN_SCALE_FACTOR=8
export TIME_SCALE_FACTOR=1
export MAX_THINK_TIME=1000000
export TRAJECTORY_START_MIN_RATIO=0
export TRAJECTORY_START_MAX_RATIO=0
export SYSTEM_IDLE_GAP_CAP_SECONDS=1000000
export BENCHMARK_DURATION=600
export REQUEST_COUNT=
export NUM_SESSIONS=4
export CONCURRENCY=4
export GPU_BLOCKS="${GPU_BLOCKS:-1024}"

"$experiment_dir/run_matrix.sh"
