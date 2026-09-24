#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
matrix_status=0

for case_dir in \
  a-no-hints \
  b-parent-retain \
  c-root-final-evict \
  d-combined; do
  if ! "$script_dir/ablations/$case_dir/run.sh"; then
    matrix_status=1
  fi
done

python "$script_dir/common/summarize_matrix.py" "$script_dir/artifacts/$RUN_ID"
python "$script_dir/common/summarize_parent_resumes.py" "$script_dir/artifacts/$RUN_ID"
echo "Matrix results written to $script_dir/artifacts/$RUN_ID"
exit "$matrix_status"
