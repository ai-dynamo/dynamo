#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common.sh"

if [[ $# -ne 3 ]]; then
  echo "usage: $0 VARIANT CAMPAIGN_PHASE RUN_DIR" >&2
  exit 2
fi

variant="$1"
campaign_phase="$2"
run_dir="$3"
validate_variant "${variant}"
validate_campaign_phase full "${campaign_phase}"
require_install

if [[ ! -d "${run_dir}/result" ]]; then
  echo "No generated BFCL results under ${run_dir}/result." >&2
  exit 1
fi

export BFCL_PROJECT_ROOT="${run_dir}"
categories="$(
  "${BFCL_PYTHON}" "${SCRIPT_DIR}/validate_run.py" metadata-value \
    --run-dir "${run_dir}" \
    --field categories
)"
model="$(
  "${BFCL_PYTHON}" "${SCRIPT_DIR}/validate_run.py" metadata-value \
    --run-dir "${run_dir}" \
    --field model
)"
validation_common=(
  "${BFCL_PYTHON}" "${SCRIPT_DIR}/validate_run.py" validate
  --run-dir "${run_dir}"
  --expected-commit "${BFCL_GORILLA_COMMIT}"
  --expected-variant "${variant}"
  --expected-campaign-phase "${campaign_phase}"
  --campaign-source-metadata "${CAMPAIGN_SOURCE_METADATA}"
  --campaign-source-root "${CAMPAIGN_SOURCE_ROOT}"
)
if [[ "${BFCL_REQUIRE_FULL:-0}" == "1" ]]; then
  validation_common+=(--require-full)
fi
run_logged "${run_dir}/logs/validate-generation.log" \
  "${validation_common[@]}" --phase generation

run_logged "${run_dir}/logs/evaluate.log" \
  "${BFCL_BIN}" evaluate \
  --model "${model}" \
  --test-category "${categories}"

"${BFCL_PYTHON}" "${SCRIPT_DIR}/summarize_results.py" "${run_dir}"
run_logged "${run_dir}/logs/validate-complete.log" \
  "${validation_common[@]}" --phase complete
echo "RUN_DIR=${run_dir}"
