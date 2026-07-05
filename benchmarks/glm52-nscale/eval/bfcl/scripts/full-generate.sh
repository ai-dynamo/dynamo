#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common.sh"

if [[ $# -lt 2 || $# -gt 3 ]]; then
  echo "usage: $0 VARIANT CAMPAIGN_PHASE [RUN_DIR]" >&2
  exit 2
fi

variant="$1"
campaign_phase="$2"
validate_variant "${variant}"
validate_campaign_phase full "${campaign_phase}"
require_install
require_endpoint

categories="${BFCL_CATEGORIES:-all_scoring}"
require_serpapi_for_categories "${categories}"
run_dir="$(new_run_dir "${variant}" full "${campaign_phase}" "${3:-}")"
prepare_run "${variant}" full "${campaign_phase}" "${categories}" "${run_dir}"

command=(
  "${BFCL_BIN}" generate
  --model "${BFCL_MODEL}"
  --test-category "${categories}"
  --temperature "${BFCL_TEMPERATURE}"
  --num-threads "${BFCL_NUM_THREADS}"
)
if [[ "${BFCL_INCLUDE_INPUT_LOG}" == "1" ]]; then
  command+=(--include-input-log)
fi
if [[ "${BFCL_ALLOW_OVERWRITE:-0}" == "1" ]]; then
  command+=(--allow-overwrite)
fi

run_logged "${run_dir}/logs/generate.log" "${command[@]}"

validation_command=(
  "${BFCL_PYTHON}" "${SCRIPT_DIR}/validate_run.py" validate
  --run-dir "${run_dir}"
  --phase generation
  --expected-commit "${BFCL_GORILLA_COMMIT}"
  --expected-variant "${variant}"
  --expected-campaign-phase "${campaign_phase}"
  --campaign-source-metadata "${CAMPAIGN_SOURCE_METADATA}"
  --campaign-source-root "${CAMPAIGN_SOURCE_ROOT}"
)
if [[ "${BFCL_REQUIRE_FULL:-0}" == "1" ]]; then
  validation_command+=(--require-full)
fi
run_logged "${run_dir}/logs/validate-generation.log" "${validation_command[@]}"
echo "RUN_DIR=${run_dir}"
