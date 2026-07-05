#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common.sh"

if [[ $# -ne 2 ]]; then
  echo "usage: $0 VARIANT RUN_DIR" >&2
  exit 2
fi

variant="$1"
run_dir="$2"
validate_variant "${variant}"
require_install
require_endpoint

if [[ ! -d "${run_dir}/result" ]]; then
  echo "No generated BFCL results under ${run_dir}/result." >&2
  exit 1
fi

export BFCL_PROJECT_ROOT="${run_dir}"
categories="simple_python,parallel,irrelevance,multi_turn_base"
validation_common=(
  "${BFCL_PYTHON}" "${SCRIPT_DIR}/validate_run.py" validate
  --run-dir "${run_dir}"
  --expected-commit "${BFCL_GORILLA_COMMIT}"
  --expected-variant "${variant}"
  --expected-campaign-phase validation
  --expected-mode smoke
  --campaign-source-metadata "${CAMPAIGN_SOURCE_METADATA}"
  --campaign-source-root "${CAMPAIGN_SOURCE_ROOT}"
  --population-config "${ROOT_DIR}/config/smoke-cases.json"
  --require-perfect
)
run_logged "${run_dir}/logs/validate-generation.log" \
  "${validation_common[@]}" --phase generation

run_logged "${run_dir}/logs/evaluate.log" \
  "${BFCL_BIN}" evaluate \
  --model "${BFCL_MODEL}" \
  --test-category "${categories}" \
  --partial-eval

"${BFCL_PYTHON}" "${SCRIPT_DIR}/summarize_results.py" "${run_dir}"
run_logged "${run_dir}/logs/validate-complete.log" \
  "${validation_common[@]}" --phase complete
echo "RUN_DIR=${run_dir}"
