#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common.sh"

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "usage: $0 VARIANT [RUN_DIR]" >&2
  exit 2
fi

variant="$1"
validate_variant "${variant}"
require_install
require_endpoint

campaign_phase=validation
categories="simple_python,parallel,irrelevance,multi_turn_base"
run_dir="$(new_run_dir "${variant}" smoke "${campaign_phase}" "${2:-}")"
prepare_run "${variant}" smoke "${campaign_phase}" "${categories}" "${run_dir}"
cp "${ROOT_DIR}/config/smoke-cases.json" "${run_dir}/test_case_ids_to_generate.json"

command=(
  "${BFCL_BIN}" generate
  --model "${BFCL_MODEL}"
  --run-ids
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
echo "RUN_DIR=${run_dir}"
