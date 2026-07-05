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
if [[ "${BFCL_CATEGORIES:-all_scoring}" != "all_scoring" ]]; then
  echo "run-full requires the pinned 5,106-case all_scoring population; use the generate/evaluate scripts for partial diagnostics." >&2
  exit 2
fi
if [[ "${BFCL_MODEL}" != "zai-org/GLM-5.2-FC" ]]; then
  echo "run-full requires BFCL_MODEL=zai-org/GLM-5.2-FC, got ${BFCL_MODEL}." >&2
  exit 2
fi
export BFCL_CATEGORIES=all_scoring
export BFCL_REQUIRE_FULL=1
run_dir="$(new_run_dir "${variant}" full "${campaign_phase}" "${3:-}")"
"${SCRIPT_DIR}/full-generate.sh" "${variant}" "${campaign_phase}" "${run_dir}"
"${SCRIPT_DIR}/full-evaluate.sh" "${variant}" "${campaign_phase}" "${run_dir}"
