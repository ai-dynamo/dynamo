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
run_dir="$(new_run_dir "${variant}" smoke validation "${2:-}")"
"${SCRIPT_DIR}/smoke-generate.sh" "${variant}" "${run_dir}"
"${SCRIPT_DIR}/smoke-evaluate.sh" "${variant}" "${run_dir}"
