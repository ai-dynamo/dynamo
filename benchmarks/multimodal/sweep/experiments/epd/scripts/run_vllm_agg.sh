#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Launch a one-GPU vLLM Aggregated service.

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=config.sh
source "$SCRIPT_DIR/config.sh"

setup_launcher vllm aggregate "$@"
launch frontend "${FRONTEND_ENV[@]}" \
    python3 -m dynamo.frontend "${FRONTEND_ARGS[@]}"
launch aggregate "${AGG_ENV[@]}" \
    python3 -m dynamo.vllm "${COMMON_ARGS[@]}" "${AGG_MEM_ARGS[@]}"
wait_for_exit
