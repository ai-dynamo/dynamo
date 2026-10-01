#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Launch a one-GPU vLLM colocated EPD service.

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=config.sh
source "$SCRIPT_DIR/config.sh"

setup_launcher vllm epd "$@"
launch frontend "${FRONTEND_ENV[@]}" \
    python3 -m dynamo.frontend "${FRONTEND_ARGS[@]}"

# Encoder workers
launch encoder-0 "${ENCODER0_ENV[@]}" \
    python3 -m dynamo.vllm "${ENCODER_ARGS[@]}" --disaggregation-mode encode
wait_for_worker_log encoder-0 "Starting to serve the encode worker endpoint"

launch encoder-1 "${ENCODER1_ENV[@]}" \
    python3 -m dynamo.vllm "${ENCODER_ARGS[@]}" --disaggregation-mode encode
wait_for_worker_log encoder-1 "Starting to serve the encode worker endpoint"

# PD worker
launch pd "${PD_ENV[@]}" \
    python3 -m dynamo.vllm \
    "${COMMON_ARGS[@]}" "${PD_MEM_ARGS[@]}" \
    --route-to-encoder --enable-mm-embeds --disaggregation-mode pd
wait_for_exit
