#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Launch a one-GPU SGLang colocated EPD service.

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=config.sh
source "$SCRIPT_DIR/config.sh"

setup_launcher sglang epd "$@"
launch frontend "${FRONTEND_ENV[@]}" \
    python3 -m dynamo.frontend "${FRONTEND_ARGS[@]}"

# Encoder workers
launch encoder-0 "${ENCODER0_ENV[@]}" \
    python3 -m dynamo.sglang "${ENCODER_ARGS[@]}"

launch encoder-1 "${ENCODER1_ENV[@]}" \
    python3 -m dynamo.sglang "${ENCODER_ARGS[@]}"

# PD worker
launch pd "${PD_ENV[@]}" \
    python3 -m dynamo.sglang \
    "${COMMON_ARGS[@]}" "${PD_MEM_ARGS[@]}" \
    --dedicated-mm-encoder --disaggregation-mode pd
wait_for_exit
