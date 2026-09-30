#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Full benchmark run. NOT executed during setup — trigger this yourself.
#   bench/run_aiperf.sh
# Requires bench/start.sh to have been run first (frontend + workers up).
set -euo pipefail
cd "$(dirname "$0")"
source ./env.sh

STAMP="$(date +%Y%m%d-%H%M%S)"
ART="$RESULTS_DIR/aiperf-$STAMP"
echo "[aiperf] artifacts -> $ART"

# Optional warmup: set WARMUP_REQUESTS to prime the shared-prefix cache + warm
# the allocator/steady-state before the measured phase (removes cold-start skew).
WARMUP_ARGS=()
if [[ -n "${WARMUP_REQUESTS:-}" ]]; then
    WARMUP_ARGS=(--warmup-request-count "$WARMUP_REQUESTS")
    echo "[aiperf] warmup: $WARMUP_REQUESTS requests"
fi

# aiperf client pinned to the non-frontend cores so it never competes with the
# 4-core frontend under test.
taskset -c "$OTHER_CORES" "$AIPERF" profile \
    --model "$MODEL" \
    --tokenizer "$MODEL" \
    --url "http://localhost:${HTTP_PORT}" \
    --endpoint-type chat \
    --streaming \
    --shared-system-prompt-length 48000 \
    --user-context-prompt-length 12000 \
    --num-dataset-entries 10000 \
    --output-tokens-mean 500 \
    --conversation-turn-mean 4 \
    --concurrency "${CONCURRENCY:-256}" \
    --request-count "${REQUEST_COUNT:-2048}" \
    "${WARMUP_ARGS[@]}" \
    --extra-inputs "ignore_eos:true" \
    --artifact-dir "$ART"

echo "[aiperf] done. Results in $ART"
