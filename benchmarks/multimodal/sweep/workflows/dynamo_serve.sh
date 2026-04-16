#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Dynamo serve workflow: frontend + backend.
# Prerequisites: nats-server and etcd must already be running (via MCP restart_infra).
#
# Usage (by orchestrator): bash dynamo_serve.sh --model <model> [extra vllm args...]

MODEL=""
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model) MODEL="$2"; shift 2 ;;
        *) EXTRA_ARGS+=("$1"); shift ;;
    esac
done

if [[ -z "$MODEL" ]]; then
    echo "ERROR: --model is required" >&2
    exit 1
fi

# Kill all children on exit so ServerManager.stop() cleanly shuts everything down.
trap 'kill 0; wait' SIGTERM SIGINT EXIT

python -m dynamo.frontend &

python -m dynamo.vllm \
    --model "$MODEL" \
    "${EXTRA_ARGS[@]}" &

wait
