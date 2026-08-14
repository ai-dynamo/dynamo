#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

export DYN_DISCOVERY_BACKEND="${DYN_DISCOVERY_BACKEND:-file}"
export DYN_EVENT_PLANE="${DYN_EVENT_PLANE:-zmq}"

worker_pids=()

cleanup() {
    for pid in "${worker_pids[@]}"; do
        kill "$pid" 2>/dev/null || true
    done
    wait "${worker_pids[@]}" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

for stage in encoder classifier generator; do
    python3 -m examples.custom_backend.workflow_remote.worker "$stage" &
    worker_pids+=("$!")
done

python3 -m examples.custom_backend.workflow_remote.client
