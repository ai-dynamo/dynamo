#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail
SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"
source "$SCRIPT_DIR/launch_utils.sh"
trap dynamo_exit_trap EXIT
trap 'dynamo_reap_and_exit 0' TERM INT

# The optional pure-Python wheel is mounted, not installed into the base image.
if [ -f "$SCRIPT_DIR/kvcr.whl" ]; then
    export PYTHONPATH="$SCRIPT_DIR/kvcr.whl${PYTHONPATH:+:$PYTHONPATH}"
fi

health_args=()
if [ "$POD_INDEX" = "0" ]; then
    state_namespace="${DYN_NAMESPACE}${DYN_NAMESPACE_WORKER_SUFFIX:+-${DYN_NAMESPACE_WORKER_SUFFIX}}"
    env DYN_NAMESPACE="$state_namespace" DYN_HEALTH_CHECK_ENABLED=false \
        python3 -m dynamo.kv_state_agent --max-slots "$KVCR_CACHE_SLOT_COUNT" &
    health_args+=(--health-dependency-url "http://127.0.0.1:${DYN_SYSTEM_PORT:-9091}/live")
fi

python3 -m kvcr.kvcr_service \
    --socket-path /run/kvcr/memory.sock \
    --pool-dir /dev/shm/kvcr \
    --guard-count 1 \
    --pool-sizes-gb 2 \
    --compatibility-digest "$DYNAMO_KVCR_COMPATIBILITY_DIGEST" \
    --health-host 0.0.0.0 --health-port "${KVCR_HEALTH_PORT:-9092}" \
    "${health_args[@]}" &

# Use Dynamo's first-child-exit and process-group cleanup semantics. Kubernetes'
# terminationGracePeriodSeconds is the final bound for unresponsive children.
wait_any_exit
