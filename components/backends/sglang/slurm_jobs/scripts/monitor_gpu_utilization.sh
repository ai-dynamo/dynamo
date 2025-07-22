#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Usage: ./monitor_gpu_utilization.sh [interval_seconds]

# Default interval is 2 seconds
INTERVAL=${1:-2}

# Check if nvidia-smi is available
if ! command -v nvidia-smi &> /dev/null; then
    echo "$(date '+%Y-%m-%d %H:%M:%S') Error: nvidia-smi not found"
    exit 1
fi

echo "Starting GPU utilization monitoring (checking every ${INTERVAL}s, printing only on changes)..."

PREV_UTILIZATION=""
while true; do
    CURRENT_UTILIZATION=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,nounits | paste -sd ' ' -)
    if [ $? -ne 0 ]; then
        echo "$(date '+%Y-%m-%d %H:%M:%S') Error: nvidia-smi command failed"
    else
        if [ "$CURRENT_UTILIZATION" != "$PREV_UTILIZATION" ]; then
            echo "$(date '+%Y-%m-%d %H:%M:%S') GPU Utilization: $CURRENT_UTILIZATION"
            PREV_UTILIZATION="$CURRENT_UTILIZATION"
        fi
    fi

    sleep $INTERVAL
done
