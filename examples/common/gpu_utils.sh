#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Shared GPU utility functions for launch scripts.
#
# Usage:
#   source "$(dirname "$(readlink -f "$0")")/../common/gpu_utils.sh"
#   # or with SCRIPT_DIR already set:
#   source "$SCRIPT_DIR/../common/gpu_utils.sh"
#
# Functions:
#   gpu_gb_to_fraction <gib>          Convert absolute GiB to a fraction of total VRAM
#   gpu_memory_fraction <gib>   Like gpu_gb_to_fraction but respects DYN_GPU_MEMORY_FRACTION_OVERRIDE

# gpu_gb_to_fraction <gib> [gpu_index]
#
# Prints the fraction of total GPU VRAM that <gib> GiB represents.
# Useful for converting portable absolute memory requirements to
# engine-specific fraction parameters (--gpu-memory-utilization, etc).
#
# Examples:
#   gpu_gb_to_fraction 4        # on 48 GiB GPU → 0.09
#   gpu_gb_to_fraction 16       # on 48 GiB GPU → 0.34
#   gpu_gb_to_fraction 4 1      # query GPU index 1 instead of 0
#
# The result is ceil-rounded to 2 decimal places with a minimum of 0.05
# and a maximum of 0.95.
gpu_gb_to_fraction() {
    local gib=${1:?usage: gpu_gb_to_fraction <gib> [gpu_index]}
    local gpu_idx=${2:-0}

    local total_mib
    total_mib=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits -i "$gpu_idx" 2>/dev/null)
    if [[ -z "$total_mib" || "$total_mib" -eq 0 ]]; then
        echo "gpu_gb_to_fraction: failed to query GPU $gpu_idx total memory" >&2
        return 1
    fi

    local total_gib
    total_gib=$(awk -v t="$total_mib" 'BEGIN { printf "%.1f", t / 1024 }')

    if awk -v gib="$gib" -v total="$total_mib" 'BEGIN { exit (gib * 1024 > total) ? 0 : 1 }'; then
        echo "" >&2
        echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!" >&2
        echo "WARNING: Requested ${gib} GiB but GPU $gpu_idx only has ${total_gib} GiB total." >&2
        echo "The model likely won't fit. Consider a GPU with more VRAM" >&2
        echo "or reduce the model size (quantization, smaller model, etc)." >&2
        echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!" >&2
        echo "" >&2
    fi

    # fraction = gib * 1024 / total_mib, ceil to 2 decimals, clamp [0.05, 0.95]
    awk -v gib="$gib" -v total="$total_mib" 'BEGIN {
        frac = (gib * 1024) / total
        # ceil to 2 decimal places
        frac = int(frac * 100 + 0.99) / 100
        if (frac < 0.05) frac = 0.05
        if (frac > 0.95) frac = 0.95
        printf "%.2f\n", frac
    }'
}

