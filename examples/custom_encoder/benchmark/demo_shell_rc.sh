#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Interactive shell setup for the side-by-side custom-encoder demo.

case "${DYN_DEMO_SIDE:-}" in
    control)
        DYN_DEMO_LABEL="CONTROL"
        DYN_DEMO_COLOR="1;36"
        ;;
    dynamo-vllm)
        DYN_DEMO_LABEL="DYNAMO.VLLM"
        DYN_DEMO_COLOR="1;32"
        ;;
    *)
        printf >&2 'DYN_DEMO_SIDE must be control or dynamo-vllm\n'
        return 2
        ;;
esac

DYN_DEMO_BENCHMARK_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

demo-aiperf() {
    "$DYN_DEMO_BENCHMARK_DIR/run_qwen2_5_vl_demo_aiperf.sh" \
        "$DYN_DEMO_SIDE"
}

PS1="\[\033[${DYN_DEMO_COLOR}m\][${DYN_DEMO_LABEL}]\[\033[0m\] \w \\$ "

printf '\n\033[%sm%s READY\033[0m\n' "$DYN_DEMO_COLOR" "$DYN_DEMO_LABEL"
printf 'Run the live benchmark with: \033[1mdemo-aiperf\033[0m\n\n'
