#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Shared helpers for the SGLang sidecar launch scripts.

# Usage: sglang_check_listener_ports <http-port> <grpc-port> <listeners>
sglang_check_listener_ports() {
    if [[ ! $1 =~ ^[0-9]+$ || ! $2 =~ ^[0-9]+$ || ! $3 =~ ^[0-9]+$ ]]; then
        echo "sglang_check_listener_ports: expected decimal numbers, got '$1' '$2' '$3'" >&2
        return 1
    fi
    if (( 10#$2 - 10#$1 < 10#$3 && 10#$1 - 10#$2 < 10#$3 )); then
        echo "SGLang HTTP port $1 and gRPC port $2 must be at least $3 apart: listener k binds HTTP port + k and gRPC port + k" >&2
        return 1
    fi
}

# Usage: sglang_listener_count [SGLang options...]
# Prints how many Rust listeners one single-node SGLang server runs: its
# --attn-dp-size with attention DP, else its --dp-size replicas.
sglang_listener_count() {
    local dp=1 attn_dp=1 previous= arg
    for arg in "$@"; do
        case $previous in
            --dp-size|--data-parallel-size) dp=$arg ;;
            --attn-dp-size|--attention-data-parallel-size) attn_dp=$arg ;;
        esac
        case $arg in
            --dp-size=*|--data-parallel-size=*) dp=${arg#*=} ;;
            --attn-dp-size=*|--attention-data-parallel-size=*) attn_dp=${arg#*=} ;;
        esac
        previous=$arg
    done
    if [[ $attn_dp != 1 ]]; then echo "$attn_dp"; else echo "$dp"; fi
}
