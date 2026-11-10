#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Usage: sglang_check_listener_ports <http-port> <grpc-port> <listeners>
sglang_check_listener_ports() {
    if [[ ! $1 =~ ^[0-9]+$ || ! $2 =~ ^[0-9]+$ || ! $3 =~ ^[0-9]+$ ]]; then
        echo "sglang_check_listener_ports: expected decimal numbers, got '$1' '$2' '$3'" >&2
        return 1
    fi
    if (( 10#$2 - 10#$1 < 10#$3 && 10#$1 - 10#$2 < 10#$3 )); then
        echo "SGLANG_HTTP_PORT and SGLANG_GRPC_PORT must be at least $3 apart: each of the $3 listeners binds both + k" >&2
        return 1
    fi
}
