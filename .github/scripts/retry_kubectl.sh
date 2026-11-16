#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Retry idempotent kubectl operations while the vCluster watchdog restores its
# tunnel. Use -f <file>, not stdin, so every attempt receives the same manifest.
retry_kubectl() (
    local error_log attempt status
    error_log=$(mktemp)
    trap 'rm -f "$error_log"' EXIT
    for attempt in 1 2 3 4; do
        if kubectl --request-timeout=30s "$@" 2>"$error_log"; then
            cat "$error_log" >&2
            return 0
        else
            status=$?
        fi
        cat "$error_log" >&2
        if (( attempt == 4 )) || ! grep -Eiq \
            'Unable to connect to the server:.*(EOF|connection refused|connection reset|i/o timeout|TLS handshake timeout)|The connection to the server .* was refused|read: connection reset by peer' \
            "$error_log"; then
            return "$status"
        fi
        echo "vCluster connection lost; retrying in 5s ($attempt/3)" >&2
        sleep 5
    done
)
