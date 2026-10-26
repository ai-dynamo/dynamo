#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Retry only transient BuildKit transport failures, preserving each attempt's log.
set -euo pipefail

destination=$1
shift
attempt_root=$(mktemp -d "${RUNNER_TEMP:-${TMPDIR:-/tmp}}/compliance-extract.XXXXXX")
echo "Compliance extraction logs: ${attempt_root}"

for attempt in 1 2 3; do
    output="${attempt_root}/output-${attempt}"
    log="${attempt_root}/attempt-${attempt}.log"
    mkdir -p "$output"
    status=0
    "$@" --output "type=local,dest=${output}" 2>&1 | tee "$log" || status=$?
    if [ "$status" -eq 0 ]; then
        # Publish only a complete export; failed attempts never reach consumers.
        rm -rf -- "$destination"
        mv -- "$output" "$destination"
        exit 0
    fi
    rm -rf -- "$output"

    # Inspect the terminal build error, not earlier warnings or command output.
    error=$(grep '^ERROR:' "$log" | tail -n 1 || true)
    if printf '%s\n' "$error" | grep -Eiq \
        '(process .*did not complete successfully|401 Unauthorized|403 Forbidden|denied:|manifest unknown)'; then
        exit "$status"
    fi
    if [ "$attempt" -eq 3 ] || ! printf '%s\n' "$error" | grep -Eiq \
        '(Unavailable:.*error reading from server: (unexpected )?EOF|connection reset by peer|transport is closing|unexpected status.*(502 Bad Gateway|503 Service Unavailable|504 Gateway Timeout))'; then
        exit "$status"
    fi
    delay=10
    [ "$attempt" -eq 2 ] && delay=30
    echo "::warning::Compliance extraction attempt ${attempt}/3 failed with a transient transport error; retrying in ${delay}s"
    sleep "$delay"
done
