#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Temporary diagnostic wrapper; install only in this run's snapshot-agent.
set -uo pipefail

helper=/usr/local/bin/cuda-checkpoint-helper.real
pid=
action=unknown
args=("$@")
for ((index=0; index<${#args[@]}; index++)); do
    case "${args[index]}" in
        --pid|-p) pid=${args[index+1]:-} ;;
        --action) action=${args[index+1]:-unknown} ;;
        --get-state|--get-restore-tid) action=${args[index]} ;;
    esac
done
if [[ ! "$pid" =~ ^[1-9][0-9]{0,9}$ ]] || ((10#$pid > 2147483647)); then
    printf 'matched-helper: missing or invalid numeric PID\n' >&2
    exit 64
fi
case "$action" in
    lock|checkpoint|restore|unlock|--get-state|--get-restore-tid) ;;
    *) printf 'matched-helper: unsupported action\n' >&2; exit 64 ;;
esac
record_failure() {
    result=$?
    if ((result != 0)) && { [[ "$action" != --get-* ]] || ((result == 124 || result == 137 || result == 70)); }; then
        printf 'pid=%s action=%s result=%s\n' "$pid" "$action" "$result" > /tmp/checkpoint-diagnostic-helper-failed
    fi
}
trap record_failure EXIT

# Maps contains the actual loaded file, not a compatibility-directory guess.
# Reject deleted/ambiguous mappings and loader separators rather than silently
# selecting another driver. Do not inspect any target environment or argv.
if ! driver=$(awk '
    $6 ~ /\/libcuda[.]so([.][0-9]+)+$/ {
        if (NF != 6) { bad=1; exit }
        if (!seen[$6]++) print $6
    }
    END { if (bad) exit 1 }
' "/proc/$pid/maps"); then
    printf 'matched-helper: cannot read a usable driver mapping for pid=%s\n' "$pid" >&2
    exit 70
fi
command=("$helper")
selected=original-no-driver-mapping
if [[ -n "$driver" ]]; then
    if [[ ! "$driver" =~ ^/[a-zA-Z0-9_./-]+/libcuda\.so(\.[0-9]+)+$ || "$driver" == */../* ]]; then
        printf 'matched-helper: ambiguous or unsafe driver mapping for pid=%s\n' "$pid" >&2
        exit 70
    fi
    selected="/proc/$pid/root$driver"
    if [[ ! -r "$selected" ]]; then
        printf 'matched-helper: mapped driver unavailable for pid=%s\n' "$pid" >&2
        exit 70
    fi
    # Set preload only for the original helper, not timeout or this shell.
    command=(env "LD_PRELOAD=$selected" "$helper")
elif [[ "$action" != --get-state && "$action" != --get-restore-tid ]]; then
    printf 'matched-helper: no mapped CUDA driver for action=%s pid=%s\n' "$action" "$pid" >&2
    exit 70
fi

log_action() {
    printf 'matched-helper %s pid=%s action=%s driver=%s result=%s\n' \
        "$1" "$pid" "$action" "$selected" "$2" \
        | tee -a /tmp/checkpoint-helper-actions.log >&2
}
log_action START pending
timeout --signal=TERM --kill-after=5s 60s "${command[@]}" "$@"
result=$?
log_action END "$result"
exit "$result"
