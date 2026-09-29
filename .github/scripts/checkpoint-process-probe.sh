#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Run once through: kubectl exec -i <this-run-agent> -- bash -s < this-file
# The caller supplies the outer deadline. No signals, CUDA calls, environ, or
# process memory are read. /proc races and permission failures are expected.
set -u
export LC_ALL=C

own_cgroup=$(<"/proc/$$/cgroup")
if [[ -z "$own_cgroup" ]]; then
    printf 'probe: cannot establish the current container cgroup; stopping\n'
    exit 1
fi

same_cgroup() {
    local pid=$1 cgroup
    [[ -r /proc/$pid/cgroup ]] || return 1
    cgroup=$(<"/proc/$pid/cgroup")
    [[ "$cgroup" == "$own_cgroup" ]]
}

start_time() {
    local stat
    local -a fields
    [[ -r /proc/$1/stat ]] || return 1
    stat=$(<"/proc/$1/stat")
    # Fields after the last ')' start with state (field 3); starttime is 22.
    read -r -a fields <<< "${stat##*) }"
    [[ ${#fields[@]} -ge 20 ]] || return 1
    printf '%s' "${fields[19]}"
}

same_process() {
    [[ "$(start_time "$1")" == "$2" ]]
}

bounded_file() {
    local label=$1 path=$2 bytes=$3
    printf '%s: ' "$label"
    if [[ -r "$path" ]]; then
        head -c "$bytes" -- "$path" 2>/dev/null || true
        printf '\n'
    else
        printf 'unavailable\n'
    fi
}

inspect_process() {
    local pid=$1 identity=$2 role=$3 thread count=0
    same_process "$pid" "$identity" || return 0
    if [[ "$role" != target ]]; then
        same_cgroup "$pid" || return 0
    fi
    printf '\nPROCESS role=%s pid=%s starttime=%s\n' "$role" "$pid" "$identity"
    # Only diagnostic executables have their argv logged. A target's arguments
    # might contain application credentials; its PID and comm suffice here.
    if [[ "$role" != target ]]; then
        printf 'cmdline: '
        head -c 4096 "/proc/$pid/cmdline" 2>/dev/null | tr '\0' ' '
        printf '\n'
    fi
    awk '/^(Name|State|Tgid|Pid|PPid|TracerPid|Threads|NSpid):/' "/proc/$pid/status" 2>/dev/null || true
    bounded_file wchan "/proc/$pid/wchan" 1024
    bounded_file syscall "/proc/$pid/syscall" 2048
    printf 'loaded CUDA/NCCL/MPI libraries (mapped paths include versions):\n'
    awk '/\/(libcuda|libcudart|libnvidia|libnccl|libmpi|libopen-pal|libopen-rte|libpmix|libprrte|libuc[mpst]|libnixl)[^/]*\.so/ {print; if (++n == 128) exit}' "/proc/$pid/maps" 2>/dev/null || true
    # Do not follow absolute symlinks below /proc/PID/root: they can resolve
    # against this container's root. The mapped libcuda path is authoritative.
    printf 'CUDA directory symlink (not followed): '
    readlink -- "/proc/$pid/root/usr/local/cuda" 2>/dev/null || printf 'absent or not a symlink\n'
    for thread in /proc/"$pid"/task/[0-9]*; do
        [[ -d "$thread" ]] || continue
        same_process "$pid" "$identity" || break
        count=$((count + 1))
        if (( count > 32 )); then
            printf 'thread output capped at 32\n'
            break
        fi
        printf '\nTHREAD tid=%s\n' "${thread##*/}"
        awk '/^(Name|State|Pid|PPid):/' "$thread/status" 2>/dev/null || true
        bounded_file wchan "$thread/wchan" 1024
        bounded_file syscall "$thread/syscall" 2048
        bounded_file kernel_stack "$thread/stack" 8192
    done
    same_process "$pid" "$identity" || printf 'process exited or changed during observation\n'
}

printf 'checkpoint process probe UTC=%s\n' "$(date -u +%FT%TZ)"
matches=0
for process in /proc/[0-9]*; do
    pid=${process##*/}
    # Do not read another container's command, status, or maps during discovery.
    same_cgroup "$pid" || continue
    executable=$(readlink -- "$process/exe" 2>/dev/null) || continue
    case "${executable##*/}" in
        snapshot-agent|cuda-checkpoint-helper|criu) ;;
        *) continue ;;
    esac
    identity=$(start_time "$pid") || continue
    same_cgroup "$pid" || continue
    argv=()
    mapfile -d '' -t -n 128 argv < "$process/cmdline" 2>/dev/null || continue
    target=
    operation=unknown
    for (( index=1; index<${#argv[@]}; index++ )); do
        case "${argv[index]}" in
            --pid|-p) target=${argv[index+1]:-} ;;
            --action) operation=${argv[index+1]:-unknown} ;;
            --get-state|--get-restore-tid|dump|restore|pre-dump) operation=${argv[index]} ;;
        esac
    done
    same_process "$pid" "$identity" || continue
    matches=$((matches + 1))
    if (( matches > 8 )); then
        printf 'diagnostic process output capped at 8\n'
        break
    fi
    printf '\nDIAGNOSTIC operation=%q\n' "$operation"
    inspect_process "$pid" "$identity" diagnostic
    # The sole cross-cgroup exception is the explicit numeric target supplied
    # to this container's still-live helper. Never follow arbitrary CRIU PIDs.
    if [[ "${executable##*/}" == cuda-checkpoint-helper && "$target" =~ ^[1-9][0-9]{0,9}$ ]] && (( 10#$target <= 2147483647 )); then
        same_cgroup "$pid" || continue
        same_process "$pid" "$identity" || continue
        current_argv=()
        mapfile -d '' -t -n 128 current_argv < "$process/cmdline" 2>/dev/null || continue
        [[ "${current_argv[*]}" == "${argv[*]}" ]] || continue
        target_identity=$(start_time "$target") || continue
        inspect_process "$target" "$target_identity" target
    fi
done
printf '\nprobe complete; matching diagnostic processes observed=%s\n' "$matches"
