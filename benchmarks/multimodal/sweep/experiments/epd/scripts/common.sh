#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Shared runtime setup and process lifecycle for the EPD launchers.

PIDS=()
MPS_ROOT=
ROLE_CACHE_ENV=()

setup_dynamo_network() {
    local default_namespace=$1

    export DYN_NAMESPACE=${DYN_NAMESPACE:-$default_namespace}
    export DYN_DISCOVERY_BACKEND=${DYN_DISCOVERY_BACKEND:-file}
    if [[ $DYN_DISCOVERY_BACKEND == file ]]; then
        export DYN_FILE_KV=${DYN_FILE_KV:-"$LOG_DIR/discovery"}
        mkdir -p "$DYN_FILE_KV"
    fi

    export DYN_MM_ALLOW_INTERNAL=1
}

setup_nixl_libs() {
    local nixl_libs
    nixl_libs=$(python3 - <<'PY'
import importlib.util
from pathlib import Path

spec = importlib.util.find_spec("nixl_cu13")
if spec is None or spec.origin is None:
    raise SystemExit("nixl_cu13 is required for frontend decoding")
root = Path(spec.origin).resolve().parent.parent / ".nixl_cu13.mesonpy.libs"
if not (root / "plugins").is_dir() or not any(root.glob("libnixl.so*")):
    raise SystemExit(f"NIXL libraries/plugins not found under {root}")
print(root)
PY
    )
    export NIXL_PLUGIN_DIR="$nixl_libs/plugins"
    export LD_LIBRARY_PATH="$nixl_libs:$NIXL_PLUGIN_DIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
}

launch() {
    local name=$1
    shift
    local -a command=("$@")
    setsid "${command[@]}" >"$LOG_DIR/$name.log" 2>&1 &
    PIDS+=("$!")
}

cleanup() {
    local status=$?
    trap - EXIT INT TERM
    set +e
    local pid
    for pid in "${PIDS[@]}"; do kill -TERM -- "-$pid" 2>/dev/null; done
    for pid in "${PIDS[@]}"; do
        for _ in {1..100}; do kill -0 "$pid" 2>/dev/null || break; sleep 0.1; done
        kill -KILL -- "-$pid" 2>/dev/null
    done
    wait 2>/dev/null
    if [[ -n $MPS_ROOT ]]; then
        timeout --signal=TERM --kill-after=2s 10s env \
            CUDA_MPS_PIPE_DIRECTORY="$MPS_ROOT/pipe" \
            CUDA_MPS_LOG_DIRECTORY="$MPS_ROOT/log" \
            bash -c 'echo quit | nvidia-cuda-mps-control' >/dev/null 2>&1
    fi
    [[ $MPS_ROOT == /tmp/dynamo-epd-mps.* ]] && rm -r -- "$MPS_ROOT"
    exit "$status"
}

install_cleanup_traps() {
    trap cleanup EXIT
    trap 'exit 130' INT
    trap 'exit 143' TERM
}

start_mps() {
    local minimum=${DYN_EPD_MIN_SHM_BYTES:-68719476736}
    [[ -w /dev/shm && $(stat -f -c %T /dev/shm) == tmpfs ]] \
        || die "EPD requires writable tmpfs at /dev/shm"
    local size
    size=$(df -PB1 /dev/shm | awk 'NR==2 {print $2}')
    ((size >= minimum)) || die "EPD requires /dev/shm >= $minimum bytes"

    MPS_ROOT=$(mktemp -d /tmp/dynamo-epd-mps.XXXXXX)
    mkdir -p "$MPS_ROOT/pipe" "$MPS_ROOT/log"
    local -a command=(env CUDA_VISIBLE_DEVICES="$GPU"
        CUDA_MPS_PIPE_DIRECTORY="$MPS_ROOT/pipe"
        CUDA_MPS_LOG_DIRECTORY="$MPS_ROOT/log" nvidia-cuda-mps-control -d)
    "${command[@]}"
    for _ in {1..100}; do
        if echo get_server_list | env CUDA_MPS_PIPE_DIRECTORY="$MPS_ROOT/pipe" \
            nvidia-cuda-mps-control >/dev/null 2>&1; then
            return
        fi
        sleep 0.1
    done
    die "CUDA MPS did not become ready"
}

wait_for_worker_log() {
    local name=$1 pattern=$2 timeout=${DYN_WORKER_READY_TIMEOUT_SECONDS:-1800}
    local pid=${PIDS[$((${#PIDS[@]} - 1))]} deadline=$((SECONDS + timeout))
    while ((SECONDS < deadline)); do
        grep -Fq "$pattern" "$LOG_DIR/$name.log" && return
        kill -0 "$pid" 2>/dev/null || die "$name exited before becoming ready"
        sleep 1
    done
    die "$name was not ready within ${timeout}s"
}

prepare_role_cache() {
    local role=$1 cache_root="$LOG_DIR/cache/$1"
    mkdir -p "$cache_root"/{home,xdg,triton,torchinductor,flashinfer}
    ROLE_CACHE_ENV=(
        HOME="$cache_root/home"
        XDG_CACHE_HOME="$cache_root/xdg"
        TRITON_CACHE_DIR="$cache_root/triton"
        TORCHINDUCTOR_CACHE_DIR="$cache_root/torchinductor"
        FLASHINFER_WORKSPACE_BASE="$cache_root/flashinfer"
    )
}
