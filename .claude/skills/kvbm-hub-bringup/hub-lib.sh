#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0
#
# Sourceable helpers for bringing up kvbm_hub + kvbmctl. Pure functions — no
# side effects at source time. Shared by `start-hub.sh` (this skill) and by
# connector launchers that render their `--kv-transfer-config` via kvbmctl.
#
#   . "$REPO/.claude/skills/kvbm-hub-bringup/hub-lib.sh"

# Build both bins (`kvbm_hub` + `kvbmctl`) in the given repo, logging to a file.
# kvbmctl is a default feature of kvbm-hub, so a plain build produces it (it
# does pull CUDA via kvbm-config — fine in a smoke environment that has it).
# Args: <repo> <log>
kvbm_hub_build() {
    local repo=$1 log=$2
    if [ -d /usr/local/cuda/bin ] && [[ ":$PATH:" != *":/usr/local/cuda/bin:"* ]]; then
        export PATH="/usr/local/cuda/bin:$PATH"
    fi
    export CUDA_PATH=${CUDA_PATH:-/usr/local/cuda} CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}
    echo "[kvbm-hub-bringup] cargo build --bin kvbm_hub --bin kvbmctl (incremental)…" > "$log"
    ( cd "$repo" && cargo build --bin kvbm_hub --bin kvbmctl ) >> "$log" 2>&1
}

# Poll the hub's control-port /health until ready, the hub PID dies, or timeout.
# Args: <control_port> <timeout_s> <hub_pid> [log]
kvbm_hub_wait_health() {
    local port=$1 timeout=$2 pid=$3 log=${4:-}
    local deadline=$(( $(date +%s) + timeout ))
    until curl -fsS -m 5 "http://127.0.0.1:$port/health" >/dev/null 2>&1; do
        if [ -n "$pid" ] && ! kill -0 "$pid" 2>/dev/null; then
            echo "[kvbm-hub-bringup] hub exited before ready" >&2
            [ -n "$log" ] && [ -f "$log" ] && tail -n 25 "$log" >&2
            return 1
        fi
        if [ "$(date +%s)" -ge "$deadline" ]; then
            echo "[kvbm-hub-bringup] hub not ready within ${timeout}s" >&2
            return 1
        fi
        sleep 1
    done
}

# Render a vLLM connector CLI fragment from a live hub via kvbmctl. Prints
# `--block-size <N> --max-model-len <M> --kv-transfer-config '{…}'` to stdout;
# returns non-zero on failure. Extra args pass straight through to kvbmctl
# (e.g. --role, --kv-connector-module-path, repeated --kvbm overrides).
#
# Consume it with the eval-array idiom so the shell-quoted (space-free) JSON
# stays a single argv element:
#   RENDERED=$(kvbm_hub_render_vllm "$KVBMCTL" "$HUB" indexer --kvbm a=b) || exit 1
#   eval "KV_ARGS=( $RENDERED )"
#   exec python -m vllm... "${KV_ARGS[@]}"
#
# Args: <kvbmctl_bin> <hub_url> <features_csv> [extra kvbmctl args...]
kvbm_hub_render_vllm() {
    local kvbmctl=$1 hub=$2 feats=$3
    shift 3
    if [ ! -x "$kvbmctl" ]; then
        echo "[kvbm-hub-bringup] kvbmctl missing at $kvbmctl (start-hub.sh builds it)" >&2
        return 1
    fi
    "$kvbmctl" get config vllm --hub "$hub" --features "$feats" "$@"
}
