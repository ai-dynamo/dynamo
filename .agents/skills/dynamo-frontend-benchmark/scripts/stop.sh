#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Stop the benchmark topology (frontend + workers). Leaves etcd/nats running.
# Waits until processes are gone, :8000 is free, and worker instances have
# drained from etcd, so a subsequent ./start.sh sees a clean slate.
set -uo pipefail
cd "$(dirname "$0")"
source ./env.sh

# If the frontend was started isolated (ISOLATE=1 → root-owned in bench.slice),
# a plain kill can't reap it. Try a non-interactive stop of the slice; if that
# needs a password it's skipped (stop it yourself: sudo systemctl stop bench.slice).
if systemctl is-active --quiet bench.slice 2>/dev/null; then
    sudo -n systemctl stop bench.slice 2>/dev/null \
        && echo "[stop] stopped isolated frontend (bench.slice)" \
        || echo "[stop] NOTE: isolated frontend in bench.slice — run: sudo systemctl stop bench.slice"
fi

for name in frontend workers; do
    pidfile="$LOG_DIR/$name.pid"
    if [[ -f "$pidfile" ]]; then
        pid="$(cat "$pidfile")"
        if kill -0 "$pid" 2>/dev/null; then
            echo "[stop] terminating $name (pid $pid) ..."
            kill "$pid" 2>/dev/null || true
            pkill -P "$pid" 2>/dev/null || true
            # give it up to 10s to exit gracefully, then SIGKILL
            for i in $(seq 1 10); do kill -0 "$pid" 2>/dev/null || break; sleep 1; done
            kill -0 "$pid" 2>/dev/null && { echo "[stop] $name still alive; SIGKILL"; kill -9 "$pid" 2>/dev/null || true; }
        fi
        rm -f "$pidfile"
    fi
done
# Belt-and-suspenders: kill any stragglers from this worktree.
pkill -f "dynamo.mocker --model-path $MODEL" 2>/dev/null || true
pkill -f "dynamo.frontend --router-mode kv" 2>/dev/null || true

# Wait for :8000 to free.
for i in $(seq 1 15); do ss -ltn 2>/dev/null | grep -q ":${HTTP_PORT}\b" || break; sleep 1; done
ss -ltn 2>/dev/null | grep -q ":${HTTP_PORT}\b" && echo "[stop] WARNING: :${HTTP_PORT} still held." || echo "[stop] :${HTTP_PORT} free."

# Wait for worker instances to drain from etcd (lease expiry).
for i in $(seq 1 20); do n="$(count_workers)"; [[ "$n" -eq 0 ]] && break; sleep 1; done
n="$(count_workers)"
[[ "$n" -eq 0 ]] && echo "[stop] etcd worker instances drained." || echo "[stop] WARNING: $n worker instance(s) still in etcd."
echo "[stop] done."
