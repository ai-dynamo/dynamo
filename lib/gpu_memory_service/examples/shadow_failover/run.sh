#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Bare-bones single-GPU GMS shadow-engine failover, no Kubernetes.
#
# Two vLLM engines share one GPU and one set of GMS-resident weights, both in
# autonomous shadow mode: each boots, parks itself, and blocks on a shared
# kernel flock. ENGINE_ID 0 wins the lock and serves; ENGINE_ID 1 waits. Kill
# the primary and the kernel releases the lock, so the shadow takes over with no
# weight reload. Follow /tmp/gms-demo-shadow.log to watch the takeover.
set -euo pipefail

# Tunables (override via env).
MODEL="${MODEL:-Qwen/Qwen3-0.6B}"
export GMS_SOCKET_DIR="${GMS_SOCKET_DIR:-/tmp/gms-demo}"
export ETCD_ENDPOINTS="${ETCD_ENDPOINTS:-http://localhost:2379}"
export NATS_SERVER="${NATS_SERVER:-nats://localhost:4222}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
LOCK="${FAILOVER_LOCK_PATH:-$GMS_SOCKET_DIR/failover.lock}"

PIDS=()
trap 'kill -KILL -"${PRIMARY_PGID:-0}" -"${SHADOW_PGID:-0}" 2>/dev/null; kill "${PIDS[@]}" 2>/dev/null' EXIT

mkdir -p "$GMS_SOCKET_DIR"

# A GMS client in shadow mode. ENGINE_ID 0 is the RW writer (loads the model
# from disk and publishes it into GMS); any other id imports it read-only.
# Per-engine ports keep the two processes from colliding on one host.
engine() { # engine <engine_id>
  setsid env \
    ENGINE_ID="$1" \
    DYN_VLLM_GMS_SHADOW_MODE=true \
    FAILOVER_LOCK_PATH="$LOCK" \
    DYN_SYSTEM_PORT="$((8080 + $1))" \
    VLLM_NIXL_SIDE_CHANNEL_PORT="$((8280 + $1))" \
    python -m dynamo.vllm --model "$MODEL" --load-format gms \
    --enforce-eager --enable-sleep-mode
}

# Discovery + message bus that dynamo.vllm registers with.
etcd --data-dir /tmp/gms-demo-etcd >/tmp/gms-demo-etcd.log 2>&1 & PIDS+=($!)
nats-server -js --store_dir /tmp/gms-demo-nats >/tmp/gms-demo-nats.log 2>&1 & PIDS+=($!)

# Per-GPU weight/KV server that holds the model for both engines.
python -m gpu_memory_service.cli.server >/tmp/gms-demo-gms.log 2>&1 & PIDS+=($!)

# Primary (writer) wins the flock and serves; shadow (reader) parks behind it.
engine 0 >/tmp/gms-demo-primary.log 2>&1 & PRIMARY_PGID=$!
engine 1 >/tmp/gms-demo-shadow.log 2>&1 & SHADOW_PGID=$!

# Wait until the shadow has loaded weights and parked on the flock (the standby
# is only "warm" once it is blocked waiting for the lock).
until grep -q "waiting for lock" /tmp/gms-demo-shadow.log 2>/dev/null; do sleep 2; done

# Crash the primary's process group; the kernel releases its flock and the
# shadow takes over without reloading weights (see /tmp/gms-demo-shadow.log).
kill -KILL -"$PRIMARY_PGID"

# Keep the shadow running so you can observe the takeover; Ctrl-C to tear down.
wait
