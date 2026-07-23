#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Bare-bones GMS shadow-engine failover, no Kubernetes. Run this SAME script on
# every node; set NODE_RANK per node (0 = leader). For a single node, just run
# it with the defaults (NNODES=1).
#
# Each node runs a GMS server. The leader node also runs etcd + nats + the
# Dynamo frontend. Two engines start per node in autonomous shadow mode sharing
# one flock: cohort 0 is the primary (ENGINE_ID=0, loads + publishes the
# weights), cohort 1 is the shadow (ENGINE_ID=1, imports them). The primary
# serves; the shadow parks on the lock.
#
# This script does NOT kill anything. To trigger failover, kill the primary
# yourself; the kernel then hands the lock to the shadow automatically:
#   kill -KILL -"$(cat /tmp/gms-demo-engine-0.pgid)"
set -euo pipefail

# Tunables (override via env).
MODEL="${MODEL:-Qwen/Qwen3-0.6B}"
TP="${TP:-1}"                            # tensor-parallel size (total GPUs per engine)
NNODES="${NNODES:-1}"                    # nodes per engine group
NODE_RANK="${NODE_RANK:-0}"              # this node's rank; 0 = leader
LEADER_ADDR="${LEADER_ADDR:-127.0.0.1}"  # leader host that workers dial
FRONTEND_PORT="${FRONTEND_PORT:-8000}"
export GMS_SOCKET_DIR="${GMS_SOCKET_DIR:-/tmp/gms-demo}"
export ETCD_ENDPOINTS="${ETCD_ENDPOINTS:-http://${LEADER_ADDR}:2379}"
export NATS_SERVER="${NATS_SERVER:-nats://${LEADER_ADDR}:4222}"

# Reap every process group we start on exit.
PIDS=()
cleanup() {
  local p
  [ "${#PIDS[@]}" -eq 0 ] && return 0
  for p in "${PIDS[@]}"; do kill -KILL "-$p" 2>/dev/null || true; done
}
trap cleanup EXIT
mkdir -p "$GMS_SOCKET_DIR"

# Launch one shadow-mode engine in its own process group. cohort 0 = primary
# (writer); others import read-only. Multi-node groups (NNODES>1) get the
# distributed flags, and non-leader ranks run headless.
engine() { # engine <cohort_id>
  local id="$1"
  local args=(--model "$MODEL" --load-format gms --enforce-eager --enable-sleep-mode
              --tensor-parallel-size "$TP")
  if [ "$NNODES" -gt 1 ]; then
    args+=(--distributed-executor-backend mp --nnodes "$NNODES" --node-rank "$NODE_RANK"
           --master-addr "$LEADER_ADDR" --master-port "$((29500 + id * 100))")
    [ "$NODE_RANK" -ne 0 ] && args+=(--headless)
  fi
  setsid env \
    ENGINE_ID="$id" \
    DYN_VLLM_GMS_SHADOW_MODE=true \
    FAILOVER_LOCK_PATH="$GMS_SOCKET_DIR/failover.lock" \
    DYN_SYSTEM_PORT="$((8080 + id))" \
    VLLM_NIXL_SIDE_CHANNEL_PORT="$((8280 + id))" \
    python -m dynamo.vllm "${args[@]}" >"/tmp/gms-demo-engine-$id.log" 2>&1 &
  PIDS+=("$!")
  printf '%s\n' "$!" >"/tmp/gms-demo-engine-$id.pgid"  # so you can kill this engine's group
}

# Every node runs a GMS server (holds this node's weights for both engines).
setsid python -m gpu_memory_service.cli.server >/tmp/gms-demo-gms.log 2>&1 & PIDS+=("$!")

# The leader node also runs discovery + the frontend; engines register there.
if [ "$NODE_RANK" -eq 0 ]; then
  setsid etcd --data-dir /tmp/gms-demo-etcd >/tmp/gms-demo-etcd.log 2>&1 & PIDS+=("$!")
  setsid nats-server -js --store_dir /tmp/gms-demo-nats >/tmp/gms-demo-nats.log 2>&1 & PIDS+=("$!")
  setsid python -m dynamo.frontend --http-port "$FRONTEND_PORT" >/tmp/gms-demo-frontend.log 2>&1 & PIDS+=("$!")
fi

engine 0  # primary
engine 1  # shadow

# Everything is up. Kill the primary yourself (see the header) to trigger
# failover, then watch /tmp/gms-demo-engine-1.log.
wait
