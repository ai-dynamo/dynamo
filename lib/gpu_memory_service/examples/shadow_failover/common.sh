#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Shared environment defaults and helpers for the GMS shadow-failover demo.
# Source this from the other scripts; do not execute it directly.

set -euo pipefail

# --- Model ------------------------------------------------------------------
export MODEL="${MODEL:-Qwen/Qwen3-0.6B}"

# --- Shared GMS / infra env (server + both engines MUST agree on these) ------
# All three GMS clients (server + primary + shadow) must use the SAME socket
# dir. Socket files are named gms_<GPU-UUID>_<tag>.sock for tags weights and
# kv_cache.
export GMS_SOCKET_DIR="${GMS_SOCKET_DIR:-/tmp/gms-demo}"
export ETCD_ENDPOINTS="${ETCD_ENDPOINTS:-http://localhost:2379}"
export NATS_SERVER="${NATS_SERVER:-nats://localhost:4222}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export DYN_LOG="${DYN_LOG:-info}"

# --- Frontend ---------------------------------------------------------------
export FRONTEND_PORT="${FRONTEND_PORT:-8000}"

# --- Per-engine ports (must be unique per engine to avoid collisions) -------
# Primary engine.
export PRIMARY_SYSTEM_PORT="${PRIMARY_SYSTEM_PORT:-8081}"
export PRIMARY_NIXL_PORT="${PRIMARY_NIXL_PORT:-8281}"
export PRIMARY_KV_EVENT_PORT="${PRIMARY_KV_EVENT_PORT:-8181}"
# Shadow engine.
export SHADOW_SYSTEM_PORT="${SHADOW_SYSTEM_PORT:-8082}"
export SHADOW_NIXL_PORT="${SHADOW_NIXL_PORT:-8282}"
export SHADOW_KV_EVENT_PORT="${SHADOW_KV_EVENT_PORT:-8182}"

# --- Engine knobs -----------------------------------------------------------
export GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.8}"

# --- Local data dirs for the external infra binaries ------------------------
export ETCD_DATA_DIR="${ETCD_DATA_DIR:-/tmp/gms-demo-etcd}"
export NATS_STORE_DIR="${NATS_STORE_DIR:-/tmp/gms-demo-nats}"

# --- Where this demo records the PGIDs / PIDs of its children ---------------
export RUN_DIR="${RUN_DIR:-/tmp/gms-demo-run}"

# wait_for_ready <port> [timeout_seconds] [path]
#
# Poll an engine/frontend system endpoint until it reports ready, or until the
# timeout elapses. Engines expose GET /health that returns JSON
# {"status":"ready"}; we accept any 2xx body containing the literal "ready".
wait_for_ready() {
  local port="$1"
  local timeout="${2:-300}"
  local path="${3:-/health}"
  local url="http://localhost:${port}${path}"
  local deadline=$((SECONDS + timeout))

  echo "    waiting for ${url} (timeout ${timeout}s) ..."
  while (( SECONDS < deadline )); do
    # Prefer a strict JSON parse; fall back to a substring match on the body.
    if curl -sf "${url}" 2>/dev/null | python3 -c '
import json, sys
try:
    sys.exit(0 if json.load(sys.stdin).get("status") == "ready" else 1)
except Exception:
    sys.exit(1)
' 2>/dev/null; then
      echo "    ready: ${url}"
      return 0
    fi
    sleep 1
  done

  echo "    ERROR: ${url} did not become ready within ${timeout}s" >&2
  return 1
}

# wait_for_port <port> [timeout_seconds]
#
# Poll until a TCP port accepts connections. Used for plain infra (etcd, nats)
# that does not speak the /health JSON contract.
wait_for_port() {
  local port="$1"
  local timeout="${2:-60}"
  local deadline=$((SECONDS + timeout))

  echo "    waiting for tcp localhost:${port} (timeout ${timeout}s) ..."
  while (( SECONDS < deadline )); do
    if (exec 3<>"/dev/tcp/localhost/${port}") 2>/dev/null; then
      exec 3>&- 3<&- 2>/dev/null || true
      echo "    listening: localhost:${port}"
      return 0
    fi
    sleep 1
  done

  echo "    ERROR: localhost:${port} not listening within ${timeout}s" >&2
  return 1
}
