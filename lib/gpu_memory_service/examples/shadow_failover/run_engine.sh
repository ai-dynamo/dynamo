#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Launch one vLLM engine against GMS. The PRIMARY and the SHADOW differ ONLY by
# environment / flags selected from the role:
#   primary -> RW writer (publishes weights into GMS), no gms_read_only.
#   shadow  -> RO importer (gms_read_only=true), imports the already-resident
#              weights with no second disk load.
#
# Usage:
#   ./run_engine.sh primary
#   ./run_engine.sh shadow
#   ROLE=shadow ./run_engine.sh
#
# Each engine is started with setsid so it leads its own process group; the
# PGID is recorded to $RUN_DIR/<role>.pgid so kill_primary.sh can group-kill it.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

ROLE="${1:-${ROLE:-}}"
if [[ "${ROLE}" != "primary" && "${ROLE}" != "shadow" ]]; then
  echo "ERROR: role must be 'primary' or 'shadow' (got '${ROLE:-<empty>}')" >&2
  echo "usage: $0 <primary|shadow>" >&2
  exit 1
fi

mkdir -p "${RUN_DIR}" "${GMS_SOCKET_DIR}"

# Select the per-engine port triple + the read-only flag from the role.
declare -a extra_args=()
if [[ "${ROLE}" == "primary" ]]; then
  system_port="${PRIMARY_SYSTEM_PORT}"
  nixl_port="${PRIMARY_NIXL_PORT}"
  kv_event_port="${PRIMARY_KV_EVENT_PORT}"
  # PRIMARY is the RW writer: do NOT pass gms_read_only.
else
  system_port="${SHADOW_SYSTEM_PORT}"
  nixl_port="${SHADOW_NIXL_PORT}"
  kv_event_port="${SHADOW_KV_EVENT_PORT}"
  # SHADOW is the RO importer: it attaches to the resident weights, no reload.
  extra_args+=(--model-loader-extra-config '{"gms_read_only": true}')
fi

# KV-event publisher config (ZMQ). Each engine needs a unique endpoint port.
kv_events_cfg="{\"publisher\":\"zmq\",\"topic\":\"kv-events\",\"endpoint\":\"tcp://*:${kv_event_port}\",\"enable_kv_cache_events\":true}"

echo "    launching ${ROLE} engine: system=${system_port} nixl=${nixl_port} kv-event=${kv_event_port}"

# NOTE: this manual flavor deliberately does NOT set DYN_VLLM_GMS_SHADOW_MODE,
# ENGINE_ID, FAILOVER_LOCK_PATH, or DYN_GMS_SCRATCH_KV_ENABLED. Those drive the
# autonomous flock-based path used by the Kubernetes operator. Here we drive
# pause/resume by hand via the engine control endpoints (see README).
#
# setsid => the engine leads its own process group, so a later
# `kill -KILL -<PGID>` reaps the whole vLLM process tree (mirrors the pytest
# os.killpg(os.getpgid(pid), SIGKILL) failover injection).
setsid env \
  GMS_SOCKET_DIR="${GMS_SOCKET_DIR}" \
  ETCD_ENDPOINTS="${ETCD_ENDPOINTS}" \
  NATS_SERVER="${NATS_SERVER}" \
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
  DYN_LOG="${DYN_LOG}" \
  DYN_SYSTEM_PORT="${system_port}" \
  VLLM_NIXL_SIDE_CHANNEL_PORT="${nixl_port}" \
  python -m dynamo.vllm \
  --model "${MODEL}" \
  --load-format gms \
  --enforce-eager \
  --enable-sleep-mode \
  --max-num-seqs 1 \
  --gpu-memory-utilization "${GPU_MEM_UTIL}" \
  --kv-events-config "${kv_events_cfg}" \
  "${extra_args[@]}" \
  > "${RUN_DIR}/${ROLE}.log" 2>&1 &

engine_pid=$!
# setsid made the engine a group leader, so its PID == its PGID.
echo "${engine_pid}" > "${RUN_DIR}/${ROLE}.pid"
echo "${engine_pid}" > "${RUN_DIR}/${ROLE}.pgid"
echo "${system_port}" > "${RUN_DIR}/${ROLE}.system_port"

# Block until the engine reports ready. For the primary this is also when its
# weights have been published into GMS; for the shadow it means the resident
# weights were imported (no second disk load).
if ! wait_for_ready "${system_port}" 300; then
  echo "ERROR: ${ROLE} engine did not become ready; see ${RUN_DIR}/${ROLE}.log" >&2
  exit 1
fi

echo "    ${ROLE} engine ready: pid ${engine_pid} (pgid ${engine_pid}), system port ${system_port}"
