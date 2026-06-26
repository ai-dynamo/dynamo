#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Start the production GMS server supervisor. It auto-discovers GPUs and
# launches one server process per (device, tag) for tags "weights" and
# "kv_cache". The supervisor itself leads a process group so we can clean up
# all of its children at once.
#
# Writes PGID to: $RUN_DIR/gms.pgid  (and the supervisor PID to $RUN_DIR/gms.pid)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

mkdir -p "${RUN_DIR}" "${GMS_SOCKET_DIR}"

echo "    starting GMS supervisor (socket dir ${GMS_SOCKET_DIR}, CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}) ..."

# setsid => the supervisor leads its own process group; killing -PGID later
# reaps every per-(device,tag) child it spawned.
setsid env \
  GMS_SOCKET_DIR="${GMS_SOCKET_DIR}" \
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
  DYN_LOG="${DYN_LOG}" \
  python -m gpu_memory_service.cli.server \
  > "${RUN_DIR}/gms.log" 2>&1 &

gms_pid=$!
echo "${gms_pid}" > "${RUN_DIR}/gms.pid"
# The supervisor's PID is its own PGID because setsid made it a group leader.
echo "${gms_pid}" > "${RUN_DIR}/gms.pgid"

# The per-GPU servers open Unix sockets under GMS_SOCKET_DIR. There is no HTTP
# health endpoint; give the supervisor a moment to spawn its children and bind
# their sockets. The first engine launch will block on the RW lock until the
# weights server is actually serving, so this short wait is just a sanity gate.
echo "    waiting for GMS sockets to appear ..."
deadline=$((SECONDS + 60))
while (( SECONDS < deadline )); do
  if ! kill -0 "${gms_pid}" 2>/dev/null; then
    echo "ERROR: GMS supervisor exited early; see ${RUN_DIR}/gms.log" >&2
    exit 1
  fi
  # shellcheck disable=SC2012
  if ls "${GMS_SOCKET_DIR}"/gms_*_weights.sock >/dev/null 2>&1; then
    echo "    GMS up: supervisor pid ${gms_pid}, sockets in ${GMS_SOCKET_DIR}"
    exit 0
  fi
  sleep 1
done

echo "ERROR: GMS sockets did not appear in ${GMS_SOCKET_DIR} within 60s" >&2
echo "       (check that a CUDA GPU + driver + pynvml are available)" >&2
exit 1
