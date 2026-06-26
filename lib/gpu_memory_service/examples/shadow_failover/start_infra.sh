#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Start the external control-plane infra: etcd + nats-server (with JetStream).
# Both are external binaries you must install yourself (see README).
#
# Writes PIDs to:
#   $RUN_DIR/etcd.pid
#   $RUN_DIR/nats.pid

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

mkdir -p "${RUN_DIR}" "${ETCD_DATA_DIR}" "${NATS_STORE_DIR}"

if ! command -v etcd >/dev/null 2>&1; then
  echo "ERROR: 'etcd' not found on PATH. Install it first (see README)." >&2
  exit 1
fi
if ! command -v nats-server >/dev/null 2>&1; then
  echo "ERROR: 'nats-server' not found on PATH. Install it first (see README)." >&2
  exit 1
fi

echo "    starting etcd (data-dir ${ETCD_DATA_DIR}) ..."
etcd --data-dir "${ETCD_DATA_DIR}" \
  > "${RUN_DIR}/etcd.log" 2>&1 &
echo $! > "${RUN_DIR}/etcd.pid"

echo "    starting nats-server -js (store-dir ${NATS_STORE_DIR}) ..."
# -js enables JetStream, which Dynamo requires for KV-event streaming.
nats-server -js --store_dir "${NATS_STORE_DIR}" \
  > "${RUN_DIR}/nats.log" 2>&1 &
echo $! > "${RUN_DIR}/nats.pid"

# etcd default client port 2379, nats default 4222.
wait_for_port 2379 60
wait_for_port 4222 60

echo "    infra up: etcd(pid $(cat "${RUN_DIR}/etcd.pid")) nats(pid $(cat "${RUN_DIR}/nats.pid"))"
