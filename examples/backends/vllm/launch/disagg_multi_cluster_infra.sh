#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Development-only coordination services for the cross-cluster disaggregated
# serving example. Production deployments should use managed or highly
# available etcd and NATS services instead.

set -euo pipefail

if [[ "${BASH_VERSINFO[0]}" -lt 4 || ( "${BASH_VERSINFO[0]}" -eq 4 && "${BASH_VERSINFO[1]}" -lt 3 ) ]]; then
    echo "This script requires bash 4.3+ (for wait -n), found ${BASH_VERSION}" >&2
    exit 1
fi

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    cat <<'EOF'
Usage: INFRA_IP=<reachable-ip> ./disagg_multi_cluster_infra.sh

Required:
  INFRA_IP          Address reachable from the prefill and decode clusters.

Optional:
  ETCD_BIN          etcd executable (default: etcd)
  NATS_BIN          nats-server executable (default: nats-server)
  INFRA_STATE_DIR   State directory (default: /tmp/dynamo-cross-cluster-infra)
  ETCD_CLIENT_PORT  etcd client port (default: 2379)
  ETCD_PEER_PORT    etcd peer port (default: 2380)
  NATS_PORT         NATS client port (default: 4222)
  NATS_MONITOR_PORT NATS monitoring port (default: 8222)
EOF
    exit 0
fi

: "${INFRA_IP:?INFRA_IP must be an address reachable from both clusters}"

ETCD_BIN="${ETCD_BIN:-etcd}"
NATS_BIN="${NATS_BIN:-nats-server}"
INFRA_STATE_DIR="${INFRA_STATE_DIR:-${TMPDIR:-/tmp}/dynamo-cross-cluster-infra}"
ETCD_CLIENT_PORT="${ETCD_CLIENT_PORT:-2379}"
ETCD_PEER_PORT="${ETCD_PEER_PORT:-2380}"
NATS_PORT="${NATS_PORT:-4222}"
NATS_MONITOR_PORT="${NATS_MONITOR_PORT:-8222}"

command -v "$ETCD_BIN" >/dev/null || {
    echo "etcd executable not found: $ETCD_BIN" >&2
    exit 1
}
command -v "$NATS_BIN" >/dev/null || {
    echo "nats-server executable not found: $NATS_BIN" >&2
    exit 1
}
command -v curl >/dev/null || {
    echo "curl is required for readiness checks" >&2
    exit 1
}

mkdir -p "$INFRA_STATE_DIR/etcd" "$INFRA_STATE_DIR/nats"

etcd_pid=""
nats_pid=""
cleanup() {
    trap - EXIT INT TERM
    [[ -n "$etcd_pid" ]] && kill "$etcd_pid" 2>/dev/null || true
    [[ -n "$nats_pid" ]] && kill "$nats_pid" 2>/dev/null || true
    wait "$etcd_pid" "$nats_pid" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

"$ETCD_BIN" \
    --name cross-cluster \
    --data-dir "$INFRA_STATE_DIR/etcd" \
    --listen-client-urls "http://0.0.0.0:${ETCD_CLIENT_PORT}" \
    --advertise-client-urls "http://${INFRA_IP}:${ETCD_CLIENT_PORT}" \
    --listen-peer-urls "http://127.0.0.1:${ETCD_PEER_PORT}" \
    --initial-advertise-peer-urls "http://127.0.0.1:${ETCD_PEER_PORT}" \
    --initial-cluster "cross-cluster=http://127.0.0.1:${ETCD_PEER_PORT}" &
etcd_pid=$!

"$NATS_BIN" \
    --jetstream \
    --store_dir "$INFRA_STATE_DIR/nats" \
    --addr 0.0.0.0 \
    --port "$NATS_PORT" \
    --http_port "$NATS_MONITOR_PORT" &
nats_pid=$!

for _ in $(seq 1 30); do
    if curl --fail --silent "http://127.0.0.1:${ETCD_CLIENT_PORT}/health" >/dev/null \
        && curl --fail --silent "http://127.0.0.1:${NATS_MONITOR_PORT}/healthz" >/dev/null; then
        break
    fi
    sleep 1
done

curl --fail --silent --show-error "http://127.0.0.1:${ETCD_CLIENT_PORT}/health"
echo
curl --fail --silent --show-error "http://127.0.0.1:${NATS_MONITOR_PORT}/healthz"
echo
echo "ETCD_ENDPOINTS=http://${INFRA_IP}:${ETCD_CLIENT_PORT}"
echo "NATS_SERVER=nats://${INFRA_IP}:${NATS_PORT}"

# Stop both services if either one exits. Bare `wait -n` covers exactly the
# two children above; `wait -n <pid>...` would require bash 5.1.
wait -n
