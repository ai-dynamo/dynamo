#! /bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

echo "commit id: $TRT_LLM_GIT_COMMIT"
echo "ucx info: $(ucx_info -v)"
echo "hostname: $(hostname)"

hostname=$(hostname)
short_hostname=$(echo "$hostname" | awk -F'.' '{print $1}')
echo "short_hostname: ${short_hostname}"

# Start NATS
nats-server -js &

export ETCD_CLIENT_PORT=2379
export ETCD_PEER_PORT=2380
export NATS_PORT=4222
export DIST_INIT_PORT=29500
export ETCD_LISTEN_ADDR=http://0.0.0.0

etcd \
  --listen-client-urls "${ETCD_LISTEN_ADDR}:${ETCD_CLIENT_PORT}" \
  --advertise-client-urls "${ETCD_LISTEN_ADDR}:${ETCD_CLIENT_PORT}" \
  --listen-peer-urls "${ETCD_LISTEN_ADDR}:${ETCD_PEER_PORT}" \
  --initial-advertise-peer-urls "http://${HEAD_NODE}:${ETCD_PEER_PORT}" \
  --initial-cluster "default=http://${HEAD_NODE}:${ETCD_PEER_PORT}" \
  --data-dir /tmp/etcd &


# Wait for NATS/etcd to startup
sleep 2

# Start OpenAI Frontend which will dynamically discover workers when they startup
# NOTE: This is a blocking call.
python3 -m dynamo.frontend --http-port 8000

