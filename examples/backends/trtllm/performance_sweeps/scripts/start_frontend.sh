#! /bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

echo "commit id: $TRT_LLM_GIT_COMMIT"
echo "ucx info: $(ucx_info -v)"
echo "hostname: $(hostname)"

hostname=$(hostname)
short_hostname=$(echo "$hostname" | awk -F'.' '{print $1}')
echo "short_hostname: ${short_hostname}"

# Start NATS
nats-server -js &

# Start etcd
etcd --listen-client-urls http://0.0.0.0:2379 --advertise-client-urls http://0.0.0.0:2379 --data-dir /tmp/etcd &

# Wait for NATS/etcd to startup
sleep 2

# Start OpenAI Frontend which will dynamically discover workers when they startup
# dynamo.frontend accepts either --http-port flag or DYN_HTTP_PORT env var (defaults to 8000)
# NOTE: This is a blocking call.
python3 -m dynamo.frontend

