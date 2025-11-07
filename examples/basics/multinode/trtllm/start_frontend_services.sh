#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Start NATS
nats-server -js &

# Start etcd
etcd --listen-client-urls http://0.0.0.0:2379 --advertise-client-urls http://0.0.0.0:2379 --data-dir /tmp/etcd &

# Wait for NATS/etcd to startup
sleep 3

# Start OpenAI Frontend which will dynamically discover workers when they startup
# NOTE: This is a blocking call.
python3 -m dynamo.frontend --http-port 8000
