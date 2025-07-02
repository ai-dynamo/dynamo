#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Helper script to generate environment variables for each node during a multinode SGLang deployment

echo "=== USAGE ==="
echo "1. Run this script: ./gen_env_vars.sh"
echo "2. Enter the IP addresses when prompted"
echo "3. Copy the commands for the head prefill node and run them"
echo "4. Copy the commands for all other nodes and run them on each node"
echo "5. Proceed with starting your prefill and decode workers"
echo ""

# Prompt for IP addresses
read -p "Enter HEAD_PREFILL_NODE IP: " HEAD_PREFILL_NODE
read -p "Enter HEAD_DECODE_NODE IP: " HEAD_DECODE_NODE

# Validate inputs
if [ -z "$HEAD_PREFILL_NODE" ] || [ -z "$HEAD_DECODE_NODE" ]; then
    echo "Error: Both IP addresses are required"
    exit 1
fi

echo "=== HEAD PREFILL NODE ($HEAD_PREFILL_NODE) ==="
echo "Run all of these commands on the head prefill node:"
echo ""
echo "nats-server -js &"
echo "etcd --listen-client-urls http://0.0.0.0:2379 \\"
echo "     --advertise-client-urls http://0.0.0.0:2379 \\"
echo "     --listen-peer-urls http://0.0.0.0:2380 \\"
echo "     --initial-cluster default=http://$HEAD_PREFILL_NODE:2380 &"
echo "export HEAD_PREFILL_NODE_IP=$HEAD_PREFILL_NODE"
echo "export HEAD_DECODE_NODE_IP=$HEAD_DECODE_NODE"
echo ""
echo "=== ALL OTHER NODES ==="
echo "Run these commands on all other nodes (prefill and decode):"
echo ""
echo "# Export environment variables"
echo "export NATS_SERVER=nats://$HEAD_PREFILL_NODE:4222"
echo "export ETCD_ENDPOINTS=http://$HEAD_PREFILL_NODE:2379"
echo "export HEAD_PREFILL_NODE_IP=$HEAD_PREFILL_NODE"
echo "export HEAD_DECODE_NODE_IP=$HEAD_DECODE_NODE"



