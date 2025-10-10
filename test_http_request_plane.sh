#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Test script for HTTP request plane demo

set -e

echo "============================================"
echo "HTTP Request Plane Demo Test"
echo "============================================"
echo ""

# Check if etcd is running
if ! pgrep -x "etcd" > /dev/null; then
    echo "⚠️  Warning: etcd doesn't appear to be running"
    echo "   Please start etcd in another terminal: etcd"
    echo ""
fi

echo "📝 This script will:"
echo "   1. Start the HTTP server (worker)"
echo "   2. Wait 3 seconds for server to be ready"
echo "   3. Run the client to send requests"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

# Start server in background
echo "🚀 Starting HTTP server..."
DYN_REQUEST_PLANE=http cargo run --example http_request_plane_demo -- server &
SERVER_PID=$!

# Wait for server to start
echo "⏳ Waiting for server to initialize..."
sleep 3

# Run client
echo ""
echo "🔌 Running client..."
echo ""
DYN_REQUEST_PLANE=http cargo run --example http_request_plane_demo -- client

# Cleanup
echo ""
echo "🛑 Stopping server..."
kill $SERVER_PID 2>/dev/null || true
wait $SERVER_PID 2>/dev/null || true

echo ""
echo "✅ Demo completed!"

