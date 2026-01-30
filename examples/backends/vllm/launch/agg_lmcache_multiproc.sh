#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
set -e

# Explicitly set PROMETHEUS_MULTIPROC_DIR (K8s-style deployment)
# Use unique directory per test run to avoid conflicts
export PROMETHEUS_MULTIPROC_DIR=${PROMETHEUS_MULTIPROC_DIR:-/tmp/prometheus_multiproc_$$_$RANDOM}
rm -rf "$PROMETHEUS_MULTIPROC_DIR"
mkdir -p "$PROMETHEUS_MULTIPROC_DIR"

# Cleanup function to remove the directory on exit
cleanup() {
    echo "Cleaning up..."
    rm -rf "$PROMETHEUS_MULTIPROC_DIR"
    kill 0
}
trap cleanup EXIT

# run ingress
# dynamo.frontend accepts either --http-port flag or DYN_HTTP_PORT env var (defaults to 8000)
python -m dynamo.frontend &

# run worker with LMCache enabled and PROMETHEUS_MULTIPROC_DIR explicitly set
DYN_SYSTEM_PORT=${DYN_SYSTEM_PORT:-8081} \
  PROMETHEUS_MULTIPROC_DIR="$PROMETHEUS_MULTIPROC_DIR" \
  python -m dynamo.vllm --model Qwen/Qwen3-0.6B --connector lmcache

