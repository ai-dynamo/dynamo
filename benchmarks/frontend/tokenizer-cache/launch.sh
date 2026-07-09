#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Terminal 1: ./launch.sh off
# Terminal 2: ./benchmark.sh
# Stop this script with Ctrl-C, then repeat with: ./launch.sh on

set -euo pipefail

case "${1:-}" in
    on)  TOKENIZER_CACHE=1 ;;
    off) TOKENIZER_CACHE=0 ;;
    *) echo "Usage: $0 <on|off>" >&2; exit 1 ;;
esac

MODEL="Qwen/Qwen3-0.6B"
OTHER_CORES="1-$(($(nproc --all) - 1))"
export DYN_FILE_KV
DYN_FILE_KV="$(mktemp -d)"

echo "Starting 4 mock workers on CPU cores ${OTHER_CORES}"
taskset -c "$OTHER_CORES" python -m dynamo.mocker \
    --model-path "$MODEL" \
    --num-workers 4 \
    --speedup-ratio 1000000 \
    --discovery-backend file \
    --request-plane tcp \
    --event-plane zmq &
MOCKER_PID=$!
trap 'kill "$MOCKER_PID" 2>/dev/null || true; rm -rf "$DYN_FILE_KV"' EXIT

echo "Starting frontend on CPU core 0 with tokenizer cache ${1}"
DYN_TOKENIZER=fastokens \
DYN_TOKENIZER_CACHE="$TOKENIZER_CACHE" \
DYN_TOKENIZER_CACHE_BYTES=1073741824 \
taskset -c 0 python -m dynamo.frontend \
    --discovery-backend file \
    --request-plane tcp \
    --event-plane zmq \
    --router-mode round-robin
