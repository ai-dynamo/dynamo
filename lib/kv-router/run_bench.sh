#!/usr/bin/env bash
set -euo pipefail

cargo bench -p dynamo-kv-router --bench mooncake_bench --features bench -- \
    /home/ubuntu/dynamo/mooncake_trace.jsonl \
    --sweep \
    --sweep-max-ms 10000 \
    --sweep-min-ms 1000 \
    --sweep-steps 5 \
    --trace-duplication-factor 4 \
    --num-gpu-blocks 16384 \
    --trace-length-factor 16 \
    --num-event-workers 24 \
    --compare nested-map,concurrent-radix-tree
