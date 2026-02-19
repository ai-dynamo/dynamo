#!/usr/bin/env bash
set -euo pipefail

cargo bench -p dynamo-kv-router --bench mooncake_bench --features bench -- \
    /home/ubuntu/dynamo/mooncake_trace.jsonl \
    --sweep \
    --sweep-max-ms 200000 \
    --sweep-min-ms 1000 \
    --sweep-steps 12 \
    --trace-duplication-factor 4 \
    --num-gpu-blocks 16384 \
    --trace-length-factor 16 \
    --num-event-workers 24 \
    --compare naive-nested-map,inverted-index,radix-tree,concurrent-radix-tree,nested-map
