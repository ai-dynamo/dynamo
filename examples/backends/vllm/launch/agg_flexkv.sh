#!/bin/bash
set -e
trap 'echo Cleaning up...; kill 0' EXIT

# Run frontend
python -m dynamo.frontend &

# Run worker with FlexKV
DYNAMO_USE_FLEXKV=1 \
FLEXKV_CPU_CACHE_GB=32 \
  python -m dynamo.vllm --model Qwen/Qwen3-0.6B --connector flexkv
