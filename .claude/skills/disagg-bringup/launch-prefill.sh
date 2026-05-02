#!/bin/bash
# Launch the Prefill vLLM instance for the disagg-bringup smoke.
# Mirrors .claude/skills/disagg-bringup/SKILL.md step 3.
set -eu
export CUDA_VISIBLE_DEVICES=0
export DYN_KVBM_CPU_CACHE_GB=2
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
exec /home/ryan/.venvs/dynamo-kvbm/bin/python3 -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen3-0.6B \
  --max-model-len 1024 \
  --max-num-seqs 8 \
  --gpu-memory-utilization 0.15 \
  --enable-chunked-prefill \
  --no-enable-prefix-caching \
  --port 8000 \
  --kv-transfer-config '{
    "kv_connector": "DynamoConnector",
    "kv_role": "kv_both",
    "kv_connector_module_path": "kvbm.v2.vllm.schedulers.connector",
    "kv_connector_extra_config": {
      "leader": {
        "disagg": { "hub_url": "http://127.0.0.1:1337", "role": "prefill" },
        "cache":  { "host": { "cache_size_gb": 2.0 } },
        "tokio":  { "worker_threads": 2 }
      },
      "worker": {
        "nixl":  { "backends": { "UCX": {}, "POSIX": {} } },
        "tokio": { "worker_threads": 2 }
      }
    }
  }'
