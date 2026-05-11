#!/bin/bash
# Launch the Decode vLLM instance for the disagg-bringup smoke.
# Same as launch-prefill.sh but role=decode and port=8001.
#
# Env vars (mirrors start-hub.sh's pattern):
#   KVBM_VENV  (default: /home/ryan/.venvs/dynamo-kvbm)
set -eu
KVBM_VENV=${KVBM_VENV:-/home/ryan/.venvs/dynamo-kvbm}
# Host cache for G2 pinned memory. Default matches Ryan's GB10 baseline.
# Lower (e.g. 0.5) when CUDA_ERROR_INVALID_VALUE surfaces on smaller hardware (A10/A100-PCIe).
KVBM_HOST_CACHE_GB=${KVBM_HOST_CACHE_GB:-2.0}
export CUDA_VISIBLE_DEVICES=0
export DYN_KVBM_CPU_CACHE_GB=2
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
exec "$KVBM_VENV/bin/python3" -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen3-0.6B \
  --max-model-len 1024 \
  --max-num-seqs 8 \
  --gpu-memory-utilization 0.15 \
  --enable-chunked-prefill \
  --no-enable-prefix-caching \
  --port 8001 \
  --kv-transfer-config '{
    "kv_connector": "DynamoConnector",
    "kv_role": "kv_both",
    "kv_load_failure_policy": "recompute",
    "kv_connector_module_path": "kvbm.v2.vllm.schedulers.connector",
    "kv_connector_extra_config": {
      "leader": {
        "disagg": { "hub_url": "http://127.0.0.1:1337", "role": "decode" },
        "cache":  { "host": { "cache_size_gb": '"$KVBM_HOST_CACHE_GB"' } },
        "tokio":  { "worker_threads": 2 }
      },
      "worker": {
        "nixl":  { "backends": { "UCX": {}, "POSIX": {} } },
        "tokio": { "worker_threads": 2 }
      }
    }
  }'
