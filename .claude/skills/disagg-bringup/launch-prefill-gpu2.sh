#!/bin/bash
# Launch the Prefill vLLM instance on GPU 2 (TP=1) for the asymmetric
# TP=2-decode + TP=1-prefill smoke.
#
# Sibling of launch-prefill.sh — same vLLM args, just a different GPU
# and higher memutil (no longer co-located with decode).
#
# Env vars:
#   KVBM_VENV          (default: /home/ryan/.venvs/dynamo-kvbm)
#   KVBM_PREFILL_GPU   (default: 2)
#   KVBM_PREFILL_MEMUTIL (default: 0.6)
set -eu
KVBM_VENV=${KVBM_VENV:-/home/ryan/.venvs/dynamo-kvbm}
KVBM_PREFILL_GPU=${KVBM_PREFILL_GPU:-2}
KVBM_PREFILL_MEMUTIL=${KVBM_PREFILL_MEMUTIL:-0.6}

export CUDA_VISIBLE_DEVICES=$KVBM_PREFILL_GPU
export DYN_KVBM_CPU_CACHE_GB=2
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
exec "$KVBM_VENV/bin/python3" -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen3-0.6B \
  --max-model-len 1024 \
  --max-num-seqs 8 \
  --gpu-memory-utilization "$KVBM_PREFILL_MEMUTIL" \
  --enable-chunked-prefill \
  --no-enable-prefix-caching \
  --port 8000 \
  --kv-transfer-config '{
    "kv_connector": "DynamoConnector",
    "kv_role": "kv_both",
    "kv_load_failure_policy": "recompute",
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
