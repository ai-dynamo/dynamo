#!/bin/bash
# Launch the Decode vLLM instance for the disagg-bringup smoke.
# Same as launch-prefill.sh but role=decode and port=8001.
#
# Env vars (mirrors start-hub.sh's pattern):
#   KVBM_VENV          (default: /home/ryan/.venvs/dynamo-kvbm)
#   KVBM_BLOCK_LAYOUT  (default: operational)  — injected into kv_connector_extra_config
#                        so it survives vLLM's EngineCore subprocess spawn.
#                        Valid values: operational | universal
set -eu

KVBM_VENV=${KVBM_VENV:-/home/ryan/.venvs/dynamo-kvbm}
KVBM_BLOCK_LAYOUT=${KVBM_BLOCK_LAYOUT:-operational}
case "$KVBM_BLOCK_LAYOUT" in
  operational|universal) ;;
  *) echo "KVBM_BLOCK_LAYOUT must be 'operational' or 'universal', got: '$KVBM_BLOCK_LAYOUT'" >&2; exit 1 ;;
esac

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
      "default": { "block_layout": "'"$KVBM_BLOCK_LAYOUT"'" },
      "leader": {
        "disagg":  { "hub_url": "http://127.0.0.1:1337", "role": "decode" },
        "cache":   { "host": { "cache_size_gb": 2.0 } },
        "tokio":   { "worker_threads": 2 },
        "control": { "metrics": true }
      },
      "worker": {
        "nixl":  { "backends": { "UCX": {}, "POSIX": {} } },
        "tokio": { "worker_threads": 2 }
      }
    }
  }'
