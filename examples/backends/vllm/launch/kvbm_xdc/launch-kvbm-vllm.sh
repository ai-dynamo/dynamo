#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
. "$SCRIPT_DIR/hardware-profiles.sh"
kvbm_xdc_apply_hardware_profile worker

ROLE=${ROLE:?ROLE must be prefill or decode}
WORKTREE=${WORKTREE:-/workspace}
NAMESPACE=${NAMESPACE:-dynamo}
ENDPOINT_TYPES=${ENDPOINT_TYPES:-chat}
KV_EVENTS_PORT=${KV_EVENTS_PORT:-20081}
ENFORCE_EAGER=${ENFORCE_EAGER:-0}
ENABLE_PREFIX_CACHING=${ENABLE_PREFIX_CACHING:-1}
VLLM_RUNNER=${VLLM_RUNNER:-}
VLLM_USE_AOT_COMPILE=${VLLM_USE_AOT_COMPILE:-}
VLLM_USE_STANDALONE_COMPILE=${VLLM_USE_STANDALONE_COMPILE:-}
VLLM_ENABLE_V1_MULTIPROCESSING=${VLLM_ENABLE_V1_MULTIPROCESSING:-}
KVBM_SKIP_VLLM_VERSION_CHECK=${KVBM_SKIP_VLLM_VERSION_CHECK:-1}
KVBM_CONNECTOR_MODULE_PATH=${KVBM_CONNECTOR_MODULE_PATH:-kvbm.v2.vllm.connector}
PREFILL_KV_CONNECTOR_WRAPPER=${PREFILL_KV_CONNECTOR_WRAPPER:-PdConnector}
DECODE_KV_CONNECTOR=${DECODE_KV_CONNECTOR:-NixlConnector}
VALIDATE_CONNECTORS=${VALIDATE_CONNECTORS:-1}
VENV=${VENV:-/opt/dynamo/venv}

case "$ROLE" in
  prefill|decode) ;;
  *) echo "ROLE must be prefill or decode, got $ROLE" >&2; exit 2 ;;
esac

: "${MODEL:?MODEL must be set by KVBM_HARDWARE_PROFILE or env override}"
: "${MAX_MODEL_LEN:?MAX_MODEL_LEN must be set by KVBM_HARDWARE_PROFILE or env override}"
: "${MAX_NUM_SEQS:?MAX_NUM_SEQS must be set by KVBM_HARDWARE_PROFILE or env override}"
: "${GPU_MEMORY_UTILIZATION:?GPU_MEMORY_UTILIZATION must be set by KVBM_HARDWARE_PROFILE or env override}"
: "${CPU_CACHE_GB:?CPU_CACHE_GB must be set by KVBM_HARDWARE_PROFILE or env override}"

case "$ROLE" in
  prefill) CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-${PREFILL_CUDA_VISIBLE_DEVICES:-}} ;;
  decode) CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-${DECODE_CUDA_VISIBLE_DEVICES:-}} ;;
esac
if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  echo "CUDA_VISIBLE_DEVICES is required for ROLE=$ROLE when the selected hardware profile does not define placement" >&2
  exit 2
fi
export CUDA_VISIBLE_DEVICES
export DYN_KVBM_CPU_CACHE_GB="$CPU_CACHE_GB"
export VLLM_ATTENTION_BACKEND=${VLLM_ATTENTION_BACKEND:-FLASH_ATTN}
export KVBM_SKIP_VLLM_VERSION_CHECK
export PYTHONHASHSEED=${PYTHONHASHSEED:-0}
export PATH="$VENV/bin:/usr/local/cargo/bin:/usr/local/cuda/bin:$PATH"
for lib_dir in "$WORKTREE/.image-target/debug/deps" "$WORKTREE/.image-target-kvbm/debug/deps" /usr/local/lib/python*/site-packages/nixl_cu*.libs; do
  if [[ -d "$lib_dir" ]]; then
    export LD_LIBRARY_PATH="$lib_dir:${LD_LIBRARY_PATH:-}"
  fi
done
if [[ -d "$WORKTREE/lib/bindings/python/src" ]]; then
  export PYTHONPATH="$WORKTREE/lib/bindings/python/src:${PYTHONPATH:-}"
fi
if [[ -d "$WORKTREE/lib/bindings/kvbm/python" ]]; then
  export PYTHONPATH="$WORKTREE/lib/bindings/kvbm/python:${PYTHONPATH:-}"
fi
if [[ -d "$WORKTREE/components/src" ]]; then
  export PYTHONPATH="$WORKTREE/components/src:${PYTHONPATH:-}"
fi

if [[ -n "${SIDE_CHANNEL_HOST:-}" ]]; then
  export VLLM_NIXL_SIDE_CHANNEL_HOST="$SIDE_CHANNEL_HOST"
fi
if [[ -n "${SIDE_CHANNEL_PORT:-}" ]]; then
  export VLLM_NIXL_SIDE_CHANNEL_PORT="$SIDE_CHANNEL_PORT"
fi
if [[ -n "$VLLM_USE_AOT_COMPILE" ]]; then
  export VLLM_USE_AOT_COMPILE
fi
if [[ -n "$VLLM_USE_STANDALONE_COMPILE" ]]; then
  export VLLM_USE_STANDALONE_COMPILE
fi
if [[ -n "$VLLM_ENABLE_V1_MULTIPROCESSING" ]]; then
  export VLLM_ENABLE_V1_MULTIPROCESSING
fi

flags=()
if [[ "$ENFORCE_EAGER" == "1" ]]; then
  flags+=(--enforce-eager)
fi
if [[ "$ENABLE_PREFIX_CACHING" == "0" ]]; then
  flags+=(--no-enable-prefix-caching)
fi
if [[ -n "$VLLM_RUNNER" ]]; then
  flags+=(--runner "$VLLM_RUNNER")
fi

if [[ "$VALIDATE_CONNECTORS" == "1" ]]; then
  "$VENV/bin/python" - "$ROLE" "$DECODE_KV_CONNECTOR" "$PREFILL_KV_CONNECTOR_WRAPPER" "$KVBM_CONNECTOR_MODULE_PATH" <<'PY'
import importlib
import sys

role, decode_connector, prefill_wrapper, kvbm_module = sys.argv[1:5]
module = importlib.import_module(kvbm_module)
getattr(module, "DynamoConnector")
if role == "prefill":
    getattr(module, prefill_wrapper)

from vllm.distributed.kv_transfer.kv_connector.factory import KVConnectorFactory

registry = set(getattr(KVConnectorFactory, "_registry", {}).keys())
required = {decode_connector}
if role == "prefill":
    required.add("NixlConnector")
missing = sorted(required - registry)
if missing:
    raise SystemExit(
        "missing vLLM KV connector registry entries: "
        + ", ".join(missing)
        + "; registry="
        + ",".join(sorted(registry))
    )
print(
    "connector preflight OK:"
    f" role={role}"
    f" decode_connector={decode_connector}"
    f" prefill_wrapper={prefill_wrapper}"
    f" kvbm_module={kvbm_module}"
)
PY
fi

DECODE_KV_TRANSFER_CONFIG='{"kv_connector":"'"$DECODE_KV_CONNECTOR"'","kv_role":"kv_both"}'
PREFILL_KV_TRANSFER_CONFIG='{"kv_connector":"'"$PREFILL_KV_CONNECTOR_WRAPPER"'","kv_role":"kv_both","kv_connector_module_path":"'"$KVBM_CONNECTOR_MODULE_PATH"'","kv_connector_extra_config":{"connectors":[{"kv_connector":"DynamoConnector","kv_connector_module_path":"'"$KVBM_CONNECTOR_MODULE_PATH"'","kv_role":"kv_both"},{"kv_connector":"NixlConnector","kv_role":"kv_both"}]}}'
KV_EVENTS_CONFIG='{"publisher":"zmq","topic":"kv-events","endpoint":"tcp://*:'"$KV_EVENTS_PORT"'","enable_kv_cache_events":true}'

args=(
  --namespace "$NAMESPACE"
  --model "$MODEL" \
  --served-model-name "$MODEL" \
  --endpoint-types "$ENDPOINT_TYPES"
  --max-model-len "$MAX_MODEL_LEN" \
  --max-num-seqs "$MAX_NUM_SEQS" \
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
)

if [[ "$ROLE" == "decode" ]]; then
  args+=(
    --disaggregation-mode decode
    --kv-transfer-config "$DECODE_KV_TRANSFER_CONFIG"
  )
else
  args+=(
    --disaggregation-mode prefill
    --kv-transfer-config "$PREFILL_KV_TRANSFER_CONFIG"
    --kv-events-config "$KV_EVENTS_CONFIG"
  )
fi

exec "$VENV/bin/python" -m dynamo.vllm "${args[@]}" "${flags[@]}"
