#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Requires matching encoder-handoff TRT-LLM and native sidecar builds.
set -euo pipefail
SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/../../../../examples/common/gpu_utils.sh"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/../../../../examples/common/launch_utils.sh"

if [[ "${1:-}" == "--help" ]]; then
    echo "Usage: MODEL=Qwen/Qwen3-VL-2B-Instruct $0"
    echo "Requires the TRT-LLM encoder-handoff build and matching native sidecar."
    echo "MODEL_REVISION: required full checkpoint commit SHA; NIXL/UCX must be available."
    echo "EPD_MODE: e_pd (default) or e_p_d; E/P/D disables conditional prefill bypass."
    echo "TRTLLM_PYTHON: engine Python."
    echo "ENCODER_GPU/PD_GPU/PREFILL_GPU/DECODE_GPU: explicit CUDA device IDs."
    echo "Defaults use the first inherited CUDA_VISIBLE_DEVICES entry; otherwise E/PD=0, P=1, D=2."
    echo "ENCODER_BIND_HOST: engine interface; ENCODER_CONNECT_HOST: local sidecar destination."
    echo "ENCODER_ADVERTISE_HOST: reachable encoder control address for consumers."
    echo "DYN_HTTP_PORT: frontend port; ENCODER_GRPC_PORT/PD_GRPC_PORT/DECODE_GRPC_PORT: engine ports."
    echo "Managed ports: DYN_ENGINE_GRPC_PORT1/2/3 and DYN_SYSTEM_PORT1/2/3."
    exit 0
fi
if [[ $# != 0 ]]; then
    echo "Configure this image encoder-disaggregation example through environment variables; see --help." >&2
    exit 1
fi
MODEL="${MODEL:-Qwen/Qwen3-VL-2B-Instruct}"
EPD_MODE="${EPD_MODE:-e_pd}"
case "$EPD_MODE" in
    e_pd|e_p_d) ;;
    *) echo "EPD_MODE must be e_pd or e_p_d." >&2; exit 1 ;;
esac
: "${MODEL_REVISION:?Set MODEL_REVISION to the immutable checkpoint revision}"
if [[ ! "$MODEL_REVISION" =~ ^[[:xdigit:]]{40}$ ]]; then
    echo "MODEL_REVISION must be a full immutable checkpoint commit SHA (40 hex digits)." >&2
    exit 1
fi
# Reuse the allocator's physical ID (including GPU/MIG UUIDs), rather than
# replacing a managed allocation with physical defaults 0/1/2. Explicit role
# overrides still allow operators to spread workers across several devices.
DEFAULT_ENCODER_GPU=0
DEFAULT_PREFILL_GPU=1
DEFAULT_DECODE_GPU=2
if [[ -v CUDA_VISIBLE_DEVICES ]]; then
    DEFAULT_ENCODER_GPU=$(inherited_gpu_id)
    DEFAULT_PREFILL_GPU="$DEFAULT_ENCODER_GPU"
    DEFAULT_DECODE_GPU="$DEFAULT_ENCODER_GPU"
fi
ENCODER_GPU="${ENCODER_GPU:-$DEFAULT_ENCODER_GPU}"
ENCODER_BIND_HOST="${ENCODER_BIND_HOST:-127.0.0.1}"
ENCODER_CONNECT_HOST="${ENCODER_CONNECT_HOST:-$ENCODER_BIND_HOST}"
case "$ENCODER_CONNECT_HOST" in
    0.0.0.0) ENCODER_CONNECT_HOST=127.0.0.1 ;;
    ::|'[::]') ENCODER_CONNECT_HOST=::1 ;;
esac
ENCODER_ADVERTISE_HOST="${ENCODER_ADVERTISE_HOST:-$ENCODER_CONNECT_HOST}"
# Bracket IPv6 literals in gRPC authorities.
grpc_address() {
    local host="$1" port="$2"
    if [[ "$host" == *:* && "$host" != \[*\] ]]; then
        host="[$host]"
    fi
    printf '%s:%s\n' "$host" "$port"
}
ENCODER_GRPC_PORT=$(dyn_port DYN_ENGINE_GRPC_PORT 1 "${ENCODER_GRPC_PORT:-30001}")
PD_GRPC_PORT=$(dyn_port DYN_ENGINE_GRPC_PORT 2 "${PD_GRPC_PORT:-30011}")
ENCODER_SYSTEM_PORT=$(dyn_port DYN_SYSTEM_PORT 1 8081)
CONSUMER_SYSTEM_PORT=$(dyn_port DYN_SYSTEM_PORT 2 8082)
if [[ "$EPD_MODE" == e_p_d ]]; then
    DECODE_GRPC_PORT=$(dyn_port DYN_ENGINE_GRPC_PORT 3 "${DECODE_GRPC_PORT:-30021}")
    DECODE_SYSTEM_PORT=$(dyn_port DYN_SYSTEM_PORT 3 8083)
fi
TRTLLM_PYTHON="${TRTLLM_PYTHON:-python3}"
"$TRTLLM_PYTHON" -c 'from tensorrt_llm.grpc.openengine.bindings import generation_pb2; assert "encoder" in generation_pb2.GenerateRequest.DESCRIPTOR.fields_by_name, "Install the encoder-handoff TRT-LLM build"'
# Resolve once, before launching children: all engines, sidecar tokenizers and
# frontend-discovered templates must read the same immutable snapshot.
MODEL_PATH=$("$TRTLLM_PYTHON" - "$MODEL" "$MODEL_REVISION" <<'PY'
import sys

from huggingface_hub import snapshot_download

print(snapshot_download(repo_id=sys.argv[1], revision=sys.argv[2]))
PY
)
resolve_sidecar trtllm SIDECAR_CMD
RUN_DIR="$(mktemp -d "${TMPDIR:-/tmp}/trtllm-epd.XXXXXXXX")"
cleanup() {
    local result=$?
    rm -rf -- "$RUN_DIR" || true
    dynamo_reap_and_exit "$result"
}
trap cleanup EXIT
export TRTLLM_OPENENGINE_ENCODER_TRANSFER=nixl
export TRTLLM_OPENENGINE_ENCODER_MODEL_REVISION="$MODEL_REVISION"

CONTEXT="${TRTLLM_CONTEXT_LENGTH:-4096}"
cat > "$RUN_DIR/encode.yaml" <<'YAML'
mm_encoder_only: true
max_batch_size: 1
dtype: bfloat16
YAML
CONSUMER_OPTIONS='{"disable_mm_encoder":true,"guided_decoding_backend":"xgrammar","max_batch_size":1,"dtype":"bfloat16"}'
CONSUMER_ARGS=()
CONSUMER_GPU="${PD_GPU:-$DEFAULT_ENCODER_GPU}"
TOPOLOGY="E+PD"
if [[ "$EPD_MODE" == e_p_d ]]; then
    CONSUMER_OPTIONS='{"disable_mm_encoder":true,"guided_decoding_backend":"xgrammar","max_batch_size":1,"dtype":"bfloat16","cache_transceiver_config":{"backend":"NIXL"}}'
    CONSUMER_ARGS+=(--disaggregation-mode prefill)
    CONSUMER_GPU="${PREFILL_GPU:-$DEFAULT_PREFILL_GPU}"
    TOPOLOGY="E+P+D"
fi
build_trtllm_override_args_with_mem --merge-with-json "$CONSUMER_OPTIONS" > "$RUN_DIR/pd.yaml"
print_launch_banner --multimodal "TensorRT-LLM sidecar image $TOPOLOGY" "$MODEL" "${DYN_HTTP_PORT:-8000}"
python3 -m dynamo.frontend &
TRTLLM_OPENENGINE_ENCODER_ENDPOINT="$(grpc_address "$ENCODER_ADVERTISE_HOST" "$ENCODER_GRPC_PORT")" \
CUDA_VISIBLE_DEVICES="$ENCODER_GPU" "$TRTLLM_PYTHON" -m tensorrt_llm.commands.serve "$MODEL_PATH" \
    --grpc --grpc-protocol openengine --host "$ENCODER_BIND_HOST" --port "$ENCODER_GRPC_PORT" \
    --served_model_name "$MODEL" --max_seq_len "$CONTEXT" --extra_llm_api_options "$RUN_DIR/encode.yaml" &
CUDA_VISIBLE_DEVICES="$CONSUMER_GPU" "$TRTLLM_PYTHON" -m tensorrt_llm.commands.serve "$MODEL_PATH" \
    --grpc --grpc-protocol openengine --host 127.0.0.1 --port "$PD_GRPC_PORT" \
    --served_model_name "$MODEL" --max_seq_len "$CONTEXT" --extra_llm_api_options "$RUN_DIR/pd.yaml" &
DYN_SYSTEM_PORT="$ENCODER_SYSTEM_PORT" "${SIDECAR_CMD[@]}" \
    --grpc-endpoint "$(grpc_address "$ENCODER_CONNECT_HOST" "$ENCODER_GRPC_PORT")" --model-path "$MODEL_PATH" \
    --disaggregation-mode encode --context-length "$CONTEXT" &
DYN_SYSTEM_PORT="$CONSUMER_SYSTEM_PORT" "${SIDECAR_CMD[@]}" \
    --grpc-endpoint "127.0.0.1:$PD_GRPC_PORT" --model-path "$MODEL_PATH" \
    --route-to-encoder --context-length "$CONTEXT" "${CONSUMER_ARGS[@]}" &
if [[ "$EPD_MODE" == e_p_d ]]; then
    CUDA_VISIBLE_DEVICES="${DECODE_GPU:-$DEFAULT_DECODE_GPU}" "$TRTLLM_PYTHON" -m tensorrt_llm.commands.serve "$MODEL_PATH" \
        --grpc --grpc-protocol openengine --host 127.0.0.1 --port "$DECODE_GRPC_PORT" \
        --served_model_name "$MODEL" --max_seq_len "$CONTEXT" --extra_llm_api_options "$RUN_DIR/pd.yaml" &
    DYN_SYSTEM_PORT="$DECODE_SYSTEM_PORT" "${SIDECAR_CMD[@]}" \
        --grpc-endpoint "127.0.0.1:$DECODE_GRPC_PORT" --model-path "$MODEL_PATH" \
        --disaggregation-mode decode --context-length "$CONTEXT" &
fi
wait_any_exit
