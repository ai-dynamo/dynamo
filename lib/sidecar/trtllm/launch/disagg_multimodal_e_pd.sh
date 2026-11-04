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
    echo "MODEL_REVISION: required immutable checkpoint revision; NIXL/UCX must be available."
    echo "TRTLLM_PYTHON: engine Python; ENCODER_GPU/PD_GPU: device indices (both default 0)."
    echo "DYN_HTTP_PORT: frontend port; ENCODER_GRPC_PORT/PD_GRPC_PORT: engine ports."
    exit 0
fi
if [[ $# != 0 ]]; then
    echo "Configure this image E+PD example through environment variables; see --help." >&2
    exit 1
fi
MODEL="${MODEL:-Qwen/Qwen3-VL-2B-Instruct}"
TRTLLM_PYTHON="${TRTLLM_PYTHON:-python3}"
"$TRTLLM_PYTHON" -c 'from tensorrt_llm.grpc.openengine.bindings import generation_pb2; assert "encoder" in generation_pb2.GenerateRequest.DESCRIPTOR.fields_by_name, "Install the encoder-handoff TRT-LLM build"'
resolve_sidecar trtllm SIDECAR_CMD
RUN_DIR="$(mktemp -d "${TMPDIR:-/tmp}/trtllm-epd.XXXXXXXX")"
cleanup() {
    local result=$?
    find "$RUN_DIR" -xdev -depth -delete
    dynamo_reap_and_exit "$result"
}
trap cleanup EXIT
export TRTLLM_OPENENGINE_ENCODER_TRANSFER=nixl
export TRTLLM_OPENENGINE_ENCODER_MODEL_REVISION="${MODEL_REVISION:?Set MODEL_REVISION to the immutable checkpoint revision}"

ENCODER_GRPC_PORT="${ENCODER_GRPC_PORT:-30001}"
PD_GRPC_PORT="${PD_GRPC_PORT:-30011}"
CONTEXT="${TRTLLM_CONTEXT_LENGTH:-4096}"
cat > "$RUN_DIR/encode.yaml" <<'YAML'
mm_encoder_only: true
max_batch_size: 1
dtype: bfloat16
YAML
build_trtllm_override_args_with_mem --merge-with-json '{"disable_mm_encoder":true,"guided_decoding_backend":"xgrammar","max_batch_size":1,"dtype":"bfloat16"}' > "$RUN_DIR/pd.yaml"
print_launch_banner --multimodal "TensorRT-LLM sidecar image E+PD" "$MODEL" "${DYN_HTTP_PORT:-8000}"
python3 -m dynamo.frontend &
TRTLLM_OPENENGINE_ENCODER_ENDPOINT="${ENCODER_ADVERTISE_HOST:-127.0.0.1}:$ENCODER_GRPC_PORT" \
CUDA_VISIBLE_DEVICES="${ENCODER_GPU:-0}" "$TRTLLM_PYTHON" -m tensorrt_llm.commands.serve "$MODEL" \
    --grpc --grpc-protocol openengine --host "${ENCODER_BIND_HOST:-127.0.0.1}" --port "$ENCODER_GRPC_PORT" \
    --hf_revision "$MODEL_REVISION" --max_seq_len "$CONTEXT" --extra_llm_api_options "$RUN_DIR/encode.yaml" &
CUDA_VISIBLE_DEVICES="${PD_GPU:-0}" "$TRTLLM_PYTHON" -m tensorrt_llm.commands.serve "$MODEL" \
    --grpc --grpc-protocol openengine --host 127.0.0.1 --port "$PD_GRPC_PORT" \
    --hf_revision "$MODEL_REVISION" --max_seq_len "$CONTEXT" --extra_llm_api_options "$RUN_DIR/pd.yaml" &
DYN_SYSTEM_PORT="${DYN_SYSTEM_PORT1:-8081}" "${SIDECAR_CMD[@]}" \
    --grpc-endpoint "127.0.0.1:$ENCODER_GRPC_PORT" --model-path "$MODEL" \
    --disaggregation-mode encode --context-length "$CONTEXT" &
DYN_SYSTEM_PORT="${DYN_SYSTEM_PORT2:-8082}" "${SIDECAR_CMD[@]}" \
    --grpc-endpoint "127.0.0.1:$PD_GRPC_PORT" --model-path "$MODEL" \
    --route-to-encoder --context-length "$CONTEXT" &
wait_any_exit
