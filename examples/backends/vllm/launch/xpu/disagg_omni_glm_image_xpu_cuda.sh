#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Cross-node disaggregated GLM-Image (AR on CUDA/XPU, DiT on CUDA/XPU).
# Uses NIXL/UCX over InfiniBand for inter-stage tensor transfer.
#
# Usage - run on BOTH nodes with ROLE set:
#   AR node:   ROLE=ar  bash disagg_omni_glm_image_xpu_cuda.sh
#   DiT node:  ROLE=dit bash disagg_omni_glm_image_xpu_cuda.sh
#
# Environment variables:
#   AR_TP          - Tensor parallel size for AR stage (default: 1)
#   AR_DEVICE_TYPE - Device type for AR stage: "xpu" or "cuda" (default: "cuda")
#   DIT_TP         - Tensor parallel size for DiT stage (default: 1)
#   DIT_DEVICE_TYPE- Device type for DiT stage: "xpu" or "cuda" (default: "xpu")
#
#   Device selection: set ZE_AFFINITY_MASK (xpu) or CUDA_VISIBLE_DEVICES (cuda) externally.
#   UCX settings: set UCX_NET_DEVICES, UCX_TLS, UCX_MEMTYPE_CACHE externally if needed.
#
# Examples:
#   AR on XPU TP2:  ZE_AFFINITY_MASK=2,3 ROLE=ar AR_TP=2 AR_DEVICE_TYPE=xpu
#   AR on CUDA:     CUDA_VISIBLE_DEVICES=0 ROLE=ar AR_TP=1 AR_DEVICE_TYPE=cuda
#   DiT on CUDA:    CUDA_VISIBLE_DEVICES=5 ROLE=dit DIT_TP=1 DIT_DEVICE_TYPE=cuda
#   DiT on XPU:     ZE_AFFINITY_MASK=0 ROLE=dit DIT_TP=1 DIT_DEVICE_TYPE=xpu

set -e
trap dynamo_exit_trap EXIT

SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"
source "$SCRIPT_DIR/../../../../common/launch_utils.sh"

# Default device types for cross-device disaggregation (CUDA AR + XPU DiT)
AR_DEVICE_TYPE="${AR_DEVICE_TYPE:-cuda}"
DIT_DEVICE_TYPE="${DIT_DEVICE_TYPE:-xpu}"

MODEL="${MODEL:-zai-org/GLM-Image}"
HTTP_PORT="${DYN_HTTP_PORT:-8000}"

NATS_SERVER="${NATS_SERVER:-nats://127.0.0.1:4222}"
ETCD_ENDPOINTS="${ETCD_ENDPOINTS:-http://127.0.0.1:2379}"

if [ -z "${STAGE_CONFIG:-}" ]; then
    STAGE_CONFIG="$(readlink -f "$SCRIPT_DIR/../stage_configs/glm_image_nixl.yaml")"
fi

# Namespace must match on both nodes for discovery
if [ -z "${DYN_NAMESPACE:-}" ]; then
    export DYN_NAMESPACE="dynamo-omni-glm-xpu-cuda-$(date +%s)"
    echo "NOTE: Auto-generated DYN_NAMESPACE=${DYN_NAMESPACE}"
    echo "      Set this explicitly on BOTH nodes before starting."
fi

AR_TP="${AR_TP:-1}"
DIT_TP="${DIT_TP:-1}"

EXTRA_ARGS=()
while [[ $# -gt 0 ]]; do
    case $1 in
        --model) MODEL="$2"; shift 2 ;;
        --stage-configs-path) STAGE_CONFIG="$2"; shift 2 ;;
        *) EXTRA_ARGS+=("$1"); shift ;;
    esac
done

if [[ -z "${ROLE:-}" ]]; then
    echo "ERROR: ROLE must be set to 'ar' or 'dit'"
    echo "  AR node:  ROLE=ar  $0"
    echo "  DiT node: ROLE=dit $0"
    exit 1
fi

echo "Namespace:    ${DYN_NAMESPACE}"
echo "Stage config: ${STAGE_CONFIG}"
echo "Role:         ${ROLE}"
echo "NATS:         ${NATS_SERVER}"
echo "etcd:         ${ETCD_ENDPOINTS}"

# NIXL settings (UCX_NET_DEVICES, UCX_TLS, UCX_MEMTYPE_CACHE can be set externally if needed)
export FLASHINFER_DISABLE_VERSION_CHECK=1
export PYTHONHASHSEED=0

if [[ "${ROLE,,}" == "ar" ]]; then
    export VLLM_TARGET_DEVICE=${AR_DEVICE_TYPE}

    # Compute 0-based logical device list from TP size (e.g. TP=2 would give "0,1")
    ar_devices=$(seq -s, 0 $(( AR_TP - 1 )))

    print_launch_banner --no-curl "Disaggregated GLM-Image - AR stage (${AR_DEVICE_TYPE^^} TP=${AR_TP})" "$MODEL" "$HTTP_PORT"
    print_curl_footer <<CURL
curl -s http://localhost:${HTTP_PORT}/v1/images/generations \\
  -H 'Content-Type: application/json' \\
  -d '{
    "model": "${MODEL}",
    "prompt": "a red apple on a white table",
    "size": "1024x1024"
  }' | jq
CURL

    echo "Starting Stage 0 (AR) on ${AR_DEVICE_TYPE^^} with TP=${AR_TP}..."

    AR_TP=${AR_TP} \
    AR_GPUS=${ar_devices} \
    DYN_SYSTEM_PORT=8081 \
    NATS_SERVER=${NATS_SERVER} \
    ETCD_ENDPOINTS=${ETCD_ENDPOINTS} \
        python -m dynamo.vllm.omni \
        --model "$MODEL" \
        --stage-id 0 \
        --stage-configs-path "$STAGE_CONFIG" \
        --output-modalities image \
        --media-output-fs-url file:///tmp/dynamo_media \
        "${EXTRA_ARGS[@]}" &
    sleep 20

    # Router
    echo "Starting Router..."
    AR_TP=${AR_TP} \
    AR_GPUS=${ar_devices} \
    DIT_GPUS=0 \
    DIT_TP=${DIT_TP} \
    DYN_SYSTEM_PORT=8083 \
    NATS_SERVER=${NATS_SERVER} \
    ETCD_ENDPOINTS=${ETCD_ENDPOINTS} \
        python -m dynamo.vllm.omni \
        --model "$MODEL" \
        --omni-router \
        --stage-configs-path "$STAGE_CONFIG" \
        --output-modalities image \
        --media-output-fs-url file:///tmp/dynamo_media \
        "${EXTRA_ARGS[@]}" &
    sleep 5

    # Frontend
    echo "Starting Frontend..."
    NATS_SERVER=${NATS_SERVER} \
    ETCD_ENDPOINTS=${ETCD_ENDPOINTS} \
        python -m dynamo.frontend &

    wait_any_exit

elif [[ "${ROLE,,}" == "dit" ]]; then
    export VLLM_TARGET_DEVICE=${DIT_DEVICE_TYPE}

    # Compute 0-based logical device list from TP size (e.g. TP=2 would give "0,1")
    dit_devices=$(seq -s, 0 $(( DIT_TP - 1 )))

    echo "Starting Stage 1 (DiT) on ${DIT_DEVICE_TYPE^^} with TP=${DIT_TP}..."

    DIT_GPUS=${dit_devices} \
    DIT_TP=${DIT_TP} \
    DYN_SYSTEM_PORT=8082 \
    NATS_SERVER=${NATS_SERVER} \
    ETCD_ENDPOINTS=${ETCD_ENDPOINTS} \
        python -m dynamo.vllm.omni \
        --model "$MODEL" \
        --stage-id 1 \
        --stage-configs-path "$STAGE_CONFIG" \
        --output-modalities image \
        --media-output-fs-url file:///tmp/dynamo_media \
        "${EXTRA_ARGS[@]}" &

    wait_any_exit

else
    echo "ERROR: Unknown ROLE '${ROLE}'. Must be 'ar' or 'dit'."
    exit 1
fi
