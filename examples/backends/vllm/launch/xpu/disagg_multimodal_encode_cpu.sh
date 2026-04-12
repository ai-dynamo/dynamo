#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Pure CPU Multimodal Encode Workers (vLLM backend)
# Launches multiple CPU-based encode workers with device-aware weighted routing
# CPUs: 1+ (configurable via NUM_ENCODE_WORKERS)
#
# This script launches ONLY encode workers on CPU (no prefill, no decode).
# It can work alongside XPU encode/prefill/decode workers from disagg_multimodal_epd_xpu.sh
#
# Device Detection: CPU mode is triggered by setting ZE_AFFINITY_MASK="" (XPU) or CUDA_VISIBLE_DEVICES="" (CUDA)
# The device-aware-weighted router automatically balances load between CPU and GPU/XPU workers

set -e
trap 'echo Cleaning up...; kill 0' EXIT

SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"
source "$SCRIPT_DIR/../../../../common/launch_utils.sh"

# Default values
MODEL_NAME="llava-hf/llava-1.5-7b-hf"
NUM_ENCODE_WORKERS="${NUM_ENCODE_WORKERS:-1}"

# Device platform and affinity env name.
# DEVICE_PLATFORM supports: cuda, xpu
DEVICE_PLATFORM="${DEVICE_PLATFORM:-xpu}"
if [[ -z "${DEVICE_AFFINITY_ENV:-}" ]]; then
    if [[ "${DEVICE_PLATFORM,,}" == "xpu" ]]; then
        DEVICE_AFFINITY_ENV="ZE_AFFINITY_MASK"
    else
        DEVICE_AFFINITY_ENV="CUDA_VISIBLE_DEVICES"
    fi
fi

# CUDA to CPU throughput ratio for device-aware routing (default: 8)
# Higher values give more weight to GPU/XPU workers
export DYN_ENCODER_CUDA_TO_CPU_RATIO="${DYN_ENCODER_CUDA_TO_CPU_RATIO:-8}"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            MODEL_NAME=$2
            shift 2
            ;;
        --num-workers)
            NUM_ENCODE_WORKERS=$2
            shift 2
            ;;
        --cuda-to-cpu-ratio)
            DYN_ENCODER_CUDA_TO_CPU_RATIO=$2
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Launch multiple CPU-based multimodal encode workers for vLLM"
            echo ""
            echo "Options:"
            echo "  --model <model_name>          Specify the VLM model to use (default: $MODEL_NAME)"
            echo "                                LLaVA 1.5 7B, Qwen2.5-VL, and Phi3V models are supported"
            echo "  --num-workers <N>             Number of CPU encode workers to launch (default: $NUM_ENCODE_WORKERS)"
            echo "  --cuda-to-cpu-ratio <ratio>   GPU-to-CPU throughput ratio for routing (default: $DYN_ENCODER_CUDA_TO_CPU_RATIO)"
            echo "  -h, --help                    Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0 --model llava-hf/llava-1.5-7b-hf"
            echo "  $0 --model Qwen/Qwen2.5-VL-7B-Instruct --num-workers 4"
            echo "  $0 --model microsoft/Phi-3.5-vision-instruct --num-workers 2 --cuda-to-cpu-ratio 10"
            echo ""
            echo "Note: This script launches encode workers on CPU only (no GPU/XPU)."
            echo "      Device platform (cuda/xpu) is auto-detected or set via DEVICE_PLATFORM env var."
            echo "      For XPU machines: Uses ZE_AFFINITY_MASK=\"\" for CPU detection"
            echo "      For CUDA machines: Uses CUDA_VISIBLE_DEVICES=\"\" for CPU detection"
            echo "      Can work alongside XPU encode/prefill/decode workers."
            echo "      Uses device-aware-weighted routing to balance CPU vs GPU/XPU load."
            echo ""
            echo "Environment Variables:"
            echo "  DEVICE_PLATFORM               - Device type: cuda or xpu (default: xpu)"
            echo "  DYN_ENCODER_CUDA_TO_CPU_RATIO - GPU-to-CPU throughput ratio (default: 8)"
            echo "  DYN_HTTP_PORT                 - Frontend HTTP port (default: 8000)"
            echo "  NUM_ENCODE_WORKERS            - Number of CPU workers (default: 1)"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Validate number of workers
if [[ "$NUM_ENCODE_WORKERS" -lt 1 ]]; then
    echo "Error: --num-workers must be at least 1"
    exit 1
fi

HTTP_PORT="${DYN_HTTP_PORT:-8000}"
print_launch_banner --multimodal "Launching CPU Encode Workers ($NUM_ENCODE_WORKERS workers)" "$MODEL_NAME" "$HTTP_PORT"

echo "=================================================="
echo "Configuration:"
echo "  Device Platform: $DEVICE_PLATFORM"
echo "  Affinity Env: $DEVICE_AFFINITY_ENV"
echo "  Model: $MODEL_NAME"
echo "  CPU Encode Workers: $NUM_ENCODE_WORKERS"
echo "  CUDA-to-CPU Ratio: $DYN_ENCODER_CUDA_TO_CPU_RATIO"
echo "  Router Mode: device-aware-weighted"
echo "=================================================="

# Start frontend with device-aware weighted routing
#echo "Starting frontend with device-aware weighted routing..."
#DYN_ROUTER_MODE=device-aware-weighted python -m dynamo.frontend &

# Give frontend time to start
sleep 2

# Start multiple CPU-based encode workers
# Set affinity env to empty string to force CPU mode (triggers device detection)
# For XPU machines: ZE_AFFINITY_MASK=""
# For CUDA machines: CUDA_VISIBLE_DEVICES=""
# Use port range 20100+ for CPU workers to avoid conflicts with GPU/XPU workers (20097-20099)
for i in $(seq 0 $((NUM_ENCODE_WORKERS - 1))); do
    NIXL_PORT=$((20100 + i * 2))
    KV_EVENTS_PORT=$((20200 + i))
    echo "Starting CPU encode worker $i (NIXL port: $NIXL_PORT, KV events port: $KV_EVENTS_PORT)..."

    # Key: Set device affinity env to "" to force CPU device detection
    VLLM_NIXL_SIDE_CHANNEL_PORT=$NIXL_PORT \
    env $DEVICE_AFFINITY_ENV="" \
    python -m dynamo.vllm \
      --multimodal-encode-worker \
      --enable-multimodal \
      --enable-mm-embeds \
      --model "$MODEL_NAME" \
      --kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_both","kv_buffer_device":"cpu"}' \
      --kv-events-config '{"publisher":"zmq","topic":"kv-events","endpoint":"tcp://*:'"$KV_EVENTS_PORT"'"}' &

    # Small delay between workers to avoid race conditions during startup
    sleep 2
done

echo "=================================================="
echo "All $NUM_ENCODE_WORKERS CPU encode workers started."
echo ""
echo "Device-aware routing is enabled:"
echo "  - CPU workers detected via $DEVICE_AFFINITY_ENV=\"\""
echo "  - If GPU/XPU workers are also running, load will be"
echo "    balanced based on device capabilities"
echo "  - GPU/XPU workers get ${DYN_ENCODER_CUDA_TO_CPU_RATIO}x weight vs CPU"
echo "=================================================="

# Exit on first worker failure; kill 0 in the EXIT trap tears down the rest
wait_any_exit
