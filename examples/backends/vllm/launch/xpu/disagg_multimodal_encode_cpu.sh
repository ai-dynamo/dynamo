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

# CPU NUMA binding configuration
# Format: "core_range1|core_range2|core_range3|..."
# VLLM_CPU_DISABLE_HT: Set to 1 to use only physical cores (no hyperthreads)
# Note: vLLM's CPU parser only supports simple ranges (e.g., "0-31"), not comma-separated
# ranges (e.g., "0-31,64-95"). Therefore, we default to physical cores only.
VLLM_CPU_DISABLE_HT="${VLLM_CPU_DISABLE_HT:-1}"

# Convert space-separated CPU list to range format
# E.g., "0 1 2 3 ... 31 64 65 ... 95" -> "0-31,64-95"
function cpulist_to_ranges() {
    local cpus=($1)
    if [[ ${#cpus[@]} -eq 0 ]]; then
        echo ""
        return
    fi

    local ranges=""
    local range_start=${cpus[0]}
    local range_end=${cpus[0]}

    for ((i=1; i<${#cpus[@]}; i++)); do
        local curr=${cpus[$i]}
        local prev=$range_end

        if [[ $((curr - prev)) -eq 1 ]]; then
            # Consecutive, extend current range
            range_end=$curr
        else
            # Gap detected, close current range and start new one
            if [[ -n "$ranges" ]]; then
                ranges="${ranges},"
            fi
            if [[ $range_start -eq $range_end ]]; then
                ranges="${ranges}${range_start}"
            else
                ranges="${ranges}${range_start}-${range_end}"
            fi
            range_start=$curr
            range_end=$curr
        fi
    done

    # Close last range
    if [[ -n "$ranges" ]]; then
        ranges="${ranges},"
    fi
    if [[ $range_start -eq $range_end ]]; then
        ranges="${ranges}${range_start}"
    else
        ranges="${ranges}${range_start}-${range_end}"
    fi

    echo "$ranges"
}

# Auto-detect NUMA topology if not set
function detect_numa_bindings() {
    local disable_ht=$1

    # Check if lscpu is available (preferred for range format)
    if command -v lscpu &> /dev/null; then
        local num_nodes=$(lscpu | grep "NUMA node(s):" | awk '{print $3}')
        if [[ -n "$num_nodes" ]] && [[ "$num_nodes" -gt 0 ]]; then
            local bindings=""
            for node in $(seq 0 $((num_nodes - 1))); do
                local cpu_line=$(lscpu | grep "NUMA node${node} CPU(s):" | awk '{print $4}')
                if [[ -n "$cpu_line" ]]; then
                    local node_binding=""
                    if [[ $disable_ht -eq 1 ]]; then
                        # Take only the first range (physical cores, no hyperthreads)
                        # E.g., "0-31,64-95" -> "0-31"
                        node_binding=$(echo "$cpu_line" | cut -d',' -f1)
                    else
                        # Take all ranges (physical cores + hyperthreads)
                        node_binding="$cpu_line"
                    fi

                    if [[ -n "$bindings" ]]; then
                        bindings="${bindings}|${node_binding}"
                    else
                        bindings="${node_binding}"
                    fi
                fi
            done
            echo "$bindings"
            return
        fi
    fi

    # Fallback to numactl if lscpu not available
    if ! command -v numactl &> /dev/null; then
        echo ""
        return
    fi

    local num_nodes=$(numactl --hardware 2>/dev/null | grep "available:" | awk '{print $2}')
    if [[ -z "$num_nodes" ]] || [[ "$num_nodes" -le 0 ]]; then
        echo ""
        return
    fi

    # Build binding string from NUMA nodes
    local bindings=""
    for node in $(seq 0 $((num_nodes - 1))); do
        # Get CPU list for this NUMA node (space-separated numbers)
        local cpu_line=$(numactl --hardware 2>/dev/null | grep "node $node cpus:" | cut -d: -f2)
        if [[ -n "$cpu_line" ]]; then
            # Convert to range format
            local node_binding=$(cpulist_to_ranges "$cpu_line")

            if [[ $disable_ht -eq 1 ]]; then
                # Take only the first range (physical cores, no hyperthreads)
                node_binding=$(echo "$node_binding" | cut -d',' -f1)
            fi

            if [[ -n "$bindings" ]]; then
                bindings="${bindings}|${node_binding}"
            else
                bindings="${node_binding}"
            fi
        fi
    done

    echo "$bindings"
}

# Set NUMA bindings with auto-detection fallback
if [[ -z "${VLLM_CPU_OMP_THREADS_BIND:-}" ]]; then
    DETECTED_BINDINGS=$(detect_numa_bindings $VLLM_CPU_DISABLE_HT)
    if [[ -n "$DETECTED_BINDINGS" ]]; then
        VLLM_CPU_OMP_THREADS_BIND="$DETECTED_BINDINGS"
        if [[ $VLLM_CPU_DISABLE_HT -eq 1 ]]; then
            echo "Auto-detected NUMA topology (physical cores only): $VLLM_CPU_OMP_THREADS_BIND"
        else
            echo "Auto-detected NUMA topology (physical + hyperthreads): $VLLM_CPU_OMP_THREADS_BIND"
        fi
    else
        # Fallback: no default binding (let workers use all cores)
        VLLM_CPU_OMP_THREADS_BIND=""
    fi
else
    VLLM_CPU_OMP_THREADS_BIND="${VLLM_CPU_OMP_THREADS_BIND}"
fi

VLLM_CPU_SGL_KERNEL="${VLLM_CPU_SGL_KERNEL:-0}"
VLLM_CPU_KVCACHE_SPACE="${VLLM_CPU_KVCACHE_SPACE:-40}"

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
        --cpu-bind)
            VLLM_CPU_OMP_THREADS_BIND=$2
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
            echo "  --cpu-bind <bindings>         NUMA CPU core bindings per worker (e.g., \"0-30|32-62|64-94\")"
            echo "  -h, --help                    Show this help message"
            echo ""
            echo "Examples:"
            echo "  # Auto-detect NUMA topology (defaults to physical cores only)"
            echo "  $0 --model llava-hf/llava-1.5-7b-hf --num-workers 2"
            echo ""
            echo "  # Auto-detect with all cores (physical + hyperthreads) - NOT RECOMMENDED"
            echo "  # Note: May fail if NUMA has non-contiguous ranges (e.g., 0-31,64-95)"
            echo "  VLLM_CPU_DISABLE_HT=0 $0 --num-workers 2"
            echo ""
            echo "  # Manual NUMA binding (simple ranges only)"
            echo "  $0 --num-workers 2 --cpu-bind \"0-31|32-63\""
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
            echo "  VLLM_CPU_OMP_THREADS_BIND     - NUMA CPU core bindings (auto-detected if not set)"
            echo "                                  Uses numactl/taskset for binding, format: \"0-31|32-63\""
            echo "  VLLM_CPU_DISABLE_HT           - Disable hyperthreading (1=physical cores only, 0=all)"
            echo "                                  Default: 1 (physical cores only, recommended)"
            echo "  VLLM_CPU_SGL_KERNEL           - Enable SGL kernel (default: 1)"
            echo "  VLLM_CPU_KVCACHE_SPACE        - KV cache space in GB (default: 40)"
            echo ""
            echo "Note: CPU binding uses numactl (preferred) or taskset (fallback)"
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

# Parse NUMA bindings if provided
if [[ -n "$VLLM_CPU_OMP_THREADS_BIND" ]]; then
    IFS='|' read -ra CPU_BINDINGS <<< "$VLLM_CPU_OMP_THREADS_BIND"
    if [[ ${#CPU_BINDINGS[@]} -lt $NUM_ENCODE_WORKERS ]]; then
        echo "Warning: Only ${#CPU_BINDINGS[@]} CPU bindings provided for $NUM_ENCODE_WORKERS workers."
        echo "         Workers beyond binding count will not have NUMA affinity set."
    fi
else
    CPU_BINDINGS=()
fi

echo "=================================================="
echo "Configuration:"
echo "  Device Platform: $DEVICE_PLATFORM"
echo "  Affinity Env: $DEVICE_AFFINITY_ENV"
echo "  Model: $MODEL_NAME"
echo "  CPU Encode Workers: $NUM_ENCODE_WORKERS"
echo "  CUDA-to-CPU Ratio: $DYN_ENCODER_CUDA_TO_CPU_RATIO"
echo "  Router Mode: device-aware-weighted"
if [[ ${#CPU_BINDINGS[@]} -gt 0 ]]; then
    echo "  CPU NUMA Bindings: ${VLLM_CPU_OMP_THREADS_BIND}"
    echo "  VLLM CPU SGL Kernel: $VLLM_CPU_SGL_KERNEL"
    echo "  VLLM CPU KVCache Space: ${VLLM_CPU_KVCACHE_SPACE} GB"
else
    echo "  CPU NUMA Bindings: Not configured (all workers share all cores)"
fi
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

    # Get CPU binding for this worker if available
    WORKER_CPU_BIND=""
    if [[ $i -lt ${#CPU_BINDINGS[@]} ]]; then
        WORKER_CPU_BIND="${CPU_BINDINGS[$i]}"
        echo "Starting CPU encode worker $i on cores $WORKER_CPU_BIND (NIXL: $NIXL_PORT, KV events: $KV_EVENTS_PORT)..."
    else
        echo "Starting CPU encode worker $i (NIXL: $NIXL_PORT, KV events: $KV_EVENTS_PORT)..."
    fi

    # Launch worker with device affinity env set to empty for CPU detection
    # Use env to properly set dynamic env variable name to empty
    if [[ -n "$WORKER_CPU_BIND" ]]; then
        echo "  Setting VLLM_CPU_OMP_THREADS_BIND=$WORKER_CPU_BIND and $DEVICE_AFFINITY_ENV= (empty for CPU)"
        env "$DEVICE_AFFINITY_ENV=" \
            VLLM_NIXL_SIDE_CHANNEL_PORT=$NIXL_PORT \
            VLLM_CPU_SGL_KERNEL=$VLLM_CPU_SGL_KERNEL \
            VLLM_CPU_KVCACHE_SPACE=$VLLM_CPU_KVCACHE_SPACE \
            VLLM_CPU_OMP_THREADS_BIND=$WORKER_CPU_BIND \
            python -m dynamo.vllm \
              --multimodal-encode-worker \
              --enable-multimodal \
              --enable-mm-embeds \
              --model "$MODEL_NAME" \
              --kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_both","kv_buffer_device":"cpu"}' \
              --kv-events-config '{"publisher":"zmq","topic":"kv-events","endpoint":"tcp://*:'"$KV_EVENTS_PORT"'"}' &
    else
        echo "  Setting $DEVICE_AFFINITY_ENV= (empty for CPU)"
        env "$DEVICE_AFFINITY_ENV=" \
            VLLM_NIXL_SIDE_CHANNEL_PORT=$NIXL_PORT \
            VLLM_CPU_SGL_KERNEL=$VLLM_CPU_SGL_KERNEL \
            VLLM_CPU_KVCACHE_SPACE=$VLLM_CPU_KVCACHE_SPACE \
            python -m dynamo.vllm \
              --multimodal-encode-worker \
              --enable-multimodal \
              --enable-mm-embeds \
              --model "$MODEL_NAME" \
              --kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_both","kv_buffer_device":"cpu"}' \
              --kv-events-config '{"publisher":"zmq","topic":"kv-events","endpoint":"tcp://*:'"$KV_EVENTS_PORT"'"}' &
    fi

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
