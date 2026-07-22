#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0
# Start script for Dynamo SGLang Disaggregated Prefill/Decode (Intel XPU Encode Workers)
# Model: Qwen3-VL-32B-Instruct-FP8
# This starts: N Encode workers on Intel XPU devices for multimodal processing
# Run this AFTER starting the CUDA side (start_sglang_pd_cuda_32b_fp8.sh)
# Each encoder uses its own dedicated NIXL side channel port
#
# Usage:
#   ./disagg_encoder_xpu.sh [OPTIONS]
#
# Options:
#   --num-encoders N           Number of encoders to start (default: 1)
#   --remote-ip IP             Remote CUDA server IP (default: 172.26.46.162)
#   --nats-port PORT           NATS server port (default: 14222)
#   --etcd-port PORT           etcd server port (default: 12379)
#   --local-ip IP              Local XPU server IP (default: auto-detect)
#   --xpu-devices "D1 D2 ..."  Space-separated XPU device IDs (default: "2 4 6 0 1 3 5 7")
#   --side-channel-base PORT   Base port for NIXL side channels (default: 22098)
#   --kv-event-base PORT       Base port for KV events (default: 22080)
#   --ucx-nics "N1 N2 ..."     Space-separated UCX NICs, cycled (default: "mlx5_1:1 mlx5_2:1")
#   --model-path PATH          Model path (default: /mnt/weka/data/llm-d-models-pv/models--Qwen--Qwen3-VL-32B-Instruct-FP8)
#   --mem-fraction FRAC        Memory fraction static (default: 0.7)
#   --page-size SIZE           Page size (default: 16)
#   --startup-delay SECS       Delay between encoder starts (default: 10)
#   --help                     Show this help message
#
# Environment variables (can also be used):
#   NUM_ENCODERS, IP_REMOTE, PORT_NATS, PORT_ETCD, IP_LOCAL, XPU_DEVICES,
#   SIDE_CHANNEL_BASE_PORT, KV_EVENT_BASE_PORT, UCX_NICS, MODEL_PATH,
#   MEM_FRACTION, PAGE_SIZE, STARTUP_DELAY

set -e

# ========================================
# DEFAULT CONFIGURATION
# ========================================

# Parse command line arguments and set defaults
NUM_ENCODERS=${NUM_ENCODERS:-1}
IP_REMOTE=${IP_REMOTE:-"172.26.46.162"}
PORT_NATS=${PORT_NATS:-14222}
PORT_ETCD=${PORT_ETCD:-12379}
IP_LOCAL=${IP_LOCAL:-$(hostname -I | awk '{print $1}')}
XPU_DEVICES=${XPU_DEVICES:-"1 0 2 3 4 5 6 7"}
SIDE_CHANNEL_BASE_PORT=${SIDE_CHANNEL_BASE_PORT:-22098}
KV_EVENT_BASE_PORT=${KV_EVENT_BASE_PORT:-22080}
UCX_NICS=${UCX_NICS:-"mlx5_1:1 mlx5_2:1"}
MODEL_PATH=${MODEL_PATH:-"/mnt/weka/data/llm-d-models-pv/models--Qwen--Qwen3-VL-32B-Instruct-FP8"}
MEM_FRACTION=${MEM_FRACTION:-0.7}
PAGE_SIZE=${PAGE_SIZE:-16}
STARTUP_DELAY=${STARTUP_DELAY:-10}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --num-encoders)
            NUM_ENCODERS="$2"
            shift 2
            ;;
        --remote-ip)
            IP_REMOTE="$2"
            shift 2
            ;;
        --nats-port)
            PORT_NATS="$2"
            shift 2
            ;;
        --etcd-port)
            PORT_ETCD="$2"
            shift 2
            ;;
        --local-ip)
            IP_LOCAL="$2"
            shift 2
            ;;
        --xpu-devices)
            XPU_DEVICES="$2"
            shift 2
            ;;
        --side-channel-base)
            SIDE_CHANNEL_BASE_PORT="$2"
            shift 2
            ;;
        --kv-event-base)
            KV_EVENT_BASE_PORT="$2"
            shift 2
            ;;
        --ucx-nics)
            UCX_NICS="$2"
            shift 2
            ;;
        --model-path)
            MODEL_PATH="$2"
            shift 2
            ;;
        --mem-fraction)
            MEM_FRACTION="$2"
            shift 2
            ;;
        --page-size)
            PAGE_SIZE="$2"
            shift 2
            ;;
        --startup-delay)
            STARTUP_DELAY="$2"
            shift 2
            ;;
        --help)
            grep "^#" "$0" | grep -E "^# (Usage|Options|Environment)" -A 100 | head -n 30
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Run with --help for usage information"
            exit 1
            ;;
    esac
done

# Convert space-separated strings to arrays
IFS=' ' read -r -a XPU_DEVICE_ARRAY <<< "$XPU_DEVICES"
IFS=' ' read -r -a UCX_NIC_ARRAY <<< "$UCX_NICS"

# ========================================

LOG_DIR="$(pwd)/logs"
mkdir -p "$LOG_DIR"

# Validate number of encoders
if [[ $NUM_ENCODERS -lt 1 ]] || [[ $NUM_ENCODERS -gt 8 ]]; then
    echo "ERROR: NUM_ENCODERS must be between 1 and 8 (got: $NUM_ENCODERS)"
    exit 1
fi

# Validate we have enough XPU devices
if [[ ${#XPU_DEVICE_ARRAY[@]} -lt $NUM_ENCODERS ]]; then
    echo "ERROR: Not enough XPU devices specified (need $NUM_ENCODERS, got ${#XPU_DEVICE_ARRAY[@]})"
    echo "XPU_DEVICES: $XPU_DEVICES"
    exit 1
fi

echo "=========================================="
echo "Dynamo SGLang ${NUM_ENCODERS}x Intel XPU Encode Workers"
echo "Model: Qwen3-VL-32B-Instruct-FP8"
echo "=========================================="
echo ""
echo "Configuration:"
echo "  Number of Encoders: $NUM_ENCODERS"
echo "  Remote CUDA server: $IP_REMOTE"
echo "  Local XPU server:   $IP_LOCAL"
echo "  NATS Port:          $PORT_NATS"
echo "  etcd Port:          $PORT_ETCD"
echo "  Model Path:         $MODEL_PATH"
echo "  Memory Fraction:    $MEM_FRACTION"
echo "  Page Size:          $PAGE_SIZE"
echo "  Startup Delay:      ${STARTUP_DELAY}s"
echo ""
echo "XPU Devices: ${XPU_DEVICE_ARRAY[*]:0:$NUM_ENCODERS}"
echo "UCX NICs:    ${UCX_NIC_ARRAY[*]}"
echo ""

# Verify remote server is reachable
echo "Checking connectivity to CUDA server..."
if ! timeout 2 bash -c "cat < /dev/null > /dev/tcp/$IP_REMOTE/$PORT_NATS" 2>/dev/null; then
    echo "WARNING: Cannot reach NATS at $IP_REMOTE:$PORT_NATS"
    echo "Make sure CUDA side is running: ./start_sglang_pd_cuda_32b_fp8.sh"
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Array to store PIDs
ENCODE_PIDS=()

# Start encoders in a loop
for i in $(seq 1 $NUM_ENCODERS); do
    ENCODER_IDX=$((i - 1))
    XPU_DEVICE=${XPU_DEVICE_ARRAY[$ENCODER_IDX]}
    UCX_NIC=${UCX_NIC_ARRAY[$((ENCODER_IDX % ${#UCX_NIC_ARRAY[@]}))]}
    SIDE_CHANNEL_PORT=$((SIDE_CHANNEL_BASE_PORT + ENCODER_IDX))
    KV_EVENT_PORT=$((KV_EVENT_BASE_PORT + ENCODER_IDX * 3))

    if [[ $i -gt 1 ]]; then
        echo ""
        echo "Waiting ${STARTUP_DELAY} seconds before starting encoder $i..."
        sleep $STARTUP_DELAY
    fi

    echo ""
    echo "Starting Intel XPU Encode Worker $i (Device $XPU_DEVICE)..."
    if [[ $i -eq 1 ]]; then
        echo "This takes several minutes for model loading..."
    fi

    ZE_AFFINITY_MASK=$XPU_DEVICE \
    NATS_SERVER=nats://${IP_REMOTE}:${PORT_NATS} \
    ETCD_ENDPOINTS=http://${IP_REMOTE}:${PORT_ETCD} \
    ETCD_REQUEST_TIMEOUT=600 \
    ETCD_LEASE_TTL=600 \
    DYN_REQUEST_PLANE=tcp \
    TRANSFER_LOCAL=0 \
    PYTHONHASHSEED=0 \
    VLLM_NIXL_SIDE_CHANNEL_HOST=${IP_LOCAL} \
    VLLM_NIXL_SIDE_CHANNEL_PORT=${SIDE_CHANNEL_PORT} \
    DYN_VLLM_KV_EVENT_PORT=${KV_EVENT_PORT} \
    UCX_MEMTYPE_CACHE=0 \
    UCX_TLS=ze_copy,rc,tcp \
    UCX_NET_DEVICES=${UCX_NIC} \
    DYN_VLLM_EMBEDDING_TRANSFER_MODE=nixl-read \
    ENABLE_ENCODER_CACHE=0 \
    VISION_ENCODE_SERIALIZE=1 \
    NIXL_USE_CPU_HOST_MEMORY=1 \
    python3 -m dynamo.sglang \
        --model "$MODEL_PATH" \
        --enable-multimodal \
        --multimodal-encode-worker \
        --chat-template qwen2-vl \
        --dtype auto \
        --kv-cache-dtype auto \
        --mem-fraction-static $MEM_FRACTION \
        --page-size $PAGE_SIZE \
        --kv-events-config '{"publisher":"zmq","topic":"kv-events","endpoint":"tcp://*:'$KV_EVENT_PORT'","enable_kv_cache_events":true}' \
        2>&1 | tee -a "$LOG_DIR/encode_xpu_32b_$i.log" &

    ENCODE_PIDS+=($!)
    echo "Encode Worker $i started with PID: ${ENCODE_PIDS[$ENCODER_IDX]}"
done

echo ""
echo "=========================================="
echo "${NUM_ENCODERS}x Intel XPU Encode Workers Started"
echo "Model: Qwen3-VL-32B-Instruct-FP8"
echo "=========================================="
echo ""
echo "Configuration:"
echo "  - Model: $MODEL_PATH"
echo "  - Remote CUDA Server: $IP_REMOTE"
echo "  - NATS: nats://$IP_REMOTE:$PORT_NATS"
echo "  - etcd: http://$IP_REMOTE:$PORT_ETCD"
echo "  - Local XPU IP: $IP_LOCAL"
echo ""

# Print encoder details
for i in $(seq 1 $NUM_ENCODERS); do
    ENCODER_IDX=$((i - 1))
    XPU_DEVICE=${XPU_DEVICE_ARRAY[$ENCODER_IDX]}
    UCX_NIC=${UCX_NIC_ARRAY[$((ENCODER_IDX % ${#UCX_NIC_ARRAY[@]}))]}
    SIDE_CHANNEL_PORT=$((SIDE_CHANNEL_BASE_PORT + ENCODER_IDX))
    KV_EVENT_PORT=$((KV_EVENT_BASE_PORT + ENCODER_IDX * 3))

    echo "Encoder $i:"
    echo "  - XPU Device (ZE_AFFINITY_MASK): $XPU_DEVICE"
    echo "  - InfiniBand NIC: $UCX_NIC"
    echo "  - Side Channel: $IP_LOCAL:$SIDE_CHANNEL_PORT"
    echo "  - KV Events Port: $KV_EVENT_PORT"
    echo "  - Process PID: ${ENCODE_PIDS[$ENCODER_IDX]}"
    echo "  - Log: $LOG_DIR/encode_xpu_32b_$i.log"
    echo ""
done
echo "Intel XPU Optimizations:"
echo "  ✓ ZE_AFFINITY_MASK (Level Zero device selection)"
echo "  ✓ UCX_TLS=ze_copy (XPU memory copy)"
echo "  ✓ Balanced InfiniBand NICs (cycled across encoders)"
echo "  ✓ Dedicated NIXL side channel ports per encoder"
echo "  ✓ VISION_ENCODE_SERIALIZE=1"
echo "  ✓ NIXL_USE_CPU_HOST_MEMORY=1"
echo "  ✓ dtype=auto, kv-cache-dtype=auto (FP8 support)"
echo "  ✓ encoder_only mode enabled (loads vision encoder ONLY, not full LLM)"
echo "  ✓ ETCD_REQUEST_TIMEOUT=600 (improved stability)"
echo "  ✓ ETCD_LEASE_TTL=600 (prevents registration expiration under load)"
echo ""
echo "Model Notes:"
echo "  • With encoder_only fix: only vision encoder loaded (~2-4GB per device)"
echo "  • Total XPU memory: ~$((NUM_ENCODERS * 2))-$((NUM_ENCODERS * 4))GB for $NUM_ENCODERS encoders"
echo "  • Load time: ~5-10 minutes per encoder (~$((NUM_ENCODERS * 5))-$((NUM_ENCODERS * 10 + (NUM_ENCODERS - 1) * STARTUP_DELAY / 60)) min total with delays)"
echo "  • Same model as CUDA decode worker for consistency"
echo ""
echo "=========================================="
echo ""
echo "To monitor encode workers:"
if [[ $NUM_ENCODERS -eq 1 ]]; then
    echo "  tail -f $LOG_DIR/encode_xpu_32b_1.log"
elif [[ $NUM_ENCODERS -le 4 ]]; then
    echo "  tail -f $LOG_DIR/encode_xpu_32b_{1..$NUM_ENCODERS}.log"
else
    echo "  tail -f $LOG_DIR/encode_xpu_32b_*.log"
fi
echo ""
echo "To check when ready:"
if [[ $NUM_ENCODERS -eq 1 ]]; then
    echo "  grep -i 'registered\\|ready\\|succeeded' $LOG_DIR/encode_xpu_32b_1.log"
elif [[ $NUM_ENCODERS -le 4 ]]; then
    echo "  grep -i 'registered\\|ready\\|succeeded' $LOG_DIR/encode_xpu_32b_{1..$NUM_ENCODERS}.log"
else
    echo "  grep -i 'registered\\|ready\\|succeeded' $LOG_DIR/encode_xpu_32b_*.log"
fi
echo ""
echo "To stop all encode workers:"
echo "  kill ${ENCODE_PIDS[*]}"
echo "  # or: pkill -f 'dynamo.sglang.*multimodal-encode-worker'"
echo ""
if [[ $NUM_ENCODERS -gt 1 ]]; then
    echo "Performance tip:"
    echo "  Monitor all $NUM_ENCODERS encoders to ensure balanced load distribution"
    echo "  Each encoder should process ~$((100 / NUM_ENCODERS))% of incoming multimodal requests"
    echo ""
fi
