#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Startup script for the streaming ASR services
#
# Environment Variables:
#   ENABLE_REMOTE_ACCESS=true  - Auto-configure for remote client access
#   DYN_TCP_RPC_HOST           - Server IP for remote access (auto-detected if not set)
#   DYN_STORE_KV               - KV store backend: 'etcd' (remote) or 'file' (local)
#   ASR_CUDA_DEVICE            - CUDA device index (default: 0)

set -e

# Configuration from environment variables with defaults
ASR_CUDA_DEVICE=${ASR_CUDA_DEVICE:-0}
WORKER_LOG_LEVEL=${WORKER_LOG_LEVEL:-INFO}

# Auto-detect server IP for remote access
detect_server_ip() {
    # Try multiple methods to get the server's IP address
    local ip=""
    
    # Method 1: hostname -I (Linux)
    if command -v hostname &> /dev/null; then
        ip=$(hostname -I 2>/dev/null | awk '{print $1}')
    fi
    
    # Method 2: ip route (Linux)
    if [ -z "$ip" ] && command -v ip &> /dev/null; then
        ip=$(ip route get 1 2>/dev/null | awk '{print $7; exit}')
    fi
    
    # Method 3: ifconfig (fallback)
    if [ -z "$ip" ] && command -v ifconfig &> /dev/null; then
        ip=$(ifconfig | grep -Eo 'inet (addr:)?([0-9]*\.){3}[0-9]*' | grep -Eo '([0-9]*\.){3}[0-9]*' | grep -v '127.0.0.1' | head -1)
    fi
    
    echo "$ip"
}

# Enable remote access if requested
if [ "${ENABLE_REMOTE_ACCESS:-false}" = "true" ]; then
    echo "Remote access enabled - configuring for external clients..."
    export DYN_STORE_KV=etcd
    
    # Auto-detect server IP if not explicitly set
    if [ -z "$DYN_TCP_RPC_HOST" ] || [ "$DYN_TCP_RPC_HOST" = "0.0.0.0" ]; then
        SERVER_IP=$(detect_server_ip)
        if [ -n "$SERVER_IP" ]; then
            export DYN_TCP_RPC_HOST="$SERVER_IP"
            echo "Auto-detected server IP: $SERVER_IP"
        else
            echo "WARNING: Could not auto-detect server IP. Set DYN_TCP_RPC_HOST manually."
            export DYN_TCP_RPC_HOST="0.0.0.0"
        fi
    fi
fi

# Dynamo runtime configuration
export DYN_STORE_KV=${DYN_STORE_KV:-file}
export DYN_REQUEST_PLANE=${DYN_REQUEST_PLANE:-tcp}
export DYN_LOG=${DYN_LOG:-$WORKER_LOG_LEVEL}
# Bind TCP endpoints - use detected IP for remote access, 0.0.0.0 for local
export DYN_TCP_RPC_HOST=${DYN_TCP_RPC_HOST:-0.0.0.0}

# Export CUDA device for inference worker
export ASR_CUDA_DEVICE

# Optional: OpenTelemetry tracing
# Set OTEL_EXPORT_ENABLED=true to enable
export OTEL_EXPORT_ENABLED=${OTEL_EXPORT_ENABLED:-false}
export OTEL_EXPORTER_OTLP_TRACES_ENDPOINT=${OTEL_EXPORTER_OTLP_TRACES_ENDPOINT:-http://localhost:4317}

echo "============================================"
echo "Starting Streaming ASR Services"
echo "============================================"
echo "CUDA Device: $ASR_CUDA_DEVICE"
echo "KV Store: $DYN_STORE_KV"
echo "Request Plane: $DYN_REQUEST_PLANE"
echo "TCP RPC Host: $DYN_TCP_RPC_HOST"
echo "Log Level: $WORKER_LOG_LEVEL"
echo "Tracing Enabled: $OTEL_EXPORT_ENABLED"
echo "State Directory: ${DYN_STATE_DIR:-/app/state}"
echo "============================================"

# Ensure state directory exists and is clean
export DYN_STATE_DIR=${DYN_STATE_DIR:-/app/state}
mkdir -p "$DYN_STATE_DIR"
rm -rf "$DYN_STATE_DIR"/* 2>/dev/null || true

# Start etcd if available (required by Dynamo runtime for endpoint registration)
ETCD_PID=""
if command -v etcd &> /dev/null; then
    ETCD_DATA_DIR=${ETCD_DATA_DIR:-/tmp/etcd-data}
    mkdir -p "$ETCD_DATA_DIR"
    rm -rf "$ETCD_DATA_DIR"/* 2>/dev/null || true

    echo "[0/3] Starting etcd..."
    etcd --listen-client-urls http://0.0.0.0:2379 \
         --advertise-client-urls http://0.0.0.0:2379 \
         --data-dir "$ETCD_DATA_DIR" \
         --log-level warn &
    ETCD_PID=$!

    # Wait for etcd to be ready
    sleep 2
    if ! kill -0 $ETCD_PID 2>/dev/null; then
        echo "ERROR: etcd failed to start"
        exit 1
    fi
    echo "etcd started (PID: $ETCD_PID)"
else
    echo "[0/3] etcd not found, skipping (may already be running or not needed)"
fi

cd /app/src

# Start the ASR inference worker in background
echo "[1/3] Starting ASR Inference Worker..."
python3 asr_inference.py &
INFERENCE_PID=$!

# Wait for inference worker to initialize (model loading takes time)
# The parakeet-rnnt-1.1b model is ~4.3GB and takes 30+ seconds to load
echo "Waiting for inference worker to load model (this may take 30-60 seconds)..."
sleep 30

# Check if inference worker is still running
if ! kill -0 $INFERENCE_PID 2>/dev/null; then
    echo "ERROR: Inference worker failed to start"
    exit 1
fi

# Start the audio chunker worker in background
echo "[2/3] Starting Audio Chunker Worker..."
python3 audio_chunker.py &
CHUNKER_PID=$!

# Wait a moment for chunker to start
sleep 3

# Start the streaming ASR service for real-time microphone streaming
echo "[3/3] Starting Streaming ASR Service (for real-time mic streaming)..."
python3 streaming_asr_service.py &
STREAMING_PID=$!

# Wait for streaming service to load model
echo "Waiting for streaming service to load model..."
sleep 30

# Check if streaming service is still running
if ! kill -0 $STREAMING_PID 2>/dev/null; then
    echo "WARNING: Streaming service failed to start (continuing anyway)"
    STREAMING_PID=""
fi

echo "============================================"
echo "All services started successfully!"
echo "  - etcd PID: $ETCD_PID"
echo "  - Inference Worker PID: $INFERENCE_PID"
echo "  - Chunker Worker PID: $CHUNKER_PID"
echo "  - Streaming Service PID: $STREAMING_PID"
echo "============================================"
echo ""
echo "Endpoints available:"
echo "  File-based transcription:"
echo "    - streaming_asr/inference/process"
echo "    - streaming_asr/chunker/transcribe"
echo "  Real-time microphone streaming:"
echo "    - streaming_asr/realtime/transcribe_stream"
echo ""
echo "To test file-based transcription (on this server):"
echo "  python3 asr_client.py /path/to/audio.wav"
echo ""

# Show remote client instructions if using etcd
if [ "$DYN_STORE_KV" = "etcd" ]; then
    echo "============================================"
    echo "REMOTE CLIENT CONNECTION"
    echo "============================================"
    echo "To connect from a remote machine with a microphone:"
    echo ""
    echo "  1. Install dependencies:"
    echo "     pip install ai-dynamo sounddevice uvloop"
    echo ""
    echo "  2. Set environment variables:"
    echo "     export DYN_STORE_KV=etcd"
    echo "     export ETCD_ENDPOINTS=http://${DYN_TCP_RPC_HOST}:2379"
    echo "     export DYN_REQUEST_PLANE=tcp"
    echo ""
    echo "  3. Run the microphone client:"
    echo "     python3 mic_client.py"
    echo ""
fi
echo "============================================"

# Function to handle shutdown
cleanup() {
    echo "Shutting down services..."
    [ -n "$STREAMING_PID" ] && kill $STREAMING_PID 2>/dev/null || true
    kill $CHUNKER_PID 2>/dev/null || true
    kill $INFERENCE_PID 2>/dev/null || true
    [ -n "$ETCD_PID" ] && kill $ETCD_PID 2>/dev/null || true
    wait
    echo "Services stopped."
    exit 0
}

# Trap signals for graceful shutdown
trap cleanup SIGTERM SIGINT

# Wait for all processes
if [ -n "$ETCD_PID" ]; then
    if [ -n "$STREAMING_PID" ]; then
        wait $INFERENCE_PID $CHUNKER_PID $STREAMING_PID $ETCD_PID
    else
        wait $INFERENCE_PID $CHUNKER_PID $ETCD_PID
    fi
else
    if [ -n "$STREAMING_PID" ]; then
        wait $INFERENCE_PID $CHUNKER_PID $STREAMING_PID
    else
        wait $INFERENCE_PID $CHUNKER_PID
    fi
fi
