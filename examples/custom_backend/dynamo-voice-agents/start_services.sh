#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Startup script for the streaming ASR services

set -e

# Configuration from environment variables with defaults
ASR_CUDA_DEVICE=${ASR_CUDA_DEVICE:-0}
WORKER_LOG_LEVEL=${WORKER_LOG_LEVEL:-INFO}

# Dynamo runtime configuration
export DYN_STORE_KV=${DYN_STORE_KV:-file}
export DYN_REQUEST_PLANE=${DYN_REQUEST_PLANE:-tcp}
export DYN_LOG=${DYN_LOG:-$WORKER_LOG_LEVEL}
# Bind TCP endpoints to all interfaces for external access
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
python asr_inference.py &
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
python audio_chunker.py &
CHUNKER_PID=$!

# Wait a moment for chunker to start
sleep 3

echo "============================================"
echo "All services started successfully!"
echo "  - etcd PID: $ETCD_PID"
echo "  - Inference Worker PID: $INFERENCE_PID"
echo "  - Chunker Worker PID: $CHUNKER_PID"
echo "============================================"
echo ""
echo "Endpoints available:"
echo "  - streaming_asr/inference/process"
echo "  - streaming_asr/chunker/transcribe"
echo ""
echo "To test, run the client with an audio file:"
echo "  python asr_client.py /path/to/audio.wav"
echo "============================================"

# Function to handle shutdown
cleanup() {
    echo "Shutting down services..."
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
    wait $INFERENCE_PID $CHUNKER_PID $ETCD_PID
else
    wait $INFERENCE_PID $CHUNKER_PID
fi
