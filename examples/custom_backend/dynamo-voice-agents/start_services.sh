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
echo "============================================"

cd /app/src

# Start the ASR inference worker in background
echo "[1/2] Starting ASR Inference Worker..."
python asr_inference.py &
INFERENCE_PID=$!

# Wait for inference worker to initialize (model loading takes time)
echo "Waiting for inference worker to load model..."
sleep 10

# Check if inference worker is still running
if ! kill -0 $INFERENCE_PID 2>/dev/null; then
    echo "ERROR: Inference worker failed to start"
    exit 1
fi

# Start the audio chunker worker in background
echo "[2/2] Starting Audio Chunker Worker..."
python audio_chunker.py &
CHUNKER_PID=$!

# Wait a moment for chunker to start
sleep 3

echo "============================================"
echo "All services started successfully!"
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
    wait
    echo "Services stopped."
    exit 0
}

# Trap signals for graceful shutdown
trap cleanup SIGTERM SIGINT

# Wait for both processes
wait $INFERENCE_PID $CHUNKER_PID
