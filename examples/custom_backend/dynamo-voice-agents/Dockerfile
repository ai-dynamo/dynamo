# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Dockerfile for Streaming ASR Voice Agents
#
# This image runs the streaming ASR pipeline with NVIDIA Parakeet RNNT model
# using the Dynamo distributed runtime. Supports both file-based transcription
# and real-time microphone streaming from remote clients.
#
# Build:
#   docker build -t dynamo-voice-agents .
#
# Run (local file transcription):
#   docker run --gpus '"device=0"' \
#       -e ASR_CUDA_DEVICE=0 \
#       dynamo-voice-agents
#
# Run (remote microphone streaming):
#   docker run --gpus '"device=0"' \
#       -p 2379:2379 \
#       -e ASR_CUDA_DEVICE=0 \
#       -e ENABLE_REMOTE_ACCESS=true \
#       dynamo-voice-agents
#
# Then connect from a remote machine:
#   export DYN_STORE_KV=etcd
#   export ETCD_ENDPOINTS=http://<SERVER_IP>:2379
#   export DYN_REQUEST_PLANE=tcp
#   python3 mic_client.py

# Use Triton Server image as base (includes PyTorch and CUDA)
ARG BASE_IMAGE=nvcr.io/nvidia/tritonserver
ARG BASE_IMAGE_TAG=24.12-py3

FROM ${BASE_IMAGE}:${BASE_IMAGE_TAG}

# Labels
LABEL maintainer="NVIDIA Corporation"
LABEL description="Streaming ASR Voice Agents with NVIDIA Dynamo"
LABEL version="1.0"

# Environment variables with defaults
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    # Dynamo runtime configuration
    DYN_STORE_KV=file \
    DYN_REQUEST_PLANE=tcp \
    # Remote access: set to 'true' to enable remote microphone clients
    # This auto-detects server IP and uses etcd for service discovery
    ENABLE_REMOTE_ACCESS=false \
    # ASR configuration
    ASR_CUDA_DEVICE=0 \
    ASR_MODEL_NAME=nvidia/parakeet-rnnt-1.1b \
    # Logging
    WORKER_LOG_LEVEL=INFO \
    # OpenTelemetry (disabled by default)
    OTEL_EXPORT_ENABLED=false \
    OTEL_EXPORTER_OTLP_TRACES_ENDPOINT=http://localhost:4317

WORKDIR /app

# Install build tools, etcd, and dependencies for NeMo
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    g++ \
    wget \
    git \
    libsndfile1 \
    sox \
    libsox-dev \
    ffmpeg \
    libavcodec-extra \
    && rm -rf /var/lib/apt/lists/*

# Install etcd (required by Dynamo runtime for endpoint registration)
ARG ETCD_VERSION=v3.5.12
RUN wget -qO- https://github.com/etcd-io/etcd/releases/download/${ETCD_VERSION}/etcd-${ETCD_VERSION}-linux-amd64.tar.gz | \
    tar -xz -C /tmp && \
    mv /tmp/etcd-${ETCD_VERSION}-linux-amd64/etcd /usr/local/bin/ && \
    mv /tmp/etcd-${ETCD_VERSION}-linux-amd64/etcdctl /usr/local/bin/ && \
    rm -rf /tmp/etcd-${ETCD_VERSION}-linux-amd64

# Install Cython and numpy first (required for some NeMo dependencies)
# Note: Skipping pip upgrade as Triton image has pip installed via apt
RUN pip install --no-cache-dir \
    Cython \
    numpy

# Install PyTorch audio and torchcodec (matching the PyTorch version in Triton 24.12)
RUN pip install --no-cache-dir \
    torchaudio \
    torchcodec

# Install NeMo toolkit 2.4 with ASR support and all dependencies
# NeMo 2.4 requires specific dependencies for ASR functionality
RUN pip install --no-cache-dir \
    "nemo_toolkit[asr]==2.4.0"

# Install Dynamo runtime (since we're not using the Dynamo frontend image)
RUN pip install --no-cache-dir \
    ai-dynamo

# Install additional application dependencies
RUN pip install --no-cache-dir \
    pydantic>=2.0.0 \
    uvloop>=0.19.0 \
    opentelemetry-api>=1.20.0 \
    opentelemetry-sdk>=1.20.0 \
    opentelemetry-exporter-otlp-proto-grpc>=1.20.0

# Copy application source code
COPY src/ /app/src/

# Copy startup script
COPY start_services.sh /app/
RUN chmod +x /app/start_services.sh

# Create directory for Dynamo runtime state
RUN mkdir -p /app/state && chmod 777 /app/state
ENV DYN_STATE_DIR=/app/state

# Health check - verify Python and imports work
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "from dynamo.runtime import DistributedRuntime; print('OK')" || exit 1

# Default command: start both workers
CMD ["/app/start_services.sh"]
