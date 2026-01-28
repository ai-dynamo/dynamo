# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Dockerfile for Streaming ASR Voice Agents
#
# This image runs the streaming ASR pipeline with NVIDIA Parakeet RNNT model
# using the Dynamo distributed runtime.
#
# Build:
#   docker build -t dynamo-voice-agents .
#
# Run:
#   docker run --gpus '"device=0"' \
#       -e ASR_CUDA_DEVICE=0 \
#       -e WORKER_LOG_LEVEL=INFO \
#       dynamo-voice-agents

# Use NVIDIA NeMo container as base (includes NeMo toolkit with ASR models)
ARG BASE_IMAGE=nvcr.io/nvidia/nemo
ARG BASE_IMAGE_TAG=25.04

FROM ${BASE_IMAGE}:${BASE_IMAGE_TAG}

# Build arguments
ARG PYTHON_VERSION=3.12

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
    # ASR configuration
    ASR_CUDA_DEVICE=0 \
    ASR_MODEL_NAME=nvidia/parakeet-rnnt-1.1b \
    # Logging
    WORKER_LOG_LEVEL=INFO \
    # OpenTelemetry (disabled by default)
    OTEL_EXPORT_ENABLED=false \
    OTEL_EXPORTER_OTLP_TRACES_ENDPOINT=http://localhost:4317

WORKDIR /app

# Install uv package manager for fast dependency installation
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Create virtual environment
RUN uv venv /opt/venv --python ${PYTHON_VERSION}
ENV VIRTUAL_ENV=/opt/venv \
    PATH="/opt/venv/bin:${PATH}"

# Copy requirements and install dependencies
COPY requirements.txt /app/
RUN uv pip install --no-cache -r requirements.txt

# Copy application source code
COPY src/ /app/src/

# Copy startup script
COPY start_services.sh /app/
RUN chmod +x /app/start_services.sh

# Create directory for Dynamo runtime state
RUN mkdir -p /app/state && chmod 777 /app/state
ENV DYN_STATE_DIR=/app/state

# Create non-root user for running the application
RUN useradd -m -s /bin/bash asr || true
RUN chown -R asr:asr /app /opt/venv 2>/dev/null || true

# Health check - verify Python and imports work
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "from dynamo.runtime import DistributedRuntime; print('OK')" || exit 1

# Default command: start both workers
CMD ["/app/start_services.sh"]
