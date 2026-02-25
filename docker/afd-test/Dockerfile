# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Dockerfile for AFD (Attention-FFN Disaggregation) testing
# Based on Dynamo development image

ARG BASE_IMAGE=nvcr.io/nvidia/cuda:12.4.1-devel-ubuntu22.04
FROM ${BASE_IMAGE}

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-venv \
    python3.11-dev \
    python3-pip \
    git \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.11 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Install test dependencies
RUN pip install --no-cache-dir \
    pytest==7.4.3 \
    pytest-asyncio==0.21.1 \
    pytest-cov==4.1.0 \
    numpy==1.26.4 \
    torch==2.2.0 --index-url https://download.pytorch.org/whl/cpu \
    prometheus-client==0.19.0

# Set working directory
WORKDIR /workspace

# Copy Dynamo source (will be mounted)
# RUN git clone https://github.com/ai-dynamo/dynamo.git /workspace/dynamo

# Set up Python path
ENV PYTHONPATH="/workspace/dynamo/components/src:${PYTHONPATH}"

# Default command
CMD ["pytest", "-v", "components/src/dynamo/sglang/tests/test_afd.py"]
