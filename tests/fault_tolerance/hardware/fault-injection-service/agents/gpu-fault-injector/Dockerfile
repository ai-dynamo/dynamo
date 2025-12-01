# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0
#
# GPU Fault Injector Agent - XID 79 Injection via nsenter+kmsg
# Runs as privileged DaemonSet on GPU nodes to inject XID errors
#
# NOTE: GPU nodes are AMD64/x86_64 architecture
# Build with: docker buildx build --platform linux/amd64 --load -t <image> .

FROM nvcr.io/nvidia/cuda:12.3.0-devel-ubuntu22.04

# Install system dependencies (nsenter, nvidia-smi, journalctl)
RUN DEBIAN_FRONTEND=noninteractive apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    curl \
    util-linux \
    systemd \
    kmod \
    pciutils \
    && rm -rf /var/lib/apt/lists/*

# Install Python packages
COPY requirements.txt /tmp/
RUN pip3 install --no-cache-dir -r /tmp/requirements.txt

# Create working directory
WORKDIR /app

# Copy agent code
COPY agent.py /app/
COPY gpu_xid_injector.py /app/

# Create log directory
RUN mkdir -p /var/log/gpu-fault-injector

# Set environment
ENV PYTHONUNBUFFERED=1

# Expose port
EXPOSE 8083

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8083/health || exit 1

# Run agent
CMD ["python3", "agent.py"]
