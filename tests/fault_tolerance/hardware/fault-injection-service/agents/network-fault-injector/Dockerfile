# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

FROM python:3.12-slim

WORKDIR /app

# Install system dependencies (tc, iptables, iproute2)
RUN apt-get update && apt-get install -y --no-install-recommends \
    iproute2 \
    iptables \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy agent code
COPY agent.py .

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8082/health || exit 1

EXPOSE 8082

CMD ["python", "agent.py"]

