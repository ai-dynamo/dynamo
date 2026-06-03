# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Power Agent image — per-node NVML power-cap enforcement DaemonSet.
# Referenced by deploy/helm/charts/power-agent/values.yaml (image.repository)
# and built by the `power-agent` CI job (.github/workflows/pr.yaml,
# post-merge-ci.yml).
#
#   docker build -t power-agent:dev -f components/power_agent/Dockerfile \
#       components/power_agent

# DOCKER_PROXY lets CI route the base image through the ECR Docker Hub mirror
# (e.g. <acct>.dkr.ecr.<region>.amazonaws.com/dockerhub/). Empty by default so
# `docker build` works unchanged for local/dev use straight from Docker Hub.
ARG DOCKER_PROXY=

# Slim base — the NVIDIA Container Toolkit injects libnvidia-ml.so at runtime
# via runtimeClassName: nvidia (set by the Helm chart), so the image does NOT
# bake pynvml's native dependency. nvmlInit() resolves the host driver lib.
FROM ${DOCKER_PROXY}python:3.11-slim-bookworm AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Runtime deps. These match power_agent.py's three optional imports:
# pynvml (NVML bindings), kubernetes (pod listing), prometheus_client (metrics).
RUN pip install --no-cache-dir \
    pynvml==12.0.* \
    kubernetes==30.* \
    prometheus-client==0.20.*

WORKDIR /app
COPY power_agent.py /app/power_agent.py

# NVML power-cap writes require root, and the agent persists managed-GPU UUID
# state to a hostPath volume. The Helm chart also sets runAsUser: 0 / privileged.
USER 0

# The Helm chart's `command:` owns the real argv (it injects --safe-default-watts,
# --node-name, --prometheus-port, etc.). This ENTRYPOINT is for operators who run
# the image directly outside the chart (dev / debugging).
ENTRYPOINT ["python", "/app/power_agent.py"]
CMD ["--help"]
