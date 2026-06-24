# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Power Agent image — per-node NVML power-cap enforcement DaemonSet.
# Referenced by deploy/helm/charts/power-agent/values.yaml (image.repository)
# and built by the `power-agent` CI job (.github/workflows/pr.yaml,
# post-merge-ci.yml).
#
# Two stages:
#   * `runtime` — the slim, shipped image (built/pushed with `--target runtime`).
#   * `test`    — `FROM runtime` + pytest + tests; its `RUN pytest` executes the
#                 unit suite *inside the shipped image layers* so CI validates
#                 the container itself (same Python 3.12 + same baked deps), not
#                 a separately pip-installed runner env. Never pushed.
#
#   # shipped image:
#   docker build --target runtime -t power-agent:dev -f deploy/power-agent/Dockerfile \
#       deploy/power-agent
#   # run unit tests inside the image (CI `power-agent-test` job):
#   docker build --target test -f deploy/power-agent/Dockerfile deploy/power-agent

# DOCKER_PROXY lets CI route the base image through the ECR Docker Hub mirror
# (e.g. <acct>.dkr.ecr.<region>.amazonaws.com/dockerhub/). Empty by default so
# `docker build` works unchanged for local/dev use straight from Docker Hub.
ARG DOCKER_PROXY=

# Slim base — the NVIDIA Container Toolkit injects libnvidia-ml.so at runtime
# via runtimeClassName: nvidia (set by the Helm chart), so the image does NOT
# bake pynvml's native dependency. nvmlInit() resolves the host driver lib.
#
# Keep the explicit `library/` namespace: ECR's Docker Hub pull-through cache
# does not auto-expand bare official-image names (`python` -> `library/python`)
# the way a direct Docker Hub pull does, so the proxied tag fails to pull
# without it. `library/python` is also a valid fully-qualified reference when
# DOCKER_PROXY is empty, so local/dev builds are unaffected.
FROM ${DOCKER_PROXY}library/python:3.12-slim-bookworm AS runtime

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

# ---------------------------------------------------------------------------
# test stage — CI-only. Builds FROM the shipped `runtime` image so the unit
# suite runs against the exact Python 3.12 + baked pynvml/kubernetes/
# prometheus-client versions that ship, then fails the build if any test fails.
# Not referenced by `--target runtime`, so it never lands in the pushed image.
# ---------------------------------------------------------------------------
FROM runtime AS test

RUN pip install --no-cache-dir pytest

# pytest.ini sets pythonpath=. so `from power_agent import …` resolves against
# /app (WORKDIR) where power_agent.py already lives in the runtime layer.
COPY pytest.ini /app/pytest.ini
COPY tests/ /app/tests/

RUN python -m pytest tests -v
