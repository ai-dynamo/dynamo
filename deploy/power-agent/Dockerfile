# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Power Agent image — supports both NVML and DCGM actuator paths.
# Referenced by deploy/helm/charts/power-agent/values.yaml (image.repository)
# and built by the `power-agent` CI job (.github/workflows/pr.yaml,
# post-merge-ci.yml).
#
# Three stages:
#   * `dcgm-vendor` — copies pydcgm Python bindings + libdcgm.so from the
#                 upstream nvcr.io/nvidia/cloud-native/dcgm image. We vendor
#                 (rather than pip-install) for two reasons: (1) pydcgm isn't
#                 on PyPI, and (2) version pinning the Python bindings to the
#                 libdcgm.so they ship with eliminates a class of dlopen
#                 mismatch failures.
#   * `runtime` — the slim, shipped image (built/pushed with `--target runtime`).
#                 Layers the agent's Python deps + sources on a slim Python base,
#                 then copies the vendored DCGM artifacts into /opt/dcgm/.
#                 PYTHONPATH + LD_LIBRARY_PATH wire them up. The image still works
#                 in actuator: nvml mode (the chart default) — pydcgm is imported
#                 lazily by DcgmActuator methods, so an NVML deployment never opens
#                 libdcgm.so. The vendoring is dormant overhead (~12 MB image bloat)
#                 for NVML users and the only deployment path for DCGM users.
#   * `test`    — `FROM runtime` + pytest + tests; its `RUN pytest` executes the
#                 unit suite *inside the shipped image layers* so CI validates
#                 the container itself (same Python 3.12 + same baked deps), not
#                 a separately pip-installed runner env. Never pushed.
#
#   # shipped image:
#   docker build --target runtime -t power-agent:dev -f deploy/power-agent/Dockerfile \
#       deploy/power-agent
#   # run unit tests inside the image (CI `power-agent` job):
#   docker build --target test -f deploy/power-agent/Dockerfile deploy/power-agent

# DOCKER_PROXY lets CI route the base image through the ECR Docker Hub mirror
# (e.g. <acct>.dkr.ecr.<region>.amazonaws.com/dockerhub/). Empty by default so
# `docker build` works unchanged for local/dev use straight from Docker Hub.
ARG DOCKER_PROXY=

# -----------------------------------------------------------------------------
# Stage 1: extract pydcgm + libdcgm.so from the upstream DCGM image
# -----------------------------------------------------------------------------
# Pin the vendored client to a version >= the operator's nvidia-dcgm
# hostengine. NVIDIA's compatibility guidance is client >= hostengine (NOT
# merely the same major), so pin the client to the `dcgm` tag your GPU Operator
# deploys for its nvidia-dcgm hostengine (see the operator's values:
# `dcgm.version` / the nvidia-dcgm image tag) and bump this in lockstep when
# that tag moves. The default below (4.5.3) matches the nvidia-dcgm hostengine
# shipped by recent GPU Operator releases; confirm against your operator.
# Requirements:
#   - DcgmActuator uses `dcgm_structs.c_dcgmDeviceConfig_v2`, which only exists
#     in DCGM 4.x (no 3.x fallback path), so this MUST stay on a 4.x tag;
#   - keep it >= the deployed hostengine version.
# The vendor image's OS only supplies the extracted libdcgm.so.4 + pydcgm .py;
# it is independent of the runtime base (python:3.12-slim-bookworm). The
# ubuntu24.04 libdcgm.so.4 loads cleanly on bookworm (glibc 2.36) — verified via
# `ctypes.CDLL` + `import pydcgm` inside the shipped runtime image.
# NGC publishes newer 4.x tags over time (e.g. 4.6.0-1-ubuntu24.04); override
# DCGM_IMAGE to match your operator. Validate any new tag resolves before merge:
#   docker manifest inspect nvcr.io/nvidia/cloud-native/dcgm:<tag>
ARG DCGM_IMAGE=nvcr.io/nvidia/cloud-native/dcgm:4.5.3-1-ubuntu24.04
FROM ${DCGM_IMAGE} AS dcgm-vendor

# pydcgm bindings location changed between DCGM majors:
#   3.x       → /usr/local/dcgm/bindings/python3/
#   4.5+      → /usr/share/datacenter-gpu-manager-4/bindings/python3/
# We target 4.5+ because that's what the DCGM_IMAGE default resolves to.
# If you override DCGM_IMAGE to a 3.x tag, this COPY will fail loudly
# (the *.py glob expands to nothing under `set -eux`, exit 1), which is
# the correct failure mode — DcgmActuator's c_dcgmDeviceConfig_v2 usage
# would crash mid-reconcile on 3.x bindings anyway.
RUN set -eux; \
    mkdir -p /vendor/dcgm-python /vendor/dcgm-lib; \
    cp -a /usr/share/datacenter-gpu-manager-4/bindings/python3/*.py \
        /vendor/dcgm-python/; \
    cp -a /usr/lib/x86_64-linux-gnu/libdcgm.so* /vendor/dcgm-lib/

# -----------------------------------------------------------------------------
# Stage 2: runtime image
# -----------------------------------------------------------------------------
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
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONPATH=/opt/dcgm/python \
    LD_LIBRARY_PATH=/opt/dcgm/lib

# Runtime deps. pynvml (NVML bindings), kubernetes (pod listing),
# prometheus_client (metrics). pydcgm bindings are vendored below.
RUN pip install --no-cache-dir \
    pynvml==12.0.* \
    kubernetes==30.* \
    prometheus-client==0.20.*

COPY --from=dcgm-vendor /vendor/dcgm-python /opt/dcgm/python
COPY --from=dcgm-vendor /vendor/dcgm-lib    /opt/dcgm/lib

# Logger shim: the upstream NGC DCGM runtime image's pydcgm bindings
# (`DcgmGroup.py:20`) `import logger` and call `logger.debug(...)` /
# `logger.warning(...)`. The `logger.py` module that satisfies those
# imports lives in DCGM's source tree under
# `datacenter-gpu-manager-4/testing/python3/logger.py` and is NOT
# packaged into the runtime image. Without this shim every DcgmGroup
# construction fails with `ModuleNotFoundError: No module named 'logger'`.
# `deploy/power-agent/logger.py` is a 10-line stdlib-logging adapter.
# Tracked in ai-dynamo/dynamo#11192 — drop this shim once the upstream NGC
# DCGM runtime image packages `logger.py` alongside its pydcgm bindings.
COPY logger.py /opt/dcgm/python/logger.py

WORKDIR /app
# managed_state.py holds the single source of truth for managed-GPU state.
# It MUST ship alongside power_agent.py / actuator.py — power_agent.py does
# `import managed_state` at module load, so a missing file fails loudly at
# startup with `ModuleNotFoundError: No module named 'managed_state'`.
COPY power_agent.py actuator.py managed_state.py /app/

USER 0

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
