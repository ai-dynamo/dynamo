#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Build the DIS-2172 ROUTER-E2E instrumented dynamo image and push to a registry.
# RUN ON A COMPUTELAB BUILD HOST (computelab-build-{1..6}, docker + big NVMe).
# Get one via the compute-session MCP:  start_session(preset="amd64" | "arm64").
#
# Image = standalone Dynamo (no vLLM/CUDA engine) with the DIS-2172 receive-side
# event-plane counters (RecvCounter at the 3 real consume sites + FPM
# get_throughput_stats). Provides:
#   * python -m dynamo.frontend  (kv-mode KV Router: kv-events + active_sequences)
#   * python -m dynamo.mocker    (publishes kv-events + FPM)
#   * python -m dynamo.common.recv_forward_pass_metrics --mode throughput (FPM recv)
#   * nats-server + etcd (bundled by the dynamo Dockerfile)
#
# NOTE: the `dev` target uninstalls ai-dynamo and never builds the dynamo._core
# extension, so we rebuild the bindings + reinstall INSIDE the image (ABI-safe).
set -euo pipefail

BRANCH="${BRANCH:-zhongdaor/event_plane_router_e2e}"
REPO_URL="${REPO_URL:-https://github.com/ai-dynamo/dynamo.git}"
WORKDIR="${WORKDIR:-/tmp/dynamo-dis2172-e2e}"
IMAGE="${IMAGE:-dynamo:dis2172-e2e}"
PUSH_TO="${PUSH_TO:-}"     # e.g. gitlab-master.nvidia.com:5005/zhongdaor/workplace/dynamo:dis2172-e2e
# Host arch -> render platform. On an arm64 build host pass PLATFORM=linux/arm64.
PLATFORM="${PLATFORM:-linux/$(dpkg --print-architecture 2>/dev/null || echo amd64)}"
CUDA_VERSION="${CUDA_VERSION:-13.0}"
# On arm64, the wheel-builder base must be the aarch64 variant.
WHEEL_BUILDER_IMAGE="${WHEEL_BUILDER_IMAGE:-}"

# 1) Source on local NVMe.
cd /tmp
if [ ! -d "$WORKDIR/.git" ]; then
    GIT_TERMINAL_PROMPT=0 git clone --depth 1 --branch "$BRANCH" "$REPO_URL" "$WORKDIR"
fi
cd "$WORKDIR"
COMMIT=$(git rev-parse --short HEAD)

# 2) Render the standalone-dynamo dev Dockerfile.
RENDER_ARGS=(--framework dynamo --device cuda --cuda-version "$CUDA_VERSION"
             --target dev --platform "$PLATFORM" --output-short-filename)
python3 container/render.py "${RENDER_ARGS[@]}"

# 3) Build (BuildKit). Pass the aarch64 wheel-builder base on arm64 if provided.
BUILD_ARGS=()
if [ -n "$WHEEL_BUILDER_IMAGE" ]; then
    BUILD_ARGS+=(--build-arg "WHEEL_BUILDER_IMAGE=$WHEEL_BUILDER_IMAGE")
fi
DOCKER_BUILDKIT=1 docker build -f container/rendered.Dockerfile \
    "${BUILD_ARGS[@]}" -t "$IMAGE" .

# 4) Rebuild bindings + reinstall the python dynamo package INSIDE the image.
#    (the dev target's uninstall leaves mocker/frontend broken otherwise).
#    maturin + uv pip is the proven fix; --no-deps avoids re-resolving the world.
cat > /tmp/Dockerfile.e2e <<DOCKER
FROM $IMAGE
USER root
RUN cd /workspace/lib/bindings/python \\
 && maturin build --release --out /tmp/dynamo-wheels \\
 && uv pip install --no-deps /tmp/dynamo-wheels/ai_dynamo_runtime-*.whl \\
 && uv pip install --no-deps /workspace \\
 && python -c "from dynamo.llm import FpmEventSubscriber; print('FpmEventSubscriber OK')" \\
 && rm -rf /tmp/dynamo-wheels
DOCKER
docker build -f /tmp/Dockerfile.e2e -t "$IMAGE" "$WORKDIR"

# 5) Push.
if [ -n "$PUSH_TO" ]; then
    docker tag "$IMAGE" "$PUSH_TO"
    docker push "$PUSH_TO"
    echo "pushed $PUSH_TO"
fi

cat <<EOF

[build] DONE  (commit $COMMIT).  Local tag: $IMAGE${PUSH_TO:+   Pushed: $PUSH_TO}

In bench.sbatch set:
  export DYN_BENCH_IMAGE=${PUSH_TO:-$IMAGE}
  export DYN_BENCH_PY=/opt/dynamo/venv/bin/python
EOF
