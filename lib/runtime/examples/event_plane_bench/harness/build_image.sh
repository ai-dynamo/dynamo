#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Build the DIS-2172 STOCK-TRANSPORT dynamo image and push to a registry.
# RUN ON A COMPUTELAB BUILD HOST (computelab-build-{1..6} x86_64, or
# computelab-armbuild-{1,2,4} arm64; docker + local NVMe). Get one via the
# compute-session MCP:  start_session(preset="amd64"|"arm64")
#
# Image = standalone Dynamo (no vLLM/CUDA engine) on Ubuntu 24.04 / glibc 2.39:
#   * ai-dynamo wheel  -> `python -m dynamo.mocker` + `python -m dynamo.frontend`
#   * STOCK event-plane transport (NO published_at_ns instrumentation) — built
#     from a clean branch off main, so the publish hot path is un-perturbed.
#   * nats-server v2.10.28 + etcd v3.5.30 (bundled by the dynamo Dockerfile)
#   * Rust/cargo toolchain + full source (dev target) -> builds event_plane_bench_sub
#
# VERIFIED RECIPE (2026-06-05, computelab-build-5). The clean MCP path
#   build_image(session, framework=VLLM, target=dev, push=true)
# is BLOCKED until a scratch volume is provisioned for the user (it rsyncs to
# {scratch_root}/worktrees/... and /home/scratch.<user>_* does not exist).
# $HOME is a 5G NFS quota -> too small. So we build on the host's local NVMe (/tmp)
# from a fresh shallow clone of the (public) PR branch instead.
set -euo pipefail

BRANCH="${BRANCH:-zhongdaor/event_plane_dynamic}"
WORKDIR="${WORKDIR:-/tmp/dynamo-dis2172-dynamic}"  # on the build host's local NVMe
IMAGE="${IMAGE:-dynamo:transport-dynamic-dev}"     # local tag; retag/push below
# Push to the INTERNAL GITLAB registry ONLY (do NOT use nvcr.io).
# e.g. gitlab-master.nvidia.com:5005/<your-namespace>/dynamo:dis2172-dev
PUSH_TO="${PUSH_TO:-}"

# 1) Source on local NVMe (public repo -> no auth; shallow single-branch).
cd /tmp
[ -d "$WORKDIR/.git" ] || GIT_TERMINAL_PROMPT=0 git clone --depth 1 --branch "$BRANCH" \
    https://github.com/ai-dynamo/dynamo.git "$WORKDIR"
cd "$WORKDIR"
COMMIT=$(git rev-parse --short HEAD)

# 2) Render the standalone-dynamo dev Dockerfile. NOTE: framework=dynamo only
#    permits device=cuda (no pure-CPU standalone target) -> CUDA base, but no
#    vLLM/torch engine. CUDA libs are unused by the CPU mocker; no GPU needed to
#    build OR run. Ubuntu 24.04 base (glibc 2.39) matches computelab compute nodes.
python3 container/render.py --framework dynamo --device cuda --cuda-version 13.0 \
    --target dev --output-short-filename

# 3) Build (BuildKit). ~20-40 min cold; layer-cached on a pinned host after.
#    The rendered Dockerfile hardcodes WHEEL_BUILDER_IMAGE=...manylinux_2_28_x86_64;
#    on an arm64 build host that stage dies with "exec format error". Override it to
#    match the host arch (manylinux_2_28_{x86_64,aarch64} both exist) so this recipe
#    is arch-agnostic — needed for the lego-c2 (ARM/Grace) arm64 image.
WHEEL_BUILDER_IMAGE="quay.io/pypa/manylinux_2_28_$(uname -m)"
DOCKER_BUILDKIT=1 docker build -f container/rendered.Dockerfile \
  --build-arg WHEEL_BUILDER_IMAGE="$WHEEL_BUILDER_IMAGE" -t "$IMAGE" .

# 4) Restore the python `dynamo` package AND add event_plane_bench_sub — both built
#    INSIDE the image (ABI-consistent). NOTE: the `dev` target deliberately runs
#    `uv pip uninstall ai-dynamo ai-dynamo-runtime kvbm` and never builds the
#    `dynamo._core` extension, so `python -m dynamo.mocker/.frontend` is BROKEN out of
#    the box. Rebuilding the bindings + reinstalling restores mocker/frontend. The image
#    presets VIRTUAL_ENV=/opt/dynamo/venv and CARGO_TARGET_DIR=/workspace/target.
cat > /tmp/Dockerfile.bench <<DOCKER
FROM $IMAGE
USER root
RUN cd /workspace/lib/bindings/python \\
 && maturin build --release --out /tmp/dynamo-wheels \\
 && uv pip install --no-deps /tmp/dynamo-wheels/ai_dynamo_runtime-*.whl \\
 && uv pip install --no-deps /workspace \\
 && rm -rf /tmp/dynamo-wheels
RUN cargo build --release \\
      --manifest-path /workspace/lib/runtime/examples/event_plane_bench/Cargo.toml -p event_plane_bench \\
 && BIN="\$(find /workspace/target -type f -name event_plane_bench_sub -path '*/release/*' | head -1)" \\
 && test -n "\$BIN" && install -m755 "\$BIN" /usr/local/bin/event_plane_bench_sub
DOCKER
docker build -f /tmp/Dockerfile.bench -t "$IMAGE" "$WORKDIR"

# 5) Push so multi-node pyxis can pull (registry creds must be configured).
if [ -n "$PUSH_TO" ]; then
    docker tag "$IMAGE" "$PUSH_TO"
    docker push "$PUSH_TO"
    echo "pushed $PUSH_TO"
fi

cat <<EOF

[build] DONE  (commit $COMMIT).  Local tag: $IMAGE${PUSH_TO:+   Pushed: $PUSH_TO}

In bench.sbatch set:
  export DYN_BENCH_IMAGE=${PUSH_TO:-$IMAGE}
  export DYN_BENCH_PY=/opt/dynamo/venv/bin/python          # in-container python
  export DYN_BENCH_SUB_BIN=/usr/local/bin/event_plane_bench_sub
EOF
