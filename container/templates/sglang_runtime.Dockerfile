{#
# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#}
# === BEGIN templates/sglang_runtime.Dockerfile ===
##################################
########## Runtime Image #########
##################################

FROM ${RUNTIME_IMAGE}:${RUNTIME_IMAGE_TAG} AS runtime

WORKDIR /workspace

# Install NATS and ETCD
COPY --from=dynamo_base /usr/bin/nats-server /usr/bin/nats-server
COPY --from=dynamo_base /usr/local/bin/etcd/ /usr/local/bin/etcd/

ENV PATH=/usr/local/bin/etcd:$PATH

# Create dynamo user with group 0 for OpenShift compatibility
RUN userdel -r ubuntu > /dev/null 2>&1 || true \
    && useradd -m -s /bin/bash -g 0 dynamo \
    && [ `id -u dynamo` -eq 1000 ] \
    && mkdir -p /home/dynamo/.cache /opt/dynamo \
    # Non-recursive chown - only the directories themselves, not contents
    && chown dynamo:0 /home/dynamo /home/dynamo/.cache /opt/dynamo /workspace \
    # No chmod needed: umask 002 handles new files, COPY --chmod handles copied content
    # Set umask globally for all subsequent RUN commands (must be done as root before USER dynamo)
    # NOTE: Setting ENV UMASK=002 does NOT work - umask is a shell builtin, not an environment variable
    && mkdir -p /etc/profile.d && echo 'umask 002' > /etc/profile.d/00-umask.sh

{% if context.sglang.enable_media_ffmpeg == "true" %}
# Copy ffmpeg
RUN --mount=type=bind,from=wheel_builder,source=/usr/local/,target=/tmp/usr/local/ \
    mkdir -p /usr/local/lib/pkgconfig && \
    cp -rnL /tmp/usr/local/include/libav* /tmp/usr/local/include/libsw* /usr/local/include/ && \
    cp -nL /tmp/usr/local/lib/libav*.so /tmp/usr/local/lib/libsw*.so /usr/local/lib/ && \
    cp -nL /tmp/usr/local/lib/pkgconfig/libav*.pc /tmp/usr/local/lib/pkgconfig/libsw*.pc /usr/local/lib/pkgconfig/ && \
    cp -r /tmp/usr/local/src/ffmpeg /usr/local/src/
{% endif %}

{% if target not in ("dev", "local-dev") %}
# Runtime target installs only the prebuilt Dynamo wheels. SGLang and its NIXL
# packages come from the upstream lmsysorg/sglang runtime image; --no-deps keeps
# pip from replacing that stack. Dev/local-dev build from source later in the
# shared dev stage after the workspace is bind-mounted.
COPY --chmod=775 --chown=dynamo:0 --from=wheel_builder /opt/dynamo/dist/*.whl /opt/dynamo/wheelhouse/

RUN --mount=type=cache,target=/root/.cache/pip,sharing=locked \
    export PIP_CACHE_DIR=/root/.cache/pip && \
    pip install --break-system-packages --no-deps \
        /opt/dynamo/wheelhouse/ai_dynamo_runtime*.whl \
        /opt/dynamo/wheelhouse/ai_dynamo*any.whl

# Install accelerate for diffusion/video worker pipelines (diffusers requires it
# for enable_model_cpu_offload but the upstream SGLang runtime image omits it)
RUN --mount=type=cache,target=/root/.cache/pip,sharing=locked \
    export PIP_CACHE_DIR=/root/.cache/pip && \
    pip install --break-system-packages --no-deps "accelerate==1.13.0"

# Install distro: openai>=1.x's _base_client imports it unconditionally, and
# sglang 0.5.11's server_args eagerly imports sglang.srt.entrypoints.openai.protocol
# which pulls in openai.types.responses → triggers openai pkg init → import distro.
# The upstream lmsysorg/sglang runtime installs openai with --no-deps so distro is
# missing; without this any dynamo.sglang worker fails to import at startup.
RUN --mount=type=cache,target=/root/.cache/pip,sharing=locked \
    export PIP_CACHE_DIR=/root/.cache/pip && \
    pip install --break-system-packages --no-deps "distro==1.9.0"

# Install gpu_memory_service wheel if enabled (all targets)
ARG ENABLE_GPU_MEMORY_SERVICE
RUN --mount=type=cache,target=/root/.cache/pip,sharing=locked \
    if [ "${ENABLE_GPU_MEMORY_SERVICE}" = "true" ]; then \
        export PIP_CACHE_DIR=/root/.cache/pip && \
        GMS_WHEEL=$(ls /opt/dynamo/wheelhouse/gpu_memory_service*.whl 2>/dev/null | head -1); \
        if [ -n "$GMS_WHEEL" ]; then pip install --no-cache-dir --break-system-packages "$GMS_WHEEL"; fi; \
    fi
{% endif %}

# Install nvtx pinned in container/deps/requirements.common.txt so DYN_NVTX=1
# profiling works in all targets (runtime, dev, local-dev) — see
# components/src/dynamo/common/utils/nvtx_utils.py. --no-deps preserves the
# upstream lmsysorg/sglang Python stack.
RUN --mount=type=bind,source=./container/deps/requirements.common.txt,target=/tmp/requirements.common.txt \
    --mount=type=cache,target=/root/.cache/pip,sharing=locked \
    export PIP_CACHE_DIR=/root/.cache/pip && \
    pip install --break-system-packages --no-deps $(grep -E '^nvtx==' /tmp/requirements.common.txt)

# Optional: pin sglang Python to a specific git ref while keeping the base
# image's sgl-kernel/FlashInfer/torch. SGLANG_REF is the commit SHA, tag, or
# branch in the sgl-project/sglang repo; empty (default) leaves the upstream
# package in place. --no-deps + --force-reinstall swaps only the sglang wheel.
# Requires git in the build image — manylinux base images bundle it; the
# upstream lmsysorg/sglang runtime does not run pip-from-git here, the install
# happens at image-build time when git is available via apt in the base.
ARG SGLANG_REF
ARG SGLANG_REPO
RUN --mount=type=cache,target=/root/.cache/pip,sharing=locked \
    if [ -n "${SGLANG_REF}" ]; then \
        export PIP_CACHE_DIR=/root/.cache/pip && \
        if ! command -v git >/dev/null 2>&1; then \
            apt-get update && apt-get install -y --no-install-recommends git && \
            rm -rf /var/lib/apt/lists/*; \
        fi && \
        pip install --break-system-packages --no-deps --force-reinstall \
            "sglang @ git+${SGLANG_REPO}@${SGLANG_REF}#subdirectory=python"; \
    fi

# Apply patches to the installed sglang.multimodal_gen runtime. Each patch
# under container/deps/sglang/patches/ is bind-mounted in and applied with
# `patch -p2` against /usr/local/lib/python3.12/dist-packages/sglang. We
# require `patch` (apt) — busybox `patch` is missing on the upstream runtime
# image. The strip prefix is p2 because each patch has paths shaped like
# a/python/sglang/...; cd-ing into the parent of `sglang/` and stripping two
# components lands at sglang/multimodal_gen/... relative to the install root.
RUN --mount=type=bind,source=./container/deps/sglang/patches,target=/tmp/sglang-patches \
    if [ -d /tmp/sglang-patches ] && [ -n "$(ls -A /tmp/sglang-patches 2>/dev/null)" ]; then \
        if ! command -v patch >/dev/null 2>&1; then \
            apt-get update && apt-get install -y --no-install-recommends patch && \
            rm -rf /var/lib/apt/lists/*; \
        fi && \
        SGLANG_INSTALL_DIR="$(python3 -c 'import sglang, pathlib; print(pathlib.Path(sglang.__file__).parent.parent)')" && \
        echo "Applying sglang patches under ${SGLANG_INSTALL_DIR}" && \
        cd "${SGLANG_INSTALL_DIR}" && \
        for p in /tmp/sglang-patches/*.patch; do \
            echo "  ${p}" && \
            patch -p2 --batch --forward < "${p}"; \
        done; \
    fi

# msgpack: required by the realtime WebSocket endpoint added in
# sgl-project/sglang#19817 (sglang.multimodal_gen.runtime.entrypoints.realtime
# .realtime_video_api). Not declared in upstream pyproject so we install it
# here. Tiny pure-C wheel — no ABI risk to the upstream stack.
RUN --mount=type=cache,target=/root/.cache/pip,sharing=locked \
    export PIP_CACHE_DIR=/root/.cache/pip && \
    pip install --break-system-packages --no-deps "msgpack>=1.0.0"

# Diffusion extras from antgroup's chunk_diffusion `[diffusion]` extra group.
# These are required by the Krea realtime video pipeline. --no-deps keeps the
# upstream torch/sglang-kernel/FlashInfer stack intact; transitive deps
# (safetensors, huggingface_hub, decorator, proglog, etc.) are already present
# in the lmsysorg/sglang runtime base image.
RUN --mount=type=cache,target=/root/.cache/pip,sharing=locked \
    export PIP_CACHE_DIR=/root/.cache/pip && \
    pip install --break-system-packages --no-deps \
        "PyYAML==6.0.1" \
        "cloudpickle==3.1.2" \
        "diffusers==0.37.0" \
        "imageio==2.36.0" \
        "imageio-ffmpeg==0.5.1" \
        "moviepy>=2.0.0" \
        "opencv-python-headless==4.10.0.84" \
        "remote-pdb==2.1.0" \
        "cache-dit==1.3.0" \
        "addict==2.4.0" \
        "av==16.1.0" \
        "scikit-image==0.25.2" \
        "trimesh>=4.0.0" \
        "xatlas" \
        "runai_model_streamer>=0.15.5"

# Native CUDA kernel extras (st_attn, vsa) from antgroup's `[diffusion]` group.
# These ship as sdists that compile against the system torch via setup.py at
# install time. --no-build-isolation makes pip reuse the installed torch
# instead of provisioning a fresh build venv (which would not pick up the
# matching CUDA-compiled torch from the base image). x86_64 only per antgroup's
# pyproject markers. If these fail to compile (missing nvcc, CUDA arch
# mismatch, etc.), the build fails — these kernels are required for Krea's
# optimized denoising path.
ARG TARGETARCH
RUN --mount=type=cache,target=/root/.cache/pip,sharing=locked \
    if [ "${TARGETARCH}" = "amd64" ]; then \
        export PIP_CACHE_DIR=/root/.cache/pip && \
        pip install --break-system-packages --no-deps --no-build-isolation \
            "st_attn==0.0.7" \
            "vsa==0.0.4"; \
    fi

# Copy tests, deploy and components for CI with correct ownership
COPY --chmod=775 --chown=dynamo:0 tests /workspace/tests
COPY --chmod=775 --chown=dynamo:0 examples /workspace/examples
COPY --chmod=775 --chown=dynamo:0 deploy /workspace/deploy
COPY --chmod=775 --chown=dynamo:0 components/src/dynamo/common /workspace/components/src/dynamo/common
COPY --chmod=775 --chown=dynamo:0 components/src/dynamo/frontend /workspace/components/src/dynamo/frontend
COPY --chmod=775 --chown=dynamo:0 components/src/dynamo/sglang /workspace/components/src/dynamo/sglang
COPY --chmod=775 --chown=dynamo:0 components/src/dynamo/mocker /workspace/components/src/dynamo/mocker
COPY --chmod=775 --chown=dynamo:0 recipes/ /workspace/recipes/
COPY --chmod=664 --chown=dynamo:0 ATTRIBUTION* LICENSE /workspace/

# Enable forceful shutdown of inflight requests
ENV SGLANG_FORCE_SHUTDOWN=1

# Setup launch banner in common directory accessible to all users
RUN --mount=type=bind,source=./container/launch_message/runtime.txt,target=/opt/dynamo/launch_message.txt \
    sed '/^#\s/d' /opt/dynamo/launch_message.txt > /opt/dynamo/.launch_screen

RUN chmod 755 /opt/dynamo/.launch_screen && \
    echo 'cat /opt/dynamo/.launch_screen' >> /etc/bash.bashrc && \
    ln -s /workspace /sgl-workspace/dynamo && \
    NSYS_BIN=$(find /opt/nvidia/nsight-compute -maxdepth 6 -type f -name nsys -executable 2>/dev/null | head -n1) && \
    if [ -n "$NSYS_BIN" ]; then ln -sf "$NSYS_BIN" /usr/local/bin/nsys; \
    else echo "WARNING: no bundled nsys found under /opt/nvidia/nsight-compute"; fi

USER dynamo
ARG DYNAMO_COMMIT_SHA
ENV DYNAMO_COMMIT_SHA=${DYNAMO_COMMIT_SHA}

ENTRYPOINT ["/opt/nvidia/nvidia_entrypoint.sh"]
CMD []
