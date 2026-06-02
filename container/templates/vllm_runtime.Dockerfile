{#
# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#}
# === BEGIN templates/vllm_runtime.Dockerfile ===
##################################
########## Runtime Image #########
##################################

{% if platform == "multi" %}
FROM --platform=linux/amd64 ${RUNTIME_IMAGE}:${RUNTIME_IMAGE_TAG} AS vllm_runtime_amd64
FROM --platform=linux/arm64 ${RUNTIME_IMAGE}:${RUNTIME_IMAGE_TAG} AS vllm_runtime_arm64
FROM vllm_runtime_${TARGETARCH} AS runtime
{% else %}
FROM ${RUNTIME_IMAGE}:${RUNTIME_IMAGE_TAG} AS runtime
{% endif %}

ARG PYTHON_VERSION
ARG ENABLE_KVBM
ARG ENABLE_GPU_MEMORY_SERVICE
ARG VLLM_OMNI_REF

WORKDIR /workspace

ENV DYNAMO_HOME=/opt/dynamo
ENV HOME=/home/dynamo
ENV PATH=/usr/local/bin/etcd:${PATH}

{% if device == "cpu" %}
ENV VIRTUAL_ENV=/opt/dynamo/venv
ENV PATH="${VIRTUAL_ENV}/bin:${PATH}"
{% endif %}

# Install NATS and ETCD
COPY --from=dynamo_base /usr/bin/nats-server /usr/bin/nats-server
COPY --from=dynamo_base /usr/local/bin/etcd/ /usr/local/bin/etcd/
COPY --from=dynamo_base /bin/uv /bin/uvx /bin/

# Create dynamo user with group 0 for OpenShift compatibility
RUN userdel -r ubuntu > /dev/null 2>&1 || true \
    && useradd -m -s /bin/bash -g 0 dynamo \
    && [ `id -u dynamo` -eq 1000 ] \
    && mkdir -p /home/dynamo/.cache /opt/dynamo \
    && ln -sf /usr/bin/python3 /usr/local/bin/python \
    && chown dynamo:0 /home/dynamo /home/dynamo/.cache /opt/dynamo /workspace \
    && mkdir -p /etc/profile.d \
    && echo 'umask 002' > /etc/profile.d/00-umask.sh

# Copy attribution files and wheels
COPY --chmod=664 --chown=dynamo:0 ATTRIBUTION* LICENSE /workspace/
COPY --chmod=775 --chown=dynamo:0 --from=wheel_builder /opt/dynamo/dist/*.whl /opt/dynamo/wheelhouse/

{% if target not in ("dev", "local-dev") %}
# Keep the upstream Python solve intact: install only Dynamo-owned wheels and
# suppress transitive dependency resolution unless a later validation proves a
# missing package must be added explicitly.
RUN --mount=type=cache,target=/root/.cache/uv,sharing=locked \
    export UV_CACHE_DIR=/root/.cache/uv && \
    uv pip install --system --no-deps /opt/dynamo/wheelhouse/ai_dynamo_runtime*.whl && \
    uv pip install --system --no-deps /opt/dynamo/wheelhouse/ai_dynamo*any.whl && \
    if [ "${ENABLE_KVBM}" = "true" ]; then \
        KVBM_WHEEL=$(ls /opt/dynamo/wheelhouse/kvbm*.whl 2>/dev/null | head -1); \
        if [ -n "$KVBM_WHEEL" ]; then uv pip install --system --no-deps "$KVBM_WHEEL"; fi; \
    fi && \
    if [ "${ENABLE_GPU_MEMORY_SERVICE}" = "true" ]; then \
        GMS_WHEEL=$(ls /opt/dynamo/wheelhouse/gpu_memory_service*.whl 2>/dev/null | head -1); \
        if [ -n "$GMS_WHEEL" ]; then uv pip install --system --no-deps "$GMS_WHEEL"; fi; \
    fi

{% if device == "cuda" %}
# The launch script examples use jq for readable curl output just like the
# upstream omni image does.
#
# NOTE: vLLM-Omni no longer shells out to the GPL SoX binary — its audio
# normalization is a pure-numpy peak_normalize() (vllm_omni/utils/audio.py), so
# sox / libsox-fmt-all (and their GPL/UNKNOWN codec deps) are intentionally not
# installed here.
RUN set -eux; \
    apt-get update; \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        jq; \
    rm -rf /var/lib/apt/lists/*

# Layer the released vLLM-Omni package matching the pinned upstream ref while
# constraining packages already solved in the upstream vLLM image.
RUN --mount=type=bind,source=./container/deps/vllm/protected_packages.txt,target=/tmp/vllm_omni_protected_packages.txt \
    --mount=type=bind,source=./container/deps/vllm/install_vllm_omni.sh,target=/tmp/install_vllm_omni.sh \
    --mount=type=cache,target=/root/.cache/uv,sharing=locked \
    set -eux; \
    export UV_CACHE_DIR=/root/.cache/uv; \
    bash /tmp/install_vllm_omni.sh

RUN set -eux; \
    adapter="$(find /usr/local/lib /usr/lib /opt -type f -path '*/lmcache/integration/vllm/vllm_v1_adapter.py' -print -quit 2>/dev/null)"; \
    test -n "${adapter}"; \
    printf '%s\n' \
        '/^        if request.status == RequestStatus.FINISHED_ABORTED:$/,/^        params = ($/c\' \
        '        if request.status == RequestStatus.FINISHED_ABORTED:\' \
        '            if self.lmcache_engine is None:\' \
        '                logger.warning(\' \
        '                    "Skipping abort-time backend cleanup for request %s: "\' \
        '                    "lmcache_engine is not initialized.",\' \
        '                    request.request_id,\' \
        '                )\' \
        '            else:\' \
        '                sm = self.lmcache_engine.storage_manager\' \
        '                if sm is not None:\' \
        '                    sm.cancel_request(request.request_id)\' \
        '\' \
        '            if self.async_loading:\' \
        '                lookup_id = request.request_id\' \
        '                if self.lookup_client is None:\' \
        '                    logger.warning(\' \
        '                        "Skipping abort-time async lookup cancel for request %s: "\' \
        '                        "lookup_client is not initialized while async_loading is enabled.",\' \
        '                        request.request_id,\' \
        '                    )\' \
        '                else:\' \
        '                    self.lookup_client.cancel_lookup(lookup_id)  # type: ignore[attr-defined]\' \
        '\' \
        '        params = (' \
        > /tmp/lmcache-request-finished.sed; \
    sed -i -f /tmp/lmcache-request-finished.sed "${adapter}"; \
    rm /tmp/lmcache-request-finished.sed; \
    grep -q 'Skipping abort-time backend cleanup' "${adapter}"; \
    grep -q 'Skipping abort-time async lookup cancel' "${adapter}"

{% endif %}
{% endif %}

# The upstream vllm/vllm-openai base image ships a GPL/GPL-3.0 ffmpeg built
# against libx264/libx265/libmp3lame. Purge that entire apt codec stack and
# replace it with the LGPL-only in-tree ffmpeg built in wheel_builder
# (--disable-gpl --disable-nonfree; H.264 via NVENC, VP9 via libvpx). PyAV,
# torchaudio, torchvision, soundfile and Pillow all bundle their own libraries
# and do not link the system ffmpeg/codecs, so removing them is safe. dpkg-query
# keeps the purge robust across base-image/arch version suffixes (e.g.
# libavcodec58 vs 60), and autoremove then sweeps the now-orphaned media deps.
#
# CRITICAL: the base image marks the CUDA math libs (libcublas/libcusolver/
# libcusparse) auto-installed, and the torch wheels here ship NO bundled cublas
# — torch loads the system copies. A bare autoremove would delete them and break
# GPU inference, so pin every CUDA/NVIDIA lib as manually-installed first.
RUN set -eux; \
    keep=$(dpkg-query -W -f='${Package}\n' 2>/dev/null \
        | grep -E '^(libcu|libnv|libnccl|cuda)' || true); \
    if [ -n "$keep" ]; then apt-mark manual $keep >/dev/null; fi; \
    purge=$(dpkg-query -W -f='${Package}\n' 2>/dev/null \
        | grep -E '^(ffmpeg|libav[a-z]|libsw[a-z]|libpostproc|libx264|libx265|libmp3lame|libaom|libdav1d|libvpx|libtheora|libvorbis|libopus|libsoxr)' \
        || true); \
    if [ -n "$purge" ]; then \
        DEBIAN_FRONTEND=noninteractive apt-get purge -y $purge; \
    fi; \
    DEBIAN_FRONTEND=noninteractive apt-get autoremove -y --purge; \
    rm -rf /var/lib/apt/lists/*

# Copy the LGPL ffmpeg from wheel_builder: versioned shared libs (libav*.so*,
# libsw*.so*) plus the LGPL CLI binary that imageio/diffusers target via
# IMAGEIO_FFMPEG_EXE for video encoding. Ungated by enable_media_ffmpeg because
# the base GPL ffmpeg was just purged, so the LGPL CLI must always be present
# for the omni video-export path to have something to encode with.
RUN --mount=type=bind,from=wheel_builder,source=/usr/local/,target=/tmp/usr/local/ \
    mkdir -p /usr/local/lib/pkgconfig && \
    cp -rnL /tmp/usr/local/include/libav* /tmp/usr/local/include/libsw* /usr/local/include/ && \
    cp -nL /tmp/usr/local/lib/libav*.so* /tmp/usr/local/lib/libsw*.so* /usr/local/lib/ && \
    cp -nL /tmp/usr/local/lib/lib*vpx*.so* /usr/local/lib/ 2>/dev/null || true && \
    cp -nL /tmp/usr/local/lib/pkgconfig/libav*.pc /tmp/usr/local/lib/pkgconfig/libsw*.pc /usr/local/lib/pkgconfig/ && \
    cp -nL /tmp/usr/local/bin/ffmpeg /usr/local/bin/ffmpeg && \
    cp -r /tmp/usr/local/src/ffmpeg /usr/local/src/ && \
    ldconfig
ENV IMAGEIO_FFMPEG_EXE=/usr/local/bin/ffmpeg

# Replace the upstream vllm/vllm-openai image's imageio-ffmpeg (which ships a
# GPL-encumbered prebuilt ffmpeg binary in <site-packages>/imageio_ffmpeg/binaries/)
# with a source install that leaves no binary on disk. IMAGEIO_FFMPEG_EXE (set
# above) points imageio at the LGPL CLI copied from wheel_builder. The
# --no-binary directive lives in the requirements file itself.
RUN --mount=type=bind,source=./container/deps/requirements.vllm.txt,target=/tmp/requirements.vllm.txt \
    --mount=type=cache,target=/root/.cache/uv,sharing=locked \
    export UV_CACHE_DIR=/root/.cache/uv && \
    uv pip install --system --reinstall-package imageio-ffmpeg --no-deps \
        --requirement /tmp/requirements.vllm.txt

USER dynamo

# Copy the workspace surface needed by the current vLLM pre-merge test image.
# Keep optional framework trees like planner out of /workspace so the upstream
# runtime does not look like a fully-expanded generic image.
COPY --chmod=775 --chown=dynamo:0 tests /workspace/tests
COPY --chmod=775 --chown=dynamo:0 examples /workspace/examples
COPY --chmod=775 --chown=dynamo:0 dev /workspace/dev
COPY --chmod=775 --chown=dynamo:0 components/src/dynamo/common /workspace/components/src/dynamo/common
COPY --chmod=775 --chown=dynamo:0 components/src/dynamo/frontend /workspace/components/src/dynamo/frontend
COPY --chmod=775 --chown=dynamo:0 components/src/dynamo/vllm /workspace/components/src/dynamo/vllm
COPY --chown=dynamo:0 lib /workspace/lib

# Setup launch banner in common directory accessible to all users
USER root
RUN --mount=type=bind,source=./container/launch_message/runtime.txt,target=/opt/dynamo/launch_message.txt \
    sed '/^#\s/d' /opt/dynamo/launch_message.txt > /opt/dynamo/.launch_screen && \
    chmod 755 /opt/dynamo/.launch_screen && \
    echo 'cat /opt/dynamo/.launch_screen' >> /etc/bash.bashrc

USER dynamo

ARG DYNAMO_COMMIT_SHA
ENV DYNAMO_COMMIT_SHA=${DYNAMO_COMMIT_SHA}

# Reset the upstream "vllm serve" entrypoint so the derived runtime behaves
# like other Dynamo images and can execute arbitrary commands directly.
ENTRYPOINT []
