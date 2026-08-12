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
FROM vllm_runtime_${TARGETARCH} AS pre_runtime
{% else %}
FROM ${RUNTIME_IMAGE}:${RUNTIME_IMAGE_TAG} AS pre_runtime
{% endif %}

ARG PYTHON_VERSION
ARG ENABLE_KVBM
ARG ENABLE_GPU_MEMORY_SERVICE
ARG VLLM_OMNI_REF
ARG NIXL_REF
{% if device == "cuda" %}
ARG CUDA_MAJOR
{% endif %}
ARG MODELEXPRESS_VERSION

WORKDIR /workspace

ENV DYNAMO_HOME=/opt/dynamo
ENV HOME=/home/dynamo
{% if device != "cuda" %}
ENV PATH=/usr/local/ucx/bin:/usr/local/bin/etcd:${PATH}
{% else %}
ENV PATH=/usr/local/bin/etcd:${PATH}
{% endif %}

{% if device != "cuda" %}
ARG SITE_PACKAGES=/usr/local/lib/python${PYTHON_VERSION}/dist-packages
ENV TORCH_LIB_DIR=${SITE_PACKAGES}/torch/lib
{% if device == "xpu" %}
ENV NIXL_PREFIX=/opt/intel/intel_nixl
ENV NIXL_LIB_DIR=${NIXL_PREFIX}/lib/x86_64-linux-gnu
# oneAPI env for XPU detection: the base bakes none of it, so device_count() is 0
# without this. ENV not setvars.sh in ENTRYPOINT, which a k8s `command:` discards.
ENV ONEAPI_ROOT=/opt/intel/oneapi
ENV CMPLR_ROOT=/opt/intel/oneapi/compiler/2025.3
ENV LD_LIBRARY_PATH=/opt/intel/oneapi/umf/1.0/lib:/opt/intel/oneapi/tcm/1.4/lib:/opt/intel/oneapi/tbb/2022.3/lib:/opt/intel/oneapi/mkl/2025.3/lib:/opt/intel/oneapi/dnnl/2025.3/lib:/opt/intel/oneapi/compiler/2025.3/opt/compiler/lib:${LD_LIBRARY_PATH:-}
ENV PATH=${PATH}:/opt/intel/oneapi/compiler/2025.3/bin:/opt/intel/oneapi/mpi/2021.15/bin
{% elif device == "cpu" %}
ENV NIXL_PREFIX=/opt/nvidia/nvda_nixl
ENV NIXL_LIB_DIR=${NIXL_PREFIX}/lib/x86_64-linux-gnu
{% endif %}
ENV NIXL_PLUGIN_DIR=${NIXL_LIB_DIR}/plugins
ENV LD_LIBRARY_PATH=${NIXL_LIB_DIR}:${NIXL_PLUGIN_DIR}:/usr/local/ucx/lib:/usr/local/ucx/lib/ucx:${TORCH_LIB_DIR}:${LD_LIBRARY_PATH:-}
ENV VIRTUAL_ENV=/opt/venv
ENV PATH="${VIRTUAL_ENV}/bin:${PATH}"
{% else %}
# Expose libnixl.so from the upstream nixl-cu${CUDA_MAJOR} PyPI wheel through a
# stable prefix so non-Python consumers use the same NIXL copy that Python imports.
# This keeps Rust nixl-sys dlopen("libnixl.so") from falling into stub mode in
# processes that do not import the nixl Python package first.
ARG SITE_PACKAGES=/usr/local/lib/python${PYTHON_VERSION}/dist-packages
ENV NIXL_PREFIX=/opt/dynamo/nixl \
    NIXL_LIB_DIR=/opt/dynamo/nixl \
    NIXL_PLUGIN_DIR=/opt/dynamo/nixl/plugins
COPY --chmod=755 container/deps/vllm/install_nixl_from_wheel.sh /usr/local/bin/install_nixl_from_wheel
RUN install_nixl_from_wheel \
    --cuda-major "${CUDA_MAJOR}" \
    --site-packages "${SITE_PACKAGES}" \
    --prefix "${NIXL_PREFIX}" \
    --skip-headers
ENV LD_LIBRARY_PATH=${NIXL_LIB_DIR}:${NIXL_PLUGIN_DIR}:${LD_LIBRARY_PATH:-}
{% endif %}

# Install NATS and ETCD
COPY --from=dynamo_base /usr/bin/nats-server /usr/bin/nats-server
COPY --from=dynamo_base /usr/local/bin/etcd/ /usr/local/bin/etcd/
COPY --from=dynamo_base /opt/uv/bin/uv /opt/uv/bin/uvx /opt/uv/bin/
ENV PATH=/opt/uv/bin:${PATH}

# Bring base-image OS packages up to the current patch releases published in
# the distro archives. --only-upgrade skips anything not already installed, so
# no new packages are added; versions are left unpinned so each rebuild picks
# up the newest patch level.
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends --only-upgrade \
        dirmngr \
        gnupg \
        gnupg-utils \
        gnupg2 \
        gpg \
        gpg-agent \
        gpgconf \
        gpgsm \
        gpgv \
        keyboxd \
        libssl3t64 \
        openssl && \
    rm -rf /var/lib/apt/lists/*

# Create dynamo user with group 0 for OpenShift compatibility.
# Pin -u 1000 explicitly: the vllm/vllm-openai >=0.22 image ships a `vllm` user at
# UID 2000, so after freeing 1000 (ubuntu) useradd would otherwise auto-assign the
# next-highest UID (2001) and fail the `id -u dynamo` == 1000 assertion below.
RUN userdel -r ubuntu > /dev/null 2>&1 || true \
    && useradd -u 1000 -m -s /bin/bash -g 0 dynamo \
    && [ `id -u dynamo` -eq 1000 ] \
    && mkdir -p /home/dynamo/.cache/vllm /opt/dynamo \
    && ln -sf /usr/bin/python3 /usr/local/bin/python \
    && chown dynamo:0 /home/dynamo /home/dynamo/.cache /home/dynamo/.cache/vllm /opt/dynamo /workspace \
    # Arbitrary OpenShift UIDs need to create the vLLM and Triton caches under $HOME.
    && chmod g+rwx /home/dynamo /home/dynamo/.cache /home/dynamo/.cache/vllm \
    && mkdir -p /etc/profile.d \
    && echo 'umask 002' > /etc/profile.d/00-umask.sh

# FlashInfer creates package-local cubin symlinks at runtime. Grant group 0
# write access so arbitrary OpenShift UIDs can initialize the cubin cache.
RUN SITE_PACKAGES="$(python3 -c 'import site; print(site.getsitepackages()[0])')" && \
    CUBINS_DIR="$SITE_PACKAGES/flashinfer_cubin/cubins" && \
    if [ -d "$CUBINS_DIR" ]; then \
        find "$CUBINS_DIR" -type d -exec chmod g+rwx {} + ; \
    fi

{% if device != "cuda" %}
# Copy UCX and NIXL from wheel_builder for CPU/XPU devices
# (CUDA devices use NIXL from upstream vLLM wheels)
COPY --from=wheel_builder /usr/local/ucx /usr/local/ucx
COPY --chown=dynamo:0 --from=wheel_builder ${NIXL_PREFIX} ${NIXL_PREFIX}
{% if device == "xpu" %}
# XPU NIXL uses lib/x86_64-linux-gnu; copy to NIXL_LIB_DIR to ensure lib dir is populated
COPY --chown=dynamo:0 --from=wheel_builder /opt/intel/intel_nixl/lib/x86_64-linux-gnu/. ${NIXL_LIB_DIR}/
{% endif %}
# Copy NIXL Python wheels
COPY --chown=dynamo:0 --from=wheel_builder /opt/dynamo/dist/nixl/ /opt/dynamo/wheelhouse/nixl/
COPY --chown=dynamo:0 --from=wheel_builder /workspace/nixl/build/src/bindings/python/nixl-meta/nixl-*.whl /opt/dynamo/wheelhouse/nixl/

# Install RDMA libraries required for UCX to find RDMA devices
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        libibverbs1 \
        rdma-core \
        ibverbs-utils \
        libibumad3 \
        libnuma1 \
        librdmacm1 \
        ibverbs-providers && \
    rm -rf /var/lib/apt/lists/*
{% endif %}

{% if device == "xpu" %}
ADD --checksum=sha256:f60e802b6f41350393e34b24793db888a8be514054769bd17e7a6e9c0c058b87 \
    https://github.com/intel/xpumanager/releases/download/v1.3.6/xpu-smi_1.3.6_20260206.143628.1004f6cb.u24.04_amd64.deb \
    /tmp/xpu-smi.deb

# Install xpu-smi in the runtime stage so dev/local-dev inherit it, without
# explicitly changing the Intel compute runtime stack.
RUN apt-get update && \
    if command -v xpu-smi >/dev/null 2>&1; then \
        echo "xpu-smi already present in base image, skipping install"; \
    else \
        if apt-cache show intel-gsc >/dev/null 2>&1; then \
            apt-get install -y --no-install-recommends /tmp/xpu-smi.deb; \
        else \
            echo "WARNING: intel-gsc is not available from configured apt sources; skipping xpu-smi install"; \
        fi; \
    fi && \
    rm -f /tmp/xpu-smi.deb && \
    apt-get clean && rm -rf /var/lib/apt/lists/*
{% endif %}

# Copy attribution files and wheels
COPY --chmod=664 --chown=dynamo:0 LICENSE /workspace/
COPY --chmod=775 --chown=dynamo:0 --from=wheel_builder /opt/dynamo/dist/*.whl /opt/dynamo/wheelhouse/

{% set pip_target = "--system" if device == "cuda" else "--python /opt/venv/bin/python" %}
{% if device != "cuda" %}
# NIXL meta package always tries to find a cuda-backend
# https://github.com/ai-dynamo/nixl/blob/v1.1.0/src/bindings/python/nixl-meta/nixl/__init__.py
#
# We therefore install nixl-cu* packages, and use LD_LIBRARY_PATH settings to point to our installation of nixl
# v1.1.0 nixl-cu13 has in-built RPATH point to conflicting built-in libs with symbols unsupported in non-cuda builds.
# we therefore avoid installing nixl-cu13

RUN --mount=type=cache,id=uv-root-{{ context.dynamo.uv_version }},target=/root/.cache/uv,sharing=locked \
    set -eu; \
    export UV_CACHE_DIR=/root/.cache/uv; \
    NIXL_VERSION="${NIXL_REF#v}"; \
    uv pip install \
        {{ pip_target }} --force-reinstall --no-deps \
        "nixl==${NIXL_VERSION}" \
        "nixl-cu12==${NIXL_VERSION}"
{% endif %}

# Install device-specific NIXL wheels for non-CUDA devices.
# These are custom-built in wheel_builder and required for dev builds to link against NIXL libraries.
{% if device != "cuda" %}
RUN --mount=type=cache,id=uv-root-{{ context.dynamo.uv_version }},target=/root/.cache/uv,sharing=locked \
    export UV_CACHE_DIR=/root/.cache/uv && \
    uv pip install {{ pip_target }} --no-deps /opt/dynamo/wheelhouse/nixl/nixl*.whl
{% endif %}

{% if target not in ("dev", "local-dev") %}
# Keep the upstream Python solve intact: install only Dynamo-owned wheels and
# suppress transitive dependency resolution unless a later validation proves a
# missing package must be added explicitly.

# Install Dynamo runtime wheels and optional KVBM/GMS wheels.
# Use --no-deps to prevent dependency conflicts (e.g., KVBM downgrading nixl).
RUN --mount=type=cache,id=uv-root-{{ context.dynamo.uv_version }},target=/root/.cache/uv,sharing=locked \
    export UV_CACHE_DIR=/root/.cache/uv && \
    uv pip install {{ pip_target }} --no-deps /opt/dynamo/wheelhouse/ai_dynamo_runtime*.whl && \
    uv pip install {{ pip_target }} --no-deps /opt/dynamo/wheelhouse/ai_dynamo*any.whl && \
    if [ "${ENABLE_KVBM}" = "true" ]; then \
        KVBM_WHEEL=$(ls /opt/dynamo/wheelhouse/kvbm*.whl 2>/dev/null | head -1); \
        if [ -n "$KVBM_WHEEL" ]; then uv pip install {{ pip_target }} --no-deps "$KVBM_WHEEL"; fi; \
    fi && \
    if [ "${ENABLE_GPU_MEMORY_SERVICE}" = "true" ]; then \
        GMS_WHEEL=$(ls /opt/dynamo/wheelhouse/gpu_memory_service*.whl 2>/dev/null | head -1); \
        if [ -n "$GMS_WHEEL" ]; then uv pip install {{ pip_target }} --no-deps "$GMS_WHEEL"; fi; \
    fi

# Launch-script examples use jq for readable curl output like the upstream omni
# image. SoX is intentionally NOT installed: vLLM-Omni replaced its sox audio path
# with a pure-numpy peak_normalize() (vllm_omni/utils/audio.py), pysox isn't
# installed, and nothing shells out to the sox binary — so `sox`/`libsox-fmt-all`
# were dead weight that only dragged in a GPL-2.0+ codec cluster (sox, libsox*,
# libao*, libmad0, libid3tag0, libltdl7) we'd then be redistributing. SoX is
# inherently GPL (no LGPL replacement), so the compliant fix is to not ship it.
# (sglang_runtime.Dockerfile is the reference codec-compliance pattern.)
RUN set -eux; \
    apt-get update; \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        jq; \
    rm -rf /var/lib/apt/lists/*

# Layer the released vLLM-Omni package matching the pinned upstream ref while
# constraining packages already solved in the upstream vLLM image.
RUN --mount=type=bind,source=./container/deps/vllm/protected_packages.txt,target=/tmp/vllm_omni_protected_packages.txt \
    --mount=type=bind,source=./container/deps/vllm/install_vllm_omni.sh,target=/tmp/install_vllm_omni.sh \
    --mount=type=cache,id=uv-root-{{ context.dynamo.uv_version }},target=/root/.cache/uv,sharing=locked \
    set -eux; \
    export UV_CACHE_DIR=/root/.cache/uv; \
    export VLLM_OMNI_TARGET_DEVICE={{ device }}; \
    bash /tmp/install_vllm_omni.sh

{% if device == "xpu" %}
# Remove conflicting standard triton package for XPU and reinstall triton-xpu
# This must be done after vLLM-Omni installation to ensure no dependencies re-install triton
# Reinstalling triton-xpu ensures the triton namespace is properly configured
RUN uv pip uninstall triton && \
    uv pip install --force-reinstall --no-deps triton-xpu
{% endif %}

{% if context.vllm.enable_modelexpress == "true" %}
# Install only the ModelExpress client package. --no-deps preserves the upstream
# vLLM runtime dependency stack. google-crc32c is imported eagerly by the MX
# vLLM plugin (>=0.5.0) and is not in the XPU base image, so install it
# alongside; the plugin-load guard later in this stage fails the build on any
# remaining --no-deps gap.
RUN --mount=type=cache,id=uv-root-{{ context.dynamo.uv_version }},target=/root/.cache/uv,sharing=locked \
    set -eux; \
    export UV_CACHE_DIR=/root/.cache/uv; \
    uv pip install {{ pip_target }} --no-deps \
        "modelexpress==${MODELEXPRESS_VERSION}"; \
    uv pip install {{ pip_target }} "google-crc32c>=1.5.0"
{% endif %}

{% endif %}

{% if device == "xpu" %}
RUN apt-get update && \
    apt-get install -y --no-install-recommends --fix-missing \
    #ffmpeg \
    libsndfile1 \
    libsm6 \
    libxext6 \
    libgl1 \
    lsb-release \
    numactl \
    wget \
    vim \
    linux-libc-dev && \
    apt-get clean && rm -rf /var/lib/apt/lists/*
{% endif %}

{% if device == "cuda" %}
# The upstream vllm/vllm-openai base image ships a GPL/GPL-3.0 ffmpeg built
# against libx264/libx265/libmp3lame. Purge ONLY the explicitly-named ffmpeg +
# codec packages and replace them with the LGPL-only in-tree ffmpeg built in
# wheel_builder (--disable-gpl --disable-nonfree; H.264 via NVENC, VP9 via
# libvpx). PyAV, torchaudio, torchvision, soundfile and Pillow all bundle their
# own libraries and do not link the system ffmpeg/codecs, so removing them is
# safe. dpkg-query keeps the match robust across base-image/arch version
# suffixes (e.g. libavcodec58 vs 60).
#
# This grep is the COMPLETE, auditable set of what leaves the image: there is
# deliberately NO apt-get autoremove, so the removal can never cascade into
# unrelated auto-installed packages. That matters because the base image marks
# both the gcc/g++/make toolchain (torch.inductor/Triton JIT shell out to it at
# runtime) and the CUDA math libs (libcublas/libcusolver/libcusparse — the torch
# wheels here ship no bundled cublas and load the system copies) as
# auto-installed. A bare `autoremove --purge` sweeps all of those as "orphaned",
# which broke runtime JIT (missing C compiler) in the 1.3.0 rc image. Any
# LGPL/BSD media libs left orphaned (libva, libvdpau, ...) are license-clean
# dead weight, not a compliance issue.
RUN set -eux; \
    purge=$(dpkg-query -W -f='${Package}\n' 2>/dev/null \
        | grep -E '^(ffmpeg|libav[a-z]|libsw[a-z]|libpostproc|libx264|libx265|libmp3lame|libaom|libdav1d|libvpx|libtheora|libvorbis|libopus|libsoxr|libcaca|libcdio|libzvbi|libgme|libvidstab|libdc1394|libraw1394|libiec61883|libtwolame|libshine|libsrt[0-9]|libudfread|libsvtav1|libbs2b|librubberband|libchromaprint|libcodec2|libgsm|libass[0-9]|libbluray|libxvidcore|libflite)' \
        || true); \
    if [ -n "$purge" ]; then \
        DEBIAN_FRONTEND=noninteractive apt-get purge -y $purge; \
    fi; \
    rm -rf /var/lib/apt/lists/*

# Regression guard for the codec purge above: torch.inductor/Triton JIT shell
# out to a host C/C++ compiler at runtime, so a missing toolchain only surfaces
# on the first compile in production. Reproduce that compile path at build time
# (CPU-only) so a missing compiler aborts the build instead of shipping.
RUN --mount=type=bind,source=./container/deps/vllm/validate_torch_compile_smoke.py,target=/tmp/validate_torch_compile_smoke.py,readonly \
    python3 /tmp/validate_torch_compile_smoke.py

# Copy the LGPL ffmpeg from wheel_builder: versioned shared libs (libav*.so*,
# libsw*.so*) + libvpx + the LGPL CLI binary that imageio/diffusers target via
# IMAGEIO_FFMPEG_EXE. Ungated by enable_media_ffmpeg because the base GPL ffmpeg
# was just purged, so the LGPL CLI must always be present for the omni
# video-export path to have something to encode with.
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

# Positive codec guard: the shipped ffmpeg MUST expose the VP9 encoder and MUST
# NOT expose any H.264/H.265/AAC/NVENC encoder. A missing/broken copy (no VP9)
# or a codec regression fails the build here rather than at runtime — closing the
# gap where an image with no working encoder passed every PR gate.
RUN set -eu; \
    ff="${IMAGEIO_FFMPEG_EXE:-ffmpeg}"; \
    "$ff" -hide_banner -encoders 2>/dev/null | grep -qiE 'libvpx[-_]vp9' \
      || { echo "ERROR: shipped ffmpeg ($ff) has no VP9 encoder" >&2; exit 1; }; \
    if "$ff" -hide_banner -encoders 2>/dev/null \
         | grep -iE 'h\.?264|h\.?265|hevc|(^| )aac|nvenc|cuvid|nvdec'; then \
        echo "ERROR: shipped ffmpeg ($ff) exposes an H.264/H.265/AAC/NVENC encoder" >&2; \
        exit 1; \
    fi
{% endif %}

# Replace the upstream vllm/vllm-openai image's imageio-ffmpeg (which ships a
# GPL-encumbered prebuilt ffmpeg binary in <site-packages>/imageio_ffmpeg/binaries/)
# with a source install that leaves no binary on disk. On cuda, IMAGEIO_FFMPEG_EXE
# (set above) points imageio at the LGPL CLI copied from wheel_builder. The
# --no-binary directive lives in the requirements file itself.
{% if device == "cuda" %}
RUN --mount=type=bind,source=./container/deps/requirements.vllm.txt,target=/tmp/requirements.vllm.txt \
    --mount=type=cache,id=uv-root-{{ context.dynamo.uv_version }},target=/root/.cache/uv,sharing=locked \
    export UV_CACHE_DIR=/root/.cache/uv && \
    uv pip install {{ pip_target }} \
        --reinstall-package imageio-ffmpeg --reinstall-package PyNvVideoCodec \
        --no-deps --requirement /tmp/requirements.vllm.txt
{% else %}
# PyNvVideoCodec decodes on NVDEC through libnvcuvid, so it is inert on a
# non-NVIDIA device. Drop it from the shared requirements rather than ship an
# unusable NVIDIA codec wheel in, for example, the Intel XPU image. The pattern
# is anchored to the line start so it cannot match inside another requirement,
# and the import check fails the build if the package arrives by another route
# -- a filter that silently stopped matching would otherwise look like success.
#
# Whole-RUN branches, rather than a conditional inside one RUN: a `{% raw %}{% if %}{% endraw %}` in the
# middle of a `\`-continued command emits a blank line that ends the command
# early, and a `#` comment there is joined onto the previous line, commenting
# out the rest. Both break the build in ways the template does not show.
RUN --mount=type=bind,source=./container/deps/requirements.vllm.txt,target=/tmp/requirements.vllm.txt \
    --mount=type=cache,id=uv-root-{{ context.dynamo.uv_version }},target=/root/.cache/uv,sharing=locked \
    export UV_CACHE_DIR=/root/.cache/uv && \
    grep -v '^PyNvVideoCodec' /tmp/requirements.vllm.txt > /tmp/requirements.vllm.nonvidia.txt && \
    uv pip install {{ pip_target }} --reinstall-package imageio-ffmpeg --no-deps \
        --requirement /tmp/requirements.vllm.nonvidia.txt && \
    ! /opt/venv/bin/python -c "import PyNvVideoCodec" 2>/dev/null
{% endif %}

# Remove the vLLM source tree shipped in the base image to avoid pytest
# collection conflicts (duplicate conftest plugin registration) and stale
# tool scripts referencing files not present in Dynamo's build context.
RUN rm -rf /workspace/vllm

{% if device == "cuda" %}
# Transitional: this exists only until the base image stops shipping these.
# The permanent statement of what may ship is
# tests/dependencies/test_no_software_video_codecs.py -- when this purge
# becomes redundant, delete it and keep those assertions.
# Remove the codec-bearing video-DECODE wheels inherited from the vllm-openai
# base. Each bundles its own full ffmpeg carrying software H.264/H.265/AAC;
# PyAV and decord additionally ship GPL libx264/libx265. Dynamo's vLLM component
# imports none of the removed wheels, so they are unused decode-side dead weight.
# (PyNvVideoCodec is KEPT for NVDEC hardware decode -- see the note below.) The in-tree
# LGPL ffmpeg + imageio-ffmpeg installed above are intentionally KEPT for the
# omni video-encode path, which uses the royalty-free VP9 (libvpx_vp9) encoder —
# no H.264 is built. Direct rm makes the removal robust regardless of how the
# base image's pip is configured; the guards fail the build if any of them survive.
RUN set -eux; \
    python3 -m pip uninstall --yes \
        av decord decord2 opencv-python opencv-python-headless torchcodec \
        || true; \
    SITE_PACKAGES="$(python3 -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"; \
    rm -rf \
        "${SITE_PACKAGES}"/av "${SITE_PACKAGES}"/av-*.dist-info "${SITE_PACKAGES}"/av.libs \
        "${SITE_PACKAGES}"/cv2 "${SITE_PACKAGES}"/opencv_python*.dist-info "${SITE_PACKAGES}"/opencv_python*.libs \
        "${SITE_PACKAGES}"/decord "${SITE_PACKAGES}"/decord-*.dist-info "${SITE_PACKAGES}"/decord.libs \
        "${SITE_PACKAGES}"/decord2 "${SITE_PACKAGES}"/decord2-*.dist-info "${SITE_PACKAGES}"/decord2.libs \
        "${SITE_PACKAGES}"/torchcodec "${SITE_PACKAGES}"/torchcodec-*.dist-info \
        /root/.cache/pip; \
    # Guard EVERY purged wheel: the uninstall above is `|| true`, so a package
    # that survived (rename, new bundling) would otherwise pass silently. decord2
    # imports as `decord`, so both are covered by the one check.
    ! python3 -c "import cv2" 2>/dev/null; \
    ! python3 -c "import av" 2>/dev/null; \
    ! python3 -c "import decord" 2>/dev/null; \
    ! python3 -c "import torchcodec" 2>/dev/null

# PyNvVideoCodec is KEPT (removed from the purge above) but UPGRADED to >=2.2.0 by
# the requirements install: the base image's 2.0.4 bundles a full FFmpeg (incl.
# libavcodec) that the codec gate rejects, while 2.2.0 bundles only libavutil +
# libavformat (container demux, no software codec). It provides the built-in
# H.264/H.265 video-input path (NVDEC hardware decode via libnvcuvid;
# common/multimodal/nvdec_decoder.py) and requires the driver "video" capability
# at runtime or it cannot import; set it in the image and ensure the K8s
# pod/runtimeClass does not drop it.
ENV NVIDIA_DRIVER_CAPABILITIES=video,compute,utility
{% endif %}

{% if target not in ("dev", "local-dev") and context.vllm.enable_modelexpress == "true" %}
# Regression guard for the --no-deps ModelExpress install above: resolve and
# invoke the vllm.general_plugins entry points exactly as vLLM does at every
# startup, so a missing transitive dependency fails the build here instead of
# at pod startup. Runs after every package/library install in this stage
# (including the XPU apt step and the cuda codec purge above) so the check is
# order-independent and sees the final image state.
RUN python3 -c "from importlib.metadata import entry_points; \
eps = [ep for ep in entry_points(group='vllm.general_plugins') if ep.name == 'modelexpress']; \
assert eps, 'modelexpress vllm.general_plugins entry point not found'; \
[ep.load()() for ep in eps]"
{% endif %}

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


{# Compliance is skipped for dev/local-dev: those images are not shipped (release
   ships runtime/frontend/operator/planner/snapshot-agent), compliance-extract
   already skips them, and their pre_runtime carries no dynamo venv to scan. #}
{% if target not in ("dev", "local-dev") %}
{% include "templates/compliance.Dockerfile" %}
{% endif %}


FROM pre_runtime AS runtime
{% if target not in ("dev", "local-dev") %}
COPY --from=licenses /legal /legal
{% endif %}
