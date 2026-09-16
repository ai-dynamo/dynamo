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
ARG TRANSFORMERS_VERSION
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
# vLLM 0.27.1's XPU image installs the oneAPI runtime and SYCL headers in
# /opt/venv through the intel-sycl-rt wheel. Do not set ONEAPI_ROOT to the
# removed /opt/intel/oneapi tree: Triton gives that variable priority over its
# wheel-metadata fallback and would search a nonexistent compiler include path.
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

{% if device == "cuda" %}
# Bring base-image OS packages up to the current patch releases published in
# the distro archives. --only-upgrade skips anything not already installed, so
# no new packages are added; versions are left unpinned so a cache-busted
# rebuild picks up the newest patch level (BuildKit reuses this layer otherwise).
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
{% endif %}

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
{% set python_executable = "python3" if device == "cuda" else "/opt/venv/bin/python" %}
{# cuda installs into the system interpreter (/usr/local/bin); xpu and cpu run out
   of ${VIRTUAL_ENV} and prepend ${VIRTUAL_ENV}/bin to PATH. #}
{% set vllm_rs_link = "/usr/local/bin/vllm-rs" if device == "cuda" else "${VIRTUAL_ENV}/bin/vllm-rs" %}
{# Inline expression, not a block tag: render.py leaves trim_blocks off, so a tag
   on its own line inside the RUN breaks the backslash continuation. #}
{% set vllm_rs_required = "1" if device == "cuda" else "0" %}
{# TODO: Remove this workaround once bundled vllm-rs accepts extra output fields. #}
{% set vllm_rs_allowlist = "1" if target not in ("dev", "local-dev") else "0" %}
{% set vllm_rs_plugins = "modelexpress" if context.vllm.enable_modelexpress == "true" else "" %}

# The vLLM 0.28.0 release images resolve the unbounded `transformers>=5.5.3`
# requirement to 5.15.1, but vLLM-Omni 0.28.0rc1 caps Transformers below 5.15.
# Omni is layered against the installed Transformers version, so install the
# compatible release first and its dependency solve sees the final Transformers
# invariant instead of resolving against 5.15.1.
RUN --mount=type=cache,id=uv-root-{{ context.dynamo.uv_version }},target=/root/.cache/uv,sharing=locked \
    export UV_CACHE_DIR=/root/.cache/uv && \
    uv pip install {{ pip_target }} --no-deps \
        "transformers==${TRANSFORMERS_VERSION}"

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
    uv pip install {{ pip_target }} --no-deps /opt/dynamo/wheelhouse/aisimulate*.whl && \
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
# libjemalloc2 lets Dynamo processes opt into jemalloc via
# LD_PRELOAD or DYN_FRONTEND_JEMALLOC; it is not preloaded by default.
RUN set -eux; \
    apt-get update; \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        jq \
        libturbojpeg \
        libjemalloc2; \
    ldconfig; \
    ldconfig -p | grep -q 'libturbojpeg.so.0'; \
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

# Resolve the same include directories Triton's XPU driver will use for its
# first-request JIT, and fail the image build if the SYCL development headers
# are not discoverable there.
RUN /opt/venv/bin/python <<'PY'
from pathlib import Path

from triton.backends.intel.driver import COMPILATION_HELPER

roots = COMPILATION_HELPER.include_dir
if not any((Path(root) / "sycl/sycl.hpp").is_file() for root in roots):
    raise RuntimeError(f"SYCL headers not found in Triton include paths: {roots}")
PY
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
{% endif %}

# Copy the LGPL ffmpeg from wheel_builder: versioned shared libs (libav*.so*,
# libsw*.so*) + libvpx + the LGPL CLI binary that imageio/diffusers target via
# IMAGEIO_FFMPEG_EXE. This remains ungated by enable_media_ffmpeg so the
# media-enabled runtime wheel and the omni video-export path always have their
# required shared libraries and CLI available.
{% if device == "cuda" or device == "xpu" %}
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

{% if device == "cuda" %}
# Delete the base image's PyNvVideoCodec before the install below replaces it.
# vllm/vllm-openai:v0.28.0-ubuntu2404 ships 2.0.4, which bundles a complete
# FFmpeg 8.1.1 -- libavcodec.so.62 plus libavdevice/libavfilter/libswscale/
# libswresample -- and the codec gate denies a wheel-vendored libavcodec on
# presence.
#
# Uninstall FIRST, then rm what is left. The order matters and the obvious
# shortcut is wrong: the wheel's RECORD is the only complete list of what it
# installed, and it reaches further than the package directory -- top-level
# `samples/` and `benchmarks/` trees (41 entries in 2.0.4) and an FFmpeg source
# tarball written outside site-packages through a
# `../../../external/ffmpeg/src/ffmpeg-<version>.tar.xz` entry. Deleting the
# dist-info first destroys that list, so a bare `rm -rf` of the package
# directory would leave those trees behind as orphans no later install owns.
# The rm is still here, after the uninstall, because it is the part that does
# not depend on the installer honouring its own metadata, and because it is what
# makes the intent legible at the point of use.
#
# `pynvvideocodec*` takes both the version-stamped dist-info and the stray
# `pynvvideocodec.` directory the wheel installs beside it (it holds a vendored
# libcudart, not a codec).
#
# Fails when the base stops shipping the package rather than quietly removing
# nothing: a no-op rm is indistinguishable from one whose paths went stale, and
# either way this block is what has to be revisited.
#
# The presence test is find_spec, not an import: PyNvVideoCodec's __init__ dlopens
# libnvidia-encode.so.1, which no builder has, so `import PyNvVideoCodec` raises
# whether or not the wheel is installed and could never fail this build -- the
# same reason the non-CUDA branch below uses find_spec.
RUN set -eu; \
    purelib="$(python3 -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"; \
    data="$(python3 -c 'import sysconfig; print(sysconfig.get_paths()["data"])')"; \
    before="$(python3 -c 'import importlib.metadata as m; print(m.version("pynvvideocodec"))' 2>/dev/null || echo none)"; \
    echo "PyNvVideoCodec in base image: ${before}"; \
    if [ "${before}" = "none" ]; then \
        echo "ERROR: the vllm-openai base no longer ships PyNvVideoCodec, so this" >&2; \
        echo "       removal has nothing to act on -- delete this RUN and keep the" >&2; \
        echo "       requirements install below, which is then the only copy." >&2; \
        exit 1; \
    fi; \
    python3 -m pip uninstall --yes pynvvideocodec; \
    rm -rf "${purelib}/PyNvVideoCodec" "${purelib}"/pynvvideocodec* "${data}/external/ffmpeg"; \
    if python3 -c 'import importlib.util, sys; sys.exit(0 if importlib.util.find_spec("PyNvVideoCodec") else 1)'; then \
        echo "ERROR: PyNvVideoCodec is still importable after the removal above; the" >&2; \
        echo "       base ships it somewhere these paths do not reach." >&2; \
        exit 1; \
    fi; \
    leftover="$(find "${data}/external" -name 'ffmpeg-*.tar.*' 2>/dev/null || true)"; \
    if [ -n "${leftover}" ]; then \
        echo "ERROR: a bundled FFmpeg source tarball survived the removal:" >&2; \
        echo "${leftover}" >&2; \
        exit 1; \
    fi
{% endif %}

# Replace the upstream vllm/vllm-openai image's imageio-ffmpeg (which ships a
# GPL-encumbered prebuilt ffmpeg binary in <site-packages>/imageio_ffmpeg/binaries/)
# with a source install that leaves no binary on disk. On cuda, IMAGEIO_FFMPEG_EXE
# (set above) points imageio at the LGPL CLI copied from wheel_builder. The
# --no-binary directive lives in the requirements file itself.
# The vllm-openai base sets UV_CACHE_DIR=/opt/uv/cache and used to bake a uv
# cache there (v0.27.1 carried archived wheel copies, including mooncake, that
# duplicated installed packages and kept stale versions on disk after floors
# refreshed them). The pinned v0.28.0 base mounts a cache over that path in
# every uv RUN and ships only the empty directory, so the rm below is a no-op
# today. It stays as a guard against a base that bakes the cache again: the
# cache would sit in an inherited layer, so removing it here does not shrink
# the pulled image, it only drops it from the final filesystem view, which is
# what the compliance scanners see.
{% if device == "cuda" %}
RUN --mount=type=bind,source=./container/deps/requirements.vllm.txt,target=/tmp/requirements.vllm.txt \
    --mount=type=cache,id=uv-root-{{ context.dynamo.uv_version }},target=/root/.cache/uv,sharing=locked \
    export UV_CACHE_DIR=/root/.cache/uv && \
    [ "$CUDA_MAJOR" = "13" ] || { echo "ERROR: requirements.vllm.txt hardcodes the mooncake-transfer-engine-cuda13 distribution; got CUDA_MAJOR=$CUDA_MAJOR" >&2; exit 1; } && \
    uv pip install {{ pip_target }} \
        --reinstall-package imageio-ffmpeg --reinstall-package PyNvVideoCodec \
        --no-deps --requirement /tmp/requirements.vllm.txt && \
    rm -rf /opt/uv/cache

# Assert what the removal and the install together left on disk. The floor in
# requirements.vllm.txt is a lower bound, so it cannot state "exactly one copy"
# or "no libavcodec" -- and those are the two properties the codec gate depends
# on. Checked here, beside the install, so a regression names its cause instead
# of surfacing as an unexplained scan violation several stages later.
#
# Reads the dist-info directories rather than importlib.metadata: a surviving
# base copy beside the new one is exactly the failure being looked for, and the
# metadata API answers with whichever one it resolves first. FLOOR is duplicated
# from requirements.vllm.txt on purpose -- this stage must not parse the file it
# is checking; tests/dependencies/test_pynvvideocodec_floor.py asserts the two
# agree, and covers the sglang and trtllm copies in the same way.
RUN python3 - <<'PYEOF'
import csv
import glob
import os
import re
import sys
from importlib.metadata import distributions

FLOOR = "2.2.3"
NAME = "pynvvideocodec"
# Mirrors the deny globs in container/compliance/policy/codec_policy.yaml. Kept as
# families rather than the four this package happens to have shed, so a future
# release vendoring libpostproc or libx264 is caught here -- beside the install
# that introduced it -- instead of as an unattributed scan violation later.
DENIED = (
    "libavcodec",
    "libavdevice",
    "libavfilter",
    "libswscale",
    "libswresample",
    "libpostproc",
    "libx264",
    "libx265",
    "libfdk-aac",
)


def canonical(name):
    return re.sub(r"[-_.]+", "-", name or "").lower()


def parse(version):
    """Compare on the numeric release only, and never raise.

    A version this cannot split (a release candidate, a dev build) must not kill
    the build with a traceback where this block has its own message to print.
    """
    parts = [int(x) for x in re.findall(r"\d+", version)[:3]]
    return tuple(parts + [0] * (3 - len(parts)))


# Enumerated over sys.path rather than one scheme directory: these images carry
# both /usr/local/lib/python3.12/dist-packages and /usr/lib/python3/dist-packages,
# and the wheel declares Root-Is-Purelib: false, so neither purelib nor platlib
# alone is guaranteed to be the install target or the only place a copy can hide.
# A surviving base copy beside the new one is the failure being looked for, which
# is also why this counts distributions instead of asking for one version.
installed = [d for d in distributions() if canonical(d.metadata["Name"]) == NAME]
versions = sorted(d.version for d in installed)
print("PyNvVideoCodec distributions on sys.path:", versions)
if len(installed) != 1:
    sys.exit(f"ERROR: expected exactly one PyNvVideoCodec, found {versions}")
if parse(versions[0]) < parse(FLOOR):
    sys.exit(f"ERROR: PyNvVideoCodec {versions[0]} is below the {FLOOR} floor")

site = os.path.normpath(str(installed[0].locate_file("")))
pkg = os.path.join(site, "PyNvVideoCodec")
bundled = sorted(
    os.path.relpath(p, pkg)
    for p in glob.glob(os.path.join(pkg, "**", "lib*.so*"), recursive=True)
)
print("PyNvVideoCodec bundles:", bundled)
# Positive first, and on both libraries: an empty package directory satisfies
# every negative check below while shipping no demuxer at all.
for required in ("libavformat", "libavutil"):
    if not any(os.path.basename(n).startswith(required) for n in bundled):
        sys.exit(
            f"ERROR: PyNvVideoCodec bundles no {required}, so the checks below "
            f"would pass vacuously; found {bundled}"
        )
denied = [n for n in bundled if os.path.basename(n).startswith(DENIED)]
if denied:
    sys.exit(f"ERROR: PyNvVideoCodec bundles libraries the codec gate denies: {denied}")

# The FFmpeg source tarball lands outside site-packages, so its directory is read
# from the wheel's own RECORD rather than guessed from a sysconfig path -- the
# RECORD is what the installer actually wrote, and it moves if the layout does.
record = installed[0].read_text("RECORD") or ""
declared = [
    row[0]
    for row in csv.reader(record.splitlines())
    if row and row[0].endswith((".tar.xz", ".tar.gz", ".tar.bz2"))
]
if len(declared) != 1:
    sys.exit(f"ERROR: expected one source tarball in the RECORD, found {declared}")
external = os.path.dirname(os.path.normpath(os.path.join(site, declared[0])))
tarballs = sorted(os.path.basename(p) for p in glob.glob(os.path.join(external, "ffmpeg-*.tar.*")))
print("bundled FFmpeg source tarballs in", external, "->", tarballs)
if len(tarballs) != 1:
    sys.exit(f"ERROR: expected exactly one bundled FFmpeg source tarball, found {tarballs}")
PYEOF
{% else %}
# PyNvVideoCodec decodes on NVDEC through libnvcuvid, so it is inert on a
# non-NVIDIA device. Drop it from the shared requirements rather than ship an
# unusable NVIDIA codec wheel in, for example, the Intel XPU image. The pattern
# is anchored to the line start so it cannot match inside another requirement,
# and the presence check fails the build if the package arrives by another route
# -- a filter that silently stopped matching would otherwise look like success.
# It asks importlib for the distribution instead of importing it: `import
# PyNvVideoCodec` dlopens libnvcuvid.so.1, which no driverless XPU or CPU
# builder has, so an import-based check would raise whether or not the wheel is
# installed and could never fail the build.
#
# mooncake goes the same way, for two reasons. The floor names the CUDA 13
# distribution, and the XPU and CPU bases carry no mooncake at all, so under
# --no-deps it would arrive here without msgpack, which vLLM does not pull in.
# The check asserts the distribution is absent rather than the `mooncake` module,
# so it tests the filter without assuming what a future base may ship. It is
# written as a positive test with no `!` and no stderr redirect: a broken
# interpreter then fails the build instead of passing it vacuously.
#
# Whole-RUN branches, rather than a conditional inside one RUN: a `{% raw %}{% if %}{% endraw %}` in the
# middle of a `\`-continued command emits a blank line that ends the command
# early, and a `#` comment there is joined onto the previous line, commenting
# out the rest. Both break the build in ways the template does not show.
RUN --mount=type=bind,source=./container/deps/requirements.vllm.txt,target=/tmp/requirements.vllm.txt \
    --mount=type=cache,id=uv-root-{{ context.dynamo.uv_version }},target=/root/.cache/uv,sharing=locked \
    export UV_CACHE_DIR=/root/.cache/uv && \
    grep -v -e '^PyNvVideoCodec' -e '^mooncake-transfer-engine-cuda13' \
        /tmp/requirements.vllm.txt > /tmp/requirements.vllm.nonvidia.txt && \
    uv pip install {{ pip_target }} --reinstall-package imageio-ffmpeg --no-deps \
        --requirement /tmp/requirements.vllm.nonvidia.txt && \
    rm -f /tmp/requirements.vllm.nonvidia.txt && \
    /opt/venv/bin/python -c "import importlib.util,sys; sys.exit(1 if importlib.util.find_spec('PyNvVideoCodec') else 0)" && \
    /opt/venv/bin/python -c "import importlib.metadata as m, re, sys; names={re.sub(r'[-_.]+', '-', n).lower() for d in m.distributions() if (n := (d.metadata or {}).get('Name'))}; sys.exit(1 if 'mooncake-transfer-engine-cuda13' in names else 0)" && \
    rm -rf /opt/uv/cache
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
# keeps OpenCV for mistral_common's cv2.resize, rebuilt at the base image's
# version with video backends disabled. Other codec-bearing wheels are removed.
# (PyNvVideoCodec is KEPT for NVDEC hardware decode -- see the note below.) The in-tree
# LGPL ffmpeg + imageio-ffmpeg installed above are intentionally KEPT for the
# omni video-encode path, which uses the royalty-free VP9 (libvpx_vp9) encoder —
# no H.264 is built. Direct rm makes the removal robust regardless of how the
# base image's pip is configured; the guards fail the build if any of them survive.
RUN set -eux; \
    OPENCV_VERSION="$(python3 -m pip show opencv-python-headless 2>/dev/null | awk '/^Version:/{print $2}')"; \
    if [ -z "${OPENCV_VERSION}" ]; then \
        OPENCV_VERSION="$(python3 -m pip show opencv-python 2>/dev/null | awk '/^Version:/{print $2}')"; \
    fi; \
    test -n "${OPENCV_VERSION}" || { echo "ERROR: base image must provide version metadata for opencv-python-headless or opencv-python" >&2; exit 1; }; \
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
    ! python3 -c "import torchcodec" 2>/dev/null; \
    ENABLE_HEADLESS=1 ENABLE_CONTRIB=0 MAKEFLAGS="-j$(nproc)" \
    CMAKE_ARGS="-DWITH_FFMPEG=OFF -DWITH_GSTREAMER=OFF -DVIDEOIO_ENABLE_PLUGINS=OFF -DWITH_1394=OFF -DWITH_V4L=OFF -DBUILD_TESTS=OFF -DBUILD_PERF_TESTS=OFF -DBUILD_EXAMPLES=OFF -DBUILD_opencv_apps=OFF -DENABLE_CCACHE=OFF" \
    python3 -m pip install --no-binary opencv-python-headless "opencv-python-headless==${OPENCV_VERSION}"; \
    rm -rf /root/.cache/pip; \
    python3 -c "import cv2; cv2.resize"; \
    ! ls -d "${SITE_PACKAGES}"/opencv_python*.libs 2>/dev/null; \
    python3 -c "import cv2,re,sys; enabled=[name for name,value in re.findall(r'^\s*(FFMPEG|GSTREAMER):\s*(\S+)', cv2.getBuildInformation(), re.M|re.I) if value.upper()=='YES']; sys.exit('ERROR: cv2 was built with video backends: '+', '.join(enabled) if enabled else 0)"

# PyNvVideoCodec is KEPT (removed from the purge above) but REPLACED at >=2.2.3 by
# the removal and requirements install above: the base image's 2.0.4 bundles a
# full FFmpeg 8.1.1 (incl. libavcodec) that the codec gate rejects, while 2.2.3
# bundles only libavutil.so.61 + libavformat.so.63 (FFmpeg 9.0.1, container
# demux, no software codec). It provides the built-in H.264/H.265 video-input
# path (NVDEC hardware decode via libnvcuvid;
# common/multimodal/nvdec_decoder.py) and requires the driver "video" capability
# at runtime or it cannot import; set it in the image and ensure the K8s
# pod/runtimeClass does not drop it.
#
# This deliberately overrides a dependency of vLLM's: `importlib.metadata.requires
# ("vllm")` lists `PyNvVideoCodec==2.0.4`, an exact pin, and the image ships 2.2.3
# against it. Unlike the equivalent override in trtllm_runtime.Dockerfile, do not
# expect a resolver complaint in the build log to confirm it -- that stage runs
# real `pip`, which prints one; this one runs `uv pip install --no-deps`, which
# does not. `uv pip check` on the finished image is what surfaces it.
#
# The override is deliberate: 2.0.4 is the version whose bundled libavcodec the
# codec gate rejects, so the override is the point. vLLM's own consumer is
# vllm/multimodal/video.py (PyNvVideoCodecVideoBackendMixin), and everything it
# touches -- SimpleDecoder, OutputColorType, get_batch_frames_by_index,
# get_stream_metadata, reconfigure_decoder -- still exists in 2.2.3. The one API
# 2.2.3 drops is SimpleDecoder.stop(), which neither vLLM nor Dynamo calls;
# `get_frames_at` in that file belongs to the torchcodec backend, not this
# package. Re-check that list when the base image moves.
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

# vLLM-Omni is installed with the current Transformers version in its protected
# constraints file, so an incompatible Omni requirement fails during dependency
# resolution. Check the completed image as well so a later package layer cannot
# silently replace the vLLM-Omni-compatible Transformers release. A global
# `uv pip check` is not appropriate here: the upstream runtime and Dynamo's
# deliberate --no-deps layers contain unrelated package-metadata conflicts.
RUN {{ python_executable }} - "${TRANSFORMERS_VERSION}" <<'PY'
import importlib.metadata as md
import sys

actual = md.version("transformers")
expected = sys.argv[1]
if actual != expected:
    raise RuntimeError(f"expected transformers {expected}, found {actual}")
PY

# Use the packaged binary to match the installed vLLM version.
RUN set -eu; \
    pkg="$({{ python_executable }} -c 'import os, vllm; print(os.path.dirname(vllm.__file__))')"; \
    if [ -f "${pkg}/vllm-rs" ] && [ -x "${pkg}/vllm-rs" ]; then \
        if [ "{{ vllm_rs_allowlist }}" = "1" ]; then \
            printf '%s\n' \
                '#!/bin/sh' \
                '# Keep Omni from changing the EngineCore output schema.' \
                'VLLM_PLUGINS="${VLLM_PLUGINS-{{ vllm_rs_plugins }}}"' \
                'export VLLM_PLUGINS' \
                "exec \"${pkg}/vllm-rs\" \"\$@\"" \
                > {{ vllm_rs_link }}; \
            chmod 755 {{ vllm_rs_link }}; \
        else \
            ln -sf "${pkg}/vllm-rs" {{ vllm_rs_link }}; \
        fi; \
        vllm-rs --help >/dev/null; \
    elif [ "{{ vllm_rs_required }}" = "1" ]; then \
        echo "ERROR: installed vllm package (${pkg}) ships no executable vllm-rs" >&2; \
        exit 1; \
    else \
        echo "WARNING: installed vllm package (${pkg}) ships no executable vllm-rs; not putting it onto PATH" >&2; \
    fi

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
   already skips them, and their pre_runtime carries no dynamo venv to scan.
   cpu likewise: unshipped, and no baseline_sbom to subtract. #}
{% if target not in ("dev", "local-dev") and device != "cpu" %}
{% include "templates/compliance.Dockerfile" %}
{% endif %}


FROM pre_runtime AS runtime
{% if target not in ("dev", "local-dev") and device != "cpu" %}
COPY --from=licenses /legal /legal
{% endif %}
