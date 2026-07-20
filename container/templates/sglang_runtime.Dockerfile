{#
# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#}
# === BEGIN templates/sglang_runtime.Dockerfile ===
##################################
########## Runtime Image #########
##################################

{% if device == "xpu" %}
FROM framework AS runtime
{% else %}
FROM ${RUNTIME_IMAGE}:${RUNTIME_IMAGE_TAG} AS pre_runtime
{% endif %}

{% if device != "xpu" %}
# SGLang 0.5.15 JIT-compiles its native HiCache hash extension, which includes
# OpenSSL headers and links libcrypto.
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        libssl-dev
{% endif %}

ARG MODELEXPRESS_VERSION

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

RUN SITE_PACKAGES="$(python3 -c 'import site; print(site.getsitepackages()[0])')" && \
    CUBINS_DIR="$SITE_PACKAGES/flashinfer_cubin/cubins" && \
    if [ -d "$CUBINS_DIR" ]; then \
        find "$CUBINS_DIR" -type d -exec chmod g+rwx {} + ; \
    fi

{% if device == "xpu" %}
{# XPU runtime: NIXL + UCX are needed for P2P transport on Intel GPUs.
   CUDA sglang runtime does NOT include NIXL/UCX (matching upstream main);
   those are only added in the dev stage for build-time linking. #}
ENV NIXL_PREFIX=/opt/intel/intel_nixl
ENV NIXL_LIB_DIR=$NIXL_PREFIX/lib/x86_64-linux-gnu
ENV NIXL_PLUGIN_DIR=$NIXL_LIB_DIR/plugins

# Copy UCX and NIXL from wheel_builder
COPY --from=wheel_builder /usr/local/ucx /usr/local/ucx
COPY --chown=dynamo:0 --from=wheel_builder $NIXL_PREFIX $NIXL_PREFIX
COPY --chown=dynamo:0 --from=wheel_builder /opt/intel/intel_nixl/lib/x86_64-linux-gnu/. ${NIXL_LIB_DIR}/

COPY --chown=dynamo:0 --from=wheel_builder /opt/dynamo/dist/nixl/ /opt/dynamo/wheelhouse/nixl/
COPY --chown=dynamo:0 --from=wheel_builder /workspace/nixl/build/src/bindings/python/nixl-meta/nixl-*.whl /opt/dynamo/wheelhouse/nixl/

ENV PATH=/usr/local/ucx/bin:$PATH

ENV LD_LIBRARY_PATH=\
$NIXL_LIB_DIR:\
$NIXL_PLUGIN_DIR:\
/usr/local/ucx/lib:\
/usr/local/ucx/lib/ucx:\
${LD_LIBRARY_PATH:-}
{% endif %}

{% if target not in ("dev", "local-dev") %}
# Runtime target installs only the prebuilt Dynamo wheels. SGLang and its NIXL
# packages come from the upstream lmsysorg/sglang runtime image; --no-deps keeps
# pip from replacing that stack. Dev/local-dev build from source later in the
# shared dev stage after the workspace is bind-mounted.
COPY --chmod=775 --chown=dynamo:0 --from=wheel_builder /opt/dynamo/dist/*.whl /opt/dynamo/wheelhouse/

{% if device == "xpu" %}
RUN pip install --no-deps \
        /opt/dynamo/wheelhouse/ai_dynamo_runtime*.whl \
        /opt/dynamo/wheelhouse/ai_dynamo*any.whl \
        /opt/dynamo/wheelhouse/nixl/nixl*.whl \
        "distro==1.9.0"
{% else %}
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
# SGLang server_args eagerly imports sglang.srt.entrypoints.openai.protocol
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

{% if context.sglang.enable_modelexpress == "true" %}
# Install only the ModelExpress client package. --no-deps preserves the upstream
# SGLang runtime dependency stack.
RUN --mount=type=cache,target=/root/.cache/pip,sharing=locked \
    set -eux; \
    export PIP_CACHE_DIR=/root/.cache/pip; \
    pip install --break-system-packages --no-deps \
        "modelexpress==${MODELEXPRESS_VERSION}"
{% endif %}
{% endif %}
{% endif %}

# Install nvtx pinned in container/deps/requirements.common.txt so DYN_NVTX=1
# profiling works in all targets (runtime, dev, local-dev) — see
# components/src/dynamo/common/utils/nvtx_utils.py. --no-deps preserves the
# upstream lmsysorg/sglang Python stack.
RUN --mount=type=bind,source=./container/deps/requirements.common.txt,target=/tmp/requirements.common.txt \
    --mount=type=cache,target=/root/.cache/pip,sharing=locked \
    export PIP_CACHE_DIR=/root/.cache/pip && \
    pip install --break-system-packages --no-deps $(grep -E '^nvtx==' /tmp/requirements.common.txt)

# Install SGLang-specific runtime dependencies without changing the upstream
# dependency solution. imageio-ffmpeg is installed from source (no bundled
# binary) for the VP9 video-encode path; see requirements.sglang.txt.
RUN --mount=type=bind,source=./container/deps/requirements.sglang.txt,target=/tmp/requirements.sglang.txt \
    --mount=type=cache,target=/root/.cache/pip,sharing=locked \
    export PIP_CACHE_DIR=/root/.cache/pip && \
    pip install --break-system-packages --force-reinstall --no-deps \
        --requirement /tmp/requirements.sglang.txt

# Remove the codec-bearing video-DECODE components from the upstream SGLang image
# (PyAV, decord, OpenCV, torchcodec + any base ffmpeg/libav*), then copy the
# VP9-only in-tree ffmpeg from wheel_builder below for the video-generation
# encode path. H.264/H.265/AAC encoders must never appear (guarded at the end).
#
# Inkling image preprocessing uses Pillow. Its audio feature extractor imports
# soundfile and uses torchaudio only for resampling, so those paths remain
# available for formats supported by libsndfile (for example WAV and FLAC).
# AAC-backed M4A stays removed; video encode is VP9 only.
RUN set -eux; \
    python3 -m pip uninstall --yes \
        av \
        decord \
        decord2 \
        opencv-python \
        opencv-python-headless \
        torchcodec; \
    SITE_PACKAGES="$(python3 -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"; \
    rm -rf \
        "${SITE_PACKAGES}"/av \
        "${SITE_PACKAGES}"/av-*.dist-info \
        "${SITE_PACKAGES}"/av.libs \
        "${SITE_PACKAGES}"/cv2 \
        "${SITE_PACKAGES}"/decord \
        "${SITE_PACKAGES}"/decord-*.dist-info \
        "${SITE_PACKAGES}"/decord.libs \
        "${SITE_PACKAGES}"/decord2 \
        "${SITE_PACKAGES}"/decord2-*.dist-info \
        "${SITE_PACKAGES}"/decord2.libs \
        "${SITE_PACKAGES}"/opencv_python*.dist-info \
        "${SITE_PACKAGES}"/opencv_python*.libs \
        "${SITE_PACKAGES}"/torchcodec \
        "${SITE_PACKAGES}"/torchcodec-*.dist-info \
        /usr/local/bin/ffmpeg \
        /usr/local/bin/ffprobe \
        /usr/local/include/libav* \
        /usr/local/include/libsw* \
        /usr/local/lib/libav* \
        /usr/local/lib/libpostproc* \
        /usr/local/lib/libsw* \
        /usr/local/lib/pkgconfig/libav*.pc \
        /usr/local/lib/pkgconfig/libpostproc*.pc \
        /usr/local/lib/pkgconfig/libsw*.pc \
        /usr/local/src/ffmpeg \
        /root/.cache/pip; \
    ldconfig

{% if device == "cuda" %}
# Copy the in-tree VP9 ffmpeg from wheel_builder: versioned shared libs
# (libav*.so*, libsw*.so*) + libvpx + the in-tree CLI binary that imageio targets
# via IMAGEIO_FFMPEG_EXE. The upstream ffmpeg was purged above, so this
# VP9-only build is the only ffmpeg present; the video-generation
# handler (CUDA DiffGenerator) encodes libvpx-vp9 with it. Same pattern as
# vllm_runtime.Dockerfile.
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
# gap where SGLang shipping no encoder passed every PR gate.
RUN set -eu; \
    ff="${IMAGEIO_FFMPEG_EXE:-ffmpeg}"; \
    "$ff" -hide_banner -encoders 2>/dev/null | grep -qiE 'libvpx[-_]vp9' \
      || { echo "ERROR: shipped ffmpeg ($ff) has no VP9 encoder" >&2; exit 1; }; \
    if "$ff" -hide_banner -encoders 2>/dev/null \
         | grep -iE 'h\.?264|h\.?265|hevc|(^| )aac|nvenc|cuvid|nvdec'; then \
        echo "ERROR: shipped ffmpeg ($ff) exposes an H.264/H.265/AAC/NVENC encoder" >&2; \
        exit 1; \
    fi
{% else %}
ENV IMAGEIO_FFMPEG_EXE=
{% endif %}

# Copy tests, deploy and components for CI with correct ownership
COPY --chmod=775 --chown=dynamo:0 tests /workspace/tests
COPY --chmod=775 --chown=dynamo:0 examples /workspace/examples
COPY --chmod=775 --chown=dynamo:0 deploy /workspace/deploy
COPY --chmod=775 --chown=dynamo:0 dev /workspace/dev
COPY --chmod=775 --chown=dynamo:0 components/src/dynamo/common /workspace/components/src/dynamo/common
COPY --chmod=775 --chown=dynamo:0 components/src/dynamo/frontend /workspace/components/src/dynamo/frontend
COPY --chmod=775 --chown=dynamo:0 components/src/dynamo/sglang /workspace/components/src/dynamo/sglang
COPY --chmod=775 --chown=dynamo:0 components/src/dynamo/mocker /workspace/components/src/dynamo/mocker
COPY --chmod=775 --chown=dynamo:0 recipes/ /workspace/recipes/
COPY --chmod=664 --chown=dynamo:0 LICENSE /workspace/

# Enable forceful shutdown of inflight requests
ENV SGLANG_FORCE_SHUTDOWN=1

# Setup launch banner in common directory accessible to all users
RUN --mount=type=bind,source=./container/launch_message/runtime.txt,target=/opt/dynamo/launch_message.txt \
    sed '/^#\s/d' /opt/dynamo/launch_message.txt > /opt/dynamo/.launch_screen

RUN chmod 755 /opt/dynamo/.launch_screen && \
    echo 'cat /opt/dynamo/.launch_screen' >> /etc/bash.bashrc && \
{%- if device == "xpu" %}
    echo '. /opt/miniforge3/bin/activate sglang' >> /etc/bash.bashrc && \
    echo 'source /opt/intel/oneapi/setvars.sh --force' >> /etc/bash.bashrc && \
    mkdir -p /sgl-workspace && \
    ln -sf /workspace /sgl-workspace/dynamo
{%- else %}
    ln -s /workspace /sgl-workspace/dynamo && \
    NSYS_BIN=$(find /opt/nvidia/nsight-compute -maxdepth 6 -type f -name nsys -executable 2>/dev/null | head -n1) && \
    if [ -n "$NSYS_BIN" ]; then ln -sf "$NSYS_BIN" /usr/local/bin/nsys; \
    else echo "WARNING: no bundled nsys found under /opt/nvidia/nsight-compute"; fi
{% endif %}

{%- if device != "xpu" %}
# Precompile Python bytecode into the image while still root. CI runs tests as
# the non-root `dynamo` user, which cannot write .pyc back to site-packages, and
# the test harness forks a fresh process per test. Without baked .pyc, every test
# process recompiles torch/transformers/sglang from source on first import (~+3.5s
# each), which previously added ~8-10 min to the sglang CI job. This was implicitly
# provided by the now-removed vendored-patch step that ran `import sglang` at build.
RUN SITE_PACKAGES="$(python3 -c 'import site; print(site.getsitepackages()[0])')" && \
    python3 -m compileall -q -j0 "$SITE_PACKAGES" && \
    (python3 -m compileall -q -j0 /sgl-workspace/sglang/python || true)
{%- endif %}

# Belt-and-suspenders guard at the end of the populated runtime stage: fail the
# build if an H.264/H.265/AAC codec *implementation* library appears. The VP9
# ffmpeg (ffmpeg + libav*/libsw* + libvpx) copied above is intentionally
# shipped for the video-encode path, so it is NOT flagged here — the in-tree
# build is VP9-only by construction (wheel_builder's post-build codec-surface
# guard) and the media-codec scan below re-permits it only under /usr/local while
# still catching any stray third-party libav*/ffmpeg. This adds the extra
# non-FFmpeg AAC/H.264 implementation names the scan's deny_globs don't list.
RUN set -eux; \
    remaining="$(find /usr /opt /workspace /sgl-workspace -xdev \
        \( -type f -o -type l \) \
        \( -name 'libx264*.so*' -o -name 'libx265*.so*' \
        -o -name 'libopenh264*.so*' -o -name 'libfdk-aac*.so*' \
        -o -name 'libfaac*.so*' -o -name 'libvo-aacenc*.so*' \
        -o -name 'libaacplus*.so*' \) -print)"; \
    if [ -n "${remaining}" ]; then \
        echo "ERROR: H.264/H.265/AAC codec libraries remain in the SGLang image:" >&2; \
        echo "${remaining}" >&2; \
        exit 1; \
    fi; \
    python3 -c 'import soundfile, torchaudio; from PIL import Image'

USER dynamo
ARG DYNAMO_COMMIT_SHA
ENV DYNAMO_COMMIT_SHA=${DYNAMO_COMMIT_SHA}

{% if device == "xpu" %}
CMD ["bash", "-c", "source /etc/bash.bashrc && exec bash"]
{% else %}
ENTRYPOINT ["/opt/nvidia/nvidia_entrypoint.sh"]
CMD []
{% endif %}


{% if device != "xpu" %}
{# Compliance is skipped for dev/local-dev: those images are not shipped (release
   ships runtime/frontend/operator/planner/snapshot-agent), compliance-extract
   already skips them, and their pre_runtime carries no dynamo venv to scan. #}
{% if target not in ("dev", "local-dev") %}
{% include "templates/compliance.Dockerfile" %}
{% endif %}


#######################################
########## Final runtime image ########
#######################################

FROM pre_runtime AS runtime
{% if target not in ("dev", "local-dev") %}
COPY --from=licenses /legal /legal
{% endif %}
ENTRYPOINT ["/opt/nvidia/nvidia_entrypoint.sh"]
CMD []
{% endif %}
