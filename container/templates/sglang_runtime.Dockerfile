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

ARG MODELEXPRESS_VERSION

WORKDIR /workspace

# Install NATS and ETCD
COPY --from=dynamo_base /usr/bin/nats-server /usr/bin/nats-server
COPY --from=dynamo_base /usr/local/bin/etcd/ /usr/local/bin/etcd/

ENV PATH=/usr/local/bin/etcd:$PATH

{% if device == "cuda" %}
# Install the TurboJPEG runtime used by frontend JPEG decoding and bring
# base-image OS packages up to the current patch releases. --only-upgrade skips
# anything not already installed while keeping both operations in one layer.
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        libturbojpeg && \
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
    rm -rf /var/lib/apt/lists/* && \
    ldconfig && \
    ldconfig -p | grep -F 'libturbojpeg.so.0' >/dev/null
{% else %}
# Install the TurboJPEG runtime used by frontend JPEG decoding.
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        libturbojpeg && \
    rm -rf /var/lib/apt/lists/* && \
    ldconfig && \
    ldconfig -p | grep -F 'libturbojpeg.so.0' >/dev/null
{% endif %}

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
        /opt/dynamo/wheelhouse/aisimulate*.whl \
        /opt/dynamo/wheelhouse/nixl/nixl*.whl \
        "distro==1.9.0"
{% else %}
RUN --mount=type=cache,target=/root/.cache/pip,sharing=locked \
    export PIP_CACHE_DIR=/root/.cache/pip && \
    pip install --break-system-packages --no-deps \
        /opt/dynamo/wheelhouse/ai_dynamo_runtime*.whl \
        /opt/dynamo/wheelhouse/ai_dynamo*any.whl \
        /opt/dynamo/wheelhouse/aisimulate*.whl

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
# SGLang runtime dependency stack. google-crc32c is imported eagerly by the MX
# sglang loader (>=0.5.0) and is not in the SGLang base image, so install it
# alongside; the import check below fails the build on any future --no-deps gap.
RUN --mount=type=cache,target=/root/.cache/pip,sharing=locked \
    set -eux; \
    export PIP_CACHE_DIR=/root/.cache/pip; \
    pip install --break-system-packages --no-deps \
        "modelexpress==${MODELEXPRESS_VERSION}"; \
    pip install --break-system-packages "google-crc32c>=1.5.0"; \
    python3 -c "import modelexpress.engines.sglang"
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
# Transitional: this exists only until the base image stops shipping these.
# The permanent statement of what may ship is
# tests/dependencies/test_no_software_video_codecs.py -- when this purge
# becomes redundant, delete it and keep those assertions.
#
# CUDA only, and it has to be: the replacement ffmpeg is copied in under
# `device == "cuda"` just below. Running the purge on XPU strips the base image's
# media stack and puts nothing back, leaving that image with no ffmpeg at all and
# an empty IMAGEIO_FFMPEG_EXE -- worse than either leaving it alone or replacing
# it properly. The XPU image is not published, so there is nothing to remove from
# it in the first place; the codec gate skips it for the same reason. Matches the
# equivalent purge in vllm_runtime.Dockerfile.
# The Muse Spark Open base also installs the Ubuntu FFmpeg/media dependency
# stack outside the Python and /usr/local paths removed below. Purge only the
# rejected media package families; do not autoremove, which could sweep unrelated
# CUDA/JIT dependencies. Keep this list scoped to packages found in that media
# stack so networking and the non-media runtime remain untouched.
{% if device == "cuda" %}
RUN set -eux; \
    media_packages="$(dpkg-query -W -f='${Package}\n' 2>/dev/null \
        | grep -E '^(ffmpeg|libass[0-9]|libavc1394|libav(codec|device|filter|format|util|resample)|libbluray|libbs2b|libcaca|libcdio|libchromaprint|libcodec2|libdc1394|libflite|libgme|libgsm|libiec61883|libjack|libmp3lame|libpocketsphinx|libpostproc|libpulse|libraw1394|librubberband|libsdl2|libshine|libsndfile|libsoxr|libsphinxbase|libsrt[0-9]|libsvtav1|libsw(resample|scale)|libtwolame|libudfread|libvidstab|libx264|libx265|libxvidcore|libzvbi)' \
        || true)"; \
    if [ -n "${media_packages}" ]; then \
        DEBIAN_FRONTEND=noninteractive apt-get purge -y ${media_packages}; \
    fi; \
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
    apt-get clean; \
    rm -rf /var/lib/apt/lists/* /var/cache/apt/*; \
    ldconfig

# The soundfile wheel carries the libsndfile implementation used by MuseSpark.
# Exercise an actual WAV encode/decode after the distro media stack is removed;
# an import-only check would not catch a missing shared-library dependency.
RUN python3 -c 'import io; import soundfile as sf; import torchaudio; b = io.BytesIO(); sf.write(b, [0.0], 16000, format="WAV"); b.seek(0); data, rate = sf.read(b); assert rate == 16000 and len(data) == 1'
{% endif %}

{% if device == "cuda" and target not in ("dev", "local-dev") %}
# The SGLang development base includes GUI, packaging, debugging and kernel
# module build tools that serving does not use. Remove these before generating
# the runtime SBOM. Keep the compiler, CUDA/JIT headers and userspace RDMA/GDRCopy
# libraries; the node supplies the kernel driver. Do not use autoremove.
# Preserve the graphics libraries present in the qualified runtime. The public
# base's Mesa version also needs libwayland-server0; its verified license is
# recorded in license_overrides.yaml rather than removing that dependency.
RUN set -eux; \
    candidates='at-spi2-common autoconf automake autopoint autotools-dev \
        bear bsdextrautils check cloc dbus-user-session dconf-gsettings-backend \
        dconf-service debhelper debugedit dh-autoreconf dh-strip-nondeterminism \
        dkms dwz fakeroot gdb gdrdrv-dkms gettext gettext-base \
        gfortran-13 gfortran-13-aarch64-linux-gnu gfortran-13-x86-64-linux-gnu \
        googletest groff-base gtk-update-icon-cache hicolor-icon-theme htop \
        humanity-icon-theme intltool-debian kmod libatk-bridge2.0-0t64 \
        libatk1.0-0t64 libatspi2.0-0t64 libavahi-client3 libavahi-common-data \
        libavahi-common3 libbabeltrace1 libc-ares-dev libcolord2 libdconf1 \
        libdebhelper-perl libdebuginfod-common libdebuginfod1t64 libdw-dev \
        libear libegl1 libelf-dev libevdev2 libevent-2.1-7t64 libevent-dev \
        libevent-extra-2.1-7t64 libevent-openssl-2.1-7t64 libfakeroot \
        libfile-stripnondeterminism-perl libgd3 libgfortran-13-dev libgrpc++-dev \
        libgrpc-dev libgtest-dev libgtk-3-0t64 libgtk-3-common libgudev-1.0-0 \
        libheif-plugin-aomdec libheif1 libhwloc-dev libinput-bin libinput10 \
        libjs-underscore libjsoncpp-dev libltdl-dev libltdl7 liblzma-dev \
        libnl-genl-3-200 libnotify4 libopenmpi-dev libpipeline1 libpmix-dev \
        libpython3-dev libqt5core5t64 libqt5dbus5t64 libqt5gui5t64 \
        libqt5network5t64 libqt5printsupport5t64 libqt5widgets5t64 \
        libsource-highlight-common libsource-highlight4t64 libsystemd-dev \
        libtirpc-common libtirpc3t64 libtool libuchardet0 libunwind-dev \
        libutempter0 libwxbase3.2-1t64 libwxgtk3.2-1t64 \
        libxcb-icccm4 libxcb-image0 libxcb-keysyms1 libxcb-render-util0 \
        libxcb-util1 libxcb-xinerama0 libxcb-xinput0 libxcb-xkb1 \
        libxcomposite1 libxdamage1 libxinerama1 libxkbcommon-x11-0 libxpm4 \
        libxtst6 lsof m4 man-db mpi-default-bin mpi-default-dev \
        nsight-systems-cli nsight-systems-cli-2026.4.1 openmpi-bin openmpi-common \
        pahole patchutils po-debconf protobuf-compiler-grpc python3-dev \
        silversearcher-ag tree ubuntu-mono uuid-dev wdiff zsh zsh-common'; \
    packages="$(dpkg-query -W -f='${Package} ${db:Status-Status}\n' ${candidates} 2>/dev/null \
        | awk '$2 == "installed" { print $1 }' || true)"; \
    if [ -n "${packages}" ]; then \
        apt-get --simulate purge ${packages} > /tmp/sglang-purge-plan; \
        cat /tmp/sglang-purge-plan; \
        if awk '$1 == "Remv" || $1 == "Purg" { print $2 }' /tmp/sglang-purge-plan \
            | grep -E '^(build-essential|cmake|make|ninja-build|pkg-config|python3|python3-minimal|python3\.[0-9]+(-dev|-minimal|-venv)?|libpython3\.[0-9]+.*|gcc(-.*)?|g\+\+(-.*)?|cuda-.*|libcublas.*|libcufft.*|libcufile.*|libcurand.*|libcusolver.*|libcusparse.*|libnpp.*|libnv.*|libnccl.*|libgdrapi.*|libibverbs.*|ibverbs-providers|librdmacm.*|libmlx.*|libnuma.*|libssl.*|libuv[0-9].*|libzmq.*|libczmq[0-9].*|libevent-(core|pthreads).*|libhwloc[0-9].*|libpmix[0-9].*|libopenmpi[0-9].*|libelf[0-9].*|libdw[0-9].*|libgbm1|libgl1|libgl1-mesa-dri|libglx-mesa0|libglx0|libwayland-server0|libc6(-dev)?|zlib1g(-dev)?|liblzma[0-9].*)$'; then \
            echo 'ERROR: SGLang cleanup would remove serving or JIT dependencies' >&2; \
            exit 1; \
        fi; \
        DEBIAN_FRONTEND=noninteractive apt-get purge -y ${packages}; \
        rm /tmp/sglang-purge-plan; \
    fi; \
    apt-get clean; \
    rm -rf /var/lib/apt/lists/* /var/cache/apt/*; \
    ldconfig; \
    for tool in c++ cmake make nvcc; do command -v "${tool}"; done; \
    test -f /usr/include/gdrapi.h; \
    printf '%s\n' '#include <Python.h>' 'int main() { return 0; }' \
        | c++ -fsyntax-only -x c++ -I"$(python3 -c 'import sysconfig; print(sysconfig.get_path("include"))')" -
ENV GDRCOPY_HOME=/usr
{% endif %}

# Drop the Nsight efa_metrics plugin the CUDA floor carries: a Go NIC sampler
# nothing in the serving path loads. On this image it arrives under Nsight
# Compute rather than Nsight Systems, and both roots move with every base bump,
# so the paths are globbed and the removal is asserted rather than pinned. This
# stage has no overlay rebase -- `runtime` is FROM pre_runtime -- so one
# deletion here ships.
RUN rm -rf \
        /usr/local/cuda-*/NsightSystems-cli-*/target-linux-*/plugins/efa_metrics \
        /opt/nvidia/nsight-systems-cli/*/target-linux-*/plugins/efa_metrics \
        /opt/nvidia/nsight-compute/*/host/target-linux-*/plugins/efa_metrics && \
    [ -z "$(find /usr/local /opt -xdev -type d -name efa_metrics 2>/dev/null)" ]

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

{% if device != "xpu" and target not in ("dev", "local-dev") %}
# Add generic UCX aliases beside NIXL's private libraries so native consumers
# use the same implementation without breaking UCX's $ORIGIN-based core and
# module lookup. The wheel's auditwheel dependency directory is deliberately
# placed first for every process; it contains only hash-mangled dependencies
# plus the two generic UCX aliases. No existing wheel file or ELF metadata is
# modified. The same script publishes NIXL's C API directory at the second path
# below and registers it with the runtime linker, so the bare dlopen of
# libnixl_capi.so in nixl-sys resolves. Both paths are passed explicitly because
# the ENV on the next line has to name the same two directories.
RUN --mount=type=bind,source=./container/deps/sglang/install_nixl_ucx_compat.sh,target=/tmp/install_nixl_ucx_compat.sh,readonly \
    --mount=type=bind,source=./container/deps/sglang/discover_nixl_ucx_layout.py,target=/tmp/discover_nixl_ucx_layout.py,readonly \
    bash /tmp/install_nixl_ucx_compat.sh /opt/dynamo/nixl-ucx-compat /opt/dynamo/nixl-capi
ENV LD_LIBRARY_PATH=/opt/dynamo/nixl-ucx-compat:/opt/dynamo/nixl-capi${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
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
# The vllm and trtllm images already copy lib/; sglang did not, so its tests had
# no lib/llm/tests/data/media fixtures and any test needing a local media clip
# could only skip. ~32 MB, negligible here and it keeps the three consistent.
COPY --chown=dynamo:0 lib /workspace/lib
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
#
# CUDA only, for the same reason as the purge above: this asserts that the purge
# worked, and on XPU there is no purge to assert. Left ungated it would fail that
# build on the base image's own libraries -- which is precisely what happened to
# the first version of this change, where gating the purge alone turned a working
# XPU build into a failing one.
{% if device == "cuda" %}
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
{% endif %}

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
# NVDEC (PyNvVideoCodec, added in requirements.sglang.txt) needs the driver
# "video" capability at runtime so libnvcuvid is exposed; without it
# PyNvVideoCodec cannot import. Ensure the K8s pod/runtimeClass does not drop it.
ENV NVIDIA_DRIVER_CAPABILITIES=video,compute,utility
{% if target not in ("dev", "local-dev") %}
COPY --from=licenses /legal /legal
{% endif %}
ENTRYPOINT ["/opt/nvidia/nvidia_entrypoint.sh"]
CMD []
{% endif %}
