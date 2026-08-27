{#
# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#}
# === BEGIN templates/trtllm_runtime.Dockerfile ===
##################################
########## Runtime Image #########
##################################

# Transport stage — runtime pulls /workspace_src/ in one bind-mount cp.
FROM scratch AS workspace_files
COPY --chmod=775 tests /workspace_src/tests
COPY --chmod=775 examples /workspace_src/examples
COPY --chmod=775 deploy /workspace_src/deploy
COPY --chmod=775 dev /workspace_src/dev
COPY --chmod=775 components/src/dynamo/common /workspace_src/components/src/dynamo/common
COPY --chmod=775 components/src/dynamo/frontend /workspace_src/components/src/dynamo/frontend
COPY --chmod=775 components/src/dynamo/trtllm /workspace_src/components/src/dynamo/trtllm
COPY --chmod=775 components/src/dynamo/mocker /workspace_src/components/src/dynamo/mocker
COPY --chmod=775 lib /workspace_src/lib
COPY --chmod=664 LICENSE /workspace_src/

# Transport stage for dynamo_base artifacts. nats-server goes to /usr/bin (not
# /bin) because upstream is usrmerged and cross-stage COPY chokes on the symlink.
FROM scratch AS dynamo_base_export
COPY --from=dynamo_base /usr/bin/nats-server /usr/bin/nats-server
COPY --from=dynamo_base /usr/local/bin/etcd/ /usr/local/bin/etcd/
COPY --from=dynamo_base /opt/uv/bin/uv /opt/uv/bin/uv
COPY --from=dynamo_base /opt/uv/bin/uvx /opt/uv/bin/uvx

{% if target in ("runtime", "dev", "local-dev") %}
# Renamed `runtime` → `runtime_full` so the final stage can re-FROM upstream
# and overlay our changes as a single layer (cuts depth for downstream wrappers,
# incl. the dev image, which otherwise overflows overlay2's ~128-layer cap).
FROM ${RUNTIME_IMAGE}:${RUNTIME_IMAGE_TAG} AS runtime_full
{% else %}
FROM ${RUNTIME_IMAGE}:${RUNTIME_IMAGE_TAG} AS pre_runtime
{% endif %}

ARG ENABLE_KVBM
ARG ENABLE_GPU_MEMORY_SERVICE
ARG TARGETARCH
ARG NIXL_REF

# LD_PRELOAD pins TRT-LLM's bundled libnixl to dodge ai-dynamo/nixl#1668
# (nixl-cu13's UCX 1.20.0 hangs with two agents/host); drop it when fixed.
# NIXL_VERSION= clears the base image's stale value (see nixl-versions.txt).
ENV DYNAMO_HOME=/workspace \
    HOME=/home/dynamo \
    PATH=/usr/local/bin/etcd:${PATH} \
    LD_PRELOAD=/opt/dynamo/libstdc++.so.6:/usr/local/lib/python3.12/dist-packages/tensorrt_llm/libs/nixl/libnixl.so \
    NIXL_PLUGIN_DIR=/usr/local/lib/python3.12/dist-packages/tensorrt_llm/libs/nixl/plugins \
    NIXL_VERSION=

WORKDIR /workspace

# Install packages missing from upstream, sanity-check libnixl, register
# TRT-LLM lib paths with ldconfig (upstream's /etc/shinit_v2 only sets them
# for shells, not K8s python3 launches), swap upstream's standalone etcd
# tooling (etcd, etcdctl, etcdutl) for dynamo_base's directory so the image
# carries a single copy of each tool, drop the unused wandb developer tooling
# the DLFW base carries (upstream removes it on main, Dockerfile.multi), and
# symlink system libstdc++ to a stable
# path for LD_PRELOAD — keeps PyInstaller-bundled tools (specifically `jet`,
# NVIDIA's internal PyInstaller-packaged CI runner) from shadowing it with an
# older copy.
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        openssh-server \
        librdmacm1 \
        rdma-core && \
    test -f /usr/local/lib/python3.12/dist-packages/tensorrt_llm/libs/nixl/libnixl.so && \
    test -d "${NIXL_PLUGIN_DIR}" && \
    ARCH_ALT=$([ "${TARGETARCH}" = "amd64" ] && echo "x86_64" || echo "aarch64") && \
    printf '%s\n' \
        "/usr/local/tensorrt/lib" \
        "/usr/local/cuda/lib64" \
        "/usr/local/ucx/lib" \
        "/opt/nvidia/nvda_nixl/lib/${ARCH_ALT}-linux-gnu" \
        "/opt/nvidia/nvda_nixl/lib64" \
        > /etc/ld.so.conf.d/00-dynamo-trtllm.conf && \
    ldconfig && \
    rm -f \
        /usr/local/bin/etcd \
        /usr/local/bin/etcdctl \
        /usr/local/bin/etcdutl && \
    /usr/bin/python3 -m pip uninstall -y --break-system-packages wandb && \
    ! /usr/bin/python3 -c "import wandb" 2>/dev/null && \
    [ ! -e /usr/local/lib/python3.12/dist-packages/wandb ] && \
    mkdir -p /opt/dynamo && \
    LIBSTDCPP=/usr/lib/${ARCH_ALT}-linux-gnu/libstdc++.so.6 && \
    test -f "$LIBSTDCPP" && ln -sf "$LIBSTDCPP" /opt/dynamo/libstdc++.so.6

# Bring base-image OS packages up to the current patch releases published in
# the distro archives. --only-upgrade skips anything not already installed, so
# no new packages are added; versions are left unpinned so a cache-busted
# rebuild picks up the newest patch level (BuildKit reuses this layer otherwise).
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && \
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
        openssl

# One COPY pulls nats-server, etcd/, uv, uvx into their final paths.
COPY --from=dynamo_base_export / /
ENV PATH=/opt/uv/bin:${PATH}

# dynamo user (group 0 for OpenShift), clear upstream /workspace baggage
# (otherwise pytest collects broken tutorial test files), and create the
# Dynamo venv on non-dev. --system-site-packages keeps upstream's solve
# importable since system Python is PEP 668 externally-managed.
RUN userdel -r ubuntu > /dev/null 2>&1 || true \
    && useradd -m -s /bin/bash -g 0 dynamo \
    && [ `id -u dynamo` -eq 1000 ] \
    && mkdir -p /home/dynamo/.cache /opt/dynamo \
    && ln -sf /usr/bin/python3 /usr/local/bin/python \
    && rm -rf /workspace && mkdir /workspace \
    && chown dynamo:0 /home/dynamo /home/dynamo/.cache /opt/dynamo /workspace \
    && mkdir -p /etc/profile.d \
    && echo 'umask 002' > /etc/profile.d/00-umask.sh{% if target not in ("dev", "local-dev") %} \
    && python3 -m venv --system-site-packages /opt/dynamo/venv \
    && ln -sf /opt/uv/bin/uv /opt/dynamo/venv/bin/uv{% endif %}

{% if target not in ("dev", "local-dev") %}
ENV VIRTUAL_ENV=/opt/dynamo/venv \
    PATH=/opt/dynamo/venv/bin:${PATH}
{% endif %}

# Place wheels in /opt/dynamo/wheelhouse unconditionally — dev/local-dev images
# install from source and skip the pip install RUN below, but they still need
# the wheels on disk because tests/dependencies/test_kvbm_imports.py greps
# this path and runs in dev-derived test images.
COPY --chmod=775 --chown=dynamo:0 --from=wheel_builder /opt/dynamo/dist/*.whl /opt/dynamo/wheelhouse/

{% if target not in ("dev", "local-dev") %}
RUN --mount=type=cache,id=uv-root-{{ context.dynamo.uv_version }},target=/root/.cache/uv,sharing=locked \
    --mount=type=bind,source=./container/deps/requirements.trtllm.txt,target=/tmp/requirements.trtllm.txt \
    export UV_CACHE_DIR=/root/.cache/uv && \
    \
    # Dynamo's own wheels — --no-deps preserves upstream's solve.
    uv pip install --no-deps /opt/dynamo/wheelhouse/ai_dynamo_runtime*.whl && \
    uv pip install --no-deps /opt/dynamo/wheelhouse/ai_dynamo*any.whl && \
    uv pip install --no-deps /opt/dynamo/wheelhouse/aisimulate*.whl && \
    \
    # nixl/nixl-cu13 for KVBM's `import nixl` ABI; version from NIXL_REF, matching
    # the nixl-sys wheel_builder links. LD_PRELOAD swaps the runtime .so (nixl#1668).
    echo "${NIXL_REF}" | grep -qE '^v[0-9]+\.[0-9]+\.[0-9]+$' || { echo "NIXL_REF must be a vX.Y.Z release tag; got '${NIXL_REF}'" >&2; exit 1; } && \
    _nixl_ver="${NIXL_REF#v}" && \
    uv pip install --no-deps "nixl==${_nixl_ver}" "nixl-cu13==${_nixl_ver}" && \
    \
    # Record effective NIXL versions (pip vs loaded .so) for diagnosis.
    { uv pip show nixl nixl-cu13 | grep -E '^(Name|Version)'; echo "loaded_libnixl: ${LD_PRELOAD##*:}"; } | tee /opt/dynamo/nixl-versions.txt && \
    \
    # Third-party deps Dynamo wheels declare but upstream lacks. The
    # requirements.trtllm.txt file itself carries a `--no-binary imageio-ffmpeg`
    # directive that keeps the GPL-encumbered prebuilt ffmpeg off disk; IMAGEIO_FFMPEG_EXE
    # below points imageio at the in-tree LGPL CLI.
    uv pip install --no-deps --requirement /tmp/requirements.trtllm.txt && \
    \
    if [ "${ENABLE_KVBM}" = "true" ]; then \
        KVBM_WHEEL=$(ls /opt/dynamo/wheelhouse/kvbm*.whl 2>/dev/null | head -1); \
        if [ -z "$KVBM_WHEEL" ]; then \
            echo "ERROR: ENABLE_KVBM=true but no kvbm*.whl found in /opt/dynamo/wheelhouse" >&2; \
            exit 1; \
        fi; \
        uv pip install --no-deps "$KVBM_WHEEL"; \
    fi && \
    if [ "${ENABLE_GPU_MEMORY_SERVICE}" = "true" ]; then \
        GMS_WHEEL=$(ls /opt/dynamo/wheelhouse/gpu_memory_service*.whl 2>/dev/null | head -1); \
        if [ -n "$GMS_WHEEL" ]; then uv pip install --no-deps "$GMS_WHEEL"; fi; \
    fi
{% endif %}

# Copy the in-tree LGPL ffmpeg from wheel_builder. The TRT-LLM diffusion handler
# always encodes video (video_handler.py:263 → encode_to_video_bytes), so the
# CLI and its libav* / libvpx runtime libs need to be present in this image and
# imageio must be pointed at it via IMAGEIO_FFMPEG_EXE. Ungated by
# enable_media_ffmpeg because TRT-LLM unconditionally needs the encoder.
RUN --mount=type=bind,from=wheel_builder,source=/usr/local/,target=/tmp/usr/local/ \
    cp -nL /tmp/usr/local/lib/libav*.so* /usr/local/lib/ 2>/dev/null || true && \
    cp -nL /tmp/usr/local/lib/libsw*.so* /usr/local/lib/ 2>/dev/null || true && \
    cp -nL /tmp/usr/local/lib/lib*vpx*.so* /usr/local/lib/ 2>/dev/null || true && \
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

# Drop upstream's preinstalled opencv-python-headless (and the libav*/ffmpeg
# copies its wheel vendors under opencv_python_headless.libs/). TRT-LLM made
# cv2 an optional video extra (NVIDIA/TensorRT-LLM#16206) and its imports are
# already function-local in 1.3.x, so nothing in the image needs it at import
# time; video-decode users can install it explicitly per upstream docs.
# Must run via the system interpreter: with VIRTUAL_ENV set, plain `pip`
# targets the venv and exits 0 WITHOUT touching the system-site install.
# The guards fail the build if the package survives. Mirrored by the
# pre_runtime whiteout below — the squash COPY cannot represent deletions.
RUN /usr/bin/python3 -m pip uninstall -y --break-system-packages opencv-python-headless && \
    ! /usr/bin/python3 -c "import cv2" 2>/dev/null && \
    [ ! -e /usr/local/lib/python3.12/dist-packages/opencv_python_headless.libs ]

# Upgrade DALI past its own media-codec cleanup. Upstream restricted DALI's
# vendored ffmpeg build to drop the software h264/hevc/aac decoders
# (NVIDIA/DALI_deps#162, NVIDIA/DALI#6352), first released in 2.1.1; the base
# image here carries 2.1.0, which predates it.
#
# Upgrade rather than remove. The vendored libav*/libsw* set cannot be trimmed
# on its own -- every one of them is DT_NEEDED by libdali.so and its siblings, so
# deleting the codec libraries breaks `import nvidia.dali` outright -- and while
# nothing in this image imports DALI today, it belongs to TensorRT-LLM rather
# than to us. A version bump removes the codec surface without taking a package
# out of someone else's image.
#
# Measured on the shipped image: 2.1.0 registers 446 decoders including h264,
# hevc, aac, aac_fixed and aac_latm; 2.1.1 and 2.2.0 each register 440 with all
# five absent and vp8/vp9/mjpeg/av1 retained.
#
# Pinned exactly, not a floor. `>=2.1.1` resolves to whatever is newest -- 2.2.0
# at the time of writing -- which is a minor-version jump into a component this
# repo does not own, taken silently at build time. 2.1.1 is the smallest change
# that removes the decoders, so it is the one to take.
#
# A pin can go stale in the one direction that matters: if TensorRT-LLM later
# ships a base image with a newer DALI, installing 2.1.1 over it is a downgrade.
# The check below turns that into a build failure with instructions rather than a
# silent regression, which also makes this block self-retiring -- when upstream
# catches up, the build tells whoever is looking to delete it.
#
# THIS BLOCK IS TRANSITIONAL. It exists because the base image predates upstream's
# own cleanup. The guards after it and the bundled-libavcodec assertion in
# tests/dependencies/test_no_software_video_codecs.py are NOT transitional: they
# are the permanent statement of what this image may contain, and they must
# outlive the workaround.
#
# System interpreter for the same reason as the opencv removal above: with
# VIRTUAL_ENV set, plain pip targets the venv and leaves the system-site copy in
# place. The guard enumerates what the library actually registers rather than
# trusting the version string, so a wheel that reintroduces a decoder fails the
# build. Mirrored by the pre_runtime whiteout below, which matters more here than
# for a deletion: DALI's libraries are hash-named, so an upgrade RENAMES them and
# the squash COPY would otherwise leave the old codec-carrying copy in place
# beside the new one.
RUN --mount=type=bind,source=./container/compliance/enumerate_bundled_decoders.py,target=/tmp/enumerate_bundled_decoders.py \
    set -eu; \
    before=$(/usr/bin/python3 -c 'import importlib.metadata as m; print(m.version("nvidia-dali-cuda130"))'); \
    echo "DALI in base image: $before"; \
    newest=$(printf '%s\n2.1.1\n' "$before" | sort -V | tail -1); \
    if [ "$newest" != "2.1.1" ]; then \
        echo "ERROR: base image already carries DALI $before, so pinning 2.1.1 would" >&2; \
        echo "       downgrade it. Upstream has caught up -- delete this RUN and let" >&2; \
        echo "       the base version stand. Keep the guards below and the bundled-" >&2; \
        echo "       libavcodec assertion in tests/dependencies/, which are what stop" >&2; \
        echo "       this from regressing." >&2; \
        exit 1; \
    fi; \
    /usr/bin/python3 -m pip install --break-system-packages --no-cache-dir \
        --extra-index-url https://pypi.nvidia.com 'nvidia-dali-cuda130==2.1.1'; \
    v=$(/usr/bin/python3 -c 'import importlib.metadata as m; print(m.version("nvidia-dali-cuda130"))'); \
    echo "DALI version: $v"; \
    [ "$v" = "2.1.1" ] || { echo "ERROR: wanted DALI 2.1.1, got $v" >&2; exit 1; }; \
    for lib in $(find /usr/local/lib/python3.12/dist-packages/nvidia/dali/.libs -name 'libavcodec*.so*'); do \
        /usr/bin/python3 /tmp/enumerate_bundled_decoders.py "$lib"; \
    done; \
    /usr/bin/python3 -c 'import nvidia.dali'

# Upgrade PyNvVideoCodec past the release that stopped shipping a separate
# libavcodec. rc24 is the first TensorRT-LLM base image to ship this package at
# all, and it ships 2.1.0, which bundles libavcodec, libavdevice, libavfilter,
# libswresample and libswscale alongside the libavformat and libavutil it
# actually uses. 2.2.0 ships only the latter two -- but libavcodec is not gone,
# it is statically linked into libavformat.so.62. That distinction drives the
# content check below: after the upgrade no file is named libavcodec*, so a
# filename test cannot see a future release that re-enables decoders inside
# libavformat.
#
# This is a surface reduction, not a codec removal, and the difference from the
# DALI block above matters because the two look identical. 2.1.0's libavcodec
# registers exactly one decoder -- vp9 -- and no encoders: none of h264, hevc,
# aac, aac_fixed or aac_latm. That is by construction, not by luck: the vendor's
# own configure line ships in the image at /usr/local/external/ffmpeg/readme.txt
# and reads --disable-encoders --disable-decoders --enable-decoder=vp9. Measured
# with enumerate_bundled_decoders.py on both arches, and 2.2.0 drops even vp9.
#
# So no prohibited *decoder implementation* ships in either version -- but that
# is a narrower statement than "nothing prohibited ships". --disable-decoders
# disables decoders, not FFmpeg code in general: the h264/hevc/aac parsers,
# demuxers, muxers and bitstream filters (h264_metadata, hevc_metadata,
# h264_mp4toannexb, h264_redundant_pps -- these do full parameter-set parse and
# rewrite) remain enabled in both. Do not read the vp9-only finding as "no
# H.264 code present".
#
# What actually fails the build is container/compliance/policy/codec_policy.yaml,
# whose deny globs match on a libavcodec being present at all, and whose two
# PyNvVideoCodec waivers are deliberately scoped to libavutil and libavformat so
# that a bundled libavcodec has to be looked at rather than absorbed silently.
#
# The version of those libraries is the main reason to move, and it is easy to
# miss because the package ships no ffmpeg executable -- only shared libraries.
# 2.1.0's libavcodec.so.61.3.100 and libavformat.so.61.1.100 embed "FFmpeg
# version 7.0.2", and they are on the runtime path rather than inert: readelf -d
# on PyNvVideoCodec_130.cpython-312-x86_64-linux-gnu.so lists libavformat.so.61,
# libavcodec.so.61, libswresample.so.5 and libavutil.so.59 as NEEDED. 2.2.0's
# libavformat.so.62.12.102 embeds "FFmpeg version 8.1.2", which is the floor
# deny_components sets for ffmpeg in codec_policy.yaml. 2.1.0 sits below it.
#
# Note that the gate did not tell us this. deny_components is evaluated against
# SBOM components (scan_codecs.scan_sbom), and the SBOM does not enumerate
# wheel-bundled .so files as an "ffmpeg" component -- the rc24 failure produced
# zero sbom: findings, and all 20 violations were filename globs. So the floor
# is unenforced for every wheel-vendored FFmpeg in every image, not just this
# one. Worth fixing in the scanner; do not assume this check has your back.
#
# Two further defects in 2.1.0 that 2.2.0 fixes: GPL-only libpostproc sources
# while NOTICES.txt declares LGPLv3 only, and on aarch64 binaries built from
# FFmpeg 7.1 shipped against a 7.0.2 source tarball -- an LGPL
# source-correspondence mismatch. Package size drops 52M -> 24M (2.1.0's
# bundled libs are real copies, not symlinks).
#
# Do NOT justify this upgrade as removing royalty exposure: the only decoder it
# removes is vp9, the royalty-free one, and royalties for the GPU-accelerated
# families are covered by the hardware licensing model.
#
# Upgrade rather than waive. A waiver has to be written as a glob, and a glob
# cannot say "this version's libavcodec is empty" -- it would carry forward to a
# future version whose libavcodec is not, disarming the check that just fired.
# Removing the file keeps the narrow waivers, and the tripwire, intact.
#
# A floor rather than an exact pin, unlike DALI above: requirements.trtllm.txt
# already installs PyNvVideoCodec>=2.2.0 into /opt/dynamo/venv, and pinning the
# system copy exactly would let the two diverge inside one image. Keep this
# specifier identical to the one there. Two assertions follow: a filename test
# that mirrors the codec gate's deny glob (so a reintroduced libavcodec fails
# here, with a clear message, rather than later in the scan), and a content test
# that enumerates what the shipped FFmpeg libraries actually register.
#
# Reading the content test's output: for 2.2.0 it reports "0 decoders, 0
# encoders" and prints every entry of its "expected present (sanity check)"
# block as ABSENT. That is the correct answer here, not a broken probe -- 2.2.0
# registers no codecs at all, vp9 included. The probe is proved live a different
# way: enumerate_bundled_decoders.py resolves av_codec_iterate through ctypes,
# so a library that does not export it raises AttributeError and fails the
# build rather than reporting a reassuring zero. The check is fail-closed by
# design; if a future release hides that symbol, this stops the build and wants
# a human, which is the right outcome for a check whose whole job is to see
# inside the file.
#
# This deliberately overrides a dependency of TensorRT-LLM's, which is why the
# build log carries a pip resolver complaint here:
#
#     tensorrt-llm 1.3.0rc24 requires PyNvVideoCodec~=2.1.0,
#     but you have pynvvideocodec 2.2.0 which is incompatible
#
# The complaint is expected and the override is deliberate. `~=2.1.0` excludes
# 2.2.0 by construction, and 2.1.0 is the version that bundles the libavcodec.
# tensorrt_llm/media/decoding.py is the only consumer; it uses CreateDemuxer,
# CreateDecoder, PyNvVCException and OutputColorType, all of which 2.2.0 still
# exports, and it imports PyNvVideoCodec function-locally so `import
# tensorrt_llm` does not touch it. Re-check that list when this base image moves:
# if upstream relaxes the pin to allow 2.2.0, this note is the thing to delete.
#
# System interpreter for the same reason as the opencv removal and the DALI
# upgrade above: with VIRTUAL_ENV set, plain pip targets the venv and leaves the
# system-site copy -- the one the scan reads -- in place.
#
# No `import PyNvVideoCodec` smoke test here, deliberately. Unlike nvidia.dali it
# dlopens libnvcuvid and needs NVIDIA_DRIVER_CAPABILITIES to include "video",
# which the builder does not have, so an import check would fail every build.
RUN --mount=type=bind,source=./container/compliance/enumerate_bundled_decoders.py,target=/tmp/enumerate_bundled_decoders.py \
    set -eu; \
    before=$(/usr/bin/python3 -c 'import importlib.metadata as m; print(m.version("pynvvideocodec"))' 2>/dev/null || echo none); \
    echo "PyNvVideoCodec in base image: $before"; \
    if [ "$before" = "none" ]; then \
        echo "ERROR: base image no longer ships PyNvVideoCodec, so this upgrade has" >&2; \
        echo "       nothing to act on -- delete this RUN and the whiteout entry that" >&2; \
        echo "       pairs with it. The venv copy installed from" >&2; \
        echo "       requirements.trtllm.txt is the one that matters." >&2; \
        exit 1; \
    fi; \
    newest=$(printf '%s\n2.2.0\n' "$before" | sort -V | tail -1); \
    if [ "$newest" != "2.2.0" ]; then \
        echo "ERROR: base image already carries PyNvVideoCodec $before, so this block" >&2; \
        echo "       is obsolete -- delete it and let the base version stand. Keep the" >&2; \
        echo "       libavcodec assertion below and the post-overlay one in" >&2; \
        echo "       pre_runtime, which are what stop this from regressing." >&2; \
        exit 1; \
    fi; \
    /usr/bin/python3 -m pip install --break-system-packages --no-cache-dir \
        'PyNvVideoCodec>=2.2.0'; \
    v=$(/usr/bin/python3 -c 'import importlib.metadata as m; print(m.version("pynvvideocodec"))'); \
    echo "PyNvVideoCodec version: $v"; \
    [ "$(printf '%s\n2.2.0\n' "$v" | sort -V | head -1)" = "2.2.0" ] \
        || { echo "ERROR: wanted PyNvVideoCodec >= 2.2.0, got $v" >&2; exit 1; }; \
    if find /usr/local/lib/python3.12/dist-packages/PyNvVideoCodec -name 'libavcodec*' | grep -q .; then \
        echo "ERROR: PyNvVideoCodec $v still bundles a libavcodec:" >&2; \
        find /usr/local/lib/python3.12/dist-packages/PyNvVideoCodec -name 'libavcodec*' >&2; \
        exit 1; \
    fi; \
    examined=0; \
    for lib in $(find /usr/local/lib/python3.12/dist-packages/PyNvVideoCodec \
            -name 'libavcodec*.so*' -o -name 'libavformat*.so*'); do \
        examined=$((examined + 1)); \
        /usr/bin/python3 /tmp/enumerate_bundled_decoders.py "$lib"; \
    done; \
    [ "$examined" -gt 0 ] \
        || { echo "ERROR: found no FFmpeg libraries under PyNvVideoCodec to examine;" >&2; \
             echo "       the package layout changed and this check went blind." >&2; exit 1; }; \
    if [ -d /opt/dynamo/venv ]; then \
        vvd=$(find /opt/dynamo/venv/lib/python3*/site-packages -maxdepth 1 \
                -name 'pynvvideocodec-*.dist-info' 2>/dev/null | head -1); \
        vv=$(basename "${vvd:-}" .dist-info); vv=${vv#pynvvideocodec-}; \
        echo "PyNvVideoCodec in /opt/dynamo/venv: ${vv:-none}"; \
        [ -n "$vv" ] \
            || { echo "ERROR: /opt/dynamo/venv exists but holds no PyNvVideoCodec;" >&2; \
                 echo "       requirements.trtllm.txt should have installed one. Read the" >&2; \
                 echo "       dist-info directly, not via the venv interpreter: the venv is" >&2; \
                 echo "       --system-site-packages, so the interpreter would resolve the" >&2; \
                 echo "       system copy this stage just upgraded and always look correct." >&2; \
                 exit 1; }; \
        [ "$(printf '%s\n2.2.0\n' "$vv" | sort -V | head -1)" = "2.2.0" ] \
            || { echo "ERROR: venv PyNvVideoCodec $vv is below the 2.2.0 floor while the" >&2; \
                 echo "       system copy is $v -- the two specifiers have drifted apart." >&2; \
                 echo "       requirements.trtllm.txt and this stage must stay in step." >&2; exit 1; }; \
    else \
        echo "NOTE: no /opt/dynamo/venv in this stage yet, so the venv/system"; \
        echo "      cross-check is skipped. Expected for the dev and local-dev"; \
        echo "      targets, whose runtime_full does not build the venv -- they"; \
        echo "      create it later, after this stage, with --system-site-packages"; \
        echo "      over the system copy asserted above."; \
    fi

# Align the base image's aiohttp with the floor requirements.common.txt sets for
# the other Dynamo images. The base ships an older release and the --no-deps
# installs above deliberately leave upstream's solve alone, so this one package is
# raised explicitly. Narrow by design: a single named package, not a re-solve.
#
# System interpreter for the same reason as the opencv removal and the DALI
# upgrade above: with VIRTUAL_ENV set, plain pip targets /opt/dynamo/venv and
# leaves the system-site copy in place. The venv is --system-site-packages, so a
# venv-only install would shadow the old version at import time while leaving it
# on disk under dist-packages, where the image inventory reads it. The guard
# checks the system interpreter specifically, so a venv-only install fails here.
#
# Mirrored by the pre_runtime whiteout below. An upgrade renames the metadata
# directory (aiohttp-3.13.5.dist-info -> aiohttp-3.14.3.dist-info) and the squash
# COPY only replaces paths that match, so without dropping the old tree first the
# base image's dist-info would ship alongside the new one.
#
# Checked against the base's own constraints (datasets, fsspec), which cap nothing
# here. cp312 manylinux_2_28 wheels exist for x86_64 and aarch64, so neither
# architecture builds from source.
# The guard asserts on dist-info directories, not on an importable version,
# because those directories are what an image inventory reads and they are what
# a venv-only install or a missed whiteout would leave wrong. It requires exactly
# one, in range, so a surviving old copy beside the new one fails the build.
# Version-compared rather than glob-matched: a glob range silently stops matching
# at the end of the series it was written for.
RUN /usr/bin/python3 -m pip install --break-system-packages --upgrade "aiohttp>=3.14.3,<4.0" && \
    /usr/bin/python3 -c 'import glob, os, sys; d = glob.glob("/usr/local/lib/python3.12/dist-packages/aiohttp-*.dist-info"); vs = [os.path.basename(p)[8:-10] for p in d]; print("aiohttp dist-info in system site:", vs); tv = lambda s: tuple(int(x) for x in s.split(".")[:3]); sys.exit(0 if len(vs) == 1 and (3, 14, 3) <= tv(vs[0]) < (4, 0, 0) else 1)'

# Pull /workspace_src (incl. LICENSE) from the transport stage and
# wire up the launch screen in a single RUN — saves the standalone workspace COPY layer.
RUN --mount=type=bind,from=workspace_files,source=/workspace_src,target=/tmp/workspace_src \
    --mount=type=bind,source=./container/launch_message/runtime.txt,target=/opt/dynamo/launch_message.txt \
    cp -a /tmp/workspace_src/. /workspace/ && \
    chown -R dynamo:0 /workspace && \
    sed '/^#\s/d' /opt/dynamo/launch_message.txt > /opt/dynamo/.launch_screen && \
    chmod 755 /opt/dynamo/.launch_screen && \
    echo 'cat /opt/dynamo/.launch_screen' >> /etc/bash.bashrc

USER dynamo

# Kept at the bottom — SHA changes per build; layers above stay cached.
ARG DYNAMO_COMMIT_SHA
ENV DYNAMO_COMMIT_SHA=${DYNAMO_COMMIT_SHA}

# Reset upstream TRT-LLM image's entrypoint so derived runtimes behave like
# other Dynamo images and can execute arbitrary commands directly.
ENTRYPOINT []
CMD ["/bin/bash"]

{% if target in ("runtime", "dev", "local-dev") %}
# Rebase on upstream so this stage inherits upstream's image config
# (ENV/WORKDIR/USER/CMD) and then overlay runtime_full's filesystem as a
# single layer. Only Dynamo-specific env needs redeclaring below.
FROM ${RUNTIME_IMAGE}:${RUNTIME_IMAGE_TAG} AS pre_runtime
# Whiteout paths runtime_full removed — COPY can't represent deletions, so
# without this, upstream's /workspace, /home/ubuntu, standalone
# /usr/local/bin/etcd* tools, and preinstalled opencv (cv2/ + vendored
# opencv_python_headless.libs/ + dist-info) would leak alongside our content.
# Keep this list in sync with any deletion RUNs in the stages above.
#
# DALI is here for a different reason: runtime_full UPGRADES it rather than
# removing it, and its vendored libraries are hash-named
# (libavcodec-847b0373.so.62 -> libavcodec-7e11e4e3.so.62). An overlay COPY only
# replaces paths that match, so without dropping the old tree first, the base
# image's 2.1.0 libraries — the ones carrying h264/hevc/aac — would ship
# alongside the upgraded ones and the upgrade would buy nothing.
#
# aiohttp is here for the DALI reason, not the opencv one: runtime_full upgrades
# it in system site, and the version-stamped metadata directory is renamed by the
# upgrade (aiohttp-3.13.5.dist-info -> aiohttp-3.14.3.dist-info). The overlay COPY
# would leave the base image's dist-info beside the new one, and the inventory
# reads that directory, so the upgrade would not show.
#
# PyNvVideoCodec is here for the DALI reason too, and it is the case the codec
# scan actually catches: the scan runs on this stage (compliance_base_stage is
# pre_runtime), not on runtime_full. runtime_full UPGRADES the package, and an
# upgrade is a deletion plus an install -- 2.1.0's libavcodec/libavdevice/
# libavfilter/libswresample/libswscale have no counterpart in 2.2.0, so the
# overlay COPY has nothing to write over them and the base image's copies would
# come straight back. Without this entry the upgrade buys nothing and the scan
# fails exactly as it did before. The version-stamped dist-info is renamed by the
# upgrade (pynvvideocodec-2.1.0.dist-info -> -2.2.0.dist-info), so the glob takes
# that and the stray `pynvvideocodec.` entry beside it.
#
# /usr/local/external/ffmpeg is the same package, in a place that is easy to
# miss: PyNvVideoCodec's wheel installs an FFmpeg source tarball through a
# `../../../external/ffmpeg/src/ffmpeg-<version>.tar.xz` RECORD entry, so it
# lands OUTSIDE dist-packages and the two globs above do not reach it. It is
# version-stamped, so this is the aiohttp rename problem again rather than the
# opencv one: runtime_full's pip upgrade removes ffmpeg-7.0.2.tar.xz and writes
# ffmpeg-8.1.2.tar.xz, the overlay COPY cannot express that deletion, and the
# image would ship BOTH.
#
# Keep the scope of that honest: a source tarball is inert, nothing on the
# runtime path reads it, and the exposure discussed above is in the shipped .so
# files, not here. This entry buys ~10.8 MB and a tree that matches what
# actually ships -- a stale 7.0.2 source sitting beside 8.1.2 binaries invites a
# future reader to conclude the wrong thing about either. The codec scan cannot
# arbitrate: its deny globs match libav*.so*/libsw*.so*, never a .tar.xz, and
# allow_paths lists /usr/local/src/ffmpeg (our in-tree build), not this path.
#
# Delete the directory rather than a single file, and note that the tarball is
# not junk: NOTICES.txt declares LGPLv3 for the bundled FFmpeg, so the source is
# the corresponding-source obligation for the .so files the extension links
# against. That is why the overlay must restore runtime_full's copy -- the point
# is to ship exactly one tarball, the one matching the shipped binaries, not to
# ship none.
#
# Its runtime dependencies are listed for the same reason. pip upgrades those too
# when the new aiohttp requires versions the base does not carry, and each rename
# has the same doubling problem. Dropping them here is free: the overlay restores
# whatever runtime_full holds, so listing one that did not move costs nothing and
# omitting one that did would leave stale metadata behind.
RUN rm -rf /workspace /home/ubuntu \
    /usr/local/bin/etcd \
    /usr/local/bin/etcdctl \
    /usr/local/bin/etcdutl \
    /usr/local/bin/wandb \
    /usr/local/lib/python3.12/dist-packages/wandb \
    /usr/local/lib/python3.12/dist-packages/wandb-* \
    /usr/local/lib/python3.12/dist-packages/cv2 \
    /usr/local/lib/python3.12/dist-packages/opencv_python_headless* \
    /usr/local/lib/python3.12/dist-packages/nvidia/dali \
    /usr/local/lib/python3.12/dist-packages/nvidia_dali_* \
    /usr/local/lib/python3.12/dist-packages/PyNvVideoCodec \
    /usr/local/lib/python3.12/dist-packages/pynvvideocodec* \
    /usr/local/external/ffmpeg \
    /usr/local/lib/python3.12/dist-packages/aiohttp \
    /usr/local/lib/python3.12/dist-packages/aiohttp-* \
    /usr/local/lib/python3.12/dist-packages/multidict \
    /usr/local/lib/python3.12/dist-packages/multidict-* \
    /usr/local/lib/python3.12/dist-packages/yarl \
    /usr/local/lib/python3.12/dist-packages/yarl-* \
    /usr/local/lib/python3.12/dist-packages/propcache \
    /usr/local/lib/python3.12/dist-packages/propcache-* \
    /usr/local/lib/python3.12/dist-packages/frozenlist \
    /usr/local/lib/python3.12/dist-packages/frozenlist-* \
    /usr/local/lib/python3.12/dist-packages/aiosignal \
    /usr/local/lib/python3.12/dist-packages/aiosignal-* \
    /usr/local/lib/python3.12/dist-packages/aiohappyeyeballs \
    /usr/local/lib/python3.12/dist-packages/aiohappyeyeballs-* \
    /usr/local/lib/python3.12/dist-packages/attr \
    /usr/local/lib/python3.12/dist-packages/attrs \
    /usr/local/lib/python3.12/dist-packages/attrs-* && \
    ! /usr/bin/python3 -c "import cv2" 2>/dev/null && \
    ! /usr/bin/python3 -c "import wandb" 2>/dev/null
COPY --from=runtime_full / /

# Post-overlay guard for the DALI whiteout above. This is the only stage where
# the failure can appear: runtime_full holds exactly one DALI, and the whiteout
# is what keeps the overlay from re-adding the base image's copy beside it.
# Because those libraries are hash-named, a missed whiteout leaves TWO -- the
# upgraded one and the 2.1.0 one that registers h264/hevc/aac -- and the image
# then reports version 2.1.0 while carrying both. Verified by building this
# stage with and without the rm -rf: without it, two copies and the codecs are
# back; with it, one copy and clean.
#
# Every match is checked, not just the first: the whole point is catching the
# case where more than one exists.
RUN --mount=type=bind,source=./container/compliance/enumerate_bundled_decoders.py,target=/tmp/enumerate_bundled_decoders.py \
    set -eu; \
    for lib in $(find /usr/local/lib/python3.12/dist-packages/nvidia/dali/.libs -name 'libavcodec*.so*' 2>/dev/null); do \
        /usr/bin/python3 /tmp/enumerate_bundled_decoders.py "$lib"; \
    done; \
    if find /usr/local/lib/python3.12/dist-packages/PyNvVideoCodec -name 'libavcodec*' 2>/dev/null | grep -q .; then \
        echo "ERROR: post-overlay PyNvVideoCodec carries a libavcodec, so the base" >&2; \
        echo "       image's 2.1.0 copy survived the overlay -- its whiteout entry is" >&2; \
        echo "       missing or misspelled. The upgrade in runtime_full cannot fix this" >&2; \
        echo "       on its own: COPY cannot represent a deletion." >&2; \
        find /usr/local/lib/python3.12/dist-packages/PyNvVideoCodec -name 'libavcodec*' >&2; \
        exit 1; \
    fi; \
    for lib in $(find /usr/local/lib/python3.12/dist-packages/PyNvVideoCodec \
            -name 'libavcodec*.so*' -o -name 'libavformat*.so*' 2>/dev/null); do \
        /usr/bin/python3 /tmp/enumerate_bundled_decoders.py "$lib"; \
    done; \
    tarballs=$(find /usr/local/external/ffmpeg -name 'ffmpeg-*.tar.*' 2>/dev/null | wc -l); \
    if [ "$tarballs" -gt 1 ]; then \
        echo "ERROR: more than one FFmpeg source tarball survived the overlay, so the" >&2; \
        echo "       base image's copy came back alongside the upgraded one -- the" >&2; \
        echo "       /usr/local/external/ffmpeg whiteout entry is missing." >&2; \
        find /usr/local/external/ffmpeg -name 'ffmpeg-*.tar.*' >&2; \
        exit 1; \
    fi

# The Nsight Systems CLI in the CUDA base ships an optional efa_metrics sampler
# for EFA network counters. Nothing in the Dynamo tree references it, and AWS
# EFA confirmed on 2026-08-18 that they do not use it. It is a static Go binary,
# so it carries its own copy of the Go standard library and is the sole carrier
# for 17 of this image's Critical and High findings.
#
# Removed after the overlay rather than in the rm -rf above, for the same reason
# that block documents in the other direction: COPY --from=runtime_full / /
# reinstates whatever the base holds, so a pre-overlay whiteout here would just
# be undone. The glob spans the CUDA version, the Nsight version and both target
# architectures, all of which move with the base image.
#
# The guard is the point of the change, not decoration: a glob that silently
# matches nothing after a base bump would leave the finding in place while the
# build stayed green.
RUN rm -rf /usr/local/cuda-*/NsightSystems-cli-*/target-linux-*/plugins/efa_metrics; \
    if find /usr/local -type d -name efa_metrics 2>/dev/null | grep -q .; then \
        echo "ERROR: an efa_metrics plugin directory survived removal; the glob no" >&2; \
        echo "       longer matches the base image's Nsight layout." >&2; \
        find /usr/local -type d -name efa_metrics >&2; \
        exit 1; \
    fi

# Mirrors runtime_full's ENV — must stay in sync. Re-declaration is required
# because `FROM ${RUNTIME_IMAGE}` here does not inherit runtime_full's config.
# dev/local-dev create their own venv in a later stage, so the venv ENV is left
# out for them — keeps this config identical to the unsquashed dev path.
{% if target in ("dev", "local-dev") %}
ENV DYNAMO_HOME=/workspace \
    HOME=/home/dynamo \
    PATH=/opt/uv/bin:/usr/local/bin/etcd:${PATH} \
    IMAGEIO_FFMPEG_EXE=/usr/local/bin/ffmpeg \
    LD_PRELOAD=/opt/dynamo/libstdc++.so.6:/usr/local/lib/python3.12/dist-packages/tensorrt_llm/libs/nixl/libnixl.so \
    NIXL_PLUGIN_DIR=/usr/local/lib/python3.12/dist-packages/tensorrt_llm/libs/nixl/plugins \
    NIXL_VERSION=
{% else %}
ENV DYNAMO_HOME=/workspace \
    HOME=/home/dynamo \
    VIRTUAL_ENV=/opt/dynamo/venv \
    PATH=/opt/dynamo/venv/bin:/opt/uv/bin:/usr/local/bin/etcd:${PATH} \
    IMAGEIO_FFMPEG_EXE=/usr/local/bin/ffmpeg \
    LD_PRELOAD=/opt/dynamo/libstdc++.so.6:/usr/local/lib/python3.12/dist-packages/tensorrt_llm/libs/nixl/libnixl.so \
    NIXL_PLUGIN_DIR=/usr/local/lib/python3.12/dist-packages/tensorrt_llm/libs/nixl/plugins \
    NIXL_VERSION=
{% endif %}

WORKDIR /workspace

ARG DYNAMO_COMMIT_SHA
ENV DYNAMO_COMMIT_SHA=${DYNAMO_COMMIT_SHA}

USER dynamo

ENTRYPOINT []
CMD ["/bin/bash"]
{% endif %}


{# Compliance is skipped for dev/local-dev: those images are not shipped (release
   ships runtime/frontend/operator/planner/snapshot-agent), compliance-extract
   already skips them, and their pre_runtime carries no dynamo venv to scan. #}
{% if target not in ("dev", "local-dev") %}
{% include "templates/compliance.Dockerfile" %}
{% endif %}


FROM pre_runtime AS runtime
# NVDEC (PyNvVideoCodec, added in requirements.trtllm.txt) needs the driver
# "video" capability at runtime so libnvcuvid is exposed; without it
# PyNvVideoCodec cannot import. Ensure the K8s pod/runtimeClass does not drop it.
ENV NVIDIA_DRIVER_CAPABILITIES=video,compute,utility
{% if target not in ("dev", "local-dev") %}
COPY --from=licenses /legal /legal
{% endif %}
