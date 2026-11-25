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

FROM ${RUNTIME_IMAGE}:${RUNTIME_IMAGE_TAG} AS pre_runtime

ARG ENABLE_GPU_MEMORY_SERVICE
ARG TARGETARCH

# Upstream uses most of overlay2's ~128-layer cap and dev/local-dev stack on
# this stage: fold new steps into an existing RUN instead of adding one.

# Setup that must precede the ENV below:
#
# - Purge upstream's git-lfs package and the nasm assembler without pruning
#   shared dependencies. Nothing installed depends on nasm; the build fails if
#   its purge would remove anything else.
#
# - Link system libstdc++ to a stable path for LD_PRELOAD. This keeps
#   PyInstaller-bundled tools (e.g. NVIDIA's `jet` CI runner) from shadowing it
#   with an older copy. It must exist before the ENV names it, or every later
#   process logs an `ld.so ... cannot be preloaded` line.
#
# - Select the Open MPI the worker runs on at /opt/dynamo/mpi.
#   amd64 uses ompi4. Since 1.3.0rc27 upstream points /opt/hpcx/ompi at ompi5,
#   where TRT-LLM's MPI.COMM_SELF.Spawn (mpi4py MpiPoolSession) dies at the
#   barrier with `ucp_ep_create ... Destination is unreachable`, so the worker
#   never binds its gRPC port. arm64 keeps ompi5: ompi4 there lacks
#   ompi_mpi_short_float, which the base's libtorch_cpu.so needs.
#   The symlink alone is not enough; the PATH, LD_LIBRARY_PATH and OPAL_PREFIX
#   entries in the ENV below are what make ompi4 launch its own runtime.
#
# - Apply the base image's two MCA edits to the selected copy (upstream edits
#   only the Open MPI it selects itself):
#   `hwloc_base_binding_policy = core` -> `none`: `import tensorrt_llm` runs
#   MPI_Init, which otherwise binds the process and all its children to CPUs 0-1.
#   `btl = self` commented out: it leaves the ob1 PML no transport between ranks,
#   and the multi-node launch path passes `--mca pml ob1`
#   (deploy/operator/internal/dynamo/backend_trtllm.go).
#   The edits are idempotent and the build fails if either setting survives.
#
# - ompi5 spawn defects, relevant where ompi5 is selected (arm64):
#   1. A singleton cannot spawn when hostname length + pid digits exceed 37: the
#      name passes through a 50-byte buffer to prte and is truncated. CI pod names
#      are longer. A .pth hook (container/deps/trtllm/_dynamo_pmix_hostname.py)
#      sets a short PMIX_HOSTNAME only for processes no MPI launcher started, on
#      hosts with names over 30 characters, and never overrides an existing value.
#      .dockerignore drops *.pth files, so the RUN writes the line itself.
#   2. UCX may pick another host interface (e.g. a k3s flannel address) and fail;
#      a bridge network or OMPI_MCA_pml=ob1 avoids it.
#
# Transitional; re-measure rather than assume:
# - The hook: fixed by open-mpi/ompi#14398 (v5.0.x: #14409; Mellanox/ompi#61),
#   not yet in a release HPC-X. When /opt/hpcx/VERSION names a newer HPC-X, run a
#   spawn with a 46-character hostname without the hook; if it passes, delete the
#   hook, its install step and its tests. Tracked in NVIDIA/TensorRT-LLM#19607.
# - The amd64 ompi4 selection and its ENV entries: remove only once ompi5 spawns
#   on amd64 without the hook, in CI and on a host with extra interfaces.
#
# tests/dependencies/test_trtllm_mpi.py checks the link per architecture, the
# library mpi4py loads, a spawn, a two-rank ob1 launch, CPU affinity after
# MPI_Init and the hook. container/dev/50-framework-paths.sh prefers the same link.
RUN --mount=type=bind,source=./container/deps/trtllm/_dynamo_pmix_hostname.py,target=/tmp/_dynamo_pmix_hostname.py \
    purged=$(apt-get -s purge nasm | awk '/^(Remv|Purg) /{print $2}') && \
    { [ "$purged" = nasm ] || { echo "ERROR: purging nasm would remove: ${purged:-nothing}" >&2; exit 1; }; } && \
    apt-get purge -y git-lfs nasm && rm -rf /var/lib/apt/lists/* && \
    test ! -e /usr/bin/git-lfs && \
    test ! -e /usr/local/bin/git-lfs && \
    ! command -v git-lfs && \
    status=$(dpkg-query -W -f='${db:Status-Status}' git-lfs 2>/dev/null || true) && \
    test "$status" != installed && \
    set -- /var/lib/dpkg/info/git-lfs.* && test ! -e "$1" && \
    test ! -e /usr/bin/nasm && \
    test ! -e /usr/bin/ndisasm && \
    ! command -v nasm && \
    status=$(dpkg-query -W -f='${db:Status-Status}' nasm 2>/dev/null || true) && \
    test "$status" != installed && \
    set -- /var/lib/dpkg/info/nasm.* && test ! -e "$1" && \
    ARCH_ALT=$([ "${TARGETARCH}" = "amd64" ] && echo "x86_64" || echo "aarch64") && \
    mkdir -p /opt/dynamo && \
    LIBSTDCPP=/usr/lib/${ARCH_ALT}-linux-gnu/libstdc++.so.6 && \
    test -f "$LIBSTDCPP" && ln -sf "$LIBSTDCPP" /opt/dynamo/libstdc++.so.6 && \
    if [ "${TARGETARCH}" = "amd64" ]; then t=/opt/hpcx/ompi4; else t=/opt/hpcx/ompi5; fi && \
    test -d "$t" && \
    conf="$t/etc/openmpi-mca-params.conf" && \
    test -f "$conf" && \
    sed -i -e 's/^\(hwloc_base_binding_policy\) = core$/\1 = none/' \
           -e 's/^\(btl = self\)$/#\1/' "$conf" && \
    ! grep -qE '^[[:space:]]*hwloc_base_binding_policy[[:space:]]*=[[:space:]]*core' "$conf" && \
    ! grep -qE '^[[:space:]]*btl[[:space:]]*=[[:space:]]*self[[:space:]]*$' "$conf" && \
    if [ "$t" = /opt/hpcx/ompi5 ]; then \
        site=/usr/local/lib/python3.12/dist-packages && \
        test -d "$site" && \
        cp /tmp/_dynamo_pmix_hostname.py "$site/" && \
        echo 'import _dynamo_pmix_hostname' > "$site/dynamo-pmix-hostname.pth"; \
    fi && \
    mkdir -p /opt/dynamo && ln -sfn "$t" /opt/dynamo/mpi && \
    echo "MPI for ${TARGETARCH}: $(readlink -f /opt/dynamo/mpi)"

# Reuse upstream's native NIXL and matching Python bindings. Only libstdc++
# needs preloading; loading NIXL into every process adds CI startup overhead.
# Clear upstream's stale NIXL_VERSION; record installed metadata below.
ENV DYNAMO_HOME=/workspace \
    HOME=/home/dynamo \
    PATH=/opt/dynamo/mpi/bin:/usr/local/bin/etcd:${PATH} \
    LD_PRELOAD=/opt/dynamo/libstdc++.so.6 \
    LD_LIBRARY_PATH=/opt/dynamo/mpi/lib:${LD_LIBRARY_PATH} \
    OPAL_PREFIX=/opt/dynamo/mpi \
    NIXL_VERSION=

# OS-level changes:
#
# - Install packages missing from upstream and register TRT-LLM lib paths with
#   ldconfig (upstream's /etc/shinit_v2 sets them only for shells, not for K8s
#   python3 launches). Remove upstream's standalone etcd/etcdctl/etcdutl so the
#   image carries only dynamo_base's copies, and drop the unused wandb tooling
#   (upstream removes it on main, Dockerfile.multi).
#
# - Upgrade already-installed OS packages to the current patch releases.
#   --only-upgrade adds no packages; versions are unpinned so a cache-busted
#   rebuild picks up the newest patch level.
#
# - Restore Triton's default CUDA header/tool paths for SSH-launched ranks that
#   lack the image ENV (GH-14864). Per-file links inside real directories, so
#   later wheel installs cannot overwrite the CUDA toolkit through a directory
#   link. The JIT check runs with path overrides unset, a fresh cache and the
#   CUDA driver stub, so the build needs no GPU.
#
# - Remove Nsight's optional efa_metrics sampler. Nothing in Dynamo uses it (AWS
#   EFA confirmed on 2026-08-18), and as a static Go binary it is the sole
#   carrier of 17 of the image's Critical and High findings. The globs span CUDA
#   version, Nsight version and architecture, and cover both the CUDA toolkit's
#   Nsight Systems and the /opt/nvidia Nsight Systems and Compute installs. The
#   guard fails the build if any efa_metrics directory survives under /usr/local
#   or /opt, so a base bump that moves the plugin cannot ship it silently.
#   The same cleanup removes the pip in the Nsight Systems CLI's own Python and
#   the onnx.egg-info that the base's PyTorch source tree keeps from its build.
#   Nothing in the image runs either, and a second guard fails the build if
#   either survives.
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    set -eu; \
    apt-get update; \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        openssh-server \
        libturbojpeg \
        librdmacm1 \
        rdma-core \
        libjemalloc2; \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends --only-upgrade \
        apt \
        apt-utils \
        ca-certificates \
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
        libapt-pkg6.0t64 \
        libbrotli1 \
        libglib2.0-0t64 \
        libgnutls30t64 \
        libperl5.38t64 \
        libssl3t64 \
        openssh-client \
        openssh-server \
        openssh-sftp-server \
        openssl \
        perl \
        perl-base \
        perl-modules-5.38 \
        zlib1g \
        zlib1g-dev; \
    ARCH_ALT=$([ "${TARGETARCH}" = "amd64" ] && echo "x86_64" || echo "aarch64"); \
    printf '%s\n' \
        "/usr/local/tensorrt/lib" \
        "/usr/local/cuda/lib64" \
        "/usr/local/ucx/lib" \
        "/opt/nvidia/nvda_nixl/lib/${ARCH_ALT}-linux-gnu" \
        "/opt/nvidia/nvda_nixl/lib64" \
        > /etc/ld.so.conf.d/00-dynamo-trtllm.conf; \
    ldconfig; \
    ldconfig -p | grep -q 'libturbojpeg.so.0'; \
    rm -f \
        /usr/local/bin/etcd \
        /usr/local/bin/etcdctl \
        /usr/local/bin/etcdutl; \
    /usr/bin/python3 -m pip uninstall -y --break-system-packages wandb; \
    if /usr/bin/python3 -c "import wandb" 2>/dev/null; then echo "ERROR: wandb still importable" >&2; exit 1; fi; \
    [ ! -e /usr/local/lib/python3.12/dist-packages/wandb ]; \
    tb=$(/usr/bin/python3 -c 'import os, triton.backends.nvidia as b; print(os.path.dirname(b.__file__))'); \
    echo "triton nvidia backend: $tb"; \
    mkdir -p "$tb/include" "$tb/bin"; \
    for hdr in cuda.h; do \
        p="$tb/include/$hdr"; \
        [ -e "$p" ] || [ -L "$p" ] || ln -s "/usr/local/cuda/include/$hdr" "$p"; \
        test -f "$p"; \
    done; \
    for tool in ptxas cuobjdump nvdisasm; do \
        p="$tb/bin/$tool"; \
        [ -e "$p" ] || [ -L "$p" ] || ln -s "/usr/local/cuda/bin/$tool" "$p"; \
        test -x "$p"; \
    done; \
    chk=/tmp/dynamo-triton-check; \
    mkdir -p "$chk/stubs" && ln -sf /usr/local/cuda/lib64/stubs/libcuda.so "$chk/stubs/libcuda.so.1"; \
    env -u TRITON_CUDACRT_PATH -u TRITON_CUDART_PATH -u TRITON_PTXAS_PATH -u TRITON_CUOBJDUMP_PATH -u TRITON_NVDISASM_PATH \
        -u CPATH -u C_INCLUDE_PATH \
        TRITON_HOME="$chk/home" TRITON_CACHE_DIR="$chk/cache" TRITON_LIBCUDA_PATH="$chk/stubs" \
        LD_LIBRARY_PATH="$chk/stubs:${LD_LIBRARY_PATH:-}" \
        /usr/bin/python3 -c 'from triton import knobs; from triton.backends.nvidia.driver import CudaUtils; CudaUtils(); print("triton env-free cuda_utils JIT ok; tools:", knobs.nvidia.ptxas.path, knobs.nvidia.cuobjdump.path, knobs.nvidia.nvdisasm.path)'; \
    rm -rf "$chk"; \
    rm -rf /usr/local/cuda-*/NsightSystems-cli-*/target-linux-*/plugins/efa_metrics \
        /opt/nvidia/nsight-systems-cli/*/target-linux-*/plugins/efa_metrics \
        /opt/nvidia/nsight-compute/*/host/target-linux-*/plugins/efa_metrics; \
    if find /usr/local /opt -xdev -type d -name efa_metrics 2>/dev/null | grep -q .; then \
        echo "ERROR: an efa_metrics plugin directory survived removal; the globs no" >&2; \
        echo "       longer match the base image's Nsight layout." >&2; \
        find /usr/local /opt -xdev -type d -name efa_metrics >&2; \
        exit 1; \
    fi; \
    rm -rf /usr/local/cuda-*/NsightSystems-cli-*/target-linux-*/python/lib/python3.12/site-packages/pip \
        /usr/local/cuda-*/NsightSystems-cli-*/target-linux-*/python/lib/python3.12/site-packages/pip-*.dist-info \
        /usr/local/cuda-*/NsightSystems-cli-*/target-linux-*/python/bin/pip3* \
        /opt/nvidia/nsight-systems-cli/*/target-linux-*/python/lib/python3.12/site-packages/pip \
        /opt/nvidia/nsight-systems-cli/*/target-linux-*/python/lib/python3.12/site-packages/pip-*.dist-info \
        /opt/nvidia/nsight-systems-cli/*/target-linux-*/python/bin/pip3* \
        /opt/pytorch/pytorch/third_party/onnx/onnx.egg-info; \
    left=$(find /usr/local /opt -xdev \( -path '*/NsightSystems-cli-*/site-packages/pip*' \
            -o -path '*/nsight-systems-cli/*/site-packages/pip*' \
            -o -path '*/nsight-compute/*/site-packages/pip*' -o -name onnx.egg-info \) 2>/dev/null); \
    if [ -n "$left" ]; then \
        echo "ERROR: an Nsight pip or the PyTorch tree's onnx.egg-info survived" >&2; \
        echo "       removal; the globs no longer match the base layout:" >&2; \
        echo "$left" >&2; \
        exit 1; \
    fi; \
    {% include "templates/verify_ca_certificates.Dockerfile" %}

ENV GRPC_DEFAULT_SSL_ROOTS_FILE_PATH=/etc/ssl/certs/ca-certificates.crt

# One COPY pulls nats-server, etcd/, uv, uvx into their final paths.
COPY --from=dynamo_base_export / /
ENV PATH=/opt/uv/bin:${PATH}

# Upstream ships matching NIXL bindings outside the standard Python search path.
# Register that directory for system Python and both Dynamo venvs instead of
# installing a second NIXL/UCX stack, and fail early if the layout changes.
#
# Then create the dynamo user (group 0 for OpenShift), clear upstream's
# /workspace (otherwise pytest collects its broken tutorial tests), and on
# runtime create the venv. --system-site-packages keeps upstream's solve
# importable, since system Python is PEP 668 externally managed.
RUN test -d /opt/nvidia/nvda_nixl/lib/python3/dist-packages/nixl_cu13 \
    && python3 -c 'import sysconfig; from pathlib import Path; Path(sysconfig.get_path("purelib"), "dynamo-nixl.pth").write_text("/opt/nvidia/nvda_nixl/lib/python3/dist-packages\n")' \
    && python3 -c 'import ctypes, importlib.metadata as m; import nixl._api, nixl._bindings; ctypes.CDLL("libnixl.so"); print("nixl:", m.version("nixl")); print("bindings:", nixl._bindings.__file__)' \
        > /opt/dynamo/nixl-versions.txt && cat /opt/dynamo/nixl-versions.txt \
    && { userdel -r ubuntu > /dev/null 2>&1 || true; } \
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
# install from source and skip the pip install RUN below.
COPY --chmod=775 --chown=dynamo:0 --from=wheel_builder /opt/dynamo/dist/*.whl /opt/dynamo/wheelhouse/

{% if target not in ("dev", "local-dev") %}
RUN --mount=type=cache,id=uv-root-{{ context.dynamo.uv_version }},target=/root/.cache/uv,sharing=locked \
    --mount=type=bind,source=./container/deps/requirements.trtllm.txt,target=/tmp/requirements.trtllm.txt \
    export UV_CACHE_DIR=/root/.cache/uv && \
    \
    # Dynamo's own wheels — --no-deps preserves upstream's solve.
    uv pip install --no-deps /opt/dynamo/wheelhouse/ai_dynamo_runtime*.whl && \
    uv pip install --no-deps /opt/dynamo/wheelhouse/ai_dynamo*any.whl && \
    uv pip install --no-deps /opt/dynamo/wheelhouse/aisim-*.whl && \
    \
    # Third-party deps Dynamo wheels declare but upstream lacks. The
    # requirements.trtllm.txt file itself carries a `--no-binary imageio-ffmpeg`
    # directive that keeps the GPL-encumbered prebuilt ffmpeg off disk; IMAGEIO_FFMPEG_EXE
    # below points imageio at the in-tree LGPL CLI.
    uv pip install --no-deps --requirement /tmp/requirements.trtllm.txt && \
    \
    if [ "${ENABLE_GPU_MEMORY_SERVICE}" = "true" ]; then \
        GMS_WHEEL=$(ls /opt/dynamo/wheelhouse/gpu_memory_service*.whl 2>/dev/null | head -1); \
        if [ -n "$GMS_WHEEL" ]; then uv pip install --no-deps "$GMS_WHEEL"; fi; \
    fi
{% endif %}

ENV IMAGEIO_FFMPEG_EXE=/usr/local/bin/ffmpeg

# Media stack. Package changes use the system interpreter: with VIRTUAL_ENV
# set, plain pip targets the venv and exits 0 without touching system site.
#
# - Copy the in-tree LGPL ffmpeg from wheel_builder. The TRT-LLM diffusion
#   handler always encodes video (video_handler.py -> encode_to_video_bytes), so
#   the CLI and its libav*/libvpx libraries must ship, with IMAGEIO_FFMPEG_EXE
#   pointing imageio at it. Not gated by enable_media_ffmpeg for that reason.
#
# - Remove upstream's opencv-python-headless and the ffmpeg copies it vendors
#   under opencv_python_headless.libs/. cv2 is an optional video extra upstream
#   (NVIDIA/TensorRT-LLM#16206) and its imports are function-local, so nothing
#   needs it at import time.
#
# - DALI comes from the base. Its vendored ffmpeg must not register the software
#   h264/hevc/aac decoders (dropped upstream from 2.1.1: NVIDIA/DALI_deps#162,
#   NVIDIA/DALI#6352). The libraries are DT_NEEDED by libdali.so, so they cannot
#   be trimmed. Enumerate what each bundled libavcodec registers rather than trust
#   the version; an empty match fails too, since a loop over nothing passes.
#
# - Replace the base's PyNvVideoCodec with the version requirements.trtllm.txt
#   pins. Uninstall first, then remove leftovers: the wheel's RECORD also covers
#   the FFmpeg source tarball under /usr/local/external/ffmpeg. 2.2.x links
#   libavcodec into libavformat, so a filename test is blind; check_pynvvideocodec
#   asserts the version, the bundled libraries against codec_policy.yaml and a
#   single source tarball, and the enumeration reads what the libraries register
#   ("0 decoders, 0 encoders" is correct for 2.2.x; the probe fails closed if
#   av_codec_iterate is missing). The venv cross-check reads dist-info directly,
#   since the --system-site-packages venv would resolve the system copy. No
#   import smoke test: it dlopens libnvcuvid, which the builder lacks. Drop the
#   replacement once the base ships the pin or newer.
#
# tests/dependencies/test_no_software_video_codecs.py repeats the codec checks
# against the shipped image.
RUN --mount=type=bind,from=wheel_builder,source=/usr/local/,target=/tmp/usr/local/ \
    --mount=type=bind,source=./container/compliance,target=/tmp/compliance/compliance \
    set -eu; \
    cp -nL /tmp/usr/local/lib/libav*.so* /usr/local/lib/ 2>/dev/null || true; \
    cp -nL /tmp/usr/local/lib/libsw*.so* /usr/local/lib/ 2>/dev/null || true; \
    cp -nL /tmp/usr/local/lib/lib*vpx*.so* /usr/local/lib/ 2>/dev/null || true; \
    cp -nL /tmp/usr/local/bin/ffmpeg /usr/local/bin/ffmpeg; \
    cp -r /tmp/usr/local/src/ffmpeg /usr/local/src/; \
    ldconfig; \
    /usr/bin/python3 -m pip uninstall -y --break-system-packages opencv-python-headless; \
    if /usr/bin/python3 -c "import cv2" 2>/dev/null; then echo "ERROR: cv2 still importable" >&2; exit 1; fi; \
    [ ! -e /usr/local/lib/python3.12/dist-packages/opencv_python_headless.libs ]; \
    v=$(/usr/bin/python3 -c 'import importlib.metadata as m; print(m.version("nvidia-dali-cuda130"))'); \
    echo "DALI version: $v"; \
    libs=$(find /usr/local/lib/python3.12/dist-packages/nvidia/dali/.libs -name 'libavcodec*.so*'); \
    if [ -z "$libs" ]; then \
        echo "ERROR: DALI $v ships no libavcodec under .libs, so the enumeration" >&2; \
        echo "       below would check nothing and pass. Either the vendored ffmpeg" >&2; \
        echo "       moved, or the codec is now linked into another library, the way" >&2; \
        echo "       PyNvVideoCodec 2.2.0 links it into libavformat. Look before this" >&2; \
        echo "       ships: widen the glob to the library that carries it." >&2; \
        exit 1; \
    fi; \
    for lib in $libs; do \
        /usr/bin/python3 /tmp/compliance/compliance/enumerate_bundled_decoders.py "$lib"; \
    done; \
    /usr/bin/python3 -c 'import nvidia.dali'; \
    /usr/bin/python3 -m pip uninstall --break-system-packages --yes pynvvideocodec; \
    rm -rf /usr/local/lib/python3.12/dist-packages/PyNvVideoCodec \
        /usr/local/lib/python3.12/dist-packages/pynvvideocodec* \
        /usr/local/external/ffmpeg; \
    /usr/bin/python3 -m pip install --break-system-packages --no-cache-dir \
        'PyNvVideoCodec==2.2.4'; \
    PYTHONPATH=/tmp/compliance /usr/bin/python3 -m compliance.check_pynvvideocodec --pinned 2.2.4; \
    v=$(/usr/bin/python3 -c 'import importlib.metadata as m; print(m.version("pynvvideocodec"))'); \
    echo "PyNvVideoCodec version: $v"; \
    examined=0; \
    for lib in $(find /usr/local/lib/python3.12/dist-packages/PyNvVideoCodec \
            -name 'libavcodec*.so*' -o -name 'libavformat*.so*'); do \
        examined=$((examined + 1)); \
        /usr/bin/python3 /tmp/compliance/compliance/enumerate_bundled_decoders.py "$lib"; \
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
                 echo "       requirements.trtllm.txt should have installed one." >&2; \
                 exit 1; }; \
        [ "$vv" = "$v" ] \
            || { echo "ERROR: venv PyNvVideoCodec is $vv but the system copy is $v --" >&2; \
                 echo "       the two have drifted apart. requirements.trtllm.txt and" >&2; \
                 echo "       this stage must pin the same version." >&2; exit 1; }; \
    else \
        echo "NOTE: no /opt/dynamo/venv in this stage yet, so the venv/system"; \
        echo "      cross-check is skipped. Expected for the dev and local-dev"; \
        echo "      targets, which create the venv later, after this stage, with"; \
        echo "      --system-site-packages over the system copy asserted above."; \
    fi

# The shipped ffmpeg MUST expose the VP9 encoder and MUST NOT expose any
# H.264/H.265/AAC/NVENC encoder, so a broken copy or codec regression fails the
# build rather than surfacing at runtime.
{% include "templates/ffmpeg_encoder_guard.Dockerfile" %}

# Raise named base-image distributions in system site, where the image
# inventory reads them. requirements.trtllm.txt only reaches the venv, and the
# --no-deps installs above leave upstream's solve alone; this is not a re-solve.
# System interpreter for the reason given above: a venv-only install would
# shadow the old version at import time while leaving it on disk.
#
# - mooncake-transfer-engine-cuda13: the specifier requirements.vllm.txt and
#   requirements.sglang.txt use, kept equal by
#   tests/dependencies/test_mooncake_floor_consistency.py. If the base's
#   tensorrt-llm pins an older build, pip check reports that pin.
# - aiohttp: the floor requirements.common.txt sets for the other images.
#   cp312 manylinux_2_28 wheels exist for both architectures.
# - certifi (CA bundle) and Pillow: no required dependencies, so each moves
#   exactly one distribution.
# - transformers: past the exact pin that tensorrt-llm declares (==5.5.4).
#   5.13.0 fixed Hub-cache remote-code imports (huggingface/transformers#46618).
#   5.13.1 shares 5.5.4's tokenizers and huggingface-hub requirements, moving no other
#   distribution. The bounds of its checks must hold the pin
#   (tests/dependencies/test_system_site_pins.py).
# - jupyter-server, jupyterlab, notebook, urllib3: the upstream developer stack.
# - datasets, fsspec, filelock, pyasn1, soupsieve, onnx, multidict, werkzeug:
#   datasets 4.7.0 caps fsspec at 2026.2.0, so datasets moves to 5.1.0.
#   tensorrt-llm declares datasets unversioned; torchtitan caps it below 4.8.0
#   but tensorrt_llm does not import torchtitan. The filelock floor matches
#   upstream TensorRT-LLM; the pyasn1 cap stays inside pyasn1-modules' range.
#   soupsieve 4 needs Python 3.11.5 and no other package; requirements.trtllm.txt
#   refreshes only the venv's own copy, so the venv check leaves it out. onnx
#   1.22.0 needs typing_extensions 4.15.0 and ml_dtypes 0.5.4, which the base
#   ships, and aiohttp caps multidict below 7.
#
# Guards assert exactly one dist-info directory in range, since that is what the
# inventory reads and what a stale copy would leave behind; they compare
# versions rather than glob, because a glob range silently stops matching. The
# multi-package upgrades fail on any unexpected transitive change. On runtime,
# the venv is checked so an older venv copy cannot shadow the refreshed ones.
RUN /usr/bin/python3 -m pip install --break-system-packages --upgrade pip==26.2.2 "aiohttp>=3.14.4,<4.0" && \
    /usr/bin/python3 -c 'import glob, os, sys; d = glob.glob("/usr/local/lib/python3.12/dist-packages/aiohttp-*.dist-info"); vs = [os.path.basename(p)[8:-10] for p in d]; print("aiohttp dist-info in system site:", vs); tv = lambda s: tuple(int(x) for x in s.split(".")[:3]); sys.exit(0 if len(vs) == 1 and (3, 14, 4) <= tv(vs[0]) < (4, 0, 0) else 1)' && \
    /usr/bin/python3 -m pip install --break-system-packages --upgrade certifi==2026.10.23 && \
    /usr/bin/python3 -c 'import glob, os, sys; d = glob.glob("/usr/local/lib/python3.12/dist-packages/certifi-*.dist-info"); vs = [os.path.basename(p)[8:-10] for p in d]; print("certifi dist-info in system site:", vs); tv = lambda s: tuple(int(x) for x in s.split(".")[:3]); sys.exit(0 if len(vs) == 1 and tv(vs[0]) >= (2026, 10, 23) else 1)' && \
    /usr/bin/python3 -m pip install --break-system-packages --upgrade "pillow>=13.0.0,<14" && \
    /usr/bin/python3 -c 'import glob, os, sys; d = glob.glob("/usr/local/lib/python3.12/dist-packages/pillow-*.dist-info"); vs = [os.path.basename(p)[7:-10] for p in d]; print("pillow dist-info in system site:", vs); tv = lambda s: tuple(int(x) for x in s.split(".")[:3]); sys.exit(0 if len(vs) == 1 and (13, 0, 0) <= tv(vs[0]) < (14, 0, 0) else 1)' && \
    /usr/bin/python3 -m pip install --break-system-packages --upgrade \
        --report /tmp/transformers-upgrade.json "transformers==5.13.1" && \
    /usr/bin/python3 -c 'import json; d = json.load(open("/tmp/transformers-upgrade.json")); changed = {p["metadata"]["name"].lower().replace("_", "-") for p in d["install"]}; print("upgraded system distributions:", sorted(changed)); assert changed <= {"transformers"}, "Unexpected transitive upgrade"' && \
    rm /tmp/transformers-upgrade.json && \
    /usr/bin/python3 -c 'import glob, importlib.metadata as m; from packaging.version import Version; bounds = {"transformers": ("5.13.1", "5.13.2")}; versions = {n: m.version(n) for n in bounds}; print("system-site versions:", versions); assert all(len(glob.glob("/usr/local/lib/python3.12/dist-packages/" + n + "-*.dist-info")) == 1 and Version(lo) <= Version(versions[n]) < Version(hi) for n, (lo, hi) in bounds.items())'{% if target not in ("dev", "local-dev") %} && \
    /opt/dynamo/venv/bin/python3 -c 'import importlib.metadata as m; from packaging.version import Version; bounds = {"transformers": ("5.13.1", "5.13.2")}; versions = {n: m.version(n) for n in bounds}; print("runtime interpreter versions:", versions); assert all(Version(lo) <= Version(versions[n]) < Version(hi) for n, (lo, hi) in bounds.items()); import transformers; print("runtime import path:", transformers.__file__); assert transformers.__file__ == "/usr/local/lib/python3.12/dist-packages/transformers/__init__.py"'{% endif %} && \
    /usr/bin/python3 -m pip install --break-system-packages --upgrade \
        --report /tmp/jupyter-upgrade.json \
        "jupyter-server==2.21.1" "jupyterlab==4.6.4" \
        "notebook==7.6.3" "urllib3==2.8.0" && \
    /usr/bin/python3 -c 'import json; d = json.load(open("/tmp/jupyter-upgrade.json")); changed = {p["metadata"]["name"].lower().replace("_", "-") for p in d["install"]}; print("upgraded system distributions:", sorted(changed)); assert changed <= {"jupyter-server", "jupyterlab", "notebook", "urllib3"}, "Unexpected transitive upgrade"' && \
    rm /tmp/jupyter-upgrade.json && \
    /usr/bin/python3 -c 'import glob, importlib.metadata as m; from packaging.version import Version; bounds = {"jupyter-server": ("2.21.0", "3"), "jupyterlab": ("4.6.4", "5"), "notebook": ("7.6.3", "8"), "urllib3": ("2.8.0", "3")}; versions = {n: m.version(n) for n in bounds}; print("system-site versions:", versions); assert all(len(glob.glob("/usr/local/lib/python3.12/dist-packages/" + n.replace("-", "_") + "-*.dist-info")) == 1 and Version(lo) <= Version(versions[n]) < Version(hi) for n, (lo, hi) in bounds.items()); from jupyter_server.serverapp import ServerApp; from jupyterlab.labapp import LabApp; from notebook.app import JupyterNotebookApp; import urllib3' && \
    /usr/bin/python3 -m pip install --break-system-packages --upgrade \
        --report /tmp/mooncake-upgrade.json \
        "mooncake-transfer-engine-cuda13>=0.3.14,<0.4" && \
    /usr/bin/python3 -c 'import json; d = json.load(open("/tmp/mooncake-upgrade.json")); changed = {p["metadata"]["name"].lower().replace("_", "-") for p in d["install"]}; print("upgraded system distributions:", sorted(changed)); assert changed <= {"mooncake-transfer-engine-cuda13"}, "Unexpected transitive upgrade"' && \
    rm /tmp/mooncake-upgrade.json && \
    /usr/bin/python3 -c 'import glob, importlib.metadata as m; from packaging.version import Version; bounds = {"mooncake-transfer-engine-cuda13": ("0.3.14", "0.4")}; versions = {n: m.version(n) for n in bounds}; print("system-site versions:", versions); assert all(len(glob.glob("/usr/local/lib/python3.12/dist-packages/" + n.replace("-", "_") + "-*.dist-info")) == 1 and Version(lo) <= Version(versions[n]) < Version(hi) for n, (lo, hi) in bounds.items()); import mooncake'{% if target not in ("dev", "local-dev") %} && \
    /opt/dynamo/venv/bin/python3 -c 'import importlib.metadata as m; from packaging.version import Version; v = m.version("mooncake-transfer-engine-cuda13"); print("runtime interpreter version:", v); assert Version("0.3.14") <= Version(v) < Version("0.4"); import mooncake; print("runtime import path:", mooncake.__file__); assert mooncake.__file__ == "/usr/local/lib/python3.12/dist-packages/mooncake/__init__.py"'{% endif %} && \
    /usr/bin/python3 -m pip install --break-system-packages --upgrade \
        --report /tmp/system-site-refresh.json \
        "datasets==5.1.0" "fsspec[http]==2026.7.0" "filelock>=4.2.1,<5" "pyasn1>=0.6.5,<0.7" \
        "soupsieve==4.1.1" "onnx==1.22.0" "multidict==6.9.1" "werkzeug==3.1.9" && \
    /usr/bin/python3 -c 'import json; d = json.load(open("/tmp/system-site-refresh.json")); changed = {p["metadata"]["name"].lower().replace("_", "-") for p in d["install"]}; print("upgraded system distributions:", sorted(changed)); assert changed <= {"datasets", "fsspec", "filelock", "pyasn1", "soupsieve", "onnx", "multidict", "werkzeug"}, "Unexpected transitive upgrade"' && \
    rm /tmp/system-site-refresh.json && \
    /usr/bin/python3 -c 'import glob, importlib.metadata as m; from packaging.version import Version; bounds = {"datasets": ("5.1.0", "5.1.1"), "fsspec": ("2026.7.0", "2026.7.1"), "filelock": ("4.2.1", "5"), "pyasn1": ("0.6.5", "0.7"), "soupsieve": ("4.1.1", "4.1.2"), "onnx": ("1.22.0", "1.22.1"), "multidict": ("6.9.1", "6.9.2"), "werkzeug": ("3.1.9", "3.1.10")}; versions = {n: m.version(n) for n in bounds}; print("system-site versions:", versions); assert all(len(glob.glob("/usr/local/lib/python3.12/dist-packages/" + n + "-*.dist-info")) == 1 and Version(lo) <= Version(versions[n]) < Version(hi) for n, (lo, hi) in bounds.items()); import datasets, fsspec, filelock, pyasn1, soupsieve, onnx, multidict, werkzeug'{% if target not in ("dev", "local-dev") %} && \
    /opt/dynamo/venv/bin/python3 -c 'import importlib.metadata as m, sys; from packaging.version import Version; bounds = {"jupyter-server": ("2.21.0", "3"), "jupyterlab": ("4.6.4", "5"), "notebook": ("7.6.3", "8"), "urllib3": ("2.8.0", "3")}; versions = {n: m.version(n) for n in bounds}; print("runtime interpreter versions:", versions); assert all(Version(lo) <= Version(versions[n]) < Version(hi) for n, (lo, hi) in bounds.items()); from jupyter_server.serverapp import ServerApp; from jupyterlab.labapp import LabApp; from notebook.app import JupyterNotebookApp; import urllib3; print("runtime import paths:", sys.modules[ServerApp.__module__].__file__, sys.modules[LabApp.__module__].__file__, sys.modules[JupyterNotebookApp.__module__].__file__, urllib3.__file__)' && \
    /opt/dynamo/venv/bin/python3 -c 'import importlib.metadata as m; from packaging.version import Version; bounds = {"datasets": ("5.1.0", "5.1.1"), "fsspec": ("2026.7.0", "2026.7.1"), "filelock": ("4.2.1", "5"), "pyasn1": ("0.6.5", "0.7"), "onnx": ("1.22.0", "1.22.1"), "multidict": ("6.9.1", "6.9.2"), "werkzeug": ("3.1.9", "3.1.10")}; versions = {n: m.version(n) for n in bounds}; print("runtime interpreter versions:", versions); assert all(Version(lo) <= Version(versions[n]) < Version(hi) for n, (lo, hi) in bounds.items()); import datasets, fsspec, filelock, pyasn1, onnx, multidict, werkzeug; paths = [datasets.__file__, fsspec.__file__, filelock.__file__, pyasn1.__file__, onnx.__file__, multidict.__file__, werkzeug.__file__]; print("runtime import paths:", *paths); assert all(p.startswith("/usr/local/lib/python3.12/dist-packages/") for p in paths)'{% endif %}

# Pull /workspace_src (incl. LICENSE) from the transport stage and
# wire up the launch screen in a single RUN — saves the standalone workspace COPY layer.
RUN --mount=type=bind,from=workspace_files,source=/workspace_src,target=/tmp/workspace_src \
    --mount=type=bind,source=./container/launch_message/runtime.txt,target=/opt/dynamo/launch_message.txt \
    cp -a /tmp/workspace_src/. /workspace/ && \
    chown -R dynamo:0 /workspace && \
    sed '/^#\s/d' /opt/dynamo/launch_message.txt > /opt/dynamo/.launch_screen && \
    chmod 755 /opt/dynamo/.launch_screen && \
    echo 'cat /opt/dynamo/.launch_screen' >> /etc/bash.bashrc

WORKDIR /workspace

USER dynamo

# Kept at the bottom — SHA changes per build; layers above stay cached.
ARG DYNAMO_COMMIT_SHA
ENV DYNAMO_COMMIT_SHA=${DYNAMO_COMMIT_SHA}

# Reset upstream TRT-LLM image's entrypoint so derived runtimes behave like
# other Dynamo images and can execute arbitrary commands directly.
ENTRYPOINT []
CMD ["/bin/bash"]



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
