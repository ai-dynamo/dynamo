{#
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#}
# === BEGIN templates/tokenspeed_runtime.Dockerfile ===

FROM ${RUNTIME_IMAGE}:${RUNTIME_IMAGE_TAG} AS tokenspeed

ARG TARGETARCH
ARG PYTHON_VERSION

USER root
WORKDIR /workspace

# TokenSpeed JIT-compiles kernels: add only the compiler/header packages its
# runner uses to the smaller NVIDIA CUDA runtime image.
RUN apt-get update && apt-get install -y --no-install-recommends \
        {{ context.tokenspeed.cuda_jit_packages | join(" ") }} \
        python${PYTHON_VERSION}-dev python${PYTHON_VERSION}-venv \
        build-essential ca-certificates git ninja-build pkg-config \
        libnuma1 libibverbs1 ibverbs-providers libopenmpi-dev libssl-dev \
        libspdlog-dev libturbojpeg libjemalloc2 && \
    rm -rf /var/lib/apt/lists/* && ldconfig

# Compile without a GPU to verify that the runtime base has the JIT headers.
RUN printf '%s\n' \
        '#include <cuda_runtime.h>' '#include <cublas_v2.h>' \
        '#include <nvrtc.h>' '#include <curand_kernel.h>' \
        '#include <spdlog/spdlog.h>' '__global__ void smoke() {}' \
        > /tmp/tokenspeed-jit-smoke.cu && \
    nvcc -std=c++17 -arch=sm_90 -c /tmp/tokenspeed-jit-smoke.cu -o /tmp/tokenspeed-jit-smoke.o && \
    rm /tmp/tokenspeed-jit-smoke.cu /tmp/tokenspeed-jit-smoke.o

COPY --from=ghcr.io/astral-sh/uv:{{ context.dynamo.uv_version }} /uv /uvx /opt/uv/bin/
RUN python${PYTHON_VERSION} -m venv /opt/dynamo/venv
ENV VIRTUAL_ENV=/opt/dynamo/venv \
    PATH=/opt/uv/bin:/opt/dynamo/venv/bin:${PATH}

# Keep subsequent installs from replacing the CUDA 13.0 PyTorch/TokenSpeed solve.
RUN printf '%s\n' \
        'torch=={{ context.tokenspeed.torch_version }}+cu130' \
        'torchvision=={{ context.tokenspeed.torchvision_version }}+cu130' \
        'tokenspeed=={{ context.tokenspeed.version }}' \
        'tokenspeed-kernel=={{ context.tokenspeed.kernel_version }}' \
        'tokenspeed-engine=={{ context.tokenspeed.engine_version }}' \
        'tokenspeed-scheduler=={{ context.tokenspeed.scheduler_version }}' \
        'flashinfer-python=={{ context.tokenspeed.flashinfer_version }}' \
        'flashinfer-cubin=={{ context.tokenspeed.flashinfer_version }}' \
        > /opt/dynamo/tokenspeed-constraints.txt
ENV UV_CONSTRAINT=/opt/dynamo/tokenspeed-constraints.txt

RUN --mount=type=cache,id=uv-root-{{ context.dynamo.uv_version }},target=/root/.cache/uv,sharing=shared \
    uv pip install \
        --index https://download.pytorch.org/whl/cu130 \
        'torch=={{ context.tokenspeed.torch_version }}+cu130' \
        'torchvision=={{ context.tokenspeed.torchvision_version }}+cu130'

# Select wheels with BuildKit's target architecture, including multi-arch builds.
RUN --mount=type=cache,id=uv-root-{{ context.dynamo.uv_version }},target=/root/.cache/uv,sharing=shared \
    case "${TARGETARCH}" in \
{% for arch, wheel in context.tokenspeed.wheels.items() %}        {{ arch }}) \
            KERNEL_PLATFORM={{ wheel.kernel_platform }}; ENGINE_PLATFORM={{ wheel.engine_platform }}; \
            KERNEL_SHA256={{ wheel.kernel_sha256 }}; \
            ENGINE_SHA256={{ wheel.engine_sha256 }} ;; \
{% endfor %}        *) echo "Unsupported TARGETARCH=${TARGETARCH}" >&2; exit 1 ;; \
    esac && \
    uv pip install \
        "{{ context.tokenspeed.wheel_base_url }}/tokenspeed-nightly-v{{ context.tokenspeed.version }}/tokenspeed-{{ context.tokenspeed.version }}-py3-none-any.whl#sha256={{ context.tokenspeed.runtime_sha256 }}" \
        "{{ context.tokenspeed.wheel_base_url }}/tokenspeed-kernel-nightly-v{{ context.tokenspeed.kernel_version }}-cu130/tokenspeed_kernel-{{ context.tokenspeed.kernel_version }}-cp312-cp312-${KERNEL_PLATFORM}.whl#sha256=${KERNEL_SHA256}" \
        "{{ context.tokenspeed.wheel_base_url }}/tokenspeed-engine-nightly-v{{ context.tokenspeed.engine_version }}/tokenspeed_engine-{{ context.tokenspeed.engine_version }}-cp312-cp312-${ENGINE_PLATFORM}.whl#sha256=${ENGINE_SHA256}" \
        'tokenspeed-scheduler=={{ context.tokenspeed.scheduler_version }}'

# Cubin/JIT-cache wheels are published on FlashInfer's indexes, not PyPI.
# This command selects those indexes and does not need a GPU at build time.
RUN python -m flashinfer download-kernels \
        --cuda-version 13.0 --flashinfer-version {{ context.tokenspeed.flashinfer_version }}

FROM tokenspeed AS runtime

COPY --from=dynamo_base /usr/bin/nats-server /usr/bin/nats-server
COPY --from=dynamo_base /usr/local/bin/etcd/ /usr/local/bin/etcd/
ENV DYNAMO_HOME=/workspace \
    HOME=/home/dynamo \
    PATH=/usr/local/bin/etcd:${PATH}

RUN if id ubuntu >/dev/null 2>&1; then userdel -r ubuntu; fi && \
    useradd --uid 1000 --create-home --shell /bin/bash --gid 0 dynamo && \
    mkdir -p /home/dynamo/.cache /etc/profile.d && \
    chown dynamo:0 /home/dynamo /home/dynamo/.cache /workspace && \
    echo 'umask 002' > /etc/profile.d/00-umask.sh

# This base has no framework dependency stack to inherit. Resolve Dynamo's
# dependencies too, constrained to the installed TokenSpeed versions above.
COPY --from=wheel_builder /opt/dynamo/dist/ai_dynamo*.whl /opt/dynamo/wheelhouse/
COPY --from=wheel_builder /opt/dynamo/dist/aisimulate*.whl /opt/dynamo/wheelhouse/
RUN --mount=type=cache,id=uv-root-{{ context.dynamo.uv_version }},target=/root/.cache/uv,sharing=shared \
    uv pip install /opt/dynamo/wheelhouse/*.whl && \
    uv pip check && \
    python -c 'import dynamo.runtime; import tokenspeed_engine'

COPY --chmod=775 --chown=dynamo:0 tests /workspace/tests
COPY --chmod=775 --chown=dynamo:0 examples /workspace/examples
COPY --chmod=775 --chown=dynamo:0 dev /workspace/dev
COPY --chmod=775 --chown=dynamo:0 components/src/dynamo/common /workspace/components/src/dynamo/common
COPY --chmod=775 --chown=dynamo:0 components/src/dynamo/frontend /workspace/components/src/dynamo/frontend
COPY --chmod=775 --chown=dynamo:0 components/src/dynamo/tokenspeed /workspace/components/src/dynamo/tokenspeed
COPY --chmod=664 --chown=dynamo:0 LICENSE /workspace/

ARG DYNAMO_COMMIT_SHA
ENV DYNAMO_COMMIT_SHA=${DYNAMO_COMMIT_SHA}

USER dynamo
ENTRYPOINT []
CMD ["/bin/bash"]
