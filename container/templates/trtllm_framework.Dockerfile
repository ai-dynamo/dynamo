{#
# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#}
# === BEGIN templates/trtllm_framework.Dockerfile ===

# Copy artifacts from NGC PyTorch image
FROM ${BASE_IMAGE}:${BASE_IMAGE_TAG} AS pytorch_base

# Empty fallback for TRTLLM wheel image copy
FROM alpine:3.20 AS trtllm_wheel_image_empty
RUN mkdir -p /app/tensorrt_llm

# Resolve TRTLLM wheel image (can be a stage name or a registry image)
FROM ${TRTLLM_WHEEL_IMAGE} AS trtllm_wheel_image

##################################################
########## Framework Builder Stage ##############
##################################################
#
# PURPOSE: Build TensorRT-LLM with root privileges
#
# This stage handles TensorRT-LLM installation which requires:
# - Root access for apt operations (CUDA repos, TensorRT installation)
# - System-level modifications in install_tensorrt.sh
# - Virtual environment population with PyTorch and TensorRT-LLM
#
# The completed venv is then copied to runtime stage with dynamo ownership

FROM ${BASE_IMAGE}:${BASE_IMAGE_TAG} AS framework

ARG ARCH_ALT
COPY --from=dynamo_base /bin/uv /bin/uvx /bin/

# Install minimal dependencies needed for TensorRT-LLM installation
ARG PYTHON_VERSION
# Cache apt downloads; sharing=locked avoids apt/dpkg races with concurrent builds.
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        python${PYTHON_VERSION}-dev \
        python3-pip \
        curl \
        git \
        git-lfs \
        ca-certificates && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN mkdir -p /opt/dynamo/venv && \
    export UV_CACHE_DIR=/root/.cache/uv && \
    uv venv /opt/dynamo/venv --python $PYTHON_VERSION

ENV VIRTUAL_ENV=/opt/dynamo/venv \
    PATH="/opt/dynamo/venv/bin:${PATH}"

# Copy pytorch installation from NGC PyTorch
ARG FLASHINFER_PYTHON_VER
ARG PYTORCH_TRITON_VER
ARG TORCHAO_VER
ARG TORCHDATA_VER
ARG TORCHTITAN_VER
ARG TORCH_VER
ARG TORCH_TENSORRT_VER
ARG TORCHVISION_VER
ARG JINJA2_VER
ARG SYMPY_VER
ARG FLASH_ATTN_VER

COPY --from=pytorch_base /usr/local/lib/python${PYTHON_VERSION}/dist-packages/torchao ${VIRTUAL_ENV}/lib/python${PYTHON_VERSION}/site-packages/torchao
COPY --from=pytorch_base /usr/local/lib/python${PYTHON_VERSION}/dist-packages/torchao-${TORCHAO_VER}.dist-info ${VIRTUAL_ENV}/lib/python${PYTHON_VERSION}/site-packages/torchao-${TORCHAO_VER}.dist-info
COPY --from=pytorch_base /usr/local/lib/python${PYTHON_VERSION}/dist-packages/torchdata ${VIRTUAL_ENV}/lib/python${PYTHON_VERSION}/site-packages/torchdata
COPY --from=pytorch_base /usr/local/lib/python${PYTHON_VERSION}/dist-packages/torchdata-${TORCHDATA_VER}.dist-info ${VIRTUAL_ENV}/lib/python${PYTHON_VERSION}/site-packages/torchdata-${TORCHDATA_VER}.dist-info
COPY --from=pytorch_base /usr/local/lib/python${PYTHON_VERSION}/dist-packages/torchtitan ${VIRTUAL_ENV}/lib/python${PYTHON_VERSION}/site-packages/torchtitan
COPY --from=pytorch_base /usr/local/lib/python${PYTHON_VERSION}/dist-packages/torchtitan-${TORCHTITAN_VER}.dist-info ${VIRTUAL_ENV}/lib/python${PYTHON_VERSION}/site-packages/torchtitan-${TORCHTITAN_VER}.dist-info
COPY --from=pytorch_base /usr/local/lib/python${PYTHON_VERSION}/dist-packages/pytorch_triton-${PYTORCH_TRITON_VER}.dist-info ${VIRTUAL_ENV}/lib/python${PYTHON_VERSION}/site-packages/pytorch_triton-${PYTORCH_TRITON_VER}.dist-info
COPY --from=pytorch_base /usr/local/lib/python${PYTHON_VERSION}/dist-packages/torch ${VIRTUAL_ENV}/lib/python${PYTHON_VERSION}/site-packages/torch
COPY --from=pytorch_base /usr/local/lib/python${PYTHON_VERSION}/dist-packages/torch-${TORCH_VER}.dist-info ${VIRTUAL_ENV}/lib/python${PYTHON_VERSION}/site-packages/torch-${TORCH_VER}.dist-info
COPY --from=pytorch_base /usr/local/lib/python${PYTHON_VERSION}/dist-packages/torchgen ${VIRTUAL_ENV}/lib/python${PYTHON_VERSION}/site-packages/torchgen
COPY --from=pytorch_base /usr/local/lib/python${PYTHON_VERSION}/dist-packages/torchvision ${VIRTUAL_ENV}/lib/python${PYTHON_VERSION}/site-packages/torchvision
COPY --from=pytorch_base /usr/local/lib/python${PYTHON_VERSION}/dist-packages/torchvision-${TORCHVISION_VER}.dist-info ${VIRTUAL_ENV}/lib/python${PYTHON_VERSION}/site-packages/torchvision-${TORCHVISION_VER}.dist-info
COPY --from=pytorch_base /usr/local/lib/python${PYTHON_VERSION}/dist-packages/torchvision.libs ${VIRTUAL_ENV}/lib/python${PYTHON_VERSION}/site-packages/torchvision.libs
COPY --from=pytorch_base /usr/local/lib/python${PYTHON_VERSION}/dist-packages/functorch ${VIRTUAL_ENV}/lib/python${PYTHON_VERSION}/site-packages/functorch
COPY --from=pytorch_base /usr/local/lib/python${PYTHON_VERSION}/dist-packages/jinja2 ${VIRTUAL_ENV}/lib/python${PYTHON_VERSION}/site-packages/jinja2
COPY --from=pytorch_base /usr/local/lib/python${PYTHON_VERSION}/dist-packages/jinja2-${JINJA2_VER}.dist-info ${VIRTUAL_ENV}/lib/python${PYTHON_VERSION}/site-packages/jinja2-${JINJA2_VER}.dist-info
COPY --from=pytorch_base /usr/local/lib/python${PYTHON_VERSION}/dist-packages/sympy ${VIRTUAL_ENV}/lib/python${PYTHON_VERSION}/site-packages/sympy
COPY --from=pytorch_base /usr/local/lib/python${PYTHON_VERSION}/dist-packages/sympy-${SYMPY_VER}.dist-info ${VIRTUAL_ENV}/lib/python${PYTHON_VERSION}/site-packages/sympy-${SYMPY_VER}.dist-info
COPY --from=pytorch_base /usr/local/lib/python${PYTHON_VERSION}/dist-packages/flash_attn ${VIRTUAL_ENV}/lib/python${PYTHON_VERSION}/site-packages/flash_attn
COPY --from=pytorch_base /usr/local/lib/python${PYTHON_VERSION}/dist-packages/flash_attn-${FLASH_ATTN_VER}.dist-info ${VIRTUAL_ENV}/lib/python${PYTHON_VERSION}/site-packages/flash_attn-${FLASH_ATTN_VER}.dist-info
COPY --from=pytorch_base /usr/local/lib/python${PYTHON_VERSION}/dist-packages/flash_attn_2_cuda.cpython-*-*-linux-gnu.so ${VIRTUAL_ENV}/lib/python${PYTHON_VERSION}/site-packages/
COPY --from=pytorch_base /usr/local/lib/python${PYTHON_VERSION}/dist-packages/torch_tensorrt ${VIRTUAL_ENV}/lib/python${PYTHON_VERSION}/site-packages/torch_tensorrt
COPY --from=pytorch_base /usr/local/lib/python${PYTHON_VERSION}/dist-packages/torch_tensorrt-${TORCH_TENSORRT_VER}.dist-info ${VIRTUAL_ENV}/lib/python${PYTHON_VERSION}/site-packages/torch_tensorrt-${TORCH_TENSORRT_VER}.dist-info

RUN --mount=type=cache,target=/root/.cache/uv \
    export UV_CACHE_DIR=/root/.cache/uv UV_HTTP_TIMEOUT=300 UV_HTTP_RETRIES=5 && \
    uv pip install flashinfer-python==${FLASHINFER_PYTHON_VER}

# Install TensorRT-LLM and related dependencies
ARG HAS_TRTLLM_CONTEXT
ARG TENSORRTLLM_PIP_WHEEL
ARG TENSORRTLLM_INDEX_URL
ARG GITHUB_TRTLLM_COMMIT

{% if context.trtllm.has_trtllm_context == "1" %}
# Copy only wheel files and commit info from trtllm_wheel stage from build_context
COPY --from=trtllm_wheel / /trtllm_wheel/
{%- endif %}
COPY --from=trtllm_wheel_image /app/tensorrt_llm /trtllm_wheel_image/

# Cache uv downloads; uv handles its own locking for this cache.
RUN --mount=type=cache,target=/root/.cache/uv \
    export UV_CACHE_DIR=/root/.cache/uv UV_HTTP_TIMEOUT=300 UV_HTTP_RETRIES=5 && \
    uv pip install "cuda-python==13.0.2"

# Note: TensorRT needs to be uninstalled before installing the TRTLLM wheel
# because there might be mismatched versions of TensorRT between the NGC PyTorch
# and the TRTLLM wheel.
RUN [ -f /etc/pip/constraint.txt ] && : > /etc/pip/constraint.txt || true && \
    # Clean up any existing conflicting CUDA repository configurations and GPG keys
    rm -f /etc/apt/sources.list.d/cuda*.list && \
    rm -f /usr/share/keyrings/cuda-archive-keyring.gpg && \
    rm -f /etc/apt/trusted.gpg.d/cuda*.gpg

RUN --mount=type=cache,target=/root/.cache/uv \
    export UV_CACHE_DIR=/root/.cache/uv UV_HTTP_TIMEOUT=300 UV_HTTP_RETRIES=5 && \
    if [ "$HAS_TRTLLM_CONTEXT" = "1" ]; then \
        # Download and run install_tensorrt.sh from TensorRT-LLM GitHub before installing the wheel
        curl -fsSL --retry 5 --retry-delay 10 --max-time 1800 -o /tmp/install_tensorrt.sh "https://github.com/NVIDIA/TensorRT-LLM/raw/${GITHUB_TRTLLM_COMMIT}/docker/common/install_tensorrt.sh" && \
        # Modify the script to use virtual environment pip instead of system pip3
        sed -i 's/pip3 install/uv pip install/g' /tmp/install_tensorrt.sh && \
        bash /tmp/install_tensorrt.sh && \
        # Install from local wheel directory in build context
        WHEEL_FILE="$(find /trtllm_wheel -name "*.whl" | head -n 1)"; \
        if [ -n "$WHEEL_FILE" ]; then \
            uv pip install "$WHEEL_FILE" triton==3.5.1; \
        else \
            echo "No wheel file found in /trtllm_wheel directory."; \
            exit 1; \
        fi; \
    elif [ -n "$(find /trtllm_wheel_image -name "*.whl" | head -n 1)" ]; then \
        # Install from wheel embedded in the TRTLLM release image
        WHEEL_FILE="$(find /trtllm_wheel_image -name "*.whl" | head -n 1)"; \
        uv pip install "$WHEEL_FILE" triton==3.5.1; \
    else \
        # Install TensorRT-LLM wheel from the provided index URL, allow dependencies from PyPI
        # TRTLLM 1.2.0rc6.post2 has issues installing from pypi with uv, installing from direct wheel link works best
        # explicitly installing triton 3.5.1 as trtllm only lists triton as dependency on x64_64 for some reason
        if echo "${TENSORRTLLM_PIP_WHEEL}" | grep -q '^tensorrt-llm=='; then \
            TRTLLM_VERSION=$(echo "${TENSORRTLLM_PIP_WHEEL}" | sed -E 's/tensorrt-llm==([0-9a-zA-Z.+-]+).*/\1/'); \
            PYTHON_TAG="cp$(echo ${PYTHON_VERSION} | tr -d '.')"; \
            DIRECT_URL="https://pypi.nvidia.com/tensorrt-llm/tensorrt_llm-${TRTLLM_VERSION}-${PYTHON_TAG}-${PYTHON_TAG}-linux_${ARCH_ALT}.whl"; \
            uv pip install --index-strategy=unsafe-best-match --extra-index-url "${TENSORRTLLM_INDEX_URL}" "${DIRECT_URL}" triton==3.5.1; \
        else \
            uv pip install --index-strategy=unsafe-best-match --extra-index-url "${TENSORRTLLM_INDEX_URL}" "${TENSORRTLLM_PIP_WHEEL}" triton==3.5.1; \
        fi; \
    fi && \
    # Run TensorRT installer that ships with the TRTLLM wheel
    TRT_INSTALLER="$(python -c "import glob, os, site; paths = []; \
        paths += site.getsitepackages() if hasattr(site, 'getsitepackages') else []; \
        user_site = site.getusersitepackages(); \
        paths.append(user_site) if user_site else None; \
        installer = ''; \
        \
        [installer:=matches[0] for base in paths \
            for matches in [glob.glob(os.path.join(base, 'tensorrt_llm', '**', 'install_tensorrt.sh'), recursive=True)] \
            if matches and not installer]; \
        print(installer)")"; \
    if [ -z "$TRT_INSTALLER" ]; then \
        echo "No install_tensorrt.sh found inside tensorrt_llm package."; \
        exit 1; \
    fi; \
    sed -i 's/pip3 install/uv pip install/g' "$TRT_INSTALLER"; \
    bash "$TRT_INSTALLER"
