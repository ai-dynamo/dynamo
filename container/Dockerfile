# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

ARG BASE_IMAGE="nvcr.io/nvidia/tritonserver"
ARG BASE_IMAGE_TAG="25.01-py3"

FROM ${BASE_IMAGE}:${BASE_IMAGE_TAG} AS triton-distributed

# TODO: non root user by default

USER root

# TODO: separate dev from runtime dependendcies

# Rust build/dev dependencies
RUN apt-get update && \
    apt-get install --no-install-recommends --yes  gdb protobuf-compiler cmake libssl-dev pkg-config

RUN curl https://sh.rustup.rs -sSf | bash -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"
RUN rustup toolchain install 1.85.0-x86_64-unknown-linux-gnu

# Install OpenAI-compatible frontend and its dependencies from triton server
# repository. These are used to have a consistent interface, schema, and FastAPI
# app between Triton Core and Triton Distributed implementations.
ARG OPENAI_SERVER_TAG="r25.01"
RUN mkdir -p /opt/tritonserver/python && \
    cd /opt/tritonserver/python && \
    rm -rf openai && \
    git clone -b ${OPENAI_SERVER_TAG} --single-branch https://github.com/triton-inference-server/server.git && \
    cd server && \
    git checkout ${SERVER_OPENAI_COMMIT} && \
    cd .. && \
    mv server/python/openai openai && \
    chown -R root:root openai && \
    chmod 755 openai && \
    chmod -R go-w openai && \
    rm -rf server && \
    python3 -m pip install -r openai/requirements.txt

# Common dependencies
RUN --mount=type=bind,source=./container/deps/requirements.txt,target=/tmp/requirements.txt \
    pip install --timeout=2000 --requirement /tmp/requirements.txt
RUN --mount=type=bind,source=./container/deps/requirements.nats.txt,target=/tmp/requirements.txt \
    pip install --timeout=2000 --requirement /tmp/requirements.txt
RUN --mount=type=bind,source=./container/deps/requirements.test.txt,target=/tmp/requirements.txt \
    pip install --timeout=2000 --requirement /tmp/requirements.txt

# Finish pyright install
RUN pyright --help > /dev/null 2>&1

# In Process Python API Install
RUN find /opt/tritonserver/python -maxdepth 1 -type f -name \
    "tritonserver-*.whl" | xargs -I {} pip3 install --force-reinstall --upgrade {}[all]

# GENAI Perf Install
# TODO: Move to tag when fix for genai-perf will be released
ARG GENAI_PERF_TAG="25d0188713adc47868d6b3f22426375237a90529"
RUN pip install "git+https://github.com/triton-inference-server/perf_analyzer.git@${GENAI_PERF_TAG}#subdirectory=genai-perf"

# Backend & Framework Specific Installation
ARG FRAMEWORK="STANDARD"
ARG TENSORRTLLM_BACKEND_REPO_TAG=
ARG TENSORRTLLM_BACKEND_REBUILD=
ARG TENSORRTLLM_SKIP_CLONE=
ENV FRAMEWORK=${FRAMEWORK}
RUN --mount=type=bind,source=./container/deps/requirements.tensorrtllm.txt,target=/tmp/requirements.txt \
    --mount=type=bind,source=./container/deps/clone_tensorrtllm.sh,target=/tmp/clone_tensorrtllm.sh \
    if [[ "$FRAMEWORK" == "TENSORRTLLM" ]] ; then pip install --timeout=2000 -r /tmp/requirements.txt; if [ ${TENSORRTLLM_SKIP_CLONE} -ne 1 ] ; then /tmp/clone_tensorrtllm.sh --tensorrtllm-backend-repo-tag ${TENSORRTLLM_BACKEND_REPO_TAG} --tensorrtllm-backend-rebuild ${TENSORRTLLM_BACKEND_REBUILD} --triton-llm-path /opt/triton/llm_binding ; fi ; fi


RUN --mount=type=bind,source=./container/deps/requirements.standard.txt,target=/tmp/requirements.txt \
    if [[ "$FRAMEWORK" == "STANDARD" ]] ; then pip install --timeout=2000 -r /tmp/requirements.txt ; fi

# Backend & Framework Specific LD_LIBRARY_PATH
ARG TENSORRTLLM_FRAMEWORK
ENV FRAMEWORK_LD_LIBRARY_PATH=${TENSORRTLLM_FRAMEWORK:+/opt/tritonserver/backends/tensorrtllm/}
ENV LD_LIBRARY_PATH=${FRAMEWORK_LD_LIBRARY_PATH}:${LD_LIBRARY_PATH}
ENV TENSORRTLLM_BACKEND_REPO_TAG=$TENSORRTLLM_BACKEND_REPO_TAG
ENV TRTLLM_USE_MPI_KVCACHE=${TENSORRTLLM_FRAMEWORK:+"1"}

# TODO set VLLM Version
# ENV VLLM_VERSION
ARG VLLM_FRAMEWORK
# DEFAULT VLLM VARIABLES
# ENV VLLM_ATTENTION_BACKEND=${VLLM_FRAMEWORK:+FLASHINFER}
ENV VLLM_WORKER_MULTIPROC_METHOD=${VLLM_FRAMEWORK:+spawn}
ENV VLLM_TORCH_HOST=${VLLM_FRAMEWORK:+localhost}
ENV VLLM_TORCH_PORT=${VLLM_FRAMEWORK:+36183}
ENV VLLM_DATA_PLANE_BACKEND=${VLLM_FRAMEWORK:+nccl}
ENV VLLM_BASELINE_WORKERS=${VLLM_FRAMEWORK:+0}
ENV VLLM_CONTEXT_WORKERS=${VLLM_FRAMEWORK:+1}
ENV VLLM_GENERATE_WORKERS=${VLLM_FRAMEWORK:+1}
ENV VLLM_BASELINE_TP_SIZE=${VLLM_FRAMEWORK:+1}
ENV VLLM_CONTEXT_TP_SIZE=${VLLM_FRAMEWORK:+1}
ENV VLLM_GENERATE_TP_SIZE=${VLLM_FRAMEWORK:+1}
ENV VLLM_KV_CAPI_PATH="/opt/triton/llm_binding/lib/libtriton_llm_capi.so"
ENV PYTHONUNBUFFERED=1

# Install NATS - pointing toward NATS github instead of binaries.nats.dev due to server instability
RUN wget https://github.com/nats-io/nats-server/releases/download/v2.10.24/nats-server-v2.10.24-amd64.deb && dpkg -i nats-server-v2.10.24-amd64.deb

# etcd
ENV ETCD_VERSION="v3.5.18"
RUN wget https://github.com/etcd-io/etcd/releases/download/$ETCD_VERSION/etcd-$ETCD_VERSION-linux-amd64.tar.gz -O /tmp/etcd.tar.gz && \
mkdir -p /usr/local/bin/etcd && \
tar -xvf /tmp/etcd.tar.gz -C /usr/local/bin/etcd --strip-components=1
ENV PATH=/usr/local/bin/etcd/:$PATH

# Enable Git operations in the /workspace directory.
RUN printf "[safe]\n      directory=/workspace\n" > /root/.gitconfig

# emacs docker-tramp requires /bin/sh to be linked to bash to operate correctly
RUN ln -sf /bin/bash /bin/sh

# Install NGINX
RUN apt-get install nginx -y
RUN rm -rf /etc/nginx/sites-enabled/default

# Install demo utils
RUN apt-get install nvtop -y
RUN apt-get install tmux -y

# Working directory
WORKDIR /workspace

# Copy Python wheel configuration files
COPY pyproject.toml /workspace/
COPY README.md /workspace/
COPY LICENSE /workspace/

# Build Rust runtime
COPY lib/runtime /workspace/lib/runtime
RUN cd lib/runtime && \
    cargo build --release --locked && cargo doc --no-deps

# Build OpenAI HTTP Service binaries
COPY lib/llm /workspace/lib/llm
COPY examples/rust /workspace/examples/rust
RUN cd examples/rust && \
    cargo build --release && \
    cp target/release/http /usr/local/bin/ && \
    cp target/release/llmctl /usr/local/bin/

# Generate C bindings. Note that this is required for TRTLLM backend re-build
COPY lib/bindings /workspace/lib/bindings
RUN cd lib/bindings/c/ && \
    cargo build --release --locked && cargo doc --no-deps

# Install uv, create virtualenv for general use, and build triton_distributed  wheel
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/
RUN mkdir /opt/triton && \
    uv venv /opt/triton/venv --python 3.12 && \
    source /opt/triton/venv/bin/activate && \
    uv build --wheel --out-dir /workspace/dist && \
    uv pip install /workspace/dist/triton_distributed*cp312*.whl

# Package the bindings
RUN mkdir -p /opt/triton/bindings/wheels && \
    mkdir /opt/triton/bindings/lib && \
    cp dist/triton_distributed*cp312*.whl /opt/triton/bindings/wheels/. && \
    cp lib/bindings/c/target/release/libtriton_distributed_llm_capi.so /opt/triton/bindings/lib/. && \
    cp -r lib/bindings/c/include /opt/triton/bindings/.

# Install triton_distributed_runtime and triton_distributed_llm wheels globally in container for tests that
# currently run without virtual environment activated.
# TODO: In future, we may use a virtualenv for everything and remove this.
RUN pip install /opt/triton/bindings/wheels/triton_distributed*cp312*.whl

# Copy everything in after install steps to avoid re-running build/install
# commands on unrelated changes in other dirs.
COPY . /workspace

# Enable system UCX
ENV RAPIDS_LIBUCX_PREFER_SYSTEM_LIBRARY=true

# Command and Entrypoint
CMD []
ENTRYPOINT ["/opt/nvidia/nvidia_entrypoint.sh"]
