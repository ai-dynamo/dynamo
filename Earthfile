# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

VERSION 0.8

############### ARTIFACTS TARGETS ##############################
# These targets are invoked in child Earthfiles to pass top-level files that are out of their build context
# https://docs.earthly.dev/earthly-0.6/best-practices#copying-files-from-outside-the-build-context

############### SHARED LIBRARY TARGETS ##############################
golang-base:
    FROM golang:1.23
    RUN apt-get update && apt-get install -y git && apt-get clean && rm -rf /var/lib/apt/lists/* && curl -sSfL https://github.com/golangci/golangci-lint/releases/download/v1.61.0/golangci-lint-1.61.0-linux-amd64.tar.gz | tar -xzv && mv golangci-lint-1.61.0-linux-amd64/golangci-lint /usr/local/bin/

operator-src:
    FROM +golang-base
    COPY ./deploy/dynamo/operator /artifacts/operator
    SAVE ARTIFACT /artifacts/operator


# artifact-base:
#     FROM python:3.12-slim-bookworm
#     WORKDIR /artifacts

# dynamo-source-artifacts:
#     FROM +artifact-base
#     COPY . /artifacts
#     SAVE ARTIFACT /artifacts

uv-source:
    FROM ghcr.io/astral-sh/uv:latest
    SAVE ARTIFACT /uv

dynamo-base:
    FROM ubuntu:24.04
    RUN apt-get update && \
        DEBIAN_FRONTEND=noninteractive apt-get install -yq python3-dev python3-pip python3-venv libucx0 curl
    COPY +uv-source/uv /bin/uv
    ENV CARGO_BUILD_JOBS=16

    RUN mkdir /opt/dynamo && \
        uv venv /opt/dynamo/venv --python 3.12 && \
        . /opt/dynamo/venv/bin/activate && \
        uv pip install pip

    ENV VIRTUAL_ENV=/opt/dynamo/venv
    ENV PATH="${VIRTUAL_ENV}/bin:${PATH}"

rust-base:
    FROM +dynamo-base
    # Rust build/dev dependencies
    RUN apt update -y && \
        apt install --no-install-recommends -y \
        wget \
        build-essential \
        protobuf-compiler \
        cmake \
        libssl-dev \
        pkg-config \
        libclang-dev \
        git

    RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb && \
        apt install -y ./cuda-keyring_1.1-1_all.deb && \
        apt update && \
        apt install -y cuda-toolkit nvidia-utils-535 nvidia-driver-535 && \
        rm cuda-keyring_1.1-1_all.deb

    # Set CUDA compute capability explicitly
    ENV CUDA_COMPUTE_CAP=80

    ENV CUDA_HOME=/usr/local/cuda
    ENV CUDA_ROOT=/usr/local/cuda
    ENV CUDA_PATH=/usr/local/cuda
    ENV CUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda
    ENV PATH=$CUDA_HOME/bin:$PATH
    ENV LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

    ENV RUSTUP_HOME=/usr/local/rustup
    ENV CARGO_HOME=/usr/local/cargo
    ENV PATH=/usr/local/cargo/bin:$PATH
    ENV RUST_VERSION=1.86.0
    ENV RUSTARCH=x86_64-unknown-linux-gnu

    RUN wget --tries=3 --waitretry=5 "https://static.rust-lang.org/rustup/archive/1.28.1/x86_64-unknown-linux-gnu/rustup-init" && \
        echo "a3339fb004c3d0bb9862ba0bce001861fe5cbde9c10d16591eb3f39ee6cd3e7f *rustup-init" | sha256sum -c - && \
        chmod +x rustup-init && \
        ./rustup-init -y --no-modify-path --profile minimal --default-toolchain 1.86.0 --default-host x86_64-unknown-linux-gnu && \
        rm rustup-init && \
        chmod -R a+w $RUSTUP_HOME $CARGO_HOME

dynamo-build:
    FROM +rust-base
    WORKDIR /workspace
    COPY Cargo.toml Cargo.lock ./
    COPY pyproject.toml README.md hatch_build.py ./
    COPY components/ components/
    COPY lib/ lib/
    COPY launch/ launch/
    COPY deploy/ deploy/

    ENV CARGO_TARGET_DIR=/workspace/target
    RUN cargo build --release --locked --features llamacpp,python,cuda && \
        cargo doc --no-deps

    # Create symlinks for wheel building
    RUN mkdir -p /workspace/deploy/dynamo/sdk/src/dynamo/sdk/cli/bin/ && \
        # Remove existing symlinks
        rm -f /workspace/deploy/dynamo/sdk/src/dynamo/sdk/cli/bin/* && \
        # Create new symlinks pointing to the correct location
        ln -sf /workspace/target/release/dynamo-run /workspace/deploy/dynamo/sdk/src/dynamo/sdk/cli/bin/dynamo-run && \
        ln -sf /workspace/target/release/http /workspace/deploy/dynamo/sdk/src/dynamo/sdk/cli/bin/http && \
        ln -sf /workspace/target/release/llmctl /workspace/deploy/dynamo/sdk/src/dynamo/sdk/cli/bin/llmctl


    RUN cd /workspace/lib/bindings/python && \
        uv build --wheel --out-dir /workspace/dist --python 3.12
    RUN cd /workspace && \
        uv build --wheel --out-dir /workspace/dist

    # Save wheels
    SAVE ARTIFACT /workspace/dist/ai_dynamo_runtime*.whl
    SAVE ARTIFACT /workspace/dist/ai_dynamo*.whl

dynamo-base-docker:
    ARG IMAGE=dynamo-base-docker
    ARG DOCKER_SERVER=my-registry
    ARG IMAGE_TAG=latest

    FROM ubuntu:24.04
    WORKDIR /workspace
    COPY container/deps/requirements.txt /tmp/requirements.txt

    # Install Python and other dependencies
    RUN apt-get update && \
        apt-get install -y --no-install-recommends \
        python3.12 \
        curl && \
        rm -rf /var/lib/apt/lists/*

    COPY +uv-source/uv /bin/uv

    # Create and activate virtual environment
    RUN mkdir -p /opt/dynamo && \
        uv venv /opt/dynamo/venv --python 3.12 && \
        . /opt/dynamo/venv/bin/activate && \
        uv pip install pip

    ENV VIRTUAL_ENV=/opt/dynamo/venv
    ENV PATH="${VIRTUAL_ENV}/bin:${PATH}"

    RUN uv pip install -r /tmp/requirements.txt

    # Copy and install wheels -- ai-dynamo-runtime first, then ai-dynamo
    COPY +dynamo-build/ai_dynamo_runtime*.whl /tmp/wheels/
    COPY +dynamo-build/ai_dynamo*.whl /tmp/wheels/
    RUN . /opt/dynamo/venv/bin/activate && \
        uv pip install /tmp/wheels/*.whl && \
        rm -rf /tmp/wheels

    SAVE IMAGE --push $DOCKER_SERVER/$IMAGE:$IMAGE_TAG

############### ALL TARGETS ##############################
all-test:
    BUILD ./deploy/dynamo/operator+test

all-docker:
    ARG DOCKER_SERVER=my-registry
    ARG IMAGE_TAG=latest
    BUILD ./deploy/dynamo/operator+docker --DOCKER_SERVER=$DOCKER_SERVER --IMAGE_TAG=$IMAGE_TAG
    BUILD ./deploy/dynamo/api-store+docker --DOCKER_SERVER=$DOCKER_SERVER --IMAGE_TAG=$IMAGE_TAG

all-lint:
    BUILD ./deploy/dynamo/operator+lint

all:
    BUILD +all-test
    BUILD +all-docker
    BUILD +all-lint

# For testing
custom:
    ARG DOCKER_SERVER=my-registry
    ARG IMAGE_TAG=latest
    BUILD +all-test
