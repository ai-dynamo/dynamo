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
        pkg-config

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


dynamo-base-docker:
    ARG IMAGE=dynamo-base-docker
    ARG CI_REGISTRY_IMAGE=my-registry
    ARG CI_COMMIT_SHA=latest
    FROM +rust-base
    WORKDIR /workspace

    COPY . /workspace/

    ENV CARGO_TARGET_DIR=/workspace/target

    RUN cargo build --release --locked --features mistralrs,sglang,vllm,python && \
        cargo doc --no-deps && \
        cp target/release/dynamo-run /usr/local/bin && \
        cp target/release/http /usr/local/bin && \
        cp target/release/llmctl /usr/local/bin && \
        cp target/release/metrics /usr/local/bin && \
        cp target/release/mock_worker /usr/local/bin

    RUN uv build --wheel --out-dir /workspace/dist && \
        uv pip install /workspace/dist/ai_dynamo*any.whl
    SAVE IMAGE --push $CI_REGISTRY_IMAGE/$IMAGE:$CI_COMMIT_SHA



############### ALL TARGETS ##############################
all-test:
    BUILD ./deploy/dynamo/operator+test

all-docker:
    ARG CI_REGISTRY_IMAGE=my-registry
    ARG CI_COMMIT_SHA=latest
    BUILD ./deploy/dynamo/operator+docker --CI_REGISTRY_IMAGE=$CI_REGISTRY_IMAGE --CI_COMMIT_SHA=$CI_COMMIT_SHA
    BUILD ./deploy/dynamo/api-store+docker --CI_REGISTRY_IMAGE=$CI_REGISTRY_IMAGE --CI_COMMIT_SHA=$CI_COMMIT_SHA

all-lint:
    BUILD ./deploy/dynamo/operator+lint

all:
    BUILD +all-test
    BUILD +all-docker
    BUILD +all-lint

# For testing
custom:
    ARG CI_REGISTRY_IMAGE=my-registry
    ARG CI_COMMIT_SHA=latest
    BUILD +all-test
