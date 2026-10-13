# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Test image for SGLang /generate through the Rust sidecar: the upstream SGLang
# nightly plus the tools to build the Dynamo frontend bindings and the sidecar
# from a mounted checkout. The build happens at run time (build.sh) so code
# changes need no image rebuild.
#
#   docker build -t dynamo-sglang-e2e lib/sidecar/sglang/e2e

ARG BASE_IMAGE=lmsysorg/sglang:dev
FROM ${BASE_IMAGE}

# bindgen (nixl-sys) needs libclang; the sidecar's build.rs needs protoc and
# the well-known protos, which the base image already has.
RUN apt-get update && apt-get install -y --no-install-recommends clang libclang-dev \
    && rm -rf /var/lib/apt/lists/*

RUN uv pip install --python /opt/sglang/bin/python3 'maturin[patchelf]>=1.8,<1.10'

COPY build.sh run.sh compare.py /opt/dynamo-e2e/

ENV VIRTUAL_ENV=/opt/sglang \
    CARGO_TARGET_DIR=/cache/target \
    DYN_DISCOVERY_BACKEND=file

WORKDIR /workspace
ENTRYPOINT []
CMD ["sleep", "infinity"]
