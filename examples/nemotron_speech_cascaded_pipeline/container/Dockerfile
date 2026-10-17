# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# Build from the repository root. The crate has its own lockfile and does not
# add speech dependencies to the LLM sidecar images or Python wheel.
ARG RUST_IMAGE=rust:1.96.1-bookworm
ARG BASE_IMAGE=nvcr.io/nvidia/base/ubuntu:noble-20250619
FROM ${RUST_IMAGE} AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
        clang cmake libclang-dev protobuf-compiler libprotobuf-dev \
        pkg-config python3 perl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /src
# Shared lib/ crates inherit versions and dependencies from the root manifest.
COPY Cargo.toml rust-toolchain.toml ./
COPY .cargo/ .cargo/
COPY lib/ lib/
COPY examples/nemotron_speech_cascaded_pipeline/speech_nim/ examples/nemotron_speech_cascaded_pipeline/speech_nim/

ARG TARGETARCH
ARG CARGO_BUILD_JOBS=6
RUN --mount=type=cache,id=speech-nim-cargo-registry-${TARGETARCH},target=/usr/local/cargo/registry,sharing=locked \
    --mount=type=cache,id=speech-nim-cargo-git-${TARGETARCH},target=/usr/local/cargo/git,sharing=locked \
    --mount=type=cache,id=speech-nim-cargo-target-${TARGETARCH},target=/src/target,sharing=locked \
    CARGO_TARGET_DIR=/src/target cargo build --release --locked \
        --manifest-path examples/nemotron_speech_cascaded_pipeline/speech_nim/Cargo.toml \
    && install -D target/release/dynamo-speech-nim-sidecar /out/dynamo-speech-nim-sidecar \
    && strip /out/dynamo-speech-nim-sidecar

FROM ${BASE_IMAGE}
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates libstdc++6 \
    && rm -rf /var/lib/apt/lists/* \
    && { userdel -r ubuntu 2>/dev/null || true; } \
    && useradd --create-home --uid 1000 --gid 0 dynamo

COPY --from=builder /out/dynamo-speech-nim-sidecar /usr/local/bin/dynamo-speech-nim-sidecar
COPY LICENSE /licenses/dynamo/LICENSE
COPY examples/nemotron_speech_cascaded_pipeline/speech_nim/proto/LICENSE /licenses/riva-protos/LICENSE
USER 1000
EXPOSE 9090
ENTRYPOINT ["/usr/local/bin/dynamo-speech-nim-sidecar"]
CMD ["--help"]
