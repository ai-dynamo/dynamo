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

# Build from the Dynamo repository root so the shared ExtProc protos remain available.
FROM rust:1.96.1-bookworm@sha256:a339861ae23e9abb272cea45dfafde21760d2ce6577a70f8a926153677902663 AS build
RUN apt-get update && apt-get install -y --no-install-recommends protobuf-compiler libprotobuf-dev cmake && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY deploy/inference-gateway/ext-proc/proto deploy/inference-gateway/ext-proc/proto
COPY examples/backends/vllm/deploy/gaie/switchyard/preproc examples/backends/vllm/deploy/gaie/switchyard/preproc
WORKDIR /app/examples/backends/vllm/deploy/gaie/switchyard/preproc
RUN --mount=type=cache,target=/usr/local/cargo/registry --mount=type=cache,target=/app/examples/backends/vllm/deploy/gaie/switchyard/preproc/target cargo build --release --locked && cp target/release/switchyard-preproc /app/switchyard-preproc

FROM debian:bookworm-slim@sha256:3783cc01769c7b2b1b83a5c5ad96c815348e28ed7da68e2e3687004faa906251
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates && rm -rf /var/lib/apt/lists/*
COPY --from=build /app/switchyard-preproc /usr/local/bin/switchyard-preproc
ENV ROUTES_CONFIG=/etc/switchyard/routes.toml
USER 10001:10001
EXPOSE 9002 9003
ENTRYPOINT ["/usr/local/bin/switchyard-preproc"]
