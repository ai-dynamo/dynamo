# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

FROM ubuntu:24.04 AS build

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libprotobuf-dev \
    protobuf-compiler \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /src
COPY . .
RUN make daemon

FROM ubuntu:24.04

RUN apt-get update && apt-get install -y --no-install-recommends libprotobuf32 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=build /src/pagebroker /usr/local/bin/pagebroker
ENTRYPOINT ["/usr/local/bin/pagebroker"]
