# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

FROM nvcr.io/nvidia/base/ubuntu:noble-20250619
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 libstdc++6 \
    && rm -rf /var/lib/apt/lists/*
ENV HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
WORKDIR /tests
COPY . /tests/
ENTRYPOINT ["python3", "/tests/run.py", "--artifacts", "/tests"]
