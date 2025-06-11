# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

ARG BASE_IMAGE="dynamo"
ARG BASE_IMAGE_TAG="latest-vllm"

FROM ${BASE_IMAGE}:${BASE_IMAGE_TAG} AS dynamo_docker

# Install prerequisites
RUN apt-get update && \
    apt-get install -y \
        ca-certificates \
        curl \
        gnupg \
        lsb-release

# Add Docker’s official GPG key
RUN mkdir -p /etc/apt/keyrings && \
    curl -fsSL https://download.docker.com/linux/ubuntu/gpg | \
    gpg --dearmor -o /etc/apt/keyrings/docker.gpg

# Add Docker’s official repository
RUN echo \
    "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] \
    https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" \
    > /etc/apt/sources.list.d/docker.list

# Install Docker Engine
RUN apt-get update && \
    apt-get install -y docker-ce docker-ce-cli containerd.io

# Verify Docker installation
RUN docker --version
