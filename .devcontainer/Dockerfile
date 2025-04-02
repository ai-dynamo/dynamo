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

# Install kubectl
FROM bitnami/kubectl:1.32.3 as kubectl

# Build stage for Rust
FROM rust:1.85.0-slim as rust-builder

# Install build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    pkg-config \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Create a dummy project to cache dependencies
RUN cargo new --bin dummy
WORKDIR /dummy

# Create a minimal Cargo.toml for the dummy project
RUN echo '[package]\nname = "dummy"\nversion = "0.1.0"\nedition = "2021"\n[dependencies]\n' > Cargo.toml
RUN cargo build --release

# Main stage
FROM ubuntu:24.04

# Avoid prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Install dependencies in a single layer
RUN apt-get update && apt-get install -y \
    # Essential build tools
    build-essential \
    python3-dev \
    python3-pip \
    python3-venv \
    pkg-config \
    libssl-dev \
    # Version control and networking
    git \
    curl \
    wget \
    # Development tools
    jq \
    protobuf-compiler \
    # System tools
    sudo \
    bash \
    ca-certificates \
    apt-transport-https \
    gnupg \
    lsb-release \
    libglib2.0-0 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Install Azure CLI
RUN curl -sL https://aka.ms/InstallAzureCLIDeb | bash

# Install Helm directly from binary
RUN curl -fsSL -o get_helm.sh https://raw.githubusercontent.com/helm/helm/main/scripts/get-helm-3 && \
    chmod 700 get_helm.sh && \
    ./get_helm.sh && \
    rm get_helm.sh && \
    # Move helm to a system location
    mv /usr/local/bin/helm /usr/bin/helm

# Create a non-root user
ARG USERNAME=vscode
ARG USER_UID=2008
ARG USER_GID=$USER_UID

RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME \
    && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME

# Set bash as default shell for the user
RUN chsh -s /bin/bash $USERNAME

# Set up Docker CLI
RUN curl -fsSL https://get.docker.com -o get-docker.sh && sh get-docker.sh
RUN usermod -aG docker $USERNAME

# Set the working directory
WORKDIR /workspaces/dynamo

# Switch to non-root user
USER $USERNAME

# Copy Rust toolchain from builder
COPY --from=rust-builder /usr/local/cargo /usr/local/cargo
ENV PATH="/home/$USERNAME/.cargo/bin:${PATH}"

# Copy kubectl binary
COPY --from=kubectl /opt/bitnami/kubectl/bin/kubectl /usr/local/bin/

# Create .kube directory with proper permissions
RUN mkdir -p /home/$USERNAME/.kube && chown -R $USERNAME:$USERNAME /home/$USERNAME/.kube