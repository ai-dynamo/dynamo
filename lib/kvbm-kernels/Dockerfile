# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Use CUDA 12.9.0 development image as base
FROM nvidia/cuda:12.9.0-devel-ubuntu22.04

# Set environment variables
ENV RUSTUP_HOME=/usr/local/rustup \
    CARGO_HOME=/usr/local/cargo \
    PATH=/usr/local/cargo/bin:$PATH \
    CUDA_PATH=/usr/local/cuda

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    build-essential \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Install Rust
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain stable

# Set working directory
WORKDIR /app

# Copy project files
COPY Cargo.toml ./
COPY src ./src
COPY build.rs ./build.rs
COPY cuda ./cuda

# Build the project
RUN cargo build --release

# Set the entrypoint to run the binary
CMD ["cargo", "test"]
