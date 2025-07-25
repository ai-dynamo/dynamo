# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

FROM nixl:v0.1.0.dev.0780842

# Install sudo, LLVM (latest stable), and configure passwordless sudo for all users
RUN apt-get update && \
    apt-get install -y sudo wget lsb-release software-properties-common gnupg ca-certificates git-lfs&& \
    # Install latest stable LLVM using official script
    bash -c "$(wget -O - https://apt.llvm.org/llvm.sh)" && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    # Add all users to sudo group and enable passwordless sudo
    echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers

RUN git lfs install
