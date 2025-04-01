FROM ubuntu:24.04

# Avoid prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Install dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    python3-dev \
    python3-pip \
    python3-venv \
    libucx0 \
    pkg-config \
    libssl-dev \
    jq \
    wget \
    sudo \
    protobuf-compiler \
    bash \
    && rm -rf /var/lib/apt/lists/*

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

# Install Rust for the vscode user
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- --default-toolchain stable --profile default -y \
    && . $HOME/.cargo/env

ENV PATH="/home/$USERNAME/.cargo/bin:${PATH}" 