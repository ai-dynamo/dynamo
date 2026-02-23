{#
# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#}
# === BEGIN templates/local_dev.Dockerfile ===
# ======================================================================
# TARGET: local-dev (non-root development with UID/GID remapping)
# ======================================================================
{% if make_efa != true %}
FROM dev AS local-dev
{% else %}
FROM aws AS local-dev
{% endif %}

ENV USERNAME=dynamo
ARG USER_UID
ARG USER_GID

# Copy rustup home into a writable per-user location so sanity_check passes.
# (dev target already has rustup/cargo/maturin from concatenated wheel_builder/dynamo_base)
RUN cp -r /usr/local/rustup /home/dynamo/.rustup && \
    chown -R dynamo:0 /home/dynamo/.rustup

# Put rustup state under the user's home (writable) while still using /usr/local/cargo/bin shims.
ENV RUSTUP_HOME=/home/${USERNAME}/.rustup
ENV CARGO_HOME=/home/${USERNAME}/.cargo
ENV PATH=/usr/local/cargo/bin:/usr/local/bin:${CARGO_HOME}/bin:${PATH}

# https://code.visualstudio.com/remote/advancedcontainers/add-nonroot-user
# Configure user with sudo access for Dev Container workflows
#
# 🚨 PERFORMANCE / PERMISSIONS MEMO (DO NOT VIOLATE)
# NEVER use `chown -R` or `chmod -R` in local-dev images.
# - It can take minutes on large mounts (and makes devcontainers feel "hung")
# - It is unnecessary: permissioning should be done via COPY --chmod/--chown and a few targeted, non-recursive ops.
# If you think you need recursion here, stop and redesign the permissions flow.
RUN mkdir -p /etc/sudoers.d \
    && echo "$USERNAME ALL=(root) NOPASSWD:ALL" > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME \
    && mkdir -p /home/$USERNAME \
    # Handle GID conflicts: if target GID exists and it's not our group, remove it
    && (getent group $USER_GID | grep -v "^$USERNAME:" && groupdel $(getent group $USER_GID | cut -d: -f1) || true) \
    # Create group if it doesn't exist, otherwise modify existing group
    && (getent group $USERNAME > /dev/null 2>&1 && groupmod -g $USER_GID $USERNAME || groupadd -g $USER_GID $USERNAME) \
    && usermod -u $USER_UID -g $USER_GID -G 0 $USERNAME \
    && chown $USERNAME:$USER_GID /home/$USERNAME \
    && chsh -s /bin/bash $USERNAME

# Set workspace directory variable
ENV WORKSPACE_DIR=${WORKSPACE_DIR}

# Development environment variables for the local-dev target
# Path configuration notes:
# - DYNAMO_HOME: Main project directory (workspace mount point)
# - CARGO_TARGET_DIR: Build artifacts in workspace/target for persistence
# - PATH: Includes cargo binaries for rust tool access
ENV HOME=/home/$USERNAME
ENV DYNAMO_HOME=${WORKSPACE_DIR}
ENV CARGO_TARGET_DIR=${WORKSPACE_DIR}/target
ENV PATH=${CARGO_HOME}/bin:$PATH

# Switch to dynamo user (dev stage has umask 002, so files should already be group-writable)
USER $USERNAME
WORKDIR $HOME

# Create user-level cargo/rustup state dirs as the target user (avoids root-owned caches).
RUN mkdir -p "${CARGO_HOME}" "${RUSTUP_HOME}"

# Ensure Python user site-packages exists and is writable (important for non-venv frameworks like SGLang).
RUN python3 -c 'import os, site; p = site.getusersitepackages(); os.makedirs(p, exist_ok=True); print(p)'

# https://code.visualstudio.com/remote/advancedcontainers/persist-bash-history
RUN SNIPPET="export PROMPT_COMMAND='history -a' && export HISTFILE=$HOME/.commandhistory/.bash_history" \
    && mkdir -p $HOME/.commandhistory \
    && chmod g+w $HOME/.commandhistory \
    && touch $HOME/.commandhistory/.bash_history \
    && echo "$SNIPPET" >> "$HOME/.bashrc"

RUN mkdir -p /home/$USERNAME/.cache/ \
    && mkdir -p /home/$USERNAME/.cache/pre-commit \
    && chmod g+w /home/$USERNAME/.cache/ \
    && chmod g+w /home/$USERNAME/.cache/pre-commit

ENTRYPOINT ["/opt/nvidia/nvidia_entrypoint.sh"]
CMD []
