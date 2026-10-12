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
ARG DEVICE

# rustup is already at /home/dynamo/.rustup from the dev stage (COPY --from=wheel_builder
# with --chown=dynamo:0 --chmod=775), so no re-copy needed here.
ENV RUSTUP_HOME=/home/${USERNAME}/.rustup
ENV CARGO_HOME=/home/${USERNAME}/.cargo
ENV PATH=${VIRTUAL_ENV}/bin:/usr/local/cargo/bin:/usr/local/bin:${CARGO_HOME}/bin:${PATH}

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
    # Ensure user-writable XDG dirs. Upstream base images (e.g. vLLM CPU/XPU) may pre-create
    # /home/dynamo/.local as root:root, which breaks `python3 -c 'site.getusersitepackages()'`
    # and other user-site installs after we switch to the non-root user below.
    # These dirs are small (a few KB), so a shallow chown is acceptable and does NOT
    # violate the "no recursive chown on large mounts" rule.
    && mkdir -p /home/$USERNAME/.local/lib /home/$USERNAME/.local/bin /home/$USERNAME/.local/share \
    && chown -R $USERNAME:$USER_GID /home/$USERNAME/.local \
    && [ ! -e /home/$USERNAME/.gitconfig ] || chown $USERNAME:$USER_GID /home/$USERNAME/.gitconfig \
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
ENV PATH=${VIRTUAL_ENV}/bin:${CARGO_HOME}/bin:$PATH

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

# Framework runtime bases bake their own uv cache location into the image env
# (vllm/vllm-openai sets UV_CACHE_DIR=/opt/uv/cache), and its entries are root-owned
# 0755. That works while the image runs as root, but this stage remaps to a non-root
# user, so afterwards every uv call dies with:
#   error: Failed to initialize cache at `/opt/uv/cache`
#   Caused by: failed to open file `/opt/uv/cache/sdists-v9/.git`: Permission denied
# which breaks the two commands the dev docs tell you to run (`maturin develop --uv`
# and `uv pip install --no-deps -e /workspace`).
#
# Point the cache at the user's own cache dir — the same /home/dynamo/.cache/uv path
# the runtime, frontend and planner stages already use. `chmod -R g+w /opt/uv/cache`
# would also work but copies a ~780MB tree into a new layer, which the permissions
# memo above rules out.
# Use a cache path this stage owns outright rather than an inherited one. uv does not
# only create new entries — it OPENS existing cache metadata on init, which is the exact
# original failure (`failed to open file .../sdists-v9/.git: Permission denied`). So a
# shallow chown of the cache root is not enough: any root-owned file nested inside an
# inherited cache re-breaks it. trtllm already ships ~27MB of root-owned content under
# /home/dynamo/.cache/uv, group-writable today but not guaranteed to stay that way.
# Recursively chmod-ing instead would copy a 781MB tree into a new layer, which the
# permissions memo above rules out.
ENV UV_CACHE_DIR=/home/${USERNAME}/.cache/uv-local-dev

# Prepare the dirs while we are still root. `usermod -u` above re-chowns only files owned
# by the OLD uid, so whatever the base left root-owned survives the remap; as the remapped
# user we could neither chmod such a dir nor mkdir inside a root-owned parent, which turns
# any base that ships 0755 into a hard build failure.
USER root
RUN set -eux; \
    for d in /home/$USERNAME/.cache /home/$USERNAME/.cache/pre-commit "$UV_CACHE_DIR"; do \
        mkdir -p "$d"; \
        chown "$USERNAME:$USER_GID" "$d"; \
        chmod g+rwx "$d"; \
    done
USER $USERNAME

{% if device == "xpu" or device == "cpu" %}
SHELL ["bash", "-c"]
CMD ["bash", "-c", "source /home/$USERNAME/.bashrc && exec bash"]
{% elif framework == "vllm" %}
# The upstream vllm/vllm-openai images do not ship the generic NVIDIA
# entrypoint path used by some CUDA bases. Keep local-dev aligned with the
# vLLM runtime template, which resets ENTRYPOINT for arbitrary commands.
ENTRYPOINT []
CMD ["bash", "-c", "source /home/$USERNAME/.bashrc && exec bash"]
{% else %}
ENTRYPOINT ["/opt/nvidia/nvidia_entrypoint.sh"]
CMD []
{% endif %}
