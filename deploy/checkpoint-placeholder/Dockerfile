# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Checkpoint placeholder image: a runtime-compatible restore target, built
# from a Dynamo framework runtime image (BASE_IMAGE), with no snapshot/CRIU
# tooling baked in.
#
# Usage:
#   docker build --build-arg BASE_IMAGE=<framework-runtime-image> -t placeholder:latest .
#
# This is the runtime image with the same default execution contract
# (entrypoint/cmd) as BASE_IMAGE, running as root so the snapshot agent can
# restore into it. It ships NO checkpoint/restore tooling: criu, nsrestore,
# the cuda-checkpoint binaries, their shared libraries, and the criu plugins
# are all bind-mounted on demand from the agent's /snapshot-binaries bundle at
# restore time. The mount destination is created by the ns-bind-mount helper
# inside this namespace, so nothing needs pre-provisioning here.
#
# Three constraints follow from that and must hold for every BASE_IMAGE:
#   - BASE_IMAGE must be glibc-compatible with the snapshot agent's own base
#     image (ubuntu 24.04, glibc 2.39), because the bundle's library closure
#     is built from that image and loaded ahead of the placeholder's own
#     libraries via LD_LIBRARY_PATH. An older, incompatible glibc builds
#     green but fails at restore with "version 'GLIBC_2.38' not found".
#   - The node kernel must be >= 5.12 (mount_setattr, used by ns-bind-mount).
#   - The placeholder pod must NOT set readOnlyRootFilesystem: true unless a
#     writable volume is mounted at /tmp. The agent bind-mounts the binary
#     bundle into the placeholder's /tmp at restore time; with a read-only
#     root and nothing mounted at /tmp, creating the mount point fails with
#     EROFS. An emptyDir at /tmp (either medium) makes it work again, since
#     volume mounts stay writable independently of the root filesystem.
#
# cuda-checkpoint wrapping is opt-in (--cuda-checkpoint-wrap in snapshotctl).
# When used, the source workload image must have cuda-checkpoint at the same
# path as it runs from -- CRIU restores file-backed mappings by path, not via
# PATH search -- and so must this placeholder image (same BASE_IMAGE).
#
# Dynamo backend entrypoints run a restore standby path when the operator
# injects DYN_SNAPSHOT_RESTORE_STANDBY=1, then sleep while the snapshot agent
# restores into the container.

ARG BASE_IMAGE=placeholder-requires-base-image-arg
ARG DOCKER_PROXY=

# cuda-checkpoint is the one piece of tooling this image cannot receive from the
# agent's bind-mounted bundle. When the target requests more than one GPU the
# operator wraps the engine command as
#   cuda-checkpoint --launch-job -- <original command>
# (checkpoint_job.go: wrapLaunchJob := gpuCount > 1), so the binary must be
# resolvable in $PATH at container start -- long before the agent mounts
# anything. Without it multi-GPU capture dies in runc with
#   exec: "cuda-checkpoint": executable file not found in $PATH
# and the checkpoint fails as StartError before the engine ever runs.
#
# It must also be present on the restore side at the same path: the wrapper is
# the engine's parent process, so CRIU dumps and restores it too, and restores
# file-backed mappings by absolute path.
FROM ${DOCKER_PROXY}library/ubuntu:24.04 AS cuda-ckpt
RUN apt-get update && apt-get install -y --no-install-recommends git ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && git clone --depth 1 https://github.com/NVIDIA/cuda-checkpoint.git /tmp/cuda-checkpoint

FROM ${BASE_IMAGE}

ARG BASE_IMAGE
ARG TARGETARCH=amd64
ENV ORIGINAL_BASE_IMAGE=${BASE_IMAGE}

USER root

RUN if [ "${TARGETARCH}" != "amd64" ]; then \
      echo "ERROR: Dynamo Snapshot requires x86_64" >&2; exit 1; \
    fi

COPY --from=cuda-ckpt /tmp/cuda-checkpoint/bin/x86_64_Linux/cuda-checkpoint /usr/local/bin/cuda-checkpoint

# Checkpoint volume mount point. This is a storage concern -- no restore
# tooling is provisioned here; see the file header above.
RUN mkdir -p /checkpoints
