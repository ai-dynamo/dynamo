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
# This image additionally ships the POSIX CUDA VMM interposer for GMS V1
# workloads: the shim at /usr/local/lib/snapshot/libcuinterposer.so and its
# coordinator at /usr/local/bin/cuinterposer-coordinator. See the cuinterposer
# build stage below for why, and agent/cmd/cuinterpose/README.md in
# ai-dynamo/snapshot for the launch contract.
#
# Dynamo backend entrypoints run a restore standby path when the operator
# injects DYN_SNAPSHOT_RESTORE_STANDBY=1, then sleep while the snapshot agent
# restores into the container.

ARG BASE_IMAGE=placeholder-requires-base-image-arg
ARG DOCKER_PROXY=
# Global build args must precede the first FROM to be visible to every stage.
# SNAPSHOT_REF must match the snapshot agent build serving these pods: the shim
# and the agent's coordinator speak a versioned wire protocol
# (CUINTERPOSER_VERSION in protocol.h).
ARG SNAPSHOT_REPO=https://github.com/ai-dynamo/snapshot.git
ARG SNAPSHOT_REF=main

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

# The POSIX CUDA VMM interposer. GMS V1 hands the engine its weights as VMM
# allocations shared with CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR, and the GMS
# servers that export them are deliberately NOT in the checkpoint. Without the
# shim, cuda-checkpoint blocks forever trying to checkpoint memory whose creator
# lives outside the target process tree.
#
# Per agent/cmd/cuinterpose/README.md the workload image must provide the shim
# at /usr/local/lib/snapshot/libcuinterposer.so; snapshot never injects it into
# the workload image. Only the engine sets LD_PRELOAD -- the GMS server must not
# preload it -- and restore pods must not preload either, because CRIU restores
# the already-interposed tree.
#
# The coordinator ships alongside it: nsrestore runs inside this container's
# mount namespace and execs /usr/local/bin/cuinterposer-coordinator by absolute
# path (agent/internal/cuda/cuinterpose.go), and it is NOT part of the agent's
# injected /tmp/snapshot-binaries bundle. Missing it fails restore with
#   restore CUDA interposition: /usr/local/bin/cuinterposer-coordinator failed:
#   no such file or directory
# after CRIU has already restored the process tree. ai-dynamo/snapshot's own
# placeholder stage installs both for the same reason.
#
# SNAPSHOT_REF must match the snapshot agent build serving these pods: the shim
# and the coordinator speak a versioned wire protocol
# (CUINTERPOSER_VERSION in protocol.h).
# SNAPSHOT_REF/SNAPSHOT_REPO are declared above the first FROM so they reach
# this stage; they are re-declared here to bring them into its scope.
FROM ${DOCKER_PROXY}nvidia/cuda:13.0.0-devel-ubuntu24.04 AS cuinterposer
ARG SNAPSHOT_REPO
ARG SNAPSHOT_REF
RUN apt-get update && apt-get install -y --no-install-recommends git ca-certificates build-essential \
    && rm -rf /var/lib/apt/lists/*
RUN git clone --filter=blob:none --no-checkout "${SNAPSHOT_REPO}" /src \
    && git -C /src fetch --depth 1 origin "${SNAPSHOT_REF}" \
    && git -C /src checkout FETCH_HEAD
RUN make -C /src/agent/cmd/cuinterpose CUDA_HOME=/usr/local/cuda \
    && test -f /src/agent/cmd/cuinterpose/build/libcuinterposer.so \
    && test -f /src/agent/cmd/cuinterpose/build/cuinterposer-coordinator

FROM ${BASE_IMAGE}

ARG BASE_IMAGE
ARG TARGETARCH=amd64
ENV ORIGINAL_BASE_IMAGE=${BASE_IMAGE}

USER root

RUN if [ "${TARGETARCH}" != "amd64" ]; then \
      echo "ERROR: Dynamo Snapshot requires x86_64" >&2; exit 1; \
    fi

COPY --from=cuda-ckpt /tmp/cuda-checkpoint/bin/x86_64_Linux/cuda-checkpoint /usr/local/bin/cuda-checkpoint
COPY --from=cuinterposer /src/agent/cmd/cuinterpose/build/libcuinterposer.so /usr/local/lib/snapshot/libcuinterposer.so
COPY --from=cuinterposer /src/agent/cmd/cuinterpose/build/cuinterposer-coordinator /usr/local/bin/cuinterposer-coordinator
RUN chmod 0644 /usr/local/lib/snapshot/libcuinterposer.so \
    && chmod 0755 /usr/local/bin/cuinterposer-coordinator

# Checkpoint volume mount point. This is a storage concern -- no restore
# tooling is provisioned here; see the file header above.
RUN mkdir -p /checkpoints
