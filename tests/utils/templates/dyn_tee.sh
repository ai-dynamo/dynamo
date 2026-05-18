#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Pipe-through wrapper: tee stdout+stderr to a per-pod logfile on the
# mounted PVC without parsing or modifying the user's command. The
# user command lives in $@ and is execve'd directly — no shell
# interpretation of args, no quoting hazard, no shlex.join needed.
#
# Wraps via:
#   container.command = ["/shared/dyn-tee.sh"]
#   container.args    = <original command> + <original args>   # raw
#
# Required env:
#   DYN_LOG_DIR  — absolute path of the per-service log directory
#                  (e.g. /tmp/service_logs/service_logs/frontend)
#   POD_NAME     — pod name (Downward API ref)
#
set -euo pipefail

# Optional per-pod resource-limit override.
# Set DYN_TEST_NOFILE_LIMIT to the desired soft+hard nofile limit (e.g. 1024)
# to force file-descriptor exhaustion at small scale — useful for
# reproducing the F-1 frontend FD-exhaustion incident class.
# Defaults to whatever the container runtime grants (typically ~1M).
if [ -n "${DYN_TEST_NOFILE_LIMIT:-}" ]; then
    # Reduce both soft and hard limits. Without -H the soft limit can
    # be raised again at runtime; we want a hard ceiling.
    ulimit -n "${DYN_TEST_NOFILE_LIMIT}" || true
    ulimit -Hn "${DYN_TEST_NOFILE_LIMIT}" 2>/dev/null || true
    echo "[dyn_tee] DYN_TEST_NOFILE_LIMIT applied: soft=$(ulimit -n) hard=$(ulimit -Hn)" >&2
fi

# Raise RLIMIT_MEMLOCK so UCX/NIXL can allocate IB completion queues.
# On AKS clusters with the cri-containerd AppArmor profile + 64KB default
# memlock, only a root container with privileged + IPC_LOCK + appArmor
# Unconfined can actually call ulimit -l. See OPS-4332.
if [ -n "${DYN_TEST_MEMLOCK_UNLIMITED:-}" ]; then
    ulimit -l unlimited 2>/dev/null && \
        echo "[dyn_tee] memlock raised: $(ulimit -l)" >&2 || \
        echo "[dyn_tee] WARN: ulimit -l unlimited failed (cap=$(ulimit -l))" >&2
fi

mkdir -p "$DYN_LOG_DIR"
TS=$(date +%s)
LOG_FILE="$DYN_LOG_DIR/${POD_NAME:-pod}_${TS}.log"
# Line-buffered tee so SIGKILL still flushes the last line.
exec > >(stdbuf -oL tee -a "$LOG_FILE")
exec 2>&1
# execve directly — replaces this bash with the user's command, so
# kubelet's SIGTERM goes straight to the user process (PID 1).
exec "$@"
