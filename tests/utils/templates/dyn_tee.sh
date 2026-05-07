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
mkdir -p "$DYN_LOG_DIR"
TS=$(date +%s)
LOG_FILE="$DYN_LOG_DIR/${POD_NAME:-pod}_${TS}.log"
# Line-buffered tee so SIGKILL still flushes the last line.
exec > >(stdbuf -oL tee -a "$LOG_FILE")
exec 2>&1
# execve directly — replaces this bash with the user's command, so
# kubelet's SIGTERM goes straight to the user process (PID 1).
exec "$@"
