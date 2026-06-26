#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Simulate a process-level engine crash: SIGKILL the primary's entire process
# group. The GPU, node, and GMS server stay healthy; only the primary engine
# process tree dies. Mirrors the pytest failover injection:
#   os.killpg(os.getpgid(pid), signal.SIGKILL)
#
# Reads the primary PGID from $RUN_DIR/primary.pgid (written by run_engine.sh).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

pgid_file="${RUN_DIR}/primary.pgid"
if [[ ! -f "${pgid_file}" ]]; then
  echo "ERROR: ${pgid_file} not found; was the primary started by run_engine.sh?" >&2
  exit 1
fi

PRIMARY_PGID="$(cat "${pgid_file}")"
echo "    SIGKILL primary process group (pgid ${PRIMARY_PGID}) ..."

# The leading '-' targets the process GROUP. Tolerate an already-dead group.
if kill -KILL -"${PRIMARY_PGID}" 2>/dev/null; then
  echo "    primary process group ${PRIMARY_PGID} killed"
else
  echo "    primary process group ${PRIMARY_PGID} was already gone"
fi
