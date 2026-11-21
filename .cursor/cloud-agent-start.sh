#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Integration tests start etcd/NATS themselves; nothing required on every boot.
set -euo pipefail

ROOT="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
if [[ -f "${ROOT}/.venv/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "${ROOT}/.venv/bin/activate"
fi
export PATH="/usr/local/bin:/usr/local/cargo/bin:${ROOT}/.venv/bin:${PATH}"

command -v cargo >/dev/null && command -v python3 >/dev/null
exit 0
