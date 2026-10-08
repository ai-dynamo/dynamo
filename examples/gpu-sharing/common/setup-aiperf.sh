#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# One-time: create a Python venv with AIPerf 0.11.0, the version used for every
# published sweep. run-sweep.sh uses ${AIPERF_VENV}/bin/aiperf by default.
#
# Environment:
#   AIPERF_VENV     venv directory          (default: $HOME/.aiperf-venv)
#   AIPERF_VERSION  AIPerf version to pin   (default: 0.11.0)
set -euo pipefail

AIPERF_VENV="${AIPERF_VENV:-${HOME}/.aiperf-venv}"
AIPERF_VERSION="${AIPERF_VERSION:-0.11.0}"

[ -d "${AIPERF_VENV}" ] || python3 -m venv "${AIPERF_VENV}"
"${AIPERF_VENV}/bin/pip" install -q --upgrade pip
"${AIPERF_VENV}/bin/pip" install -q "aiperf==${AIPERF_VERSION}"
"${AIPERF_VENV}/bin/aiperf" --version
