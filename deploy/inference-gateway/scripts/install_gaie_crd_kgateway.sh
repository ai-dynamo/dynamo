#!/usr/bin/env bash

# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

echo "install_gaie_crd_kgateway.sh is deprecated; installing agentgateway instead." >&2
exec "${SCRIPT_DIR}/install_gaie_crd_agentgateway.sh" "$@"
