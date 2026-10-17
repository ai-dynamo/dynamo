#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Serve-test entry point for the remote CustomEncoder topology.
#
# The serve harness launches "<directory>/launch/<script>" and hands each
# topology dynamic ports as DYN_HTTP_PORT and DYN_SYSTEM_PORT<N>. The remote
# example instead names its two worker ports DYN_GENERATOR_SYSTEM_PORT and
# DYN_ORCHESTRATOR_SYSTEM_PORT, so map them here; without the mapping, parallel
# topologies would collide on the 8081/8082 defaults.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export DYN_GENERATOR_SYSTEM_PORT="${DYN_GENERATOR_SYSTEM_PORT:-${DYN_SYSTEM_PORT1:-8081}}"
export DYN_ORCHESTRATOR_SYSTEM_PORT="${DYN_ORCHESTRATOR_SYSTEM_PORT:-${DYN_SYSTEM_PORT2:-8082}}"

exec "$SCRIPT_DIR/../remote/launch.sh" "$@"
