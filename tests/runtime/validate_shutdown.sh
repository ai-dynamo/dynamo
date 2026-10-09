#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# CPU shutdown validation. Requires bindings rebuilt from this checkout.
set -euo pipefail

SHUTDOWN_REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$SHUTDOWN_REPO_ROOT"
SHUTDOWN_PYTHON="${SHUTDOWN_PYTHON:-python3}"

cargo test -p dynamo-runtime --lib runtime::tests::shutdown
cargo test -p dynamo-runtime --lib draining_uses_compatible_worker_unavailable_wire_identity
cargo test -p dynamo-runtime --lib shutdown_budget_includes_post_cancellation_teardown
cargo test -p dynamo-backend-common --lib worker::tests -- --test-threads=1
cargo test -p dynamo-backend-common --lib shutdown::tests -- --test-threads=1

# Avoid unrelated optional pytest plugins and GPU/container fixtures. Each
# process probe owns its signals and ports; no inference engine is launched.
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
"$SHUTDOWN_PYTHON" -m pytest -c pyproject.toml -o addopts='' \
    -p pytest_asyncio.plugin -p pytest_timeout \
    --confcutdir=components/src/dynamo/common/utils/tests \
    components/src/dynamo/common/utils/tests/test_worker_shutdown.py -q
"$SHUTDOWN_PYTHON" -m pytest -c pyproject.toml -o addopts='' \
    -p pytest_asyncio.plugin -p pytest_timeout \
    tests/runtime/test_python_worker_shutdown.py \
    tests/runtime/test_worker_shutdown_process.py -q
