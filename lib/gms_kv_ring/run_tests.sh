#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Run the gms_kv_ring suite. Uses pytest-xdist with 4 subprocess
# workers — each worker is a fresh Python process, so CUDA driver
# state accumulated over many engine attach/detach cycles never
# crosses test boundaries. This keeps CUDA initialization isolated across workers.
#
# Why xdist and not pytest-forked: pytest-forked uses os.fork(), which
# CUDA doesn't survive (cuInit returns CUDA_ERROR_NOT_INITIALIZED in
# the child). xdist spawns subprocess workers via execnet, so each
# worker initializes CUDA fresh.
#
# Why -n 4 and not -n auto: with full parallelism, the Rust speedup
# microbench can fail on a heavily-loaded CPU since Python and Rust
# both slow down but at different rates. -n 4 keeps the worker count
# below typical CPU count.
#
# Pass-through args: anything you pass is forwarded to pytest.

set -euo pipefail

cd "$(dirname "$0")/../.."

PYTHON="${PYTHON:-python}"
WORKERS="${WORKERS:-4}"

exec "$PYTHON" -m pytest lib/gms_kv_ring/tests/ \
    --timeout=30 -n "$WORKERS" --dist worksteal \
    "$@"
