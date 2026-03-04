#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Generate all API reference docs.
#
# Usage (from repository root):
#   docs/scripts/generate_api_docs.sh                    # generate only
#   docs/scripts/generate_api_docs.sh --version 1.0.0    # with Rust version

set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

RUST_ARGS=()
if [[ "${1:-}" == "--version" ]]; then
    RUST_ARGS=("--version" "${2:?Missing version argument}")
fi

echo "=== Generating API Reference Docs ==="
echo

# 1. Python API (griffe → docs/api/python/README.md)
echo "[1/3] Python API..."
python3 docs/scripts/generate_python_api.py &
PID_PYTHON=$!

# 2. K8s CRD API (fernify → docs/kubernetes/api-reference.md)
# Note: git checkout restores the raw committed file before fernifying.
# This is safe because the fernified version is never committed — only the
# raw version is tracked. If you have local edits to api-reference.md,
# stash them first.
if ! git diff --quiet docs/kubernetes/api-reference.md 2>/dev/null; then
    echo "[2/3] K8s CRD API (restoring raw source before formatting)..."
    git checkout -- docs/kubernetes/api-reference.md
else
    echo "[2/3] K8s CRD API..."
fi
python3 docs/scripts/fernify_k8s_api.py &
PID_K8S=$!

# 3. Rust API (Cargo.toml → docs/api/rust/README.md)
echo "[3/3] Rust API..."
python3 docs/scripts/generate_rust_api.py "${RUST_ARGS[@]}" &
PID_RUST=$!

# Wait for all generators
wait "$PID_PYTHON" "$PID_K8S" "$PID_RUST"
echo
echo "=== Done ==="
