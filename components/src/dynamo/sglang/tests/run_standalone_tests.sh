#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Quick test runner for AFD standalone tests
# Run this to verify AFD implementation without full Dynamo runtime

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TEST_FILE="test_afd_standalone.py"

echo "=============================================="
echo "AFD Standalone Unit Tests"
echo "=============================================="

# Copy test to temp directory to avoid pyproject.toml conflicts
TEMP_DIR=$(mktemp -d)
cp "${SCRIPT_DIR}/${TEST_FILE}" "${TEMP_DIR}/"

# Run tests
cd "${TEMP_DIR}"
python -m pytest -v ${TEST_FILE} -k "not Performance" --tb=short

# Cleanup
rm -rf "${TEMP_DIR}"

echo ""
echo "=============================================="
echo "All tests passed!"
echo "=============================================="
