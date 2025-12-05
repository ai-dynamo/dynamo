#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Quick rebuild script for the kvbm Python bindings
# Uses local venv if exists, otherwise tries default location

set -e

echo "🔧 Rebuilding KVBM Python bindings..."
echo "=================================="

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
KVBM_BINDINGS_DIR="$REPO_ROOT/lib/bindings/kvbm"

# Find virtual environment
# Priority: local venv > local .venv > symlinked .venv
if [ -d "$SCRIPT_DIR/venv" ]; then
    VENV_PATH="$SCRIPT_DIR/venv"
elif [ -d "$SCRIPT_DIR/.venv" ] && [ ! -L "$SCRIPT_DIR/.venv" ]; then
    VENV_PATH="$SCRIPT_DIR/.venv"
elif [ -L "$SCRIPT_DIR/.venv" ] && [ -d "$(readlink -f "$SCRIPT_DIR/.venv")" ]; then
    VENV_PATH="$(readlink -f "$SCRIPT_DIR/.venv")"
else
    echo "❌ No virtual environment found!"
    echo "   Expected locations:"
    echo "     - $SCRIPT_DIR/venv"
    echo "     - $SCRIPT_DIR/.venv"
    echo ""
    echo "   Create one with: python -m venv $SCRIPT_DIR/venv"
    exit 1
fi

echo "📦 Using venv: $VENV_PATH"

# Activate the virtual environment
source "$VENV_PATH/bin/activate"

# Navigate to the kvbm Python bindings directory
cd "$KVBM_BINDINGS_DIR"

echo "📦 Building with maturin..."
echo "Build mode: dev (default)"
echo ""

# Build with maturin develop
maturin develop

echo ""
echo "✅ Build complete!"
echo "Package: kvbm"

# Verify the build
python -c "from kvbm._core import v2; print(f'kvbm._core.v2 loaded: {[a for a in dir(v2) if not a.startswith(\"_\")]}')"
