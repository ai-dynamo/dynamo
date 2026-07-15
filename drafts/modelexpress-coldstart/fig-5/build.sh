#!/usr/bin/env bash
#  SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#  SPDX-License-Identifier: Apache-2.0
# Rebuild fig-5 (NIXL registration-time scoreboard) in the Dynamo Dark aesthetic.
#
# Prerequisites: python3 with plotly, kaleido, pyyaml.
# A local .venv (created next to this script) is used if present.
#
# Usage:
#   ./build.sh
set -euo pipefail
cd "$(dirname "$0")"

# Bootstrap a local venv with the render deps if one is not already present.
if [ ! -x ".venv/bin/python" ]; then
  echo "==> Creating local .venv and installing plotly/kaleido/pyyaml..."
  python3 -m venv .venv
  .venv/bin/python -m pip install --quiet --upgrade pip
  .venv/bin/python -m pip install --quiet plotly kaleido pyyaml
fi
PY=".venv/bin/python"

echo "==> Rendering fig-5 (NIXL registration scoreboard)..."
"$PY" gen_fig_5_nixl_registration.py

echo "==> Linting sources against Dynamo Dark tokens (fails on ERROR)..."
LINT="../../../docs/skills/blog-figures/tools/lint_figures.py"
if [ -f "$LINT" ]; then
  "$PY" "$LINT" --score .
else
  echo "    (linter not found at $LINT — skipping)"
fi

echo "==> Done. Output:"
ls -lh images/*.png images/*.svg
