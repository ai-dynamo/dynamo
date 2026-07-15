#!/usr/bin/env bash
#  SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#  SPDX-License-Identifier: Apache-2.0
#
# Rebuild fig-6 (Cold Start Anatomy) from a clean checkout.
#
# Bootstraps a local .venv (plotly + kaleido + PyYAML), renders the figure to
# images/, then lints the generator sources against the Dynamo Dark tokens.
# Re-runnable without args; reuses the venv if it already exists.
#
# Prerequisites: python3 (3.10+) and network access for the first run
# (kaleido downloads a headless Chrome on first render).
#
# Usage:
#   ./build.sh
set -euo pipefail
cd "$(dirname "$0")"

VENV=.venv
if [ ! -x "$VENV/bin/python" ]; then
  echo "==> Creating venv + installing deps..."
  python3 -m venv "$VENV"
  "$VENV/bin/pip" install --quiet --upgrade pip
  "$VENV/bin/pip" install --quiet plotly==6.9.0 kaleido==1.3.0 PyYAML==6.0.3
fi

echo "==> Rendering fig-6-coldstart-anatomy.png..."
"$VENV/bin/python" gen_fig_6_coldstart_anatomy.py

LINT=../../../docs/skills/blog-figures/tools/lint_figures.py
if [ -f "$LINT" ]; then
  echo "==> Linting sources against Dynamo Dark tokens (fails on ERROR)..."
  # Lint the figure sources only (not the local .venv third-party packages).
  "$VENV/bin/python" "$LINT" gen_fig_6_coldstart_anatomy.py plotly_dynamo.py --score
fi

echo "==> Done."
ls -lh images/*.png
