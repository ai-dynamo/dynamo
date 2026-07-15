#!/usr/bin/env bash
#  SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#  SPDX-License-Identifier: Apache-2.0
#
# Rebuild the ModelExpress / cold-start HERO from a clean checkout.
#
# Bootstraps a local .venv (plotly + kaleido + PyYAML), renders the hero to
# images/ (PNG + SVG), then lints the generator sources against the Dynamo Dark
# tokens. Re-runnable without args; reuses the venv if it already exists.
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

echo "==> Rendering hero-modelexpress-coldstart.{png,svg}..."
"$VENV/bin/python" tools/gen_hero.py

LINT=../../../docs/skills/blog-figures/tools/lint_figures.py
if [ -f "$LINT" ]; then
  echo "==> Linting sources against Dynamo Dark tokens (fails on ERROR)..."
  # Lint the figure sources only (not the local .venv third-party packages).
  "$VENV/bin/python" "$LINT" tools/gen_hero.py tools/plotly_dynamo.py --score
fi

echo "==> Done."
ls -lh images/hero-modelexpress-coldstart.*
