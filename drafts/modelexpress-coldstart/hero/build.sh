#!/usr/bin/env bash
#  SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#  SPDX-License-Identifier: Apache-2.0
#
# Rebuild the ModelExpress / cold-start HERO from a clean checkout.
#
# Bootstraps a shared .venv (plotly + kaleido + PyYAML), renders the hero to
# images/ (PNG + SVG), then lints the generator sources against the Dynamo Dark
# tokens. Re-runnable without args; reuses the venv if it already exists.
#
# The venv lives in the sibling hero-concepts/ folder (../hero-concepts/.venv) so
# it is shared with the explored concepts AND kept out of this hero folder - the
# hero directory is lint-scanned recursively, and third-party packages would trip
# the raw-hex gate.
#
# Prerequisites: python3.10+ (system python3 may be too old for kaleido) and
# network access for the first run (kaleido downloads a headless Chrome).
#
# Usage:
#   ./build.sh
set -euo pipefail
cd "$(dirname "$0")"

VENV=../hero-concepts/.venv
if [ ! -x "$VENV/bin/python" ]; then
  echo "==> Creating shared venv + installing deps..."
  # kaleido 1.3.0 needs Python 3.10+; prefer a modern interpreter, fall back to python3.
  PY=python3
  for cand in python3.13 python3.12 python3.11 python3.10; do
    if command -v "$cand" >/dev/null 2>&1; then PY="$cand"; break; fi
  done
  "$PY" -m venv "$VENV"
  "$VENV/bin/pip" install --quiet --upgrade pip
  "$VENV/bin/pip" install --quiet plotly==6.9.0 kaleido==1.3.0 PyYAML==6.0.3
fi

echo "==> Rendering hero-modelexpress-coldstart.{png,svg}..."
"$VENV/bin/python" tools/gen_hero.py

LINT=../../../docs/skills/blog-figures/tools/lint_figures.py
if [ -f "$LINT" ]; then
  echo "==> Linting the hero folder against Dynamo Dark tokens (fails on ERROR)..."
  # The shared .venv lives outside hero/, so scanning the folder only hits the
  # figure sources (gen_hero.py, plotly_dynamo.py) - no third-party packages.
  "$VENV/bin/python" "$LINT" . --score
fi

echo "==> Done."
ls -lh images/hero-modelexpress-coldstart.*
