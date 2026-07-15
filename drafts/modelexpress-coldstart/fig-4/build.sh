#!/usr/bin/env bash
#  SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#  SPDX-License-Identifier: Apache-2.0
#
# One-shot rebuild for fig-4 (cold-start phase breakdown).
# Creates a local .venv on first run, then renders the PNG + SVG.
set -euo pipefail
cd "$(dirname "$0")"

VENV=.venv
if [ ! -d "$VENV" ]; then
  python3 -m venv "$VENV"
  "$VENV/bin/python" -m pip install --quiet --upgrade pip
  "$VENV/bin/python" -m pip install --quiet plotly kaleido pyyaml
fi

"$VENV/bin/python" gen_fig_4_coldstart.py

# Lint the figure SOURCES only (never the .venv third-party tree).
LINT=../../../docs/skills/blog-figures/tools/lint_figures.py
if [ -f "$LINT" ]; then
  python3 "$LINT" gen_fig_4_coldstart.py plotly_dynamo.py --score || true
fi
