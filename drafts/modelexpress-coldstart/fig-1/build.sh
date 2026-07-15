#!/usr/bin/env bash
#  SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#  SPDX-License-Identifier: Apache-2.0
# Build fig-1 (ModelExpress cold-start sequence) in the Dynamo Dark aesthetic.
#
# Uses the self-contained venv in this folder if present, else falls back to
# python3 on PATH. Renders the PNG, then lints the generator against the
# canonical Dynamo Dark tokens (fails on any ERROR).
#
# Prerequisites (already in ./.venv): plotly, kaleido, pyyaml
#   python3 -m venv .venv && ./.venv/bin/python -m pip install plotly kaleido pyyaml
#
# Usage:
#   ./build.sh
set -euo pipefail
cd "$(dirname "$0")"

if [ -x ./.venv/bin/python ]; then
  PY=./.venv/bin/python
else
  PY=python3
fi

echo "==> Rendering fig-1 (ModelExpress cold-start sequence)..."
"$PY" gen_fig_1_coldstart_sequence.py

LINTER=../../../docs/skills/blog-figures/tools/lint_figures.py
if [ -f "$LINTER" ]; then
  echo "==> Linting sources against Dynamo Dark tokens (fails on ERROR)..."
  # Scope to the figure sources — the linter recurses, and .venv/ holds
  # thousands of third-party hex literals that are not part of this figure.
  "$PY" "$LINTER" --score --tokens ./design_tokens.yaml \
    gen_fig_1_coldstart_sequence.py plotly_dynamo.py
fi

echo "==> Done. Output:"
ls -lh images/*.png
