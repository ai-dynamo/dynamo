#!/usr/bin/env bash
#  SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#  SPDX-License-Identifier: Apache-2.0
#
# Reproducible one-shot build for fig-3 (ModelExpress cold-start architecture).
#
# Bootstraps an isolated venv inside this directory (if missing), installs the
# Plotly + kaleido + pyyaml deps, renders the figure to images/, then lints the
# figure SOURCES against the Dynamo Dark tokens.
#
# The linter is pointed at the generator + plotly helper (not the whole dir) so
# it does not recurse into the local .venv (third-party package sources).
#
# Usage:
#   ./build.sh
set -euo pipefail
cd "$(dirname "$0")"

VENV=.venv
PY="$VENV/bin/python"

if [ ! -x "$PY" ]; then
  echo "==> Creating venv ($VENV)..."
  if command -v uv >/dev/null 2>&1; then
    uv venv --python 3.13 "$VENV"
    uv pip install --python "$PY" plotly kaleido pyyaml
  else
    python3 -m venv "$VENV"
    "$PY" -m pip install --quiet --upgrade pip
    "$PY" -m pip install --quiet plotly kaleido pyyaml
  fi
fi

echo "==> Rendering ModelExpress cold-start architecture..."
"$PY" gen_fig_3_modelexpress.py

echo "==> Linting figure sources against Dynamo Dark tokens (fails on ERROR)..."
LINTER=../../../docs/skills/blog-figures/tools/lint_figures.py
python3 "$LINTER" gen_fig_3_modelexpress.py plotly_dynamo.py \
  --tokens design_tokens.yaml --score

echo "==> Done. Output:"
ls -lh images/fig-3-modelexpress-coldstart.*
