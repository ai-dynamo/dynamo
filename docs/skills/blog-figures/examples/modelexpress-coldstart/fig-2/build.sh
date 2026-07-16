#!/usr/bin/env bash
#  SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#  SPDX-License-Identifier: Apache-2.0
#
# One-shot rebuild for fig-2 (ModelExpress RL training loop).
#
# Self-bootstrapping: if a local .venv exists it is reused; otherwise one is
# created here (inside fig-2/) and the render deps are installed into it. The
# .venv is a build tool, not a deliverable — keep it out of commits.
set -euo pipefail
cd "$(dirname "$0")"

VENV=".venv"
if [ ! -x "$VENV/bin/python" ]; then
  echo "Creating build venv in $VENV ..."
  python3 -m venv "$VENV"
  "$VENV/bin/python" -m pip install --quiet --upgrade pip
  "$VENV/bin/python" -m pip install --quiet plotly kaleido pyyaml
fi

"$VENV/bin/python" gen_fig_2_rl_loop.py
echo "Rendered images/fig-2-rl-loop.png"
