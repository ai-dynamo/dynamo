#!/usr/bin/env bash
#  SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#  SPDX-License-Identifier: Apache-2.0
# Build every example figure in the blog-figures skill.
#
# Prerequisites:
#   - python3 with plotly, kaleido, numpy, pyyaml   (all generators)
#   - playwright + chromium                          (gen_fig_cards.py only)
#       pip install playwright && playwright install chromium
#
# Usage:
#   ./build.sh
set -euo pipefail
cd "$(dirname "$0")"

echo "==> Architecture / data-flow diagram..."
python3 gen_fig_2_architecture.py
echo "==> Decision cascade (delta bracket)..."
python3 gen_fig_5_decision_cascade.py
echo "==> Tuning loop (phase tags + feedback loop)..."
python3 gen_fig_6_tuning_loop.py
echo "==> Concurrency sweep + Pareto (dual panel)..."
python3 gen_fig_concurrency_sweep.py
echo "==> Throughput scoreboard (compact title)..."
python3 gen_fig_throughput_bars.py
echo "==> Comparison cards (HTML -> PNG)..."
python3 gen_fig_cards.py

echo "==> Linting sources against Dynamo Dark tokens (fails on ERROR)..."
python3 ../tools/lint_figures.py --score .

echo "==> Done. Output:"
ls -lh images/*.png
