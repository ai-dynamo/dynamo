#!/usr/bin/env bash
#  SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#  SPDX-License-Identifier: Apache-2.0
# Build all agentic-harnesses blog figures.
#
# Each Plotly script writes both a 2x PNG and an SVG into ../images.
#
# Prerequisites:
#   python3 with plotly, kaleido, numpy, pyyaml
#
# Usage:
#   ./build.sh
set -euo pipefail
cd "$(dirname "$0")"

echo "==> Figure 1: agent loop"
python3 gen_fig_1_agent_loop.py

echo "==> Figure 2: TTFT prefix stability"
python3 gen_fig_2_ttft_prefix_stability.py

echo "==> Figure 3: streaming dispatch timeline"
python3 gen_fig_3_streaming_dispatch.py

echo "==> Done. Output files:"
ls -lh ../images/fig-*.{svg,png}
