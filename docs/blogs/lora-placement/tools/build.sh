#!/usr/bin/env bash
#  SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#  SPDX-License-Identifier: Apache-2.0
# Build all LoRA Placement figure assets.
#
# Prerequisites:
#   - python3 with plotly, kaleido, numpy, pyyaml
#   - rsvg-convert (optional, for SVG -> PNG of architecture diagrams)
#   - d2 (optional, only needed to re-render D2 -> SVG)
#
# Usage:
#   ./build.sh          # generate all figures
#   ./build.sh --d2     # also re-render D2 sources to SVGs first
set -euo pipefail
cd "$(dirname "$0")"

IMAGES=../images

if [[ "${1:-}" == "--d2" ]]; then
  echo "==> Rendering D2 sources to raw SVGs..."
  d2 --layout tala architecture-overview.d2  architecture-overview.svg
  d2 --layout tala control-loop.d2           control-loop.svg
  d2 --layout tala mcf-bipartite.d2          mcf-bipartite.svg
  echo "    Raw SVGs ready for legend injection."
fi

echo "==> Generating Figures 1-3 (architecture diagrams)..."
python3 gen_diagrams.py

if command -v rsvg-convert &> /dev/null; then
  echo "==> Rendering SVGs to 2x PNGs..."
  rsvg-convert -z 2 "${IMAGES}/fig-1-architecture-overview.svg" -o "${IMAGES}/fig-1-architecture-overview.png"
  rsvg-convert -z 2 "${IMAGES}/fig-2-control-loop.svg"          -o "${IMAGES}/fig-2-control-loop.png"
  rsvg-convert -z 2 "${IMAGES}/fig-3-mcf-bipartite.svg"         -o "${IMAGES}/fig-3-mcf-bipartite.png"
else
  echo "    rsvg-convert not found -- skipping PNG conversion for diagrams"
fi

echo "==> Generating Figures 4, 5, 7 (churn bar charts)..."
python3 gen_churn_bars.py

echo "==> Generating Figure 6 (spike timeline)..."
python3 gen_spike_timeline.py

echo "==> Done. Output files:"
ls -lh "${IMAGES}"/fig-*.{svg,png} 2>/dev/null || ls -lh "${IMAGES}"/fig-*
