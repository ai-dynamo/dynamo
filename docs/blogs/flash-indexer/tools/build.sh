#!/usr/bin/env bash
# Build all Flash Indexer figure assets.
#
# Prerequisites:
#   - rsvg-convert (brew install librsvg)
#   - python3 with plotly, kaleido, numpy, pyyaml
#   - d2 (optional, only needed to re-render D2 -> SVG)
#
# Usage:
#   ./build.sh          # render PNGs from SVGs + heatmap + sweep chart
#   ./build.sh --d2     # also re-render D2 sources to SVGs first
set -euo pipefail
cd "$(dirname "$0")"

IMAGES=../images

if [[ "${1:-}" == "--d2" ]]; then
  echo "==> Rendering D2 sources to raw SVGs..."
  d2 --layout tala event-flow.d2      event-flow-raw.svg
  d2 --layout elk  radix-tree.d2      radix-tree-raw.svg
  d2 --layout tala write-read-path.d2 write-read-path-raw.svg
  echo "    NOTE: Raw SVGs need post-processing before legend injection."
fi

echo "==> Injecting legends + padding into SVGs -> ${IMAGES}/"
python3 inject_legends.py

echo "==> Rendering SVGs to 2x PNGs..."
rsvg-convert -z 2 "${IMAGES}/fig-2-kv-event-flow.svg"    -o "${IMAGES}/fig-2-kv-event-flow.png"
rsvg-convert -z 2 "${IMAGES}/fig-3-prefix-tree.svg"       -o "${IMAGES}/fig-3-prefix-tree.png"
rsvg-convert -z 2 "${IMAGES}/fig-4-concurrency-model.svg" -o "${IMAGES}/fig-4-concurrency-model.png"
rsvg-convert -z 2 "${IMAGES}/fig-5-jump-search.svg"       -o "${IMAGES}/fig-5-jump-search.png"

echo "==> Generating Figure 1 (heatmap)..."
python3 gen_heatmap.py

echo "==> Generating Figure 6 (throughput chart)..."
python3 gen_throughput.py ../data/sweep_plot.json

echo "==> Done. Output files:"
ls -lh "${IMAGES}"/fig-*.{svg,png} 2>/dev/null || ls -lh "${IMAGES}"/fig-*
