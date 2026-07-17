#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Record hero-demo.sh to an asciicast and embed the GitHub Dark Default theme
# (with classic yellow kept). Folds the record + theme-injection steps into one.
#
# Usage:
#   ./record-hero.sh                       # -> hero-demo.cast at 120x32
#   ./record-hero.sh hero-demo-28.cast 28  # custom output + row count
#   ./record-hero.sh out.cast 25 120       # custom output, rows, cols
#
# Requires: asciinema (3.x), python3.
set -euo pipefail

OUT="${1:-hero-demo.cast}"
ROWS="${2:-32}"
COLS="${3:-120}"
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

asciinema rec --overwrite \
  --window-size "${COLS}x${ROWS}" \
  --idle-time-limit 2 \
  --command "bash ${DIR}/hero-demo.sh" \
  "$OUT"

python3 "${DIR}/apply-hero-theme.py" "$OUT"
echo "Recorded + themed: $OUT (${COLS}x${ROWS})"
