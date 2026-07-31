#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Record hero-demo.sh to the landing-page asciicast and embed the Dynamo Glass
# theme. Folds the record + theme-injection steps into one.
#
# Usage:
#   ./record-hero.sh                    # -> ../../assets/hero-demo-25.cast at 120x25
#   ./record-hero.sh out.cast 28        # custom output + row count
#   ./record-hero.sh out.cast 25 120    # custom output, rows, cols
#
# Requires: asciinema (3.x), python3.
set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT="${1:-${DIR}/../../assets/hero-demo-25.cast}"
ROWS="${2:-25}"
COLS="${3:-120}"

asciinema rec --overwrite \
  --window-size "${COLS}x${ROWS}" \
  --idle-time-limit 2 \
  --command "bash ${DIR}/hero-demo.sh" \
  "$OUT"

python3 "${DIR}/apply-hero-theme.py" "$OUT"
echo "Recorded + themed: $OUT (${COLS}x${ROWS})"
