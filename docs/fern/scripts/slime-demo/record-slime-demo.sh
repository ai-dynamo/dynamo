#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Record slime-demo.sh to the Slime Integration overview asciicast and embed
# the Dynamo Glass theme that the landing-page hero demo uses.
#
# Usage:
#   ./record-slime-demo.sh                    # -> ../../assets/slime-rl-demo.cast at 120x25
#   ./record-slime-demo.sh out.cast 25 120    # custom output, rows, cols
#
# Requires: asciinema (3.x), python3.
set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT="${1:-${DIR}/../../assets/slime-rl-demo.cast}"
ROWS="${2:-25}"
COLS="${3:-120}"

# Without a terminal (for example, when an agent or CI records it), asciinema
# must run headless.
HEADLESS=""
[[ -t 1 ]] || HEADLESS="--headless"

# Run from the script directory so the cast header records a relative command.
case "$OUT" in /*) ;; *) OUT="${PWD}/${OUT}" ;; esac
cd "$DIR"
asciinema rec --overwrite ${HEADLESS} \
  --window-size "${COLS}x${ROWS}" \
  --idle-time-limit 2 \
  --command "bash slime-demo.sh" \
  "$OUT"

python3 "${DIR}/../hero-demo/apply-hero-theme.py" "$OUT"
echo "Recorded + themed: $OUT (${COLS}x${ROWS})"
