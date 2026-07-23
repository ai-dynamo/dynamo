#!/usr/bin/env bash
#  SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#  SPDX-License-Identifier: Apache-2.0
# Build the DynoSim Digest hero figure.
#
# Prerequisites:
#   - python3 with plotly, kaleido, numpy, pyyaml
#
# Usage:
#   ./build.sh          # regenerates ../dynosim-hero.png
set -euo pipefail
cd "$(dirname "$0")"

echo "==> Generating DynoSim hero (Pareto frontier)..."
python3 gen_hero.py

echo "==> Done."
ls -lh ../dynosim-hero.png
