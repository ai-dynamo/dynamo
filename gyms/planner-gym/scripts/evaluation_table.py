#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Print the final evaluation tables (every method) from published results JSON.

    python scripts/evaluation_table.py \
        --results agg=runs/match-sim-agg-all-datasets/results.agg.json \
        --results disagg=runs/match-sim-disagg-all-datasets/results.disagg.json \
        --out runs/evaluation.md
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))

if __name__ == "__main__":
    from autoscaling_arena.evaluation_table import main

    raise SystemExit(main())
