#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Print mean-rank / pairwise-win-rate tables from a published results JSON.

    python scripts/rank_tables.py runs/match-sim-agg-all-datasets/results.agg.json
    python scripts/rank_tables.py results.json --sla interactive-ttft500ms-itl100ms --out tables.md
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))

if __name__ == "__main__":
    from autoscaling_arena.rank_tables import main

    raise SystemExit(main())
