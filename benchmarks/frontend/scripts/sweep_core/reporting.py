# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Per-run report generation -- wraps analysis/create_report.py."""

from __future__ import annotations

import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent.parent
ANALYSIS_DIR = SCRIPT_DIR / "analysis"

# Add analysis directory to sys.path once at import time
if str(ANALYSIS_DIR) not in sys.path:
    sys.path.insert(0, str(ANALYSIS_DIR))


def generate_report(run_dir: Path) -> None:
    """Run create_report.py on a single run directory, saving report.md."""
    try:
        from create_report import run_analysis

        report = run_analysis(run_dir)
        (run_dir / "report.md").write_text(report)
    except (ImportError, OSError) as e:
        print(f"    Report generation failed: {e}")
    except Exception as e:
        print(f"    Report generation failed: {e}")
