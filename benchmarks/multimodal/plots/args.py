# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class PlotArgs:
    dataset_dirs: list[Path]
    output_dir: Optional[Path]


def parse_args() -> PlotArgs:
    parser = argparse.ArgumentParser(
        prog="python -m benchmarks.multimodal.plots",
        description="Generic aiperf result plotter.",
    )
    parser.add_argument(
        "dataset_dirs",
        nargs="+",
        type=Path,
        help="Dataset directories containing line_name/x_value/profile_export_aiperf.json",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for plots. Creates <output_dir>/<dataset_name>/*.png. "
        "Defaults to <dataset_dir>/plots/ when not specified.",
    )
    ns = parser.parse_args()
    return PlotArgs(dataset_dirs=ns.dataset_dirs, output_dir=ns.output_dir)
