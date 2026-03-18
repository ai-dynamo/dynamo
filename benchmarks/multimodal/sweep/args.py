# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
from typing import List


def _parse_concurrencies(value: str) -> List[int]:
    return [int(x.strip()) for x in value.split(",")]


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a multimodal benchmark sweep from a YAML config.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python -m benchmarks.multimodal.sweep --config experiments/cache_sweep.yaml\n"
            "  python -m benchmarks.multimodal.sweep --config exp.yaml --osl 200 --skip-plots\n"
        ),
    )

    parser.add_argument(
        "--config",
        required=True,
        help="Path to YAML experiment config file.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Override output directory from config.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Override model name from config.",
    )
    parser.add_argument(
        "--concurrencies",
        type=_parse_concurrencies,
        default=None,
        help="Override concurrency levels (comma-separated, e.g. '1,2,4,8').",
    )
    parser.add_argument(
        "--osl",
        type=int,
        default=None,
        help="Override output sequence length.",
    )
    parser.add_argument(
        "--request-count",
        type=int,
        default=None,
        help="Override request count per concurrency level.",
    )
    parser.add_argument(
        "--skip-plots",
        action="store_true",
        default=None,
        help="Skip plot generation.",
    )

    return parser.parse_args(argv)
