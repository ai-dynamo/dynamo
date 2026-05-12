# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
from typing import List


def _parse_int_list(value: str) -> List[int]:
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

    sweep_group = parser.add_mutually_exclusive_group()
    sweep_group.add_argument(
        "--request-rates",
        type=_parse_int_list,
        default=None,
        help="Override request rates (comma-separated, e.g. '4,8,16,32').",
    )
    sweep_group.add_argument(
        "--concurrencies",
        type=_parse_int_list,
        default=None,
        help="Override concurrency levels (comma-separated, e.g. '16,32,64,128').",
    )

    parser.add_argument(
        "--osl",
        type=int,
        default=None,
        help="Override output sequence length.",
    )
    parser.add_argument(
        "--conversation-num",
        type=int,
        default=None,
        help=(
            "Override number of conversations (sessions) per sweep value. "
            "If unset, derived from the input JSONL's unique session_id count "
            "(flat JSONLs count each row as a 1-turn conversation)."
        ),
    )
    parser.add_argument(
        "--skip-plots",
        action="store_true",
        default=None,
        help="Skip plot generation.",
    )
    return parser.parse_args(argv)
