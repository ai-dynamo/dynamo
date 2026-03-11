# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
from typing import List


def _parse_concurrencies(value: str) -> List[int]:
    return [int(x.strip()) for x in value.split(",")]


def _parse_qps_rates(value: str) -> List[float]:
    return [float(x.strip()) for x in value.split(",")]


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a multimodal benchmark sweep from a YAML config.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  # Concurrency-based (default):\n"
            "  python -m benchmarks.multimodal.sweep --config exp.yaml\n"
            "  python -m benchmarks.multimodal.sweep --config exp.yaml --concurrencies 1,8,16,32\n"
            "\n"
            "  # QPS-based (fixed arrival rate):\n"
            "  python -m benchmarks.multimodal.sweep --config exp.yaml --qps-rates 4,8,12,16,20,24\n"
            "  python -m benchmarks.multimodal.sweep --config exp.yaml --qps-rates 8 --min-duration 120\n"
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

    # Load mode: concurrency or QPS (mutually exclusive at CLI level;
    # YAML can set either concurrencies or qps_rates)
    load_group = parser.add_mutually_exclusive_group()
    load_group.add_argument(
        "--concurrencies",
        type=_parse_concurrencies,
        default=None,
        help="Concurrency levels, comma-separated (e.g. '1,8,16,32,64').",
    )
    load_group.add_argument(
        "--qps-rates",
        type=_parse_qps_rates,
        default=None,
        dest="qps_rates",
        help="QPS rates, comma-separated (e.g. '4,8,12,16,20,24'). "
        "Switches to QPS mode with auto-scaled request counts.",
    )

    parser.add_argument(
        "--min-duration",
        type=int,
        default=None,
        dest="min_duration",
        help="Minimum benchmark duration in seconds for QPS mode (default: 60). "
        "Request count is auto-scaled to max(request_count, qps * min_duration).",
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
        help="Override request count (or minimum count in QPS mode).",
    )
    parser.add_argument(
        "--skip-plots",
        action="store_true",
        default=None,
        help="Skip plot generation.",
    )

    return parser.parse_args(argv)
