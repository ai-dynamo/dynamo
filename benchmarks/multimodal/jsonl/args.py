# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CLI argument parsing for JSONL benchmark generation."""

import argparse
import sys
from pathlib import Path

DEFAULT_IMAGES_PER_REQUEST = 3
USER_TEXT_TOKENS = 300
COCO_ANNOTATIONS = Path(__file__).parent / "annotations" / "image_info_test2017.json"


def _common_parser() -> argparse.ArgumentParser:
    """Args shared across all strategies."""
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output .jsonl path (default: auto-generated from parameters)",
    )
    p.add_argument(
        "--user-text-tokens",
        type=int,
        default=USER_TEXT_TOKENS,
        help=f"Target user text tokens per request (default: {USER_TEXT_TOKENS})",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducible generation (default: time-based)",
    )
    return p


def _image_parser() -> argparse.ArgumentParser:
    """Args for image generation (reusable for future video/audio parsers)."""
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument(
        "--image-size",
        type=int,
        nargs=2,
        default=[512, 512],
        metavar=("WIDTH", "HEIGHT"),
        help="Size of generated PNG images in pixels (default: 512 512)",
    )
    p.add_argument(
        "--image-dir",
        type=Path,
        default=Path("/tmp/bench_images"),
        help="Directory to save generated PNG images (default: /tmp/bench_images)",
    )
    p.add_argument(
        "--image-mode",
        choices=["base64", "http"],
        default="base64",
        help="'base64' generates local PNGs (default); 'http' uses COCO URLs",
    )
    p.add_argument(
        "--coco-annotations",
        type=Path,
        default=COCO_ANNOTATIONS,
        help=f"Path to COCO image_info JSON for --image-mode http (default: {COCO_ANNOTATIONS})",
    )
    return p


def parse_args(description: str = "") -> argparse.Namespace:
    common = _common_parser()
    image = _image_parser()

    parser = argparse.ArgumentParser(
        description=description,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="strategy")

    # --- single-turn (default) ---
    st = sub.add_parser(
        "single-turn",
        parents=[common, image],
        help="Independent requests with random image sampling (default)",
    )
    st.add_argument(
        "-n",
        "--num-requests",
        type=int,
        default=500,
        help="Number of requests to generate (default: 500)",
    )
    st.add_argument(
        "--images-per-request",
        type=int,
        default=DEFAULT_IMAGES_PER_REQUEST,
        help=f"Number of images per request (default: {DEFAULT_IMAGES_PER_REQUEST})",
    )
    st.add_argument(
        "--images-pool",
        type=int,
        default=None,
        help="Unique images in pool. Smaller pool = more cross-request reuse. "
        "Default: num_requests * images_per_request (all unique).",
    )

    # --- sliding-window ---
    sw = sub.add_parser(
        "sliding-window",
        parents=[common, image],
        help="Causal sessions with sliding-window image overlap",
    )
    sw.add_argument(
        "--num-users",
        type=int,
        default=10,
        help="Number of concurrent user sessions (default: 10)",
    )
    sw.add_argument(
        "--turns-per-user",
        type=int,
        default=20,
        help="Number of requests per user (default: 20)",
    )
    sw.add_argument(
        "--window-size",
        type=int,
        default=5,
        help="Number of images per request (default: 5)",
    )

    # Default to single-turn when no subcommand given
    known_strategies = {"single-turn", "sliding-window"}
    if len(sys.argv) < 2 or sys.argv[1] not in known_strategies:
        args = st.parse_args(sys.argv[1:])
        args.strategy = "single-turn"
    else:
        args = parser.parse_args()

    return args
