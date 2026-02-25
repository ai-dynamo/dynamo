# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CLI argument parsing for request generation scripts."""

import argparse
from pathlib import Path

DEFAULT_IMAGES_PER_REQUEST = 3
USER_TEXT_TOKENS = 300
COCO_ANNOTATIONS = Path(__file__).parent / "annotations" / "image_info_test2017.json"


def parse_args(description: str = "") -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=description,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "-n",
        "--num-requests",
        type=int,
        default=500,
        help="Number of requests to generate (default: 500)",
    )
    parser.add_argument(
        "--images-pool",
        type=int,
        default=None,
        help="Number of unique images in the pool. Each request samples from this pool, "
        "so a smaller pool means more cross-request reuse. "
        "Default: num_requests * images_per_request (all unique, no reuse).",
    )
    parser.add_argument(
        "--images-per-request",
        type=int,
        default=DEFAULT_IMAGES_PER_REQUEST,
        help=f"Number of images per request (default: {DEFAULT_IMAGES_PER_REQUEST})",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output .jsonl path (default: {n}req_{img}img_{pool}pool_{word}word_{mode}.jsonl, e.g. 100req_20img_1000pool_4000word_base64.jsonl)",
    )
    parser.add_argument(
        "--image-dir",
        type=Path,
        default=Path("/tmp/bench_images"),
        help="Directory to save generated PNG images (default: /tmp/bench_images)",
    )
    parser.add_argument(
        "--user-text-tokens",
        type=int,
        default=USER_TEXT_TOKENS,
        help=f"Target user text tokens per request (default: {USER_TEXT_TOKENS}). --isl is an alias.",
    )
    parser.add_argument(
        "--image-mode",
        choices=["base64", "http"],
        default="base64",
        help="Image loading mode: 'base64' generates local PNGs and puts file paths in "
        "the JSONL so aiperf reads and base64-encodes them before sending (default); "
        "'http' puts COCO HTTP URLs in the JSONL so the LLM server downloads images itself",
    )
    parser.add_argument(
        "--coco-annotations",
        type=Path,
        default=COCO_ANNOTATIONS,
        help=f"Path to COCO image_info JSON for --image-mode http (default: {COCO_ANNOTATIONS})",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        nargs=2,
        default=[512, 512],
        metavar=("WIDTH", "HEIGHT"),
        help="Size of generated PNG images in pixels (default: 512 512)",
    )
    return parser.parse_args()
