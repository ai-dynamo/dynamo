# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CLI argument parsing for raw replay generation."""

import argparse
from pathlib import Path

DEFAULT_SEED = 42
COCO_ANNOTATIONS = (
    Path(__file__).parent.parent / "annotations" / "image_info_test2017.json"
)


def parse_args(description: str = "") -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=description,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to YAML config describing per-turn specs (system_tokens, "
        "user_tokens, max_output_tokens, images, etc.)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory to write per-session JSONL files",
    )
    parser.add_argument(
        "--num-conversations",
        type=int,
        required=True,
        help="Number of multi-turn conversations to generate",
    )
    parser.add_argument(
        "--image-pool-size",
        type=int,
        default=None,
        help="Number of unique images in the pool. Each conversation samples from this "
        "pool, so a smaller pool means more cross-conversation reuse. "
        "Default: total images needed across all conversations (all unique, no reuse).",
    )
    parser.add_argument(
        "--image-mode",
        choices=["base64", "http"],
        default="base64",
        help="Image loading mode: 'base64' generates local PNGs and embeds them as "
        "data URLs in the JSONL (default); 'http' puts COCO HTTP URLs in the JSONL "
        "so the LLM server downloads images itself",
    )
    parser.add_argument(
        "--coco-annotations",
        type=Path,
        default=COCO_ANNOTATIONS,
        help=f"Path to COCO image_info JSON for --image-mode http (default: {COCO_ANNOTATIONS})",
    )
    parser.add_argument(
        "--wrap-sys-to-user",
        action="store_true",
        default=False,
        help="For turns after the first, prepend the system prompt into the user "
        "message instead of emitting a separate system message. Use this when "
        "the target model's chat template ignores system roles outside index 0.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"Random seed for reproducibility (default: {DEFAULT_SEED})",
    )
    return parser.parse_args()
