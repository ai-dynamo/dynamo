"""CLI argument parsing for aiperf image generation."""

import argparse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate random-noise PNGs into aiperf's source_images directory.",
    )
    parser.add_argument(
        "--images-pool",
        type=int,
        default=200,
        help="Number of unique images to generate (default: 200)",
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
