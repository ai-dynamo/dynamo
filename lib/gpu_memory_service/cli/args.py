# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Argument parsing for GPU Memory Service server."""

import argparse
import logging
from dataclasses import dataclass
from typing import Optional

from gpu_memory_service.common.utils import GMS_TAGS, get_socket_path

logger = logging.getLogger(__name__)


@dataclass
class Config:
    """Configuration for GPU Memory Service server."""

    device: int
    tag: str
    socket_path: str
    alloc_retry_interval: float
    alloc_retry_timeout: Optional[float]
    verbose: bool


def parse_args(argv: Optional[list[str]] = None) -> list[Config]:
    """Parse command line arguments into one server Config per requested tag."""
    parser = argparse.ArgumentParser(
        description="GPU Memory Service allocation server."
    )

    parser.add_argument(
        "--device",
        type=int,
        required=True,
        help="CUDA device ID to manage memory for.",
    )
    parser.add_argument(
        "--tag",
        type=str,
        action="append",
        help="Logical GMS tag to serve; may be repeated. Defaults to all "
        f"production tags ({', '.join(GMS_TAGS)}), each on its own socket.",
    )
    parser.add_argument(
        "--socket-path",
        type=str,
        default=None,
        help="Path for Unix domain socket. Default uses GPU UUID for stability. "
        "Requires exactly one --tag.",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging.",
    )
    parser.add_argument(
        "--alloc-retry-interval",
        type=float,
        default=0.5,
        help="Seconds to sleep between allocation retries on CUDA OOM (default: 0.5).",
    )
    parser.add_argument(
        "--alloc-retry-timeout",
        type=float,
        default=60.0,
        help="Max seconds to wait for allocation retries before failing (default: 60.0). "
        "Pass an explicit large value if you need essentially-unbounded retry.",
    )

    args = parser.parse_args(argv)

    tags = args.tag or list(GMS_TAGS)
    if len(tags) != len(set(tags)):
        parser.error("--tag values must be unique")
    if args.socket_path is not None and len(tags) != 1:
        parser.error("--socket-path requires exactly one --tag")
    if args.alloc_retry_interval <= 0:
        parser.error("--alloc-retry-interval must be > 0")
    if args.alloc_retry_timeout is not None and args.alloc_retry_timeout <= 0:
        parser.error("--alloc-retry-timeout must be > 0 when set")

    return [
        Config(
            device=args.device,
            tag=tag,
            # Use UUID-based socket path by default (stable across
            # CUDA_VISIBLE_DEVICES).
            socket_path=args.socket_path or get_socket_path(args.device, tag),
            alloc_retry_interval=args.alloc_retry_interval,
            alloc_retry_timeout=args.alloc_retry_timeout,
            verbose=args.verbose,
        )
        for tag in tags
    ]
