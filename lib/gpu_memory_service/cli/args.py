# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Argument parsing for GPU Memory Service server."""

import argparse
import logging
from dataclasses import dataclass

from gpu_memory_service.common.utils import get_socket_path

logger = logging.getLogger(__name__)


@dataclass
class Config:
    """Configuration for GPU Memory Service server."""

    device: int
    socket_path: str
    verbose: bool


def parse_args() -> Config:
    """Parse command line arguments for GPU Memory Service server."""
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
        "--socket-path",
        type=str,
        default=None,
        help="Path for Unix domain socket. Default uses GPU UUID for stability.",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging.",
    )

    args = parser.parse_args()

    # Use UUID-based socket path by default (stable across CUDA_VISIBLE_DEVICES)
    socket_path = args.socket_path or get_socket_path(args.device)

    return Config(
        device=args.device,
        socket_path=socket_path,
        verbose=args.verbose,
    )
