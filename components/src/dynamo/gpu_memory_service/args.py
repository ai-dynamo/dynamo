# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Argument parsing for GPU Memory Service server component."""

import argparse
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class Config:
    """Configuration for GPU Memory Service server."""

    # GMS specific
    device: int
    socket_path: str
    verbose: bool


def parse_args() -> Config:
    """Parse command line arguments for GMS server."""
    parser = argparse.ArgumentParser(
        description="GPU Memory Service allocation server for Dynamo."
    )

    # GMS specific arguments
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
        help="Path for Unix domain socket. Default: /tmp/gms_{device}.sock. "
        "Supports {device} placeholder for multi-GPU setups.",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging.",
    )

    args = parser.parse_args()

    # Generate default socket path if not provided
    socket_path = args.socket_path
    if socket_path is None:
        socket_path = f"/tmp/gms_{args.device}.sock"
    else:
        # Expand {device} placeholder
        socket_path = socket_path.format(device=args.device)

    config = Config(
        device=args.device,
        socket_path=socket_path,
        verbose=args.verbose,
    )

    return config
