# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Argument parsing for GPU Memory Service server."""

import argparse
import logging
from dataclasses import dataclass
from typing import Optional

from gpu_memory_service.common.utils import get_socket_path

logger = logging.getLogger(__name__)


@dataclass
class Config:
    """Configuration for GPU Memory Service server."""

    device: int
    socket_path: str
    verbose: bool
    # Leader-follower coordination (None = single-GPU mode, no coordination)
    state_file: Optional[str]
    is_leader: bool
    follower_poll_ms: int
    lock_timeout_s: float


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
    parser.add_argument(
        "--state-file",
        type=str,
        default=None,
        help=(
            "Path to shared state file for leader-follower coordination. "
            "If not set, runs in single-GPU mode with no coordination."
        ),
    )
    parser.add_argument(
        "--follower",
        action="store_true",
        help="Run as follower (default: leader).",
    )
    parser.add_argument(
        "--follower-poll-ms",
        type=int,
        default=100,
        help="Polling interval in milliseconds for follower waiting on leader state.",
    )
    parser.add_argument(
        "--lock-timeout-s",
        type=float,
        default=5.0,
        help="Timeout in seconds for flock acquisition on the state file.",
    )
    args = parser.parse_args()

    # Use UUID-based socket path by default (stable across CUDA_VISIBLE_DEVICES)
    socket_path = args.socket_path or get_socket_path(args.device)

    return Config(
        device=args.device,
        socket_path=socket_path,
        verbose=args.verbose,
        state_file=args.state_file,
        is_leader=not args.follower,
        follower_poll_ms=args.follower_poll_ms,
        lock_timeout_s=args.lock_timeout_s,
    )
