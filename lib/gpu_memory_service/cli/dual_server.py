# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run independent weights and KV-cache GMS instances for one GPU."""

from __future__ import annotations

import argparse

from gpu_memory_service.common.utils import get_socket_path

from .args import Config
from .runner import run

TAGS = ("weights", "kv_cache")


def make_server_configs(
    device: int,
    *,
    allocation_retry_interval: float,
    allocation_retry_timeout: float | None,
    verbose: bool,
) -> list[Config]:
    """Build one independent server configuration per production GMS tag."""
    return [
        Config(
            device=device,
            tag=tag,
            socket_path=get_socket_path(device, tag),
            alloc_retry_interval=allocation_retry_interval,
            alloc_retry_timeout=allocation_retry_timeout,
            verbose=verbose,
        )
        for tag in TAGS
    ]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run weights and KV-cache GMS servers for one CUDA device."
    )
    parser.add_argument(
        "--device",
        type=int,
        required=True,
        help="CUDA device ID to manage memory for.",
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
        help="Seconds to sleep between allocation retries on CUDA OOM.",
    )
    parser.add_argument(
        "--alloc-retry-timeout",
        type=float,
        default=60.0,
        help="Max seconds to wait for allocation retries before failing.",
    )
    return parser


def main() -> None:
    """Run the production dual-tag GMS child."""
    parser = _build_parser()
    args = parser.parse_args()
    if args.alloc_retry_interval <= 0:
        parser.error("--alloc-retry-interval must be > 0")
    if args.alloc_retry_timeout is not None and args.alloc_retry_timeout <= 0:
        parser.error("--alloc-retry-timeout must be > 0 when set")

    run(
        make_server_configs(
            args.device,
            allocation_retry_interval=args.alloc_retry_interval,
            allocation_retry_timeout=args.alloc_retry_timeout,
            verbose=args.verbose,
        )
    )


if __name__ == "__main__":
    main()
