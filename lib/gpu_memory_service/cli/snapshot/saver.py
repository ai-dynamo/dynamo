# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GMS checkpoint saver entry point.

Waits for committed GMS weights on each device, then saves GPU memory state
to the checkpoint directory. Runs as a regular Job container that exits
after save so the Job completes once tensors are on disk.
"""

from __future__ import annotations

import argparse
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from gpu_memory_service.cli.snapshot.env_args import arg_or_env
from gpu_memory_service.common.cuda_utils import list_devices
from gpu_memory_service.common.utils import get_socket_path
from gpu_memory_service.snapshot.backends.sharded_ssd import (
    GMS_SHARDED_SSD_ROOTS_ENV,
    device_sharded_ssd_roots,
    parse_sharded_ssd_roots,
)
from gpu_memory_service.snapshot.storage_client import GMSStorageClient

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# How long the saver waits for the engine to commit weights before giving up
# and failing the Job. Without a bound, an engine that crashes before commit
# would leave the saver blocked indefinitely and the Job stuck Running.
DEFAULT_SAVE_LOCK_TIMEOUT_MS = 30 * 60 * 1000  # 30 minutes

GMS_CHECKPOINT_DIR_ENV = "GMS_CHECKPOINT_DIR"
GMS_SAVE_LOCK_TIMEOUT_MS_ENV = "GMS_SAVE_LOCK_TIMEOUT_MS"
GMS_SAVE_WORKERS_ENV = "GMS_SAVE_WORKERS"
GMS_SHARD_SIZE_BYTES_ENV = "GMS_SHARD_SIZE_BYTES"


def _save_device(
    checkpoint_dir: str,
    device: int,
    max_workers: int,
    lock_timeout_ms: int,
    shard_size_bytes: int,
    sharded_ssd_roots: list[str],
) -> None:
    output_dir = os.path.join(checkpoint_dir, f"device-{device}")
    shard_roots = device_sharded_ssd_roots(
        checkpoint_dir,
        device,
        sharded_ssd_roots,
    )
    logger.info(
        "Saving GMS checkpoint: device=%d output_dir=%s lock_timeout_ms=%d "
        "shard_size_bytes=%d sharded_ssd_roots=%s",
        device,
        output_dir,
        lock_timeout_ms,
        shard_size_bytes,
        ",".join(shard_roots) or "-",
    )
    t0 = time.monotonic()
    GMSStorageClient(
        output_dir,
        socket_path=get_socket_path(device),
        device=device,
        timeout_ms=lock_timeout_ms,
        shard_size_bytes=shard_size_bytes,
        sharded_ssd_roots=shard_roots,
    ).save(max_workers=max_workers)
    elapsed = time.monotonic() - t0
    logger.info("GMS checkpoint saved: device=%d elapsed=%.2fs", device, elapsed)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Save a GMS checkpoint.")
    parser.add_argument(
        "--checkpoint-dir",
        default=None,
        help=f"Checkpoint directory. Falls back to {GMS_CHECKPOINT_DIR_ENV}.",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=None,
        help=f"Shard save workers per device. Falls back to {GMS_SAVE_WORKERS_ENV}.",
    )
    parser.add_argument(
        "--save-lock-timeout-ms",
        type=int,
        default=None,
        help=(
            "Timeout for acquiring the GMS RO lock before save. Falls back to "
            f"{GMS_SAVE_LOCK_TIMEOUT_MS_ENV}; default is "
            f"{DEFAULT_SAVE_LOCK_TIMEOUT_MS}."
        ),
    )
    parser.add_argument(
        "--shard-size-bytes",
        type=int,
        default=None,
        help=(
            "Shard size in bytes. Falls back to "
            f"{GMS_SHARD_SIZE_BYTES_ENV}; default is 4 GiB."
        ),
    )
    parser.add_argument(
        "--sharded-ssd-roots",
        default=None,
        help=(
            "Comma-separated SSD roots for sharded prototype saves. "
            f"Falls back to {GMS_SHARDED_SSD_ROOTS_ENV}."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)
    checkpoint_dir = arg_or_env(
        parser,
        args.checkpoint_dir,
        GMS_CHECKPOINT_DIR_ENV,
        required=True,
        required_flag="--checkpoint-dir",
    )
    max_workers = arg_or_env(
        parser,
        args.max_workers,
        GMS_SAVE_WORKERS_ENV,
        default=8,
        coerce=int,
    )
    lock_timeout_ms = arg_or_env(
        parser,
        args.save_lock_timeout_ms,
        GMS_SAVE_LOCK_TIMEOUT_MS_ENV,
        default=DEFAULT_SAVE_LOCK_TIMEOUT_MS,
        coerce=int,
    )
    shard_size_bytes = arg_or_env(
        parser,
        args.shard_size_bytes,
        GMS_SHARD_SIZE_BYTES_ENV,
        default=4 * 1024**3,
        coerce=int,
    )
    sharded_ssd_roots = parse_sharded_ssd_roots(
        arg_or_env(
            parser,
            args.sharded_ssd_roots,
            GMS_SHARDED_SSD_ROOTS_ENV,
            default="",
        )
    )

    devices = list_devices()
    logger.info(
        "Starting GMS save for %d devices lock_timeout_ms=%d sharded_ssd_roots=%s",
        len(devices),
        lock_timeout_ms,
        ",".join(sharded_ssd_roots) or "-",
    )
    t0 = time.monotonic()
    with ThreadPoolExecutor(max_workers=len(devices)) as pool:
        futures = {
            pool.submit(
                _save_device,
                checkpoint_dir,
                dev,
                max_workers,
                lock_timeout_ms,
                shard_size_bytes,
                sharded_ssd_roots,
            ): dev
            for dev in devices
        }
        for future in as_completed(futures):
            future.result()
    elapsed = time.monotonic() - t0
    logger.info("All %d devices saved in %.2fs", len(devices), elapsed)
    logger.info("Save complete; exiting")


if __name__ == "__main__":
    main()
