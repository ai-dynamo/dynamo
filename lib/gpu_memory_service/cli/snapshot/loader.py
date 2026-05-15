# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GMS checkpoint loader entry point.

Loads saved GMS state from a checkpoint directory into the running GMS
servers. Devices are loaded in parallel to saturate PVC bandwidth. Runs as
a regular sidecar; the GMS RO lock — not init-phase ordering — gates the
restored engine on weight load.
"""

from __future__ import annotations

import argparse
import logging
import os
import signal
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from gpu_memory_service.cli.snapshot.env_args import arg_or_env
from gpu_memory_service.common.cuda_utils import list_devices
from gpu_memory_service.common.utils import get_socket_path
from gpu_memory_service.snapshot.backends.sharded_ssd import (
    GMS_SHARDED_SSD_ROOTS_ENV,
    parse_sharded_ssd_roots,
)
from gpu_memory_service.snapshot.storage_client import GMSStorageClient
from gpu_memory_service.snapshot.transfer import (
    DEFAULT_TRANSFER_BACKEND,
    TRANSFER_BACKEND_CHOICES,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

GMS_CHECKPOINT_DIR_ENV = "GMS_CHECKPOINT_DIR"
GMS_LOAD_WORKERS_ENV = "GMS_LOAD_WORKERS"
GMS_TRANSFER_BACKEND_ENV = "GMS_TRANSFER_BACKEND"


def _load_device(
    checkpoint_dir: str,
    device: int,
    max_workers: int,
    transfer_backend: str,
    sharded_ssd_roots: list[str],
) -> None:
    input_dir = os.path.join(checkpoint_dir, f"device-{device}")
    logger.info(
        "Loading GMS checkpoint: device=%d input_dir=%s transfer_backend=%s max_workers=%d",
        device,
        input_dir,
        transfer_backend,
        max_workers,
    )
    t0 = time.monotonic()
    client = GMSStorageClient(
        socket_path=get_socket_path(device),
        device=device,
        transfer_backend=transfer_backend,
        sharded_ssd_roots=sharded_ssd_roots,
    )
    client.load_to_gms(
        input_dir,
        max_workers=max_workers,
        clear_existing=True,
    )
    elapsed = time.monotonic() - t0
    logger.info("GMS checkpoint loaded: device=%d elapsed=%.2fs", device, elapsed)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Load a GMS checkpoint into GMS.")
    parser.add_argument(
        "--checkpoint-dir",
        default=None,
        help=f"Checkpoint directory. Falls back to {GMS_CHECKPOINT_DIR_ENV}.",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=None,
        help=f"Shard load workers per device. Falls back to {GMS_LOAD_WORKERS_ENV}.",
    )
    parser.add_argument(
        "--transfer-backend",
        choices=TRANSFER_BACKEND_CHOICES,
        default=None,
        help=(
            f"Restore transfer backend. Falls back to {GMS_TRANSFER_BACKEND_ENV}; "
            f"default is {DEFAULT_TRANSFER_BACKEND!r}."
        ),
    )
    parser.add_argument(
        "--sharded-ssd-roots",
        default=None,
        help=(
            "Comma-separated SSD roots for the sharded-ssd restore backend. "
            f"Falls back to {GMS_SHARDED_SSD_ROOTS_ENV}."
        ),
    )
    return parser


def _validate_transfer_backend(
    parser: argparse.ArgumentParser,
    backend: str,
) -> str:
    if backend not in TRANSFER_BACKEND_CHOICES:
        parser.error(
            f"--transfer-backend must be one of {', '.join(TRANSFER_BACKEND_CHOICES)} "
            f"when {GMS_TRANSFER_BACKEND_ENV} is set"
        )
    return backend


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
    max_workers = int(
        arg_or_env(parser, args.max_workers, GMS_LOAD_WORKERS_ENV, default=8)
    )
    transfer_backend = _validate_transfer_backend(
        parser,
        arg_or_env(
            parser,
            args.transfer_backend,
            GMS_TRANSFER_BACKEND_ENV,
            default=DEFAULT_TRANSFER_BACKEND,
        ),
    )
    sharded_ssd_roots = parse_sharded_ssd_roots(
        arg_or_env(
            parser,
            args.sharded_ssd_roots,
            GMS_SHARDED_SSD_ROOTS_ENV,
            default="",
        )
    )
    logger.info(
        "Starting GMS load: transfer_backend=%s max_workers=%d sharded_ssd_roots=%s",
        transfer_backend,
        max_workers,
        ",".join(sharded_ssd_roots) or "-",
    )
    devices = list_devices()

    t0 = time.monotonic()
    with ThreadPoolExecutor(max_workers=len(devices)) as pool:
        futures = {
            pool.submit(
                _load_device,
                checkpoint_dir,
                dev,
                max_workers,
                transfer_backend,
                sharded_ssd_roots,
            ): dev
            for dev in devices
        }
        for future in as_completed(futures):
            dev = futures[future]
            future.result()
            logger.info("Device %d load complete", dev)
    elapsed = time.monotonic() - t0
    logger.info("All %d devices loaded in %.2fs", len(devices), elapsed)
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))

    while True:
        time.sleep(3600)


if __name__ == "__main__":
    main()
