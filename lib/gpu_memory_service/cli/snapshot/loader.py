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
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from gpu_memory_service.common.cuda_utils import list_devices
from gpu_memory_service.common.utils import get_socket_path
from gpu_memory_service.snapshot.backends.sharded_ssd import parse_sharded_ssd_roots
from gpu_memory_service.snapshot.storage_client import GMSStorageClient
from gpu_memory_service.snapshot.transfer import (
    CHECKPOINT_DIR_TRANSFER_BACKENDS,
    DEFAULT_TRANSFER_BACKEND,
    TRANSFER_BACKEND_CHOICES,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


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
        help=(
            "Checkpoint directory. Required for directory-backed transfer "
            f"backends: {', '.join(CHECKPOINT_DIR_TRANSFER_BACKENDS)}."
        ),
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=8,
        help="Shard load workers per device.",
    )
    parser.add_argument(
        "--transfer-backend",
        choices=TRANSFER_BACKEND_CHOICES,
        default=DEFAULT_TRANSFER_BACKEND,
        help=f"Restore transfer backend. Default is {DEFAULT_TRANSFER_BACKEND!r}.",
    )
    parser.add_argument(
        "--sharded-ssd-roots",
        default="",
        help=("Comma-separated SSD roots for the sharded-ssd restore backend."),
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if (
        args.transfer_backend in CHECKPOINT_DIR_TRANSFER_BACKENDS
        and not args.checkpoint_dir
    ):
        parser.error(
            f"--checkpoint-dir is required for --transfer-backend={args.transfer_backend}"
        )
    checkpoint_dir = args.checkpoint_dir
    max_workers = args.max_workers
    transfer_backend = args.transfer_backend
    sharded_ssd_roots = parse_sharded_ssd_roots(args.sharded_ssd_roots)
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

    while True:
        time.sleep(3600)


if __name__ == "__main__":
    main()
