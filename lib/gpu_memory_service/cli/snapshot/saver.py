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

from gpu_memory_service.common.utils import get_socket_path
from gpu_memory_service.common.vmm import VMMDeviceType, get_vmm, init_vmm
from gpu_memory_service.snapshot.backends.sharded_ssd import (
    device_sharded_ssd_roots,
    parse_sharded_ssd_roots,
)
from gpu_memory_service.snapshot.storage_client import GMSStorageClient

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


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
    # This runs on a ThreadPoolExecutor thread; bind its CUDA device before
    # any device work, mirroring the loader's _load_device.
    cuda_utils.cuda_runtime_set_device(device)
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
    parser = argparse.ArgumentParser(
        description="Save a GMS checkpoint.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--checkpoint-dir",
        default=None,
        help="Checkpoint directory for directory-backed save outputs.",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=8,
        help="Shard save workers per device.",
    )
    parser.add_argument(
        "--save-lock-timeout-ms",
        type=int,
        default=30 * 60 * 1000,
        help=(
            "Timeout for acquiring the GMS RO lock before save. Without a "
            "bound, an engine that crashes before commit would leave the "
            "saver blocked indefinitely and the Job stuck Running."
        ),
    )
    parser.add_argument(
        "--shard-size-bytes",
        type=int,
        default=4 * 1024**3,
        help="Shard size in bytes.",
    )
    parser.add_argument(
        "--sharded-ssd-roots",
        default="",
        help="Comma-separated SSD roots for sharded prototype saves.",
    )
    parser.add_argument(
        "--device-type",
        type=str,
        default=VMMDeviceType.CUDA.value,
        choices=[d.value for d in VMMDeviceType],
        help="VMM device type (default: cuda).",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if not args.checkpoint_dir:
        parser.error("--checkpoint-dir is required for directory-backed saves")
    checkpoint_dir = args.checkpoint_dir
    assert checkpoint_dir is not None
    max_workers = args.max_workers
    lock_timeout_ms = args.save_lock_timeout_ms
    shard_size_bytes = args.shard_size_bytes
    sharded_ssd_roots = parse_sharded_ssd_roots(args.sharded_ssd_roots)

    device_type = VMMDeviceType.from_str(args.device_type)
    init_vmm(device_type)
    vmm = get_vmm()
    vmm.ensure_initialized()
    devices = vmm.list_devices()
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
