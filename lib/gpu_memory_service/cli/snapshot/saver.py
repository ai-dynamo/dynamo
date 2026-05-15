# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GMS checkpoint saver entry point.

Waits for committed GMS weights on each device, then saves GPU memory state
to the checkpoint directory. Runs as a regular Job container that exits
after save so the Job completes once tensors are on disk.
"""

from __future__ import annotations

import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from gpu_memory_service.common.cuda_utils import list_devices
from gpu_memory_service.common.utils import get_socket_path
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


def _save_device(
    checkpoint_dir: str, device: int, max_workers: int, lock_timeout_ms: int
) -> None:
    output_dir = os.path.join(checkpoint_dir, f"device-{device}")
    logger.info("Saving GMS checkpoint: device=%d output_dir=%s", device, output_dir)
    t0 = time.monotonic()
    GMSStorageClient(
        output_dir,
        socket_path=get_socket_path(device),
        device=device,
        timeout_ms=lock_timeout_ms,
    ).save(max_workers=max_workers)
    elapsed = time.monotonic() - t0
    logger.info("GMS checkpoint saved: device=%d elapsed=%.2fs", device, elapsed)


def main() -> None:
    checkpoint_dir = os.environ["GMS_CHECKPOINT_DIR"]
    max_workers = int(os.environ.get("GMS_SAVE_WORKERS", "8"))
    lock_timeout_ms = int(
        os.environ.get("GMS_SAVE_LOCK_TIMEOUT_MS", str(DEFAULT_SAVE_LOCK_TIMEOUT_MS))
    )

    devices = list_devices()
    logger.info(
        "Starting GMS save for %d devices (lock_timeout_ms=%d)",
        len(devices),
        lock_timeout_ms,
    )
    t0 = time.monotonic()
    with ThreadPoolExecutor(max_workers=len(devices)) as pool:
        futures = {
            pool.submit(
                _save_device, checkpoint_dir, dev, max_workers, lock_timeout_ms
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
