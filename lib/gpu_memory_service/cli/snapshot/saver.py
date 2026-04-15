# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GMS checkpoint saver entry point.

Waits for committed GMS weights on each device, then saves GPU memory state
to the checkpoint directory. Writes a stop file to signal the GMS server to
shut down after save completes.
"""

from __future__ import annotations

import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from gpu_memory_service.common.cuda_utils import list_devices
from gpu_memory_service.common.utils import get_socket_path, wait_for_weights_socket
from gpu_memory_service.snapshot.storage_client import GMSStorageClient

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

def _save_device(checkpoint_dir: str, device: int, max_workers: int) -> None:
    wait_for_weights_socket(device)
    output_dir = os.path.join(checkpoint_dir, f"device-{device}")
    logger.info("Saving GMS checkpoint: device=%d output_dir=%s", device, output_dir)
    t0 = time.monotonic()
    GMSStorageClient(
        output_dir,
        socket_path=get_socket_path(device),
        device=device,
    ).save(max_workers=max_workers)
    elapsed = time.monotonic() - t0
    logger.info("GMS checkpoint saved: device=%d elapsed=%.2fs", device, elapsed)


def main() -> None:
    checkpoint_dir = os.environ["GMS_CHECKPOINT_DIR"]
    max_workers = int(os.environ.get("GMS_SAVE_WORKERS", "8"))

    devices = list_devices()
    logger.info("Starting GMS save for %d devices", len(devices))
    try:
        t0 = time.monotonic()
        with ThreadPoolExecutor(max_workers=len(devices)) as pool:
            futures = {
                pool.submit(_save_device, checkpoint_dir, dev, max_workers): dev
                for dev in devices
            }
            for future in as_completed(futures):
                future.result()
        elapsed = time.monotonic() - t0
        logger.info("All %d devices saved in %.2fs", len(devices), elapsed)
    finally:
        # Signal the GMS server to shut down, even if saving failed.
        (Path(checkpoint_dir) / "checkpoint-done").write_text("done", encoding="utf-8")


if __name__ == "__main__":
    main()
