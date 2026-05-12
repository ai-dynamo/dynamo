# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GMS checkpoint loader entry point.

Waits for the GMS server UDS socket on each device, then loads saved GMS
state from a checkpoint directory into the running GMS servers. Devices
are loaded in parallel to saturate PVC bandwidth.
"""

from __future__ import annotations

import logging
import os
import signal
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from gpu_memory_service.common.cuda_utils import list_devices
from gpu_memory_service.common.utils import get_socket_path, wait_for_weights_socket
from gpu_memory_service.snapshot.storage_client import GMSStorageClient

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def _load_device(checkpoint_dir: str, device: int, max_workers: int) -> None:
    wait_for_weights_socket(device)
    input_dir = os.path.join(checkpoint_dir, f"device-{device}")
    logger.info("Loading GMS checkpoint: device=%d input_dir=%s", device, input_dir)
    t0 = time.monotonic()
    client = GMSStorageClient(
        socket_path=get_socket_path(device),
        device=device,
    )
    client.load_to_gms(
        input_dir,
        max_workers=max_workers,
        clear_existing=True,
    )
    elapsed = time.monotonic() - t0
    logger.info("GMS checkpoint loaded: device=%d elapsed=%.2fs", device, elapsed)


def main() -> None:
    checkpoint_dir = os.environ["GMS_CHECKPOINT_DIR"]
    max_workers = int(os.environ.get("GMS_LOAD_WORKERS", "8"))
    devices = list_devices()

    t0 = time.monotonic()
    with ThreadPoolExecutor(max_workers=len(devices)) as pool:
        futures = {
            pool.submit(_load_device, checkpoint_dir, dev, max_workers): dev
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
