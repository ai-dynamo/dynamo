# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from gpu_memory_service.cli.gms_sidecar_common import (
    checkpoint_device_dir,
    list_devices,
    wait_for_weights_socket,
)
from gpu_memory_service.client.gms_storage_client import GMSStorageClient
from gpu_memory_service.common.utils import get_socket_path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

_DEFAULT_MAX_WORKERS = 8


def _load_device(checkpoint_dir: str, device: int, max_workers: int) -> None:
    """Load GMS checkpoint for a single device."""
    wait_for_weights_socket(device)
    input_dir = checkpoint_device_dir(checkpoint_dir, device)
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
    max_workers = int(os.environ.get("GMS_LOAD_WORKERS", str(_DEFAULT_MAX_WORKERS)))
    devices = list_devices()

    # Load all devices in parallel to saturate PVC bandwidth.
    t0 = time.monotonic()
    with ThreadPoolExecutor(max_workers=len(devices)) as pool:
        futures = {
            pool.submit(_load_device, checkpoint_dir, dev, max_workers): dev
            for dev in devices
        }
        for future in as_completed(futures):
            dev = futures[future]
            future.result()  # propagate exceptions
            logger.info("Device %d load complete", dev)
    elapsed = time.monotonic() - t0
    logger.info(
        "All %d devices loaded in %.2fs",
        len(devices),
        elapsed,
    )

    while True:
        time.sleep(3600)


if __name__ == "__main__":
    main()
