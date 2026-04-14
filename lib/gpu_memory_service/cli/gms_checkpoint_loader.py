# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
import os
import time

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


def main() -> None:
    checkpoint_dir = os.environ["GMS_CHECKPOINT_DIR"]
    for device in list_devices():
        wait_for_weights_socket(device)
        input_dir = checkpoint_device_dir(checkpoint_dir, device)
        logger.info("Loading GMS checkpoint: device=%d input_dir=%s", device, input_dir)
        client = GMSStorageClient(
            socket_path=get_socket_path(device),
            device=device,
        )
        client.load_to_gms(
            input_dir,
            max_workers=4,
            clear_existing=True,
        )

    while True:
        time.sleep(3600)


if __name__ == "__main__":
    main()
