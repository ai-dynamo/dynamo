# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
import time
from pathlib import Path

from gpu_memory_service.common.utils import get_socket_path

_WEIGHTS_TAG = "weights"
_READY_FILE = "gms-ready"


def ready_file_path() -> Path:
    """Return the path of the GMS server ready sentinel file."""
    return Path(os.environ.get("GMS_SOCKET_DIR", "/tmp")) / _READY_FILE


def list_devices() -> list[int]:
    import pynvml

    pynvml.nvmlInit()
    try:
        count = pynvml.nvmlDeviceGetCount()
    finally:
        pynvml.nvmlShutdown()

    if count == 0:
        raise SystemExit("no nvidia devices found")
    return list(range(count))


def checkpoint_device_dir(root: str, device: int) -> str:
    return os.path.join(root, f"device-{device}")


def wait_for_socket(device: int, tag: str) -> None:
    socket_path = get_socket_path(device, tag)
    while not os.path.exists(socket_path):
        time.sleep(1)


def wait_for_weights_socket(device: int) -> None:
    wait_for_socket(device, _WEIGHTS_TAG)


def optional_checkpoint_stop_file() -> Path | None:
    control_dir = os.environ.get("GMS_CONTROL_DIR")
    if not control_dir:
        return None
    return Path(control_dir) / "checkpoint-done"
