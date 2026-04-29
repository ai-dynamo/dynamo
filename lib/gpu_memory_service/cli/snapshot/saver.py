# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GMS checkpoint saver entry point.

Waits for committed GMS weights on each device, then saves GPU memory state
to the checkpoint directory. Runs as an init sidecar — sleeps after saving
until the pod terminates.
"""

from __future__ import annotations

import logging
import os
import signal
import sys
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

DEFAULT_GMS_SAVE_COMPLETE_FILE = "gms-save-complete"
GMS_SAVE_COMPLETE_FILE_ENV = "GMS_SAVE_COMPLETE_FILE"
GMS_POD_UID_ENV = "GMS_POD_UID"


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


def _completion_sentinel_file() -> str:
    override = os.environ.get(GMS_SAVE_COMPLETE_FILE_ENV, "").strip()
    if override:
        return override
    pod_uid = os.environ.get(GMS_POD_UID_ENV, "").strip()
    if pod_uid:
        return f"{DEFAULT_GMS_SAVE_COMPLETE_FILE}-{pod_uid}"
    return DEFAULT_GMS_SAVE_COMPLETE_FILE


def _completion_sentinel_path(checkpoint_dir: str) -> Path:
    return Path(checkpoint_dir) / _completion_sentinel_file()


def _clear_completion_sentinel(checkpoint_dir: str) -> None:
    try:
        _completion_sentinel_path(checkpoint_dir).unlink()
    except FileNotFoundError:
        pass


def _write_completion_sentinel(checkpoint_dir: str) -> None:
    path = _completion_sentinel_path(checkpoint_dir)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path.write_text("ok\n", encoding="utf-8")
    tmp_path.replace(path)
    logger.info("Wrote GMS save completion sentinel: %s", path)


def main() -> None:
    checkpoint_dir = os.environ["GMS_CHECKPOINT_DIR"]
    max_workers = int(os.environ.get("GMS_SAVE_WORKERS", "8"))
    _clear_completion_sentinel(checkpoint_dir)

    devices = list_devices()
    logger.info("Starting GMS save for %d devices", len(devices))
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
    _write_completion_sentinel(checkpoint_dir)

    logger.info("Save complete; sleeping until pod terminates")
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
    while True:
        time.sleep(3600)


if __name__ == "__main__":
    main()
