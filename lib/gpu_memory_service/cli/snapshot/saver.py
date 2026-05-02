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
GMS_WEIGHTS_CHECKPOINT_DIR_ENV = "GMS_WEIGHTS_CHECKPOINT_DIR"
GMS_LOCAL_SSD_ROOTS_ENV = "GMS_LOCAL_SSD_ROOTS"
GMS_SHARD_SIZE_BYTES_ENV = "GMS_SHARD_SIZE_BYTES"


def _parse_local_ssd_roots() -> list[str]:
    return [
        part.strip()
        for part in os.environ.get(GMS_LOCAL_SSD_ROOTS_ENV, "").split(",")
        if part.strip()
    ]


def _checkpoint_suffix(checkpoint_dir: str) -> Path:
    parts = Path(checkpoint_dir).parts
    if "versions" in parts:
        idx = parts.index("versions")
        if idx > 0 and idx + 1 < len(parts):
            return Path(parts[idx - 1]) / "versions" / parts[idx + 1]
    return Path(checkpoint_dir.strip(os.sep).replace(os.sep, "_"))


def _device_shard_roots(
    checkpoint_dir: str,
    device: int,
    local_ssd_roots: list[str],
) -> list[str]:
    suffix = _checkpoint_suffix(checkpoint_dir) / f"device-{device}"
    return [str(Path(root) / suffix) for root in local_ssd_roots]


def _save_device(
    checkpoint_dir: str,
    device: int,
    max_workers: int,
    shard_size_bytes: int,
    local_ssd_roots: list[str],
) -> None:
    wait_for_weights_socket(device)
    output_dir = os.path.join(checkpoint_dir, f"device-{device}")
    shard_roots = _device_shard_roots(checkpoint_dir, device, local_ssd_roots)
    logger.info(
        "Saving GMS checkpoint: device=%d output_dir=%s shard_size_bytes=%d local_ssd_roots=%s",
        device,
        output_dir,
        shard_size_bytes,
        ",".join(shard_roots) or "-",
    )
    t0 = time.monotonic()
    GMSStorageClient(
        output_dir,
        socket_path=get_socket_path(device),
        device=device,
        shard_size_bytes=shard_size_bytes,
        shard_roots=shard_roots,
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
    control_dir = os.environ["GMS_CHECKPOINT_DIR"]
    checkpoint_dir = os.environ.get(GMS_WEIGHTS_CHECKPOINT_DIR_ENV, control_dir)
    max_workers = int(os.environ.get("GMS_SAVE_WORKERS", "8"))
    shard_size_bytes = int(os.environ.get(GMS_SHARD_SIZE_BYTES_ENV, str(4 * 1024**3)))
    local_ssd_roots = _parse_local_ssd_roots()
    _clear_completion_sentinel(control_dir)

    devices = list_devices()
    logger.info(
        "Starting GMS save for %d devices local_ssd_roots=%s",
        len(devices),
        ",".join(local_ssd_roots) or "-",
    )
    t0 = time.monotonic()
    with ThreadPoolExecutor(max_workers=len(devices)) as pool:
        futures = {
            pool.submit(
                _save_device,
                checkpoint_dir,
                dev,
                max_workers,
                shard_size_bytes,
                local_ssd_roots,
            ): dev
            for dev in devices
        }
        for future in as_completed(futures):
            future.result()
    elapsed = time.monotonic() - t0
    logger.info("All %d devices saved in %.2fs", len(devices), elapsed)
    _write_completion_sentinel(control_dir)

    logger.info("Save complete; sleeping until pod terminates")
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
    while True:
        time.sleep(3600)


if __name__ == "__main__":
    main()
