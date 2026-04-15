# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GMS checkpoint saver entry point.

Waits for the checkpoint pod to reach Ready=True, then saves GMS state from
each device in parallel. Writes a stop file to signal the GMS server to shut
down after save completes.
"""

from __future__ import annotations

import json
import logging
import os
import ssl
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from gpu_memory_service.common.utils import get_socket_path
from gpu_memory_service.snapshot.storage_client import GMSStorageClient

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

_WEIGHTS_TAG = "weights"
_SERVICE_ACCOUNT_TOKEN = Path("/var/run/secrets/kubernetes.io/serviceaccount/token")
_SERVICE_ACCOUNT_CA = "/var/run/secrets/kubernetes.io/serviceaccount/ca.crt"


def _list_devices() -> list[int]:
    import pynvml

    pynvml.nvmlInit()
    try:
        count = pynvml.nvmlDeviceGetCount()
    finally:
        pynvml.nvmlShutdown()

    if count == 0:
        raise SystemExit("no nvidia devices found")
    return list(range(count))


def _wait_for_weights_socket(device: int) -> None:
    socket_path = get_socket_path(device, _WEIGHTS_TAG)
    while not os.path.exists(socket_path):
        time.sleep(1)


def _checkpoint_pod_ready(pod: dict[str, Any]) -> bool:
    status = pod.get("status") or {}
    if str(status.get("phase", "")).strip() != "Running":
        return False
    for condition in status.get("conditions") or []:
        if (
            condition.get("type") == "Ready"
            and str(condition.get("status", "")).strip() == "True"
        ):
            return True
    return False


def _main_terminated(pod: dict[str, Any]) -> bool:
    status = pod.get("status") or {}
    for container in status.get("containerStatuses") or []:
        if container.get("name") != "main":
            continue
        return bool((container.get("state") or {}).get("terminated"))
    return False


def main() -> None:
    service_token = _SERVICE_ACCOUNT_TOKEN.read_text(encoding="utf-8").strip()
    ssl_context = ssl.create_default_context(cafile=_SERVICE_ACCOUNT_CA)
    pod_api_url = (
        "https://"
        + os.environ["KUBERNETES_SERVICE_HOST"]
        + ":"
        + os.environ.get("KUBERNETES_SERVICE_PORT_HTTPS", "443")
        + f"/api/v1/namespaces/{os.environ['POD_NAMESPACE']}/pods/{os.environ['POD_NAME']}"
    )
    checkpoint_dir = os.environ["GMS_CHECKPOINT_DIR"]

    def checkpoint_pod() -> dict[str, Any]:
        request = urllib.request.Request(
            pod_api_url,
            headers={"Authorization": f"Bearer {service_token}"},
        )
        with urllib.request.urlopen(
            request,
            context=ssl_context,
            timeout=5,
        ) as response:
            return json.load(response)

    logger.info("Waiting for checkpoint pod Ready=True before GMS save")
    while True:
        try:
            pod = checkpoint_pod()
        except (urllib.error.URLError, TimeoutError, ssl.SSLError, OSError):
            time.sleep(1)
            continue

        if _checkpoint_pod_ready(pod):
            break
        if _main_terminated(pod):
            raise SystemExit("main container terminated before GMS save could start")
        time.sleep(1)

    def _save_device(device: int, max_workers: int) -> None:
        _wait_for_weights_socket(device)
        output_dir = os.path.join(checkpoint_dir, f"device-{device}")
        logger.info(
            "Saving GMS checkpoint: device=%d output_dir=%s",
            device,
            output_dir,
        )
        t0 = time.monotonic()
        client = GMSStorageClient(
            output_dir,
            socket_path=get_socket_path(device),
            device=device,
        )
        client.save(max_workers=max_workers)
        elapsed = time.monotonic() - t0
        logger.info("GMS checkpoint saved: device=%d elapsed=%.2fs", device, elapsed)

    max_workers = int(os.environ.get("GMS_SAVE_WORKERS", "8"))
    logger.info("Checkpoint pod is Ready; starting GMS save")
    try:
        devices = _list_devices()
        t0 = time.monotonic()
        with ThreadPoolExecutor(max_workers=len(devices)) as pool:
            futures = {
                pool.submit(_save_device, dev, max_workers): dev for dev in devices
            }
            for future in as_completed(futures):
                future.result()
        elapsed = time.monotonic() - t0
        logger.info("All %d devices saved in %.2fs", len(devices), elapsed)
    finally:
        (Path(os.environ["GMS_CONTROL_DIR"]) / "checkpoint-done").write_text(
            "done",
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
