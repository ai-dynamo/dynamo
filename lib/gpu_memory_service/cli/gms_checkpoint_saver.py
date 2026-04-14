# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import logging
import os
import ssl
import time
import urllib.request
from pathlib import Path
from typing import Any

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


_SERVICE_ACCOUNT_TOKEN = Path(
    "/var/run/secrets/kubernetes.io/serviceaccount/token"
)
_SERVICE_ACCOUNT_CA = "/var/run/secrets/kubernetes.io/serviceaccount/ca.crt"


def checkpoint_pod_ready(pod: dict[str, Any]) -> bool:
    status = pod.get("status") or {}
    if str(status.get("phase", "")).strip() != "Running":
        return False
    for condition in status.get("conditions") or []:
        if condition.get("type") == "Ready" and str(
            condition.get("status", "")
        ).strip() == "True":
            return True
    return False


def main_terminated(pod: dict[str, Any]) -> bool:
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
        except Exception:
            time.sleep(1)
            continue

        if checkpoint_pod_ready(pod):
            break
        if main_terminated(pod):
            raise SystemExit("main container terminated before GMS save could start")
        time.sleep(1)

    logger.info("Checkpoint pod is Ready; starting GMS save")
    try:
        for device in list_devices():
            wait_for_weights_socket(device)
            output_dir = checkpoint_device_dir(checkpoint_dir, device)
            logger.info(
                "Saving GMS checkpoint: device=%d output_dir=%s",
                device,
                output_dir,
            )
            client = GMSStorageClient(
                output_dir,
                socket_path=get_socket_path(device),
                device=device,
            )
            client.save(max_workers=4)
    finally:
        (Path(os.environ["GMS_CONTROL_DIR"]) / "checkpoint-done").write_text(
            "done",
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
