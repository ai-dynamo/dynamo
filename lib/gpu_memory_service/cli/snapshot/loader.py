# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GMS checkpoint loader entry point.

Loads saved GMS state from a checkpoint directory into the running GMS servers.
Devices are loaded in parallel to saturate PVC bandwidth.
"""

from __future__ import annotations

import importlib
import logging
import os
import signal
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

_STARTUP_MONOTONIC_NS = time.monotonic_ns()


def _startup_log(event: str, detail: str = "") -> None:
    elapsed_ms = (time.monotonic_ns() - _STARTUP_MONOTONIC_NS) / 1_000_000
    message = f"GMS_LOADER_STARTUP event={event} elapsed_ms={elapsed_ms:.3f}"
    if detail:
        message = f"{message} {detail}"
    print(message, flush=True)


_startup_log("module_start")
_startup_log("import_common_utils_start")
common_utils = importlib.import_module("gpu_memory_service.common.utils")
get_socket_path = common_utils.get_socket_path

_startup_log("import_common_utils_done")
_startup_log("import_storage_client_start")
storage_client_module = importlib.import_module(
    "gpu_memory_service.snapshot.storage_client"
)
GMSStorageClient = storage_client_module.GMSStorageClient

_startup_log("import_storage_client_done")
_startup_log("import_transfer_start")
transfer_module = importlib.import_module("gpu_memory_service.snapshot.transfer")
DEFAULT_TRANSFER_BACKEND = transfer_module.DEFAULT_TRANSFER_BACKEND

_startup_log("import_transfer_done")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

DEFAULT_GMS_LOAD_COMPLETE_FILE = "gms-load-complete"
GMS_LOAD_COMPLETE_FILE_ENV = "GMS_LOAD_COMPLETE_FILE"
GMS_TRANSFER_BACKEND_ENV = "GMS_TRANSFER_BACKEND"
GMS_POD_UID_ENV = "GMS_POD_UID"
GMS_WEIGHTS_CHECKPOINT_DIR_ENV = "GMS_WEIGHTS_CHECKPOINT_DIR"
GMS_RESTORE_TRIGGER_FILE_ENV = "GMS_RESTORE_TRIGGER_FILE"
GMS_RESTORE_TRIGGER_ANNOTATION_ENV = "GMS_RESTORE_TRIGGER_ANNOTATION"
RESTORE_TRIGGER_POLL_SECONDS = 0.05
K8S_API_TIMEOUT_SECONDS = 2.0
SERVICE_ACCOUNT_DIR = Path("/var/run/secrets/kubernetes.io/serviceaccount")


def _load_device(
    checkpoint_dir: str,
    device: int,
    max_workers: int,
    transfer_backend: str,
) -> None:
    input_dir = os.path.join(checkpoint_dir, f"device-{device}")
    logger.info(
        "Loading GMS checkpoint: device=%d input_dir=%s transfer_backend=%s max_workers=%d",
        device,
        input_dir,
        transfer_backend,
        max_workers,
    )
    t0 = time.monotonic()
    client = GMSStorageClient(
        socket_path=get_socket_path(device),
        device=device,
        transfer_backend=transfer_backend,
    )
    client.load_to_gms(
        input_dir,
        max_workers=max_workers,
        clear_existing=True,
        wait_for_socket=True,
    )
    elapsed = time.monotonic() - t0
    logger.info("GMS checkpoint loaded: device=%d elapsed=%.2fs", device, elapsed)


def _completion_sentinel_file() -> str:
    override = os.environ.get(GMS_LOAD_COMPLETE_FILE_ENV, "").strip()
    if override:
        return override
    pod_uid = os.environ.get(GMS_POD_UID_ENV, "").strip()
    if pod_uid:
        return f"{DEFAULT_GMS_LOAD_COMPLETE_FILE}-{pod_uid}"
    return DEFAULT_GMS_LOAD_COMPLETE_FILE


def _completion_sentinel_path(checkpoint_dir: str) -> Path:
    return Path(checkpoint_dir) / _completion_sentinel_file()


def _clear_completion_sentinel(checkpoint_dir: str) -> None:
    try:
        _completion_sentinel_path(checkpoint_dir).unlink()
    except FileNotFoundError:
        pass


def _wait_for_restore_trigger_file(path: Path) -> None:
    _startup_log("restore_trigger_file_wait_start", f"path={path}")
    while True:
        try:
            trigger = path.read_text(encoding="utf-8").strip()
        except FileNotFoundError:
            trigger = ""
        if trigger:
            logger.info("Observed GMS restore trigger: file=%s token=%s", path, trigger)
            _startup_log("restore_trigger_file_wait_done", f"token={trigger}")
            return
        time.sleep(RESTORE_TRIGGER_POLL_SECONDS)


def _k8s_pod_annotation_reader(annotation: str):
    import json
    import ssl
    from urllib import parse, request

    host = os.environ.get("KUBERNETES_SERVICE_HOST", "").strip()
    port = os.environ.get("KUBERNETES_SERVICE_PORT_HTTPS", "").strip() or "443"
    pod_name = os.environ.get("HOSTNAME", "").strip()
    namespace = (SERVICE_ACCOUNT_DIR / "namespace").read_text(encoding="utf-8").strip()
    token = (SERVICE_ACCOUNT_DIR / "token").read_text(encoding="utf-8").strip()
    if not host or not pod_name or not namespace or not token:
        raise RuntimeError(
            "missing Kubernetes service host, pod name, namespace, or service account token"
        )

    url = (
        f"https://{host}:{port}/api/v1/namespaces/"
        f"{parse.quote(namespace, safe='')}/pods/{parse.quote(pod_name, safe='')}"
    )
    ca_path = SERVICE_ACCOUNT_DIR / "ca.crt"
    context = (
        ssl.create_default_context(cafile=str(ca_path))
        if ca_path.exists()
        else ssl.create_default_context()
    )

    def read_annotation() -> str:
        req = request.Request(
            url,
            headers={"Authorization": f"Bearer {token}"},
        )
        with request.urlopen(
            req,
            context=context,
            timeout=K8S_API_TIMEOUT_SECONDS,
        ) as resp:
            pod = json.loads(resp.read())
        annotations = pod.get("metadata", {}).get("annotations") or {}
        return str(annotations.get(annotation, "")).strip()

    return namespace, pod_name, read_annotation


def _wait_for_restore_trigger_annotation(annotation: str) -> bool:
    try:
        namespace, pod_name, read_annotation = _k8s_pod_annotation_reader(annotation)
    except Exception as exc:
        logger.warning(
            "Unable to configure GMS restore trigger annotation watch: annotation=%s error=%s",
            annotation,
            exc,
        )
        return False

    _startup_log(
        "restore_trigger_k8s_wait_start",
        f"annotation={annotation} pod={namespace}/{pod_name}",
    )
    next_error_log = 0.0
    while True:
        try:
            trigger = read_annotation()
        except Exception as exc:
            now = time.monotonic()
            if now >= next_error_log:
                logger.warning(
                    "Waiting for GMS restore trigger annotation failed: annotation=%s error=%s",
                    annotation,
                    exc,
                )
                next_error_log = now + 5.0
            trigger = ""
        if trigger:
            logger.info(
                "Observed GMS restore trigger: annotation=%s token=%s",
                annotation,
                trigger,
            )
            _startup_log("restore_trigger_k8s_wait_done", f"token={trigger}")
            return True
        time.sleep(RESTORE_TRIGGER_POLL_SECONDS)


def _wait_for_restore_trigger() -> None:
    annotation = os.environ.get(GMS_RESTORE_TRIGGER_ANNOTATION_ENV, "").strip()
    if annotation and _wait_for_restore_trigger_annotation(annotation):
        return

    trigger_file = os.environ.get(GMS_RESTORE_TRIGGER_FILE_ENV, "").strip()
    if not trigger_file:
        return

    _wait_for_restore_trigger_file(Path(trigger_file))


def _list_checkpoint_devices(checkpoint_dir: str) -> list[int]:
    devices: list[int] = []
    for child in Path(checkpoint_dir).iterdir():
        if not child.is_dir() or not child.name.startswith("device-"):
            continue
        suffix = child.name.removeprefix("device-")
        if suffix.isdigit():
            devices.append(int(suffix))
    if devices:
        return sorted(set(devices))

    _startup_log("checkpoint_devices_missing_fallback_nvml")
    from gpu_memory_service.common.cuda_utils import list_devices

    return list_devices()


def _write_completion_sentinel(checkpoint_dir: str) -> None:
    path = _completion_sentinel_path(checkpoint_dir)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path.write_text("ok\n", encoding="utf-8")
    tmp_path.replace(path)
    logger.info("Wrote GMS load completion sentinel: %s", path)


def main() -> None:
    _startup_log("main_enter", f"pid={os.getpid()}")
    control_dir = os.environ["GMS_CHECKPOINT_DIR"]
    checkpoint_dir = os.environ.get(GMS_WEIGHTS_CHECKPOINT_DIR_ENV, control_dir)
    max_workers = int(os.environ.get("GMS_LOAD_WORKERS", "8"))
    transfer_backend = os.environ.get(
        GMS_TRANSFER_BACKEND_ENV,
        DEFAULT_TRANSFER_BACKEND,
    )
    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "<unset>")
    nvidia_visible = os.environ.get("NVIDIA_VISIBLE_DEVICES", "<unset>")
    _startup_log(
        "env_loaded",
        f"transfer_backend={transfer_backend} max_workers={max_workers} "
        f"cuda_visible={cuda_visible} nvidia_visible={nvidia_visible}",
    )
    _startup_log("clear_sentinel_start")
    _clear_completion_sentinel(control_dir)
    _startup_log("clear_sentinel_done")
    _wait_for_restore_trigger()
    _startup_log("discover_devices_start")
    devices = _list_checkpoint_devices(checkpoint_dir)
    _startup_log(
        "discover_devices_done",
        f"devices={','.join(str(dev) for dev in devices)}",
    )

    t0 = time.monotonic()
    _startup_log("thread_pool_submit_start", f"device_count={len(devices)}")
    with ThreadPoolExecutor(max_workers=len(devices)) as pool:
        futures = {
            pool.submit(
                _load_device,
                checkpoint_dir,
                dev,
                max_workers,
                transfer_backend,
            ): dev
            for dev in devices
        }
        _startup_log("thread_pool_submit_done", f"future_count={len(futures)}")
        for future in as_completed(futures):
            dev = futures[future]
            future.result()
            logger.info("Device %d load complete", dev)
    elapsed = time.monotonic() - t0
    logger.info("All %d devices loaded in %.2fs", len(devices), elapsed)
    _write_completion_sentinel(control_dir)
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))

    while True:
        time.sleep(3600)


if __name__ == "__main__":
    main()
