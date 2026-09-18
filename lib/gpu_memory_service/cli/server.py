# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GMS server entry point.

Launches one GMS server process per GPU, then supervises them. Restore
optionally starts one-shot loaders. Device discovery uses NVML without
initializing the CUDA driver.
"""

from __future__ import annotations

import argparse
import logging
import os
import signal
import subprocess
import sys
import time
from contextlib import closing

from gpu_memory_service.cli.snapshot import start_per_device
from gpu_memory_service.client.session import _GMSClientSession as v0_session
from gpu_memory_service.common.locks import RequestedLockType
from gpu_memory_service.common.utils import GMS_TAGS, get_socket_path
from gpu_memory_service.common.vmm import VMMDeviceType, get_vmm, init_vmm
from gpu_memory_service.v1.client.session import _GMSClientSession as v1_session
from gpu_memory_service.v1.device import get_socket_path as v1_get_socket_path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

_PROBE_TIMEOUT_SECONDS = 0.5


def _tags_from_env() -> tuple[str, ...]:
    raw = os.environ.get("GMS_SERVER_TAGS")
    if not raw:
        return tuple(GMS_TAGS)
    tags = tuple(tag.strip() for tag in raw.split(",") if tag.strip())
    if not tags:
        raise RuntimeError("GMS_SERVER_TAGS must contain at least one tag")
    unknown = sorted(set(tags) - set(GMS_TAGS))
    if unknown:
        raise RuntimeError(
            f"unsupported GMS_SERVER_TAGS: {', '.join(unknown)}; "
            f"expected a subset of {', '.join(GMS_TAGS)}"
        )
    if len(tags) != len(set(tags)):
        raise RuntimeError("GMS_SERVER_TAGS must not contain duplicate tags")
    return tags


def _directory_command(socket_path: str) -> list[str]:
    return [
        sys.executable,
        "-m",
        "gms_kv_ring.daemon.directory_server",
        socket_path,
    ]


def _terminate_all(processes: list[subprocess.Popen]) -> None:
    for process in processes:
        if process.poll() is None:
            process.terminate()


def _supervise(
    servers: list[subprocess.Popen],
    loaders: list[subprocess.Popen] | None = None,
) -> int:
    """Supervise persistent servers and optional one-shot loaders."""
    pending_loaders = list(loaders or ())
    while servers:
        for server in servers:
            exit_code = server.poll()
            if exit_code is not None:
                _terminate_all([*servers, *pending_loaders])
                return exit_code or 1

        for loader in list(pending_loaders):
            exit_code = loader.poll()
            if exit_code is not None:
                if exit_code:
                    _terminate_all([*servers, *pending_loaders])
                    return exit_code
                pending_loaders.remove(loader)

        time.sleep(1)
    return 0


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="GPU Memory Service supervisor (one server process per device).",
        allow_abbrev=False,
    )
    parser.add_argument(
        "--device-type",
        type=str,
        default=VMMDeviceType.CUDA.value,
        choices=[d.value for d in VMMDeviceType],
        help="VMM device type forwarded to server (default: cuda).",
    )
    parser.add_argument(
        "--probe-restore-ready",
        action="store_true",
        help=(
            "One-shot: try bounded RO admission on every weights socket and "
            "exit. Does not start servers or loaders."
        ),
    )
    parser.add_argument(
        "--enable-loader",
        nargs=argparse.REMAINDER,
        metavar="ARG",
        help=(
            "Start loaders after the servers. Remaining args, including "
            "--checkpoint-dir, go to the loader. Pass --device to load one GPU."
        ),
    )
    raw = argv if argv is not None else sys.argv[1:]
    if "--probe-restore-ready" in raw and "--enable-loader" in raw:
        parser.error(
            "--probe-restore-ready is one-shot and cannot start servers or loaders"
        )
    args = parser.parse_args(argv)
    use_v1 = os.environ.get("DYN_GMS_USE_V1") == "true"
    if use_v1 and args.device_type != VMMDeviceType.CUDA.value:
        parser.error("DYN_GMS_USE_V1=true only supports --device-type=cuda")
    tags = _tags_from_env()
    if use_v1 and tags != tuple(GMS_TAGS):
        parser.error(
            "GMS_SERVER_TAGS subsets are not supported by GMS V1 because its "
            "weights and kv_cache domains share one checkpoint lifecycle"
        )

    init_vmm(VMMDeviceType.from_str(args.device_type))
    vmm = get_vmm()
    devices = vmm.list_devices()
    if args.probe_restore_ready:
        timeout_ms = int(_PROBE_TIMEOUT_SECONDS * 1000)
        for device in devices:
            if use_v1:
                session = v1_session(
                    v1_get_socket_path(device, "weights"),
                    RequestedLockType.RO,
                    connect_timeout=_PROBE_TIMEOUT_SECONDS,
                    admission_timeout=_PROBE_TIMEOUT_SECONDS,
                )
            else:
                session = v0_session(
                    get_socket_path(device, "weights"),
                    RequestedLockType.RO,
                    timeout_ms,
                )
            with closing(session):
                pass
        return

    servers: list[subprocess.Popen] = []
    loaders: list[subprocess.Popen] = []

    directory_socket = os.environ.get("GMS_DIRECTORY_SOCKET")
    if directory_socket:
        process = subprocess.Popen(_directory_command(directory_socket))
        logger.info(
            "Started GMS content directory socket=%s pid=%d",
            directory_socket,
            process.pid,
        )
        servers.append(process)

    def terminate(*_args) -> None:
        _terminate_all([*servers, *loaders])
        raise SystemExit(0)

    signal.signal(signal.SIGTERM, terminate)
    signal.signal(signal.SIGINT, terminate)

    try:
        for device in devices:
            command = [sys.executable, "-m", "gpu_memory_service"]
            command.extend(["--device", str(device)])
            if not use_v1:
                command.extend(["--device-type", args.device_type])
                for tag in tags:
                    command.extend(["--tag", tag])
            process = subprocess.Popen(command)
            logger.info(
                "Started GMS%s device=%d tags=%s pid=%d",
                " V1" if use_v1 else "",
                device,
                ",".join(tags),
                process.pid,
            )
            servers.append(process)

        if args.enable_loader is not None:
            loader_argv = list(args.enable_loader)
            loaders.extend(
                start_per_device(
                    "gpu_memory_service.cli.snapshot.loader",
                    loader_argv,
                    devices,
                )
            )

        raise SystemExit(_supervise(servers, loaders))
    finally:
        _terminate_all([*servers, *loaders])


if __name__ == "__main__":
    main()
