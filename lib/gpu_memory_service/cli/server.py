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
import signal
import subprocess
import sys
import time
from contextlib import closing

from gpu_memory_service.cli.snapshot import start_per_device
from gpu_memory_service.common.locks import RequestedLockType
from gpu_memory_service.common.vmm import VMMDeviceType, get_vmm, init_vmm
from gpu_memory_service.v1.client.session import _GMSClientSession
from gpu_memory_service.v1.device import get_socket_path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

_PROBE_TIMEOUT_SECONDS = 0.5


def _child_command(device: int, device_type: str, use_v1: bool = False) -> list[str]:
    """Command for one child process serving every production tag on one GPU."""
    command = [sys.executable, "-m", "gpu_memory_service"]
    if use_v1:
        command.append("--use-v1")
    command.extend(["--device", str(device)])
    if not use_v1:
        command.extend(["--device-type", device_type])
    return command


def _probe_v1_restore_ready(devices: list[int]) -> None:
    for device in devices:
        with closing(
            _GMSClientSession(
                get_socket_path(device, "weights"),
                RequestedLockType.RO,
                connect_timeout=_PROBE_TIMEOUT_SECONDS,
                admission_timeout=_PROBE_TIMEOUT_SECONDS,
            )
        ):
            pass


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
        "--use-v1",
        action="store_true",
        help="Launch the CUDA-only V1 server for every visible device.",
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
        help="Attempt bounded RO admission on every V1 weights socket and exit.",
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
    args = parser.parse_args(argv)
    if args.use_v1 and args.device_type != VMMDeviceType.CUDA.value:
        parser.error("--use-v1 only supports --device-type=cuda")
    if args.probe_restore_ready and not args.use_v1:
        parser.error("--probe-restore-ready requires --use-v1")
    if args.probe_restore_ready and args.enable_loader is not None:
        parser.error("--probe-restore-ready cannot be combined with --enable-loader")

    init_vmm(VMMDeviceType.from_str(args.device_type))
    vmm = get_vmm()
    devices = vmm.list_devices()
    if args.probe_restore_ready:
        _probe_v1_restore_ready(devices)
        return

    servers: list[subprocess.Popen] = []
    loaders: list[subprocess.Popen] = []

    def terminate(*_args) -> None:
        _terminate_all([*servers, *loaders])
        raise SystemExit(0)

    signal.signal(signal.SIGTERM, terminate)
    signal.signal(signal.SIGINT, terminate)

    try:
        for device in devices:
            server = subprocess.Popen(
                _child_command(device, args.device_type, use_v1=args.use_v1)
            )
            logger.info(
                "Started GMS%s device=%d pid=%d",
                " V1" if args.use_v1 else "",
                device,
                server.pid,
            )
            servers.append(server)

        if args.enable_loader is not None:
            loader_argv = list(args.enable_loader)
            if args.use_v1:
                loader_argv.insert(0, "--use-v1")
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
