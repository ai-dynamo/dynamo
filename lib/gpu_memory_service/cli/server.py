# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GMS server entry point.

Launches one GMS server process per GPU serving every production GMS tag,
then supervises them. V1 restore optionally adds one one-shot loader per
visible GPU, or only ``--device`` when the loader remainder names one.
Device discovery uses NVML without initializing the CUDA driver.
"""

from __future__ import annotations

import argparse
import logging
import signal
import subprocess
import sys
import time
from contextlib import closing

from gpu_memory_service.cli.snapshot import should_fan_out_v1
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


def _loader_command(loader_args: list[str], device: int | None = None) -> list[str]:
    """Command for one V1 loader; inject ``--device`` only when fanning out."""
    command = [
        sys.executable,
        "-m",
        "gpu_memory_service.v1.snapshot.loader",
        *loader_args,
    ]
    if device is not None:
        command.extend(["--device", str(device)])
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
            "Run one V1 loader per visible device after the servers start. "
            "Pass --device in the following arguments to load only that GPU. "
            "All following arguments are forwarded to the loader, including "
            "--checkpoint-dir."
        ),
    )
    args = parser.parse_args(argv)
    if args.use_v1 and args.device_type != VMMDeviceType.CUDA.value:
        parser.error("--use-v1 only supports --device-type=cuda")
    if args.probe_restore_ready and not args.use_v1:
        parser.error("--probe-restore-ready requires --use-v1")
    if args.enable_loader is not None and not args.use_v1:
        parser.error("--enable-loader requires --use-v1")
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
            if should_fan_out_v1(args.enable_loader):
                for device in devices:
                    loader = subprocess.Popen(
                        _loader_command(args.enable_loader, device)
                    )
                    logger.info(
                        "Started GMS V1 loader device=%d pid=%d",
                        device,
                        loader.pid,
                    )
                    loaders.append(loader)
            else:
                loader = subprocess.Popen(_loader_command(args.enable_loader))
                logger.info("Started GMS V1 loader pid=%d", loader.pid)
                loaders.append(loader)

        raise SystemExit(_supervise(servers, loaders))
    finally:
        _terminate_all([*servers, *loaders])


if __name__ == "__main__":
    main()
