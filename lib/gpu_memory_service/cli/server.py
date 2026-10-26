# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GMS server entry point.

Launches one GMS server process per GPU, then supervises them. Each V1 process
is pinned to its GPU, can open no other GPU (see v1.landlock), and with
--enable-loader loads its GPU's weights in a thread. V0 can start one loader
process per GPU. Device discovery uses NVML without initializing the CUDA
driver.
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
from gpu_memory_service.common.utils import get_socket_path
from gpu_memory_service.common.vmm import VMMDeviceType, get_vmm, init_vmm
from gpu_memory_service.v1 import landlock, ranks
from gpu_memory_service.v1.client.session import _GMSClientSession as v1_session
from gpu_memory_service.v1.device import get_socket_path as v1_get_socket_path
from gpu_memory_service.v1.device import node_gpus

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

_PROBE_TIMEOUT_SECONDS = 0.5


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


def _run_v1_ranks(loader_argv: list[str] | None, confine: bool) -> int:
    """Run one V1 server process per GPU until one exits."""
    if confine:
        try:
            landlock.abi_version()
        except OSError as exc:
            raise SystemExit(
                f"GMS V1 cannot confine each device's process to its GPU: {exc}. "
                "Pass --no-isolation to run them unconfined; each then starts "
                "CUDA on every visible GPU, which is slower."
            ) from exc

    def command(device: int) -> list[str]:
        argv = [sys.executable, "-m", "gpu_memory_service", "--device", str(device)]
        if loader_argv is not None:
            argv += ["--enable-loader", *loader_argv]
        return argv

    processes: list[subprocess.Popen] = []

    def terminate(*_args) -> None:
        _terminate_all(processes)
        raise SystemExit(0)

    signal.signal(signal.SIGTERM, terminate)
    signal.signal(signal.SIGINT, terminate)
    try:
        processes.extend(ranks.start_ranks(node_gpus(), command, confine=confine))
        return _supervise(processes)
    finally:
        _terminate_all(processes)


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
        "--no-isolation",
        action="store_true",
        help=(
            "V1: do not confine each GPU's server process to its own GPU with "
            "Landlock, for kernels without it. Each process then starts CUDA on "
            "every visible GPU, which is slower."
        ),
    )
    parser.add_argument(
        "--enable-loader",
        nargs=argparse.REMAINDER,
        metavar="ARG",
        help=(
            "Load checkpoint weights into the servers at start-up. All later "
            "arguments, including --checkpoint-dir, go to the loader. V0 starts "
            "one loader process per GPU that exits once that GPU is loaded; pass "
            "--device N to load only GPU N. V1 loads each GPU in a thread of "
            "that GPU's server process."
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
    if use_v1 and not args.probe_restore_ready:
        raise SystemExit(_run_v1_ranks(args.enable_loader, not args.no_isolation))
    if use_v1:
        for gpu in node_gpus():
            with closing(
                v1_session(
                    v1_get_socket_path(gpu.uuid, "weights"),
                    RequestedLockType.RO,
                    connect_timeout=_PROBE_TIMEOUT_SECONDS,
                    admission_timeout=_PROBE_TIMEOUT_SECONDS,
                )
            ):
                pass
        return

    init_vmm(VMMDeviceType.from_str(args.device_type))
    vmm = get_vmm()
    devices = vmm.list_devices()
    if args.probe_restore_ready:
        timeout_ms = int(_PROBE_TIMEOUT_SECONDS * 1000)
        for device in devices:
            with closing(
                v0_session(
                    get_socket_path(device, "weights"),
                    RequestedLockType.RO,
                    timeout_ms,
                )
            ):
                pass
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
            command = [sys.executable, "-m", "gpu_memory_service"]
            command.extend(["--device", str(device)])
            command.extend(["--device-type", args.device_type])
            process = subprocess.Popen(command)
            logger.info("Started GMS device=%d pid=%d", device, process.pid)
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
