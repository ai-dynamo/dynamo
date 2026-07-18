# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GMS server entry point.

Launches one GMS server process per GPU serving every production GMS tag,
then supervises them: terminates the rest if any child exits, and propagates
the first non-zero exit code. Runs until SIGTERM (pod termination kills it)
or until a child exits.
"""

from __future__ import annotations

import logging
import os
import signal
import subprocess
import sys
import time
from collections.abc import Mapping

from gpu_memory_service.common.cuda_utils import list_device_uuids, list_devices
from gpu_memory_service.common.utils import (
    ENV_SERVER_DEVICE_UUID,
    ENV_SERVER_EXPECTED_GPU_UUIDS,
    ENV_SERVER_GPU_UUID_ISOLATION,
    is_truthy_env,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def _child_command(device: int) -> list[str]:
    """Command for one child process serving every production tag on one GPU."""
    return [sys.executable, "-m", "gpu_memory_service", "--device", str(device)]


def _resolve_visible_device_uuids(
    available_uuids: list[str],
    visibility: str | None,
    *,
    ordinal_uuids: list[str] | None = None,
) -> list[str]:
    """Resolve inherited CUDA visibility to full physical GPU UUIDs."""
    if visibility is None or visibility.lower() == "all":
        return list(available_uuids)
    if not visibility or visibility.lower() in {"none", "void"}:
        return []

    resolved = []
    for token in (part.strip() for part in visibility.split(",")):
        if not token:
            raise ValueError(f"invalid empty GPU in visibility value {visibility!r}")
        if token.isdigit():
            ordinal = int(token)
            if ordinal_uuids is None or ordinal >= len(ordinal_uuids):
                raise ValueError(
                    "numeric GPU visibility is ambiguous without a UUID allocation"
                )
            uuid = ordinal_uuids[ordinal]
        elif token.startswith("GPU-"):
            matches = [
                uuid
                for uuid in available_uuids
                if uuid.lower().startswith(token.lower())
            ]
            if len(matches) != 1:
                raise ValueError(
                    f"GPU UUID {token!r} does not uniquely identify an "
                    "NVML-visible GPU"
                )
            uuid = matches[0]
        else:
            raise ValueError(
                f"unsupported GPU visibility token {token!r}; full GPUs are required"
            )
        if uuid in resolved:
            raise ValueError(f"duplicate GPU UUID {uuid!r} in visibility value")
        resolved.append(uuid)
    return resolved


def _assigned_device_uuids(environ: Mapping[str, str]) -> list[str]:
    available_uuids = list_device_uuids()
    if any(not uuid.startswith("GPU-") for uuid in available_uuids):
        raise ValueError("UUID-isolated GMS requires full physical GPUs")
    nvidia_visibility = environ.get("NVIDIA_VISIBLE_DEVICES")
    normalized_nvidia = (
        nvidia_visibility.strip().lower() if nvidia_visibility is not None else None
    )
    if normalized_nvidia == "none":
        cuda_visibility = environ.get("CUDA_VISIBLE_DEVICES")
        if cuda_visibility is not None and cuda_visibility.strip().lower() not in {
            "",
            "none",
            "void",
        }:
            raise ValueError(
                "CUDA_VISIBLE_DEVICES contradicts NVIDIA_VISIBLE_DEVICES=none"
            )
        return []

    expected_uuids = None
    expected_visibility = environ.get(ENV_SERVER_EXPECTED_GPU_UUIDS)
    if expected_visibility is not None:
        expected_uuids = _resolve_visible_device_uuids(
            available_uuids,
            expected_visibility,
        )
    nvidia_uuids = None
    if nvidia_visibility and any(
        token.strip().startswith("GPU-") for token in nvidia_visibility.split(",")
    ):
        nvidia_uuids = _resolve_visible_device_uuids(
            available_uuids,
            nvidia_visibility,
        )
        if expected_uuids is not None and any(
            uuid not in expected_uuids for uuid in nvidia_uuids
        ):
            raise ValueError(
                "NVIDIA_VISIBLE_DEVICES contradicts the declared DRA allocation"
            )
    elif normalized_nvidia in {"all", "void"}:
        nvidia_uuids = expected_uuids
    elif normalized_nvidia is None:
        nvidia_uuids = expected_uuids
    else:
        raise ValueError(
            "numeric or unsupported NVIDIA_VISIBLE_DEVICES is ambiguous "
            "without an explicit UUID allocation"
        )
    cuda_visibility = environ.get("CUDA_VISIBLE_DEVICES")
    if nvidia_uuids is not None:
        if cuda_visibility is None or cuda_visibility.strip().lower() == "all":
            return nvidia_uuids
        return _resolve_visible_device_uuids(
            nvidia_uuids,
            cuda_visibility,
            ordinal_uuids=nvidia_uuids,
        )

    if cuda_visibility is not None:
        cuda_tokens = [token.strip() for token in cuda_visibility.split(",")]
        if all(token.startswith("GPU-") for token in cuda_tokens):
            return _resolve_visible_device_uuids(
                available_uuids,
                cuda_visibility,
            )
        if cuda_visibility.strip().lower() in {"", "none", "void"}:
            return []
        if cuda_visibility.strip().lower() == "all":
            raise ValueError(
                "CUDA_VISIBLE_DEVICES=all is ambiguous without an explicit "
                "UUID allocation"
            )
        if all(token.isdigit() for token in cuda_tokens):
            raise ValueError(
                "numeric CUDA_VISIBLE_DEVICES is ambiguous without an explicit "
                "UUID allocation"
            )
        return _resolve_visible_device_uuids(
            available_uuids,
            cuda_visibility,
        )

    raise ValueError("GPU allocation is ambiguous without explicit UUID visibility")


def _child_launch(
    device: int,
    *,
    device_uuid: str | None = None,
    environ: Mapping[str, str] | None = None,
) -> tuple[list[str], dict[str, str] | None]:
    """Build a child command and an optional isolated environment."""
    if device_uuid is None:
        return _child_command(device), None

    child_env = dict(os.environ if environ is None else environ)
    child_env["CUDA_VISIBLE_DEVICES"] = device_uuid
    child_env[ENV_SERVER_DEVICE_UUID] = device_uuid
    return _child_command(0), child_env


def _terminate_all(processes: list[subprocess.Popen]) -> None:
    for process in processes:
        if process.poll() is None:
            process.terminate()


def _supervise(processes: list[subprocess.Popen]) -> int:
    """Block until any child exits, terminate the rest, and return its exit code."""
    while processes:
        for process in processes:
            exit_code = process.poll()
            if exit_code is not None:
                _terminate_all(processes)
                return exit_code
        time.sleep(1)
    return 0


def main() -> None:
    processes = []
    isolate_gpu_uuid = is_truthy_env(ENV_SERVER_GPU_UUID_ISOLATION)
    if isolate_gpu_uuid:
        try:
            device_uuids = _assigned_device_uuids(os.environ)
        except ValueError as error:
            raise SystemExit(f"invalid inherited GPU visibility: {error}") from error
        if not device_uuids:
            raise SystemExit("no nvidia devices found in inherited GPU visibility")
        launches = enumerate(device_uuids)
    else:
        launches = ((device, None) for device in list_devices())

    for device, device_uuid in launches:
        command, child_env = _child_launch(
            device,
            device_uuid=device_uuid,
            environ=os.environ,
        )
        if child_env is None:
            proc = subprocess.Popen(command)
        else:
            proc = subprocess.Popen(command, env=child_env)
        logger.info(
            "Started GMS device=%d physical_uuid=%s child_device=%s pid=%d",
            device,
            device_uuid or "-",
            command[-1],
            proc.pid,
        )
        processes.append(proc)

    def terminate(*_args) -> None:
        _terminate_all(processes)
        raise SystemExit(0)

    signal.signal(signal.SIGTERM, terminate)
    signal.signal(signal.SIGINT, terminate)

    raise SystemExit(_supervise(processes))


if __name__ == "__main__":
    main()
