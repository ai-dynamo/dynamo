# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GMS checkpoint loader entry point.

Loads saved GMS state from a checkpoint directory into the running GMS
servers. Devices are loaded in parallel to saturate PVC bandwidth. Runs as
a regular sidecar; the GMS RO lock — not init-phase ordering — gates the
restored engine on weight load.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

from gpu_memory_service.common import cuda_utils
from gpu_memory_service.common.utils import get_socket_path
from gpu_memory_service.snapshot.backends.sharded_ssd import parse_sharded_ssd_roots
from gpu_memory_service.snapshot.storage_client import GMSStorageClient
from gpu_memory_service.snapshot.transfer import TransferBackendKind

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class _FileTransferGate:
    """Process-wide file barrier for synchronized restore experiments."""

    def __init__(
        self,
        *,
        ready_file: str,
        release_file: str,
        participants: list[int],
        timeout_s: float,
    ) -> None:
        if not os.path.isabs(ready_file) or not os.path.isabs(release_file):
            raise ValueError("transfer gate marker paths must be absolute")
        if ready_file == release_file:
            raise ValueError("transfer gate ready and release paths must differ")
        if timeout_s <= 0:
            raise ValueError("transfer gate timeout must be positive")
        if not participants:
            raise ValueError("transfer gate requires at least one participant")

        self._ready_file = ready_file
        self._release_file = release_file
        self._expected = frozenset(participants)
        self._timeout_s = timeout_s
        self._arrived: set[int] = set()
        self._released = False
        self._failure: Exception | None = None
        self._condition = threading.Condition()

    def wait(self, participant: int) -> None:
        started_at = time.monotonic()
        deadline = started_at + self._timeout_s
        with self._condition:
            if self._failure is not None:
                raise RuntimeError("GMS experimental transfer gate failed") from (
                    self._failure
                )
            if participant not in self._expected:
                raise RuntimeError(
                    f"unexpected transfer gate participant {participant}; "
                    f"expected={sorted(self._expected)}"
                )
            if participant in self._arrived:
                raise RuntimeError(f"duplicate transfer gate participant {participant}")
            self._arrived.add(participant)
            is_last = self._arrived == self._expected

        if is_last:
            try:
                marker_started_at = time.monotonic()
                _write_marker_atomically(
                    self._ready_file,
                    {
                        "readyAt": datetime.now(timezone.utc).isoformat(),
                        "participants": sorted(self._arrived),
                    },
                )
                marker_elapsed_s = time.monotonic() - marker_started_at
                logger.info(
                    "GMS experimental transfer-ready marker created: "
                    "path=%s participants=%s duration=%.6fs",
                    self._ready_file,
                    sorted(self._arrived),
                    marker_elapsed_s,
                )
                while not os.path.exists(self._release_file):
                    if time.monotonic() >= deadline:
                        raise TimeoutError(
                            "timed out waiting for GMS experimental transfer "
                            f"release marker {self._release_file}"
                        )
                    time.sleep(0.01)
            except Exception as exc:
                with self._condition:
                    self._failure = exc
                    self._condition.notify_all()
                raise
            with self._condition:
                self._released = True
                self._condition.notify_all()
        else:
            with self._condition:
                while not self._released and self._failure is None:
                    remaining_s = deadline - time.monotonic()
                    if remaining_s <= 0:
                        self._failure = TimeoutError(
                            "timed out waiting for all GMS transfer gate "
                            f"participants; arrived={sorted(self._arrived)} "
                            f"expected={sorted(self._expected)}"
                        )
                        self._condition.notify_all()
                        break
                    self._condition.wait(timeout=remaining_s)
                if self._failure is not None:
                    raise RuntimeError("GMS experimental transfer gate failed") from (
                        self._failure
                    )

        logger.info(
            "GMS experimental transfer gate released: device=%d "
            "release_file=%s wait=%.6fs",
            participant,
            self._release_file,
            time.monotonic() - started_at,
        )


def _write_marker_atomically(path: str, payload: dict[str, object]) -> None:
    parent = os.path.dirname(path)
    if not os.path.isdir(parent):
        raise FileNotFoundError(
            f"transfer gate marker directory does not exist: {parent}"
        )
    fd, temporary_path = tempfile.mkstemp(
        dir=parent,
        prefix=f".{os.path.basename(path)}.",
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary_path, path)
    finally:
        try:
            os.unlink(temporary_path)
        except FileNotFoundError:
            pass


def _load_device(
    checkpoint_dir: str,
    device: int,
    max_workers: int,
    transfer_backend: str,
    sharded_ssd_roots: list[str],
    sharded_ssd_queues_per_root: int,
    restore_transfer_gate: _FileTransferGate | None = None,
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
    # NIXL/POSIX staging setup may happen in background worker threads, but
    # GMSStorageClient still publishes the restored layout from this thread.
    # Ensure the loader's main per-device thread has a current CUDA context for
    # the final synchronize/unmap/commit path.
    cuda_utils.cuda_runtime_set_device(device)
    client = GMSStorageClient(
        socket_path=get_socket_path(device),
        device=device,
        transfer_backend=transfer_backend,
        sharded_ssd_roots=sharded_ssd_roots,
        sharded_ssd_queues_per_root=sharded_ssd_queues_per_root,
        restore_transfer_gate=restore_transfer_gate,
    )
    client.load_to_gms(
        input_dir,
        max_workers=max_workers,
        clear_existing=True,
    )
    elapsed = time.monotonic() - t0
    logger.info("GMS checkpoint loaded: device=%d elapsed=%.2fs", device, elapsed)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Load a GMS checkpoint into GMS.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--checkpoint-dir",
        default=None,
        help=(
            "Checkpoint directory. Required for directory-backed transfer "
            f"backends: {', '.join(backend.value for backend in TransferBackendKind)}."
        ),
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=16,
        help="Shard load workers per device.",
    )
    parser.add_argument(
        "--transfer-backend",
        choices=[backend.value for backend in TransferBackendKind],
        default=TransferBackendKind.NIXL.value,
        help="Restore transfer backend.",
    )
    parser.add_argument(
        "--sharded-ssd-roots",
        default="",
        help=("Comma-separated SSD roots for the sharded-ssd restore backend."),
    )
    parser.add_argument(
        "--sharded-ssd-queues-per-root",
        type=int,
        default=2,
        help="Number of independent sharded-ssd restore queues per SSD root.",
    )
    parser.add_argument(
        "--experiment-transfer-ready-file",
        default="",
        help=(
            "Experimental opt-in: atomically create this file after all device "
            "targets and NIXL POSIX staging workers are ready."
        ),
    )
    parser.add_argument(
        "--experiment-transfer-release-file",
        default="",
        help=(
            "Experimental opt-in: wait for this file before submitting any "
            "NIXL POSIX FILE-to-DRAM transfer."
        ),
    )
    parser.add_argument(
        "--experiment-transfer-gate-timeout-seconds",
        type=float,
        default=300.0,
        help="Timeout for the experimental transfer gate.",
    )
    return parser


def _list_checkpoint_devices(checkpoint_dir: str | None) -> list[int]:
    devices = cuda_utils.list_devices()
    if not checkpoint_dir:
        return devices

    checkpoint_path = Path(checkpoint_dir)
    checkpoint_devices: set[int] = set()
    for child in checkpoint_path.iterdir():
        if not child.is_dir() or not child.name.startswith("device-"):
            continue

        suffix = child.name.removeprefix("device-")
        if suffix.isdigit() and suffix == str(int(suffix)):
            checkpoint_devices.add(int(suffix))

    visible_devices = set(devices)
    missing_devices = sorted(visible_devices - checkpoint_devices)
    extra_devices = sorted(checkpoint_devices - visible_devices)
    if missing_devices or extra_devices:
        raise RuntimeError(
            "Checkpoint device directories under "
            f"{checkpoint_path} do not match CUDA/NVML-visible devices: "
            f"visible={devices} "
            f"checkpoint={sorted(checkpoint_devices)} "
            f"missing={','.join(str(device) for device in missing_devices) or '-'} "
            f"extra={','.join(str(device) for device in extra_devices) or '-'}"
        )

    logger.info(
        "Using CUDA/NVML-visible checkpoint devices from %s: %s",
        checkpoint_path,
        devices,
    )
    return devices


def main(argv: list[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if not args.checkpoint_dir:
        parser.error(
            f"--checkpoint-dir is required for --transfer-backend={args.transfer_backend}"
        )
    if args.sharded_ssd_queues_per_root <= 0:
        parser.error("--sharded-ssd-queues-per-root must be a positive integer")
    checkpoint_dir = args.checkpoint_dir
    max_workers = args.max_workers
    transfer_backend = args.transfer_backend
    sharded_ssd_roots = parse_sharded_ssd_roots(args.sharded_ssd_roots)
    sharded_ssd_queues_per_root = args.sharded_ssd_queues_per_root
    logger.info(
        "Starting GMS load: transfer_backend=%s max_workers=%d "
        "sharded_ssd_roots=%s sharded_ssd_queues_per_root=%d",
        transfer_backend,
        max_workers,
        ",".join(sharded_ssd_roots) or "-",
        sharded_ssd_queues_per_root,
    )
    devices = _list_checkpoint_devices(checkpoint_dir)
    gate_paths = (
        args.experiment_transfer_ready_file,
        args.experiment_transfer_release_file,
    )
    if any(gate_paths) and not all(gate_paths):
        parser.error(
            "--experiment-transfer-ready-file and "
            "--experiment-transfer-release-file must be set together"
        )
    if all(gate_paths) and transfer_backend not in {
        TransferBackendKind.NIXL.value,
        TransferBackendKind.SHARDED_SSD.value,
    }:
        parser.error(
            "the experimental transfer gate requires a NIXL POSIX staging backend"
        )
    transfer_gate = (
        _FileTransferGate(
            ready_file=args.experiment_transfer_ready_file,
            release_file=args.experiment_transfer_release_file,
            participants=devices,
            timeout_s=args.experiment_transfer_gate_timeout_seconds,
        )
        if all(gate_paths)
        else None
    )

    t0 = time.monotonic()
    with ThreadPoolExecutor(max_workers=len(devices)) as pool:
        futures = {
            pool.submit(
                _load_device,
                checkpoint_dir,
                dev,
                max_workers,
                transfer_backend,
                sharded_ssd_roots,
                sharded_ssd_queues_per_root,
                transfer_gate,
            ): dev
            for dev in devices
        }
        for future in as_completed(futures):
            dev = futures[future]
            future.result()
            logger.info("Device %d load complete", dev)
    elapsed = time.monotonic() - t0
    logger.info("All %d devices loaded in %.2fs", len(devices), elapsed)

    while True:
        time.sleep(3600)


if __name__ == "__main__":
    main()
