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
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from gpu_memory_service.common import cuda_utils
from gpu_memory_service.common.profiling import profile_log
from gpu_memory_service.common.utils import get_socket_path
from gpu_memory_service.snapshot.backends.sharded_ssd import parse_sharded_ssd_roots
from gpu_memory_service.snapshot.storage_client import GMSStorageClient
from gpu_memory_service.snapshot.transfer import TransferBackendKind

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def _load_device(
    checkpoint_dir: str,
    device: int,
    max_workers: int,
    transfer_backend: str,
    sharded_ssd_roots: list[str],
    sharded_ssd_queues_per_root: int,
) -> None:
    device_t0 = time.monotonic()
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
    phase_t0 = time.monotonic()
    cuda_utils.cuda_runtime_set_device(device)
    profile_log(
        logger,
        "loader device=%d phase cuda_set_device elapsed=%.6fs",
        device,
        time.monotonic() - phase_t0,
    )
    phase_t0 = time.monotonic()
    socket_path = get_socket_path(device)
    profile_log(
        logger,
        "loader device=%d phase resolve_socket_path path=%s elapsed=%.6fs",
        device,
        socket_path,
        time.monotonic() - phase_t0,
    )
    phase_t0 = time.monotonic()
    client = GMSStorageClient(
        socket_path=socket_path,
        device=device,
        transfer_backend=transfer_backend,
        sharded_ssd_roots=sharded_ssd_roots,
        sharded_ssd_queues_per_root=sharded_ssd_queues_per_root,
    )
    profile_log(
        logger,
        "loader device=%d phase storage_client_created elapsed=%.6fs",
        device,
        time.monotonic() - phase_t0,
    )
    phase_t0 = time.monotonic()
    client.load_to_gms(
        input_dir,
        max_workers=max_workers,
        clear_existing=True,
    )
    profile_log(
        logger,
        "loader device=%d phase load_to_gms elapsed=%.6fs",
        device,
        time.monotonic() - phase_t0,
    )
    elapsed = time.monotonic() - t0
    logger.info("GMS checkpoint loaded: device=%d elapsed=%.2fs", device, elapsed)
    profile_log(
        logger,
        "loader device=%d total since_device_thread_start=%.6fs since_logical_load_start=%.6fs",
        device,
        time.monotonic() - device_t0,
        elapsed,
    )


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
    return parser


def _list_checkpoint_devices(checkpoint_dir: str | None) -> list[int]:
    t0 = time.monotonic()
    devices = cuda_utils.list_devices()
    profile_log(
        logger,
        "loader phase list_cuda_devices devices=%s elapsed=%.6fs",
        devices,
        time.monotonic() - t0,
    )
    if not checkpoint_dir:
        return devices

    phase_t0 = time.monotonic()
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
    profile_log(
        logger,
        "loader phase scan_checkpoint_devices checkpoint_dir=%s checkpoint_devices=%s "
        "elapsed=%.6fs",
        checkpoint_path,
        sorted(checkpoint_devices),
        time.monotonic() - phase_t0,
    )
    return devices


def main(argv: list[str] | None = None) -> None:
    process_t0 = time.monotonic()
    parse_t0 = time.monotonic()
    parser = _build_parser()
    args = parser.parse_args(argv)
    profile_log(
        logger,
        "loader phase parse_args transfer_backend=%s max_workers=%d elapsed=%.6fs",
        args.transfer_backend,
        args.max_workers,
        time.monotonic() - parse_t0,
    )
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
    devices_t0 = time.monotonic()
    devices = _list_checkpoint_devices(checkpoint_dir)
    profile_log(
        logger,
        "loader phase devices_ready device_count=%d elapsed=%.6fs "
        "since_process_main=%.6fs",
        len(devices),
        time.monotonic() - devices_t0,
        time.monotonic() - process_t0,
    )

    t0 = time.monotonic()
    pool_t0 = time.monotonic()
    with ThreadPoolExecutor(max_workers=len(devices)) as pool:
        profile_log(
            logger,
            "loader phase device_pool_created workers=%d elapsed=%.6fs",
            len(devices),
            time.monotonic() - pool_t0,
        )
        submit_t0 = time.monotonic()
        futures = {
            pool.submit(
                _load_device,
                checkpoint_dir,
                dev,
                max_workers,
                transfer_backend,
                sharded_ssd_roots,
                sharded_ssd_queues_per_root,
            ): dev
            for dev in devices
        }
        profile_log(
            logger,
            "loader phase device_futures_submitted count=%d elapsed=%.6fs",
            len(futures),
            time.monotonic() - submit_t0,
        )
        for future in as_completed(futures):
            dev = futures[future]
            future.result()
            logger.info("Device %d load complete", dev)
    elapsed = time.monotonic() - t0
    logger.info("All %d devices loaded in %.2fs", len(devices), elapsed)
    profile_log(
        logger,
        "loader total devices=%d load_elapsed=%.6fs process_main_elapsed=%.6fs",
        len(devices),
        elapsed,
        time.monotonic() - process_t0,
    )

    while True:
        time.sleep(3600)


if __name__ == "__main__":
    main()
