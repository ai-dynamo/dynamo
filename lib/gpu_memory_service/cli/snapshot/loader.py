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
import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from gpu_memory_service.common import cuda_utils
from gpu_memory_service.common.snapshot_profile import SnapshotProfile
from gpu_memory_service.common.utils import get_socket_path
from gpu_memory_service.snapshot.backends.sharded_ssd import parse_sharded_ssd_roots
from gpu_memory_service.snapshot.storage_client import GMSStorageClient
from gpu_memory_service.snapshot.transfer import TransferBackendKind

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

SHARDED_SSD_CUDA_MODE_ENV = "DYN_GMS_SHARDED_SSD_CUDA_MODE"
MAPPING_FIRST_ENV = "DYN_GMS_MAPPING_FIRST"
PINNED_REGISTRATION_GROUPS_ENV = "DYN_GMS_PINNED_REGISTRATION_GROUPS"
CUDA_MODE_RUNTIME = "runtime"
CUDA_MODE_DRIVER = "driver"

_first_cuda_set_device_claim_lock = threading.Lock()
_first_cuda_set_device_claimed = False


class _MappingCoordinator:
    """Failure-aware all-device mapping completion coordinator."""

    def __init__(self, participants: int) -> None:
        self._remaining = int(participants)
        self._condition = threading.Condition()
        self._error: BaseException | None = None
        self._wall_start_ns: int | None = None
        self._wall_end_ns: int | None = None
        self._monotonic_start_ns: int | None = None
        self._monotonic_end_ns: int | None = None

    def start(self) -> None:
        wall_ns = time.time_ns()
        monotonic_ns = time.monotonic_ns()
        with self._condition:
            if self._wall_start_ns is None or wall_ns < self._wall_start_ns:
                self._wall_start_ns = wall_ns
                self._monotonic_start_ns = monotonic_ns

    def arrive(self, error: BaseException | None = None) -> None:
        wall_ns = time.time_ns()
        monotonic_ns = time.monotonic_ns()
        with self._condition:
            if self._remaining <= 0:
                raise RuntimeError("mapping coordinator received too many arrivals")
            if error is not None and self._error is None:
                self._error = error
            self._wall_end_ns = max(self._wall_end_ns or wall_ns, wall_ns)
            self._monotonic_end_ns = max(
                self._monotonic_end_ns or monotonic_ns,
                monotonic_ns,
            )
            self._remaining -= 1
            if self._remaining == 0 or self._error is not None:
                self._condition.notify_all()

    def wait(self) -> None:
        with self._condition:
            self._condition.wait_for(
                lambda: self._remaining == 0 or self._error is not None
            )
            if self._error is not None:
                raise RuntimeError(
                    "another device failed before mapping completed"
                ) from (self._error)

    def envelope(self) -> tuple[int, int, int]:
        with self._condition:
            if (
                self._remaining != 0
                or self._wall_start_ns is None
                or self._wall_end_ns is None
                or self._monotonic_start_ns is None
                or self._monotonic_end_ns is None
            ):
                raise RuntimeError("mapping envelope requested before completion")
            return (
                self._wall_start_ns,
                self._wall_end_ns,
                self._monotonic_end_ns - self._monotonic_start_ns,
            )


class _MappingParticipant:
    """Idempotently contributes one device result to a coordinator."""

    def __init__(self, coordinator: _MappingCoordinator) -> None:
        self._coordinator = coordinator
        self._lock = threading.Lock()
        self._arrived = False

    def complete(self) -> None:
        self._arrive()

    def start(self) -> None:
        self._coordinator.start()

    def fail(self, error: BaseException) -> None:
        self._arrive(error)

    def _arrive(self, error: BaseException | None = None) -> None:
        with self._lock:
            if self._arrived:
                return
            self._arrived = True
        self._coordinator.arrive(error)


def _reset_cuda_set_device_profile_state() -> None:
    global _first_cuda_set_device_claimed
    with _first_cuda_set_device_claim_lock:
        _first_cuda_set_device_claimed = False


def _claim_first_cuda_set_device_profile() -> bool:
    global _first_cuda_set_device_claimed
    with _first_cuda_set_device_claim_lock:
        first = not _first_cuda_set_device_claimed
        _first_cuda_set_device_claimed = True
    return first


def _load_device_impl(
    checkpoint_dir: str,
    device: int,
    max_workers: int,
    transfer_backend: str,
    sharded_ssd_roots: list[str],
    sharded_ssd_queues_per_root: int,
    cuda_initialization_complete: Callable[[], None] | None = None,
    *,
    sharded_ssd_cuda_mode: str = CUDA_MODE_RUNTIME,
    primary_context_retain_complete: Callable[[], None] | None = None,
    driver_process: cuda_utils.DriverCudaProcess | None = None,
    mapping_participant: _MappingParticipant | None = None,
    mapping_gate: _MappingCoordinator | None = None,
    pinned_registration_groups: int = 0,
) -> None:
    profile = SnapshotProfile(
        "loader",
        logger=logger,
        device=device,
        service="weights",
    )
    profile.ensure_profile_session_id()
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
    with profile.phase("per_device_load_total"):
        cuda_operations = None
        try:
            if sharded_ssd_cuda_mode == CUDA_MODE_RUNTIME:
                first_claimed = (
                    profile.enabled and _claim_first_cuda_set_device_profile()
                )
                with profile.phase(
                    "cuda_set_device",
                    api="cudaSetDevice",
                    first_claimed_for_process_profile=first_claimed,
                ):
                    if first_claimed:
                        with profile.phase(
                            "first_claimed_cuda_set_device",
                            api="cudaSetDevice",
                            semantics="bookkeeping_claim_before_unsynchronized_call",
                        ):
                            cuda_utils.cuda_runtime_set_device(device)
                    else:
                        cuda_utils.cuda_runtime_set_device(device)
                if profile.enabled:
                    with profile.phase("current_context_query", api="cuCtxGetCurrent"):
                        current_context = cuda_utils.cuda_current_context()
                    if current_context == 0:
                        raise RuntimeError(
                            f"cudaSetDevice({device}) did not establish a current "
                            "context"
                        )
                    with profile.phase(
                        "current_context_established",
                        current_context=current_context,
                    ):
                        pass
            else:
                if driver_process is None:
                    raise RuntimeError("Driver CUDA mode requires a DriverCudaProcess")
                try:
                    with profile.phase("cu_device_get", api="cuDeviceGet"):
                        cuda_device = driver_process.device_get(device)
                    with profile.phase(
                        "primary_context_retain",
                        api="cuDevicePrimaryCtxRetain",
                    ):
                        driver_process.primary_context_retain(
                            device,
                            cuda_device,
                        )
                finally:
                    if primary_context_retain_complete is not None:
                        primary_context_retain_complete()
                cuda_operations = driver_process.operations(device, profile)
                with profile.phase(
                    "loader_cu_ctx_set_current",
                    api="cuCtxSetCurrent",
                ):
                    current_context = cuda_operations.set_current_device(device)
                with profile.phase(
                    "current_context_established",
                    current_context=current_context,
                    api="cuCtxSetCurrent",
                ):
                    pass
        finally:
            if cuda_initialization_complete is not None:
                cuda_initialization_complete()
        with profile.phase("storage_client_construction"):
            client = GMSStorageClient(
                socket_path=get_socket_path(device),
                device=device,
                transfer_backend=transfer_backend,
                sharded_ssd_roots=sharded_ssd_roots,
                sharded_ssd_queues_per_root=sharded_ssd_queues_per_root,
                profile=profile,
                cuda_operations=cuda_operations,
                mapping_completion=(
                    mapping_participant.complete
                    if mapping_participant is not None
                    else None
                ),
                mapping_starting=(
                    mapping_participant.start
                    if mapping_participant is not None
                    else None
                ),
                mapping_gate=mapping_gate,
                pinned_registration_groups=pinned_registration_groups,
            )
        client.load_to_gms(
            input_dir,
            max_workers=max_workers,
            clear_existing=True,
        )
    elapsed = time.monotonic() - t0
    logger.info("GMS checkpoint loaded: device=%d elapsed=%.2fs", device, elapsed)


def _load_device(
    checkpoint_dir: str,
    device: int,
    max_workers: int,
    transfer_backend: str,
    sharded_ssd_roots: list[str],
    sharded_ssd_queues_per_root: int,
    cuda_initialization_complete: Callable[[], None] | None = None,
    *,
    sharded_ssd_cuda_mode: str = CUDA_MODE_RUNTIME,
    primary_context_retain_complete: Callable[[], None] | None = None,
    driver_process: cuda_utils.DriverCudaProcess | None = None,
    mapping_participant: _MappingParticipant | None = None,
    mapping_gate: _MappingCoordinator | None = None,
    pinned_registration_groups: int = 0,
) -> None:
    try:
        _load_device_impl(
            checkpoint_dir,
            device,
            max_workers,
            transfer_backend,
            sharded_ssd_roots,
            sharded_ssd_queues_per_root,
            cuda_initialization_complete,
            sharded_ssd_cuda_mode=sharded_ssd_cuda_mode,
            primary_context_retain_complete=primary_context_retain_complete,
            driver_process=driver_process,
            mapping_participant=mapping_participant,
            mapping_gate=mapping_gate,
            pinned_registration_groups=pinned_registration_groups,
        )
    except BaseException as error:
        if mapping_participant is not None:
            mapping_participant.fail(error)
        raise


def _environment_flag(name: str) -> bool:
    value = os.environ.get(name, "0")
    if value not in {"0", "1"}:
        raise ValueError(f"{name} must be 0 or 1, got {value!r}")
    return value == "1"


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
        "--sharded-ssd-cuda-mode",
        choices=[CUDA_MODE_RUNTIME, CUDA_MODE_DRIVER],
        default=os.environ.get(SHARDED_SSD_CUDA_MODE_ENV, CUDA_MODE_RUNTIME),
        help=(
            "CUDA API used only by the sharded-SSD snapshot loader staging path. "
            f"May also be set with {SHARDED_SSD_CUDA_MODE_ENV}."
        ),
    )
    parser.add_argument(
        "--mapping-first",
        action=argparse.BooleanOptionalAction,
        default=_environment_flag(MAPPING_FIRST_ENV),
        help=(
            "Wait for every device's VMM mapping to complete before any "
            "sharded-SSD staging host registration. May also be set with "
            f"{MAPPING_FIRST_ENV}=1."
        ),
    )
    parser.add_argument(
        "--pinned-registration-groups",
        type=int,
        default=int(os.environ.get(PINNED_REGISTRATION_GROUPS_ENV, "0")),
        help=(
            "Contiguous pinned-host registration groups per GPU; 0 preserves "
            "independent slot allocations. May also be set with "
            f"{PINNED_REGISTRATION_GROUPS_ENV}."
        ),
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
    profile = SnapshotProfile(
        "loader",
        logger=logger,
        scope="cli",
        service="weights",
    )
    with profile.phase("cli_entry_and_argument_parse"):
        parser = _build_parser()
        args = parser.parse_args(argv)
    if not args.checkpoint_dir:
        parser.error(
            f"--checkpoint-dir is required for --transfer-backend={args.transfer_backend}"
        )
    if args.sharded_ssd_queues_per_root <= 0:
        parser.error("--sharded-ssd-queues-per-root must be a positive integer")
    if args.pinned_registration_groups < 0:
        parser.error("--pinned-registration-groups must not be negative")
    if args.sharded_ssd_cuda_mode not in (CUDA_MODE_RUNTIME, CUDA_MODE_DRIVER):
        parser.error(
            "--sharded-ssd-cuda-mode must be one of: "
            f"{CUDA_MODE_RUNTIME}, {CUDA_MODE_DRIVER}"
        )
    if (
        args.sharded_ssd_cuda_mode != CUDA_MODE_RUNTIME
        and args.transfer_backend != TransferBackendKind.SHARDED_SSD.value
    ):
        parser.error(
            "--sharded-ssd-cuda-mode=driver requires "
            f"--transfer-backend={TransferBackendKind.SHARDED_SSD.value}"
        )
    if (
        args.mapping_first or args.pinned_registration_groups
    ) and args.transfer_backend != TransferBackendKind.SHARDED_SSD.value:
        parser.error(
            "--mapping-first and --pinned-registration-groups require "
            f"--transfer-backend={TransferBackendKind.SHARDED_SSD.value}"
        )
    checkpoint_dir = args.checkpoint_dir
    max_workers = args.max_workers
    transfer_backend = args.transfer_backend
    sharded_ssd_roots = parse_sharded_ssd_roots(args.sharded_ssd_roots)
    sharded_ssd_queues_per_root = args.sharded_ssd_queues_per_root
    sharded_ssd_cuda_mode = args.sharded_ssd_cuda_mode
    mapping_first = args.mapping_first
    pinned_registration_groups = args.pinned_registration_groups
    if sharded_ssd_cuda_mode == CUDA_MODE_DRIVER:
        cuda_utils.forbid_cuda_runtime_calls()
    logger.info(
        "Starting GMS load: transfer_backend=%s max_workers=%d "
        "sharded_ssd_roots=%s sharded_ssd_queues_per_root=%d "
        "sharded_ssd_cuda_mode=%s mapping_first=%s "
        "pinned_registration_groups=%d",
        transfer_backend,
        max_workers,
        ",".join(sharded_ssd_roots) or "-",
        sharded_ssd_queues_per_root,
        sharded_ssd_cuda_mode,
        mapping_first,
        pinned_registration_groups,
    )
    with profile.phase("device_and_checkpoint_discovery"):
        devices = _list_checkpoint_devices(checkpoint_dir)

    _reset_cuda_set_device_profile_state()
    initialization_lock = threading.Lock()
    initialization_remaining = len(devices)
    initialization_complete = threading.Event()
    primary_context_remaining = len(devices)
    primary_context_complete = threading.Event()
    mapping_coordinator = _MappingCoordinator(len(devices))
    mapping_participants = {
        device: _MappingParticipant(mapping_coordinator) for device in devices
    }
    mapping_gate = mapping_coordinator if mapping_first else None

    def mark_cuda_initialization_complete() -> None:
        nonlocal initialization_remaining
        with initialization_lock:
            initialization_remaining -= 1
            if initialization_remaining == 0:
                initialization_complete.set()

    def mark_primary_context_retain_complete() -> None:
        nonlocal primary_context_remaining
        with initialization_lock:
            primary_context_remaining -= 1
            if primary_context_remaining == 0:
                primary_context_complete.set()

    initialization_callback = (
        mark_cuda_initialization_complete if profile.enabled else None
    )
    primary_context_callback = (
        mark_primary_context_retain_complete
        if profile.enabled and sharded_ssd_cuda_mode == CUDA_MODE_DRIVER
        else None
    )
    driver_process = (
        cuda_utils.DriverCudaProcess()
        if sharded_ssd_cuda_mode == CUDA_MODE_DRIVER
        else None
    )
    t0 = time.monotonic()
    try:
        with profile.phase("all_device_barrier", count=len(devices)):
            with ThreadPoolExecutor(max_workers=len(devices)) as pool:
                if driver_process is None:
                    with profile.phase(
                        "all_device_cuda_initialization",
                        count=len(devices),
                        semantics="concurrent_envelope",
                        includes=(
                            "task_scheduling,cuda_set_device,current_context_query"
                        ),
                    ):
                        with profile.phase(
                            "per_device_thread_scheduling",
                            count=len(devices),
                        ):
                            futures = {
                                pool.submit(
                                    _load_device,
                                    checkpoint_dir,
                                    dev,
                                    max_workers,
                                    transfer_backend,
                                    sharded_ssd_roots,
                                    sharded_ssd_queues_per_root,
                                    initialization_callback,
                                    mapping_participant=mapping_participants[dev],
                                    mapping_gate=mapping_gate,
                                    pinned_registration_groups=(
                                        pinned_registration_groups
                                    ),
                                ): dev
                                for dev in devices
                            }
                        if profile.enabled:
                            initialization_complete.wait()
                else:
                    with profile.phase(
                        "loader_cuda_initialization_total",
                        count=len(devices),
                        cuda_api=CUDA_MODE_DRIVER,
                        includes=(
                            "loader_cu_init,task_scheduling,cu_device_get,"
                            "primary_context_retain,cu_ctx_set_current"
                        ),
                    ):
                        with profile.phase(
                            "loader_cu_init",
                            api="cuInit",
                            explicit=True,
                            once_per_process=True,
                        ):
                            driver_process.initialize()
                        with profile.phase(
                            "all_device_driver_initialization",
                            count=len(devices),
                            semantics="concurrent_envelope",
                        ):
                            with profile.phase(
                                "all_device_cu_ctx_set_current_envelope",
                                count=len(devices),
                                semantics="concurrent_envelope",
                                includes=(
                                    "task_scheduling,cu_device_get,"
                                    "primary_context_retain,cu_ctx_set_current"
                                ),
                            ):
                                with profile.phase(
                                    "all_device_primary_context_retain_envelope",
                                    count=len(devices),
                                    semantics="concurrent_envelope",
                                    includes=(
                                        "task_scheduling,cu_device_get,"
                                        "primary_context_retain"
                                    ),
                                ):
                                    with profile.phase(
                                        "per_device_thread_scheduling",
                                        count=len(devices),
                                    ):
                                        futures = {
                                            pool.submit(
                                                _load_device,
                                                checkpoint_dir,
                                                dev,
                                                max_workers,
                                                transfer_backend,
                                                sharded_ssd_roots,
                                                sharded_ssd_queues_per_root,
                                                initialization_callback,
                                                sharded_ssd_cuda_mode=(
                                                    sharded_ssd_cuda_mode
                                                ),
                                                primary_context_retain_complete=(
                                                    primary_context_callback
                                                ),
                                                driver_process=driver_process,
                                                mapping_participant=(
                                                    mapping_participants[dev]
                                                ),
                                                mapping_gate=mapping_gate,
                                                pinned_registration_groups=(
                                                    pinned_registration_groups
                                                ),
                                            ): dev
                                            for dev in devices
                                        }
                                    if profile.enabled:
                                        primary_context_complete.wait()
                                if profile.enabled:
                                    initialization_complete.wait()
                with profile.phase(
                    "all_device_mapping_barrier_wait",
                    count=len(devices),
                    mapping_first=mapping_first,
                ):
                    mapping_coordinator.wait()
                if profile.enabled:
                    wall_start_ns, wall_end_ns, duration_ns = (
                        mapping_coordinator.envelope()
                    )
                    profile.emit(
                        "all_device_mapping_envelope",
                        wall_start_ns=wall_start_ns,
                        wall_end_ns=wall_end_ns,
                        duration_ns=duration_ns,
                        count=len(devices),
                        semantics="concurrent_envelope",
                        mapping_first=mapping_first,
                    )
                for future in as_completed(futures):
                    dev = futures[future]
                    future.result()
                    logger.info("Device %d load complete", dev)
    finally:
        if driver_process is not None:
            with profile.phase(
                "primary_context_release",
                api="cuDevicePrimaryCtxRelease",
                count=len(devices),
                after_all_dependent_work=True,
            ):
                driver_process.close()
    elapsed = time.monotonic() - t0
    logger.info("All %d devices loaded in %.2fs", len(devices), elapsed)

    while True:
        time.sleep(3600)


if __name__ == "__main__":
    main()
