# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared NIXL helpers for GMS snapshot transfer backends."""

from __future__ import annotations

import errno
import logging
import os
import time
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Mapping, Optional, Sequence, TypeVar

from gpu_memory_service.common.snapshot_profile import SnapshotProfile
from gpu_memory_service.snapshot.transfer import FileTransferSource

NIXL_POSIX_BACKEND = "POSIX"
NIXL_GDS_BACKEND = "GDS_MT"

FILE_MEM_TYPE = "FILE"
DRAM_MEM_TYPE = "DRAM"
VRAM_MEM_TYPE = "VRAM"


@dataclass(frozen=True)
class NixlApi:
    agent_type: Any
    agent_config_type: Any


@dataclass(frozen=True)
class NixlTransferResources:
    """Resources owned by one prepared NIXL transfer handle."""

    handle: Any
    label: str
    registrations: Sequence[Any] = ()
    fds: Sequence[int] = ()


_TransferItemT = TypeVar("_TransferItemT")

NixlFileGroup = tuple[str, Sequence[FileTransferSource]]
NixlWorkGroup = tuple[str, Sequence[NixlFileGroup]]


def load_nixl_api() -> NixlApi:
    try:
        from nixl._api import nixl_agent, nixl_agent_config
    except ImportError as exc:
        raise RuntimeError(
            "NIXL Python bindings are required for the nixl GMS transfer backends"
        ) from exc
    return NixlApi(agent_type=nixl_agent, agent_config_type=nixl_agent_config)


def create_nixl_agent(
    api: NixlApi,
    *,
    agent_name: str,
    backend_name: str,
    backend_params: Optional[Mapping[str, str]] = None,
) -> Any:
    agent = api.agent_type(
        agent_name,
        api.agent_config_type(backends=[]),
    )
    if backend_params:
        agent.create_backend(backend_name, dict(backend_params))
    else:
        agent.create_backend(backend_name)
    return agent


def split_work_groups(
    work_groups: Sequence[NixlWorkGroup],
    worker_count: int,
) -> list[NixlWorkGroup]:
    """Split logical work groups into at most worker_count balanced buckets."""
    if not work_groups:
        return []

    worker_count = max(1, min(int(worker_count), len(work_groups)))
    if len(work_groups) <= worker_count:
        return list(work_groups)

    bucket_file_groups: list[list[NixlFileGroup]] = [[] for _ in range(worker_count)]
    bucket_names: list[list[str]] = [[] for _ in range(worker_count)]
    bucket_bytes = [0] * worker_count

    # Largest-first greedy bin packing keeps workers reasonably balanced while
    # staying deterministic for equal-sized groups.
    sized_groups = [
        (
            sum(
                source.byte_count
                for _path, sources in file_groups
                for source in sources
            ),
            index,
            group_name,
            file_groups,
        )
        for index, (group_name, file_groups) in enumerate(work_groups)
    ]
    sized_groups.sort(key=lambda item: (-item[0], item[1]))

    for size_bytes, _index, group_name, file_groups in sized_groups:
        bucket_index = min(range(worker_count), key=lambda idx: bucket_bytes[idx])
        bucket_file_groups[bucket_index].extend(file_groups)
        bucket_names[bucket_index].append(group_name)
        bucket_bytes[bucket_index] += size_bytes

    buckets: list[NixlWorkGroup] = []
    for index, file_groups in enumerate(bucket_file_groups):
        if not file_groups:
            continue
        buckets.append((",".join(bucket_names[index]), file_groups))
    return buckets


def open_direct_read_fd(
    path: str,
    *,
    logger: logging.Logger,
    require_direct: bool = True,
) -> int:
    odirect = getattr(os, "O_DIRECT", 0)
    if require_direct and not odirect:
        raise RuntimeError("O_DIRECT is not available on this platform")

    try:
        return os.open(path, os.O_RDONLY | odirect)
    except OSError as exc:
        if odirect and exc.errno in {errno.EINVAL, errno.EOPNOTSUPP}:
            if require_direct:
                raise RuntimeError(
                    f"O_DIRECT unavailable for {path}; refusing buffered read"
                ) from exc
            logger.warning(
                "O_DIRECT unavailable for %s; falling back to buffered reads",
                path,
            )
            return os.open(path, os.O_RDONLY)
        raise


def start_transfer(
    agent: Any,
    handle: Any,
    label: str,
    backend_name: str,
    *,
    profile: SnapshotProfile | None = None,
    profile_fields: Optional[Mapping[str, Any]] = None,
) -> None:
    if profile is None or not profile.enabled:
        state = agent.transfer(handle)
        if state == "ERR":
            raise RuntimeError(f"{backend_name} transfer failed to start: {label}")
        if state not in {"PROC", "DONE"}:
            raise RuntimeError(
                f"{backend_name} transfer returned unexpected state {state!r}"
            )
        return
    with profile.aggregate(
        "nixl_transfer_post",
        **dict(profile_fields or {}),
    ):
        state = agent.transfer(handle)
    if state == "ERR":
        raise RuntimeError(f"{backend_name} transfer failed to start: {label}")
    if state not in {"PROC", "DONE"}:
        raise RuntimeError(
            f"{backend_name} transfer returned unexpected state {state!r}"
        )


def wait_for_transfer(
    agent: Any,
    handle: Any,
    label: str,
    backend_name: str,
    *,
    profile: SnapshotProfile | None = None,
    profile_fields: Optional[Mapping[str, Any]] = None,
) -> None:
    start_transfer(
        agent,
        handle,
        label,
        backend_name,
        profile=profile,
        profile_fields=profile_fields,
    )
    wait_for_transfer_done(
        agent,
        handle,
        label,
        backend_name,
        profile=profile,
        profile_fields=profile_fields,
    )


def wait_for_transfer_done(
    agent: Any,
    handle: Any,
    label: str,
    backend_name: str,
    *,
    profile: SnapshotProfile | None = None,
    profile_fields: Optional[Mapping[str, Any]] = None,
) -> None:
    if profile is None or not profile.enabled:
        state = agent.check_xfer_state(handle)
        while state == "PROC":
            time.sleep(0.001)
            state = agent.check_xfer_state(handle)
        if state == "ERR":
            raise RuntimeError(f"{backend_name} transfer failed: {label}")
        if state != "DONE":
            raise RuntimeError(
                f"{backend_name} transfer ended in unexpected state {state!r}: {label}"
            )
        return
    with profile.aggregate(
        "nixl_transfer_poll_wait",
        **dict(profile_fields or {}),
    ):
        state = agent.check_xfer_state(handle)
        while state == "PROC":
            time.sleep(0.001)
            state = agent.check_xfer_state(handle)
    if state == "ERR":
        raise RuntimeError(f"{backend_name} transfer failed: {label}")
    if state != "DONE":
        raise RuntimeError(
            f"{backend_name} transfer ended in unexpected state {state!r}: {label}"
        )


def run_bounded_nixl_transfers(
    *,
    agent: Any,
    backend_name: str,
    items: Iterable[_TransferItemT],
    max_inflight: int,
    prepare_transfer: Callable[[_TransferItemT], NixlTransferResources],
    logger: logging.Logger,
) -> None:
    """Prepare, start, wait for, and release NIXL transfers with a bounded window."""
    pending: list[NixlTransferResources] = []
    max_inflight = max(1, int(max_inflight))
    try:
        for item in items:
            transfer = prepare_transfer(item)
            try:
                start_transfer(agent, transfer.handle, transfer.label, backend_name)
            except Exception:
                release_nixl_transfer_resources(agent, transfer)
                raise
            pending.append(transfer)

            if len(pending) >= max_inflight:
                _wait_and_release_nixl_transfer(
                    agent,
                    backend_name,
                    pending.pop(0),
                )

        while pending:
            _wait_and_release_nixl_transfer(
                agent,
                backend_name,
                pending.pop(0),
            )
    except Exception:
        _drain_pending_nixl_transfers(
            agent,
            backend_name,
            pending,
            logger,
        )
        raise


def release_nixl_transfer_resources(
    agent: Any,
    transfer: NixlTransferResources,
) -> None:
    if transfer.handle is not None:
        try:
            agent.release_xfer_handle(transfer.handle)
        except Exception:
            pass
    for registration in transfer.registrations:
        if registration is None:
            continue
        try:
            agent.deregister_memory(registration)
        except Exception:
            pass
    for fd in transfer.fds:
        try:
            os.close(fd)
        except OSError:
            pass


def _wait_and_release_nixl_transfer(
    agent: Any,
    backend_name: str,
    transfer: NixlTransferResources,
) -> None:
    try:
        wait_for_transfer_done(
            agent,
            transfer.handle,
            transfer.label,
            backend_name,
        )
    finally:
        release_nixl_transfer_resources(agent, transfer)


def _drain_pending_nixl_transfers(
    agent: Any,
    backend_name: str,
    pending: Sequence[NixlTransferResources],
    logger: logging.Logger,
) -> None:
    for transfer in pending:
        try:
            _wait_and_release_nixl_transfer(agent, backend_name, transfer)
        except Exception:
            logger.warning(
                "%s failed while draining in-flight transfer %s",
                backend_name,
                transfer.label,
                exc_info=True,
            )


def release_transfer_resources(
    agent: Any,
    handle: Any,
    first_reg: Any,
    second_reg: Any,
    fd: Optional[int] = None,
    *,
    profile: SnapshotProfile | None = None,
    profile_fields: Optional[Mapping[str, Any]] = None,
) -> None:
    if profile is None or not profile.enabled:
        if handle is not None:
            try:
                agent.release_xfer_handle(handle)
            except Exception:
                pass
        if first_reg is not None:
            try:
                agent.deregister_memory(first_reg)
            except Exception:
                pass
        if second_reg is not None:
            try:
                agent.deregister_memory(second_reg)
            except Exception:
                pass
        if fd is not None:
            try:
                os.close(fd)
            except OSError:
                pass
        return
    fields = dict(profile_fields or {})
    if handle is not None:
        try:
            with profile.aggregate("nixl_transfer_release", **fields):
                agent.release_xfer_handle(handle)
        except Exception:
            pass
    if first_reg is not None:
        try:
            with profile.aggregate("nixl_deregister_dram", **fields):
                agent.deregister_memory(first_reg)
        except Exception:
            pass
    if second_reg is not None:
        try:
            with profile.aggregate("nixl_deregister_file", **fields):
                agent.deregister_memory(second_reg)
        except Exception:
            pass
    if fd is not None:
        try:
            os.close(fd)
        except OSError:
            pass
