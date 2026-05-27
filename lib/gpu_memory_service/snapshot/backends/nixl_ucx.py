# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""NIXL UCX restore backend for peer VRAM -> local GMS VRAM transfers."""

from __future__ import annotations

import base64
import json
import logging
import os
import threading
import time
from typing import Any, Mapping, Optional, Sequence

from gpu_memory_service.snapshot.backends.nixl_common import (
    NIXL_UCX_BACKEND,
    VRAM_MEM_TYPE,
    NixlTransferResources,
    create_nixl_agent,
    load_nixl_api,
    release_nixl_transfer_resources,
    start_transfer,
    wait_for_transfer_done,
)
from gpu_memory_service.snapshot.transfer import (
    NIXL_UCX_TRANSFER_BACKEND,
    GMSSnapshotConfig,
    GMSTransferTarget,
    RemoteTransferSource,
    TransferSession,
    validate_transfer_targets,
)

logger = logging.getLogger(__name__)

GMS_NIXL_UCX_CONFIG_PATH_ENV = "GMS_NIXL_UCX_CONFIG_PATH"
NIXL_UCX_CONFIG_PATH_CONFIG_KEY = "nixl_ucx_config_path"
NIXL_UCX_REMOTE_AGENT_CONFIG_KEY = "nixl_ucx_remote_agent"
NIXL_UCX_REMOTE_METADATA_CONFIG_KEY = "nixl_ucx_remote_metadata"
NIXL_UCX_SOURCES_CONFIG_KEY = "nixl_ucx_sources"


class NixlUCXTransferBackend:
    """NIXL UCX backend for restoring GMS allocations from peer GPU memory."""

    name = NIXL_UCX_TRANSFER_BACKEND

    def __init__(self, *, config: GMSSnapshotConfig) -> None:
        api = load_nixl_api()
        self._device = config.device
        self._max_workers = config.max_workers
        self._agent_name = f"gms_ucx_loader_{self._device}_{os.getpid()}"
        self._agent = create_nixl_agent(
            api,
            agent_name=self._agent_name,
            backend_name=NIXL_UCX_BACKEND,
        )

        remote_metadata = load_remote_peer_metadata(config.backend_config)["metadata"]
        if not remote_metadata:
            raise RuntimeError(
                f"{NIXL_UCX_TRANSFER_BACKEND} requires peer NIXL metadata in "
                f"{NIXL_UCX_REMOTE_METADATA_CONFIG_KEY}"
            )
        self._remote_agent = _load_remote_metadata(self._agent, remote_metadata)
        logger.info(
            "NIXL UCX backend initialized for device %d with peer %s and %d max "
            "in-flight transfers",
            self._device,
            self._remote_agent,
            self._max_workers,
        )

    def start_restore(self, sources: Sequence[RemoteTransferSource]) -> TransferSession:
        materialized_sources = [
            (
                RemoteTransferSource(
                    allocation_id=source.allocation_id,
                    remote_agent=self._remote_agent,
                    va=source.va,
                    device=source.device,
                    byte_count=source.byte_count,
                )
                if not source.remote_agent
                else source
            )
            for source in sources
        ]
        return _NixlUCXTransferSession(
            agent=self._agent,
            remote_agent=self._remote_agent,
            device=self._device,
            max_workers=self._max_workers,
            sources=materialized_sources,
        )

    def close(self) -> None:
        if self._agent is not None:
            try:
                self._agent.remove_remote_agent(self._remote_agent)
            except Exception:
                logger.debug(
                    "failed to remove UCX remote agent %s",
                    self._remote_agent,
                    exc_info=True,
                )
        self._agent = None


class _NixlUCXTransferSession:
    def __init__(
        self,
        *,
        agent: Any,
        remote_agent: str,
        device: int,
        max_workers: int,
        sources: Sequence[RemoteTransferSource],
    ) -> None:
        self._agent = agent
        self._remote_agent = remote_agent
        self._device = device
        self._max_workers = max(1, int(max_workers))
        self._sources = list(sources)
        self._sources_by_id = {source.allocation_id: source for source in self._sources}
        self._total_bytes = sum(source.byte_count for source in self._sources)
        self._active = True
        self._condition = threading.Condition()
        self._targets: dict[str, GMSTransferTarget] = {}
        self._pending_allocation_ids = {
            source.allocation_id for source in self._sources
        }
        self._submitted_done = False
        self._scheduler_thread: Optional[threading.Thread] = None
        self._cancel_event = threading.Event()
        self._error: Optional[BaseException] = None
        self._stream_started_at: Optional[float] = None
        self._first_transfer_at: Optional[float] = None

    def submit_targets(self, targets: Mapping[str, GMSTransferTarget]) -> None:
        if not self._active:
            raise RuntimeError("NIXL UCX restore session is closed")
        if not targets:
            return
        self._validate_submitted_targets(targets)
        self._ensure_streaming_started()

        with self._condition:
            self._raise_error_locked()
            for allocation_id, target in targets.items():
                previous = self._targets.get(allocation_id)
                if previous is not None and previous != target:
                    raise RuntimeError(
                        f"NIXL UCX got duplicate target for allocation {allocation_id}"
                    )
                self._targets[allocation_id] = target
            self._condition.notify_all()

    def finish_restore(self) -> None:
        if not self._sources:
            self._active = False
            return
        self._ensure_streaming_started()
        with self._condition:
            self._submitted_done = True
            self._condition.notify_all()

        try:
            assert self._scheduler_thread is not None
            self._scheduler_thread.join()
            self._raise_error()
        finally:
            self._active = False

        now = time.monotonic()
        first_transfer_at = self._first_transfer_at or now
        transfer_elapsed = now - first_transfer_at
        total_elapsed = (
            now - self._stream_started_at
            if self._stream_started_at is not None
            else transfer_elapsed
        )
        throughput = (
            self._total_bytes / transfer_elapsed / (1024**3)
            if transfer_elapsed > 0
            else 0.0
        )
        logger.info(
            "NIXL UCX transfers complete: %.2f GiB in %.3fs "
            "(%.2f GiB/s, allocations=%d, max_inflight=%d, streaming=True, "
            "total_stream_elapsed=%.3fs)",
            self._total_bytes / (1024**3),
            transfer_elapsed,
            throughput,
            len(self._sources),
            self._max_workers,
            total_elapsed,
        )

    def restore(self, targets: Mapping[str, GMSTransferTarget]) -> None:
        validate_transfer_targets(self._sources, targets, device=self._device)
        if not self._sources:
            self._active = False
            return
        self.submit_targets(targets)
        self.finish_restore()

    def close(self) -> None:
        self._cancel_event.set()
        with self._condition:
            self._submitted_done = True
            self._condition.notify_all()
        if (
            self._scheduler_thread is not None
            and self._scheduler_thread.is_alive()
            and threading.current_thread() is not self._scheduler_thread
        ):
            self._scheduler_thread.join()
        self._active = False

    def _validate_submitted_targets(
        self,
        targets: Mapping[str, GMSTransferTarget],
    ) -> None:
        for allocation_id, target in targets.items():
            source = self._sources_by_id.get(allocation_id)
            if source is None:
                raise RuntimeError(
                    f"NIXL UCX got target for unknown allocation {allocation_id}"
                )
            if target.byte_count != source.byte_count:
                raise RuntimeError(
                    f"NIXL UCX target size mismatch for allocation {allocation_id}: "
                    f"source={source.byte_count} target={target.byte_count}"
                )
            if target.device != self._device:
                raise RuntimeError(
                    f"NIXL UCX target device mismatch for allocation {allocation_id}: "
                    f"backend={self._device} target={target.device}"
                )

    def _ensure_streaming_started(self) -> None:
        if not self._sources:
            self._active = False
            return
        if self._scheduler_thread is not None:
            return
        self._validate_remote_agents()
        self._stream_started_at = time.monotonic()
        logger.info(
            "NIXL UCX streaming restore targets started: allocations=%d "
            "bytes=%.2f GiB max_inflight=%d",
            len(self._sources),
            self._total_bytes / (1024**3),
            self._max_workers,
        )
        self._scheduler_thread = threading.Thread(
            target=self._run_streaming_scheduler,
            name="nixl-ucx-streaming-scheduler",
            daemon=True,
        )
        self._scheduler_thread.start()

    def _validate_remote_agents(self) -> None:
        for source in self._sources:
            if source.remote_agent != self._remote_agent:
                raise RuntimeError(
                    f"{NIXL_UCX_TRANSFER_BACKEND} loaded peer "
                    f"{self._remote_agent!r} but source {source.allocation_id} "
                    f"references {source.remote_agent!r}"
                )

    def _pop_ready_source_locked(
        self,
    ) -> Optional[tuple[RemoteTransferSource, GMSTransferTarget]]:
        for allocation_id in sorted(self._pending_allocation_ids):
            target = self._targets.get(allocation_id)
            if target is None:
                continue
            self._pending_allocation_ids.remove(allocation_id)
            return self._sources_by_id[allocation_id], target
        return None

    def _run_streaming_scheduler(self) -> None:
        inflight: list[NixlTransferResources] = []
        try:
            while True:
                self._raise_error()
                while len(inflight) < self._max_workers:
                    with self._condition:
                        ready = self._pop_ready_source_locked()
                    if ready is None:
                        break
                    source, target = ready
                    inflight.append(self._start_source_transfer(source, target))

                self._poll_completed_transfers(inflight)

                with self._condition:
                    if not self._pending_allocation_ids and not inflight:
                        return
                    ready_exists = any(
                        allocation_id in self._targets
                        for allocation_id in self._pending_allocation_ids
                    )
                    if ready_exists and len(inflight) < self._max_workers:
                        continue
                    if self._submitted_done and self._pending_allocation_ids:
                        if not ready_exists:
                            raise RuntimeError(
                                f"NIXL UCX missing {len(self._pending_allocation_ids)} "
                                "restore target(s) before finish_restore"
                            )
                    if self._cancel_event.is_set():
                        raise RuntimeError("NIXL UCX restore session was cancelled")
                    self._condition.wait(timeout=0.001 if inflight else None)
        except BaseException as exc:
            with self._condition:
                if self._error is None:
                    self._error = exc
                self._condition.notify_all()
        finally:
            self._drain_inflight_transfers(inflight)

    def _start_source_transfer(
        self,
        source: RemoteTransferSource,
        target: GMSTransferTarget,
    ) -> NixlTransferResources:
        transfer = self._prepare_source_transfer(source, target)
        try:
            start_transfer(
                self._agent,
                transfer.handle,
                transfer.label,
                NIXL_UCX_TRANSFER_BACKEND,
            )
            if self._first_transfer_at is None:
                self._first_transfer_at = time.monotonic()
                assert self._stream_started_at is not None
                logger.info(
                    "NIXL UCX first streaming transfer started after %.3fs "
                    "from first target submission",
                    self._first_transfer_at - self._stream_started_at,
                )
            return transfer
        except Exception:
            release_nixl_transfer_resources(self._agent, transfer)
            raise

    def _poll_completed_transfers(
        self,
        inflight: list[NixlTransferResources],
    ) -> None:
        still_running: list[NixlTransferResources] = []
        first_error: Optional[BaseException] = None
        for transfer in inflight:
            state = self._agent.check_xfer_state(transfer.handle)
            if state == "PROC":
                still_running.append(transfer)
                continue
            try:
                if state == "ERR":
                    raise RuntimeError(
                        f"NIXL UCX transfer failed: {transfer.label}"
                    )
                if state != "DONE":
                    raise RuntimeError(
                        f"NIXL UCX transfer ended in unexpected state {state!r}: "
                        f"{transfer.label}"
                    )
            except Exception as exc:
                if first_error is None:
                    first_error = exc
            finally:
                release_nixl_transfer_resources(self._agent, transfer)
        inflight[:] = still_running
        if first_error is not None:
            raise first_error

    def _drain_inflight_transfers(
        self,
        inflight: list[NixlTransferResources],
    ) -> None:
        while inflight:
            transfer = inflight.pop(0)
            try:
                wait_for_transfer_done(
                    self._agent,
                    transfer.handle,
                    transfer.label,
                    NIXL_UCX_TRANSFER_BACKEND,
                )
            except Exception:
                logger.warning(
                    "NIXL UCX failed while draining in-flight transfer %s",
                    transfer.label,
                    exc_info=True,
                )
            finally:
                release_nixl_transfer_resources(self._agent, transfer)

    def _prepare_source_transfer(
        self,
        source: RemoteTransferSource,
        target: GMSTransferTarget,
    ) -> NixlTransferResources:
        local_reg = None
        local_descs = self._agent.get_xfer_descs(
            [(target.va, target.byte_count, target.device)],
            mem_type=VRAM_MEM_TYPE,
        )
        remote_descs = self._agent.get_xfer_descs(
            [(source.va, source.byte_count, source.device)],
            mem_type=VRAM_MEM_TYPE,
        )
        handle = None
        try:
            local_reg = self._agent.register_memory(
                [(target.va, target.byte_count, target.device, "")],
                VRAM_MEM_TYPE,
                backends=[NIXL_UCX_BACKEND],
            )
            handle = self._agent.initialize_xfer(
                "READ",
                local_descs,
                remote_descs,
                source.remote_agent,
                backends=[NIXL_UCX_BACKEND],
            )
            return NixlTransferResources(
                handle=handle,
                label=source.allocation_id,
                registrations=(local_reg,),
            )
        except Exception:
            release_nixl_transfer_resources(
                self._agent,
                NixlTransferResources(
                    handle=handle,
                    label=source.allocation_id,
                    registrations=(() if local_reg is None else (local_reg,)),
                ),
            )
            raise

    def _raise_error(self) -> None:
        with self._condition:
            self._raise_error_locked()

    def _raise_error_locked(self) -> None:
        if self._error is not None:
            raise self._error


def _load_remote_metadata(agent: Any, metadata: bytes | str) -> str:
    if isinstance(metadata, str):
        metadata = _decode_metadata_bytes(metadata)
    remote_name = agent.add_remote_agent(metadata)
    if isinstance(remote_name, bytes):
        return remote_name.decode("utf-8")
    return str(remote_name)


def load_remote_peer_metadata(config: Mapping[str, Any]) -> dict[str, Any]:
    """Load and normalize UCX peer metadata from config or a JSON file."""
    payload: dict[str, Any] = {}
    path = config.get(NIXL_UCX_CONFIG_PATH_CONFIG_KEY) or os.environ.get(
        GMS_NIXL_UCX_CONFIG_PATH_ENV
    )
    if path:
        with open(path, encoding="utf-8") as handle:
            payload.update(json.load(handle))

    for key in (
        NIXL_UCX_REMOTE_AGENT_CONFIG_KEY,
        NIXL_UCX_REMOTE_METADATA_CONFIG_KEY,
        NIXL_UCX_SOURCES_CONFIG_KEY,
        "agent_name",
        "metadata",
        "sources",
    ):
        if key in config and config[key] is not None:
            payload[key] = config[key]

    agent_name = (
        payload["agent_name"]
        if "agent_name" in payload
        else payload.get(NIXL_UCX_REMOTE_AGENT_CONFIG_KEY)
    )
    metadata = (
        payload["metadata"]
        if "metadata" in payload
        else payload.get(NIXL_UCX_REMOTE_METADATA_CONFIG_KEY)
    )
    sources = (
        payload["sources"]
        if "sources" in payload
        else payload.get(NIXL_UCX_SOURCES_CONFIG_KEY)
    )

    if metadata is None:
        raise RuntimeError(
            f"{NIXL_UCX_TRANSFER_BACKEND} requires peer metadata "
            f"({NIXL_UCX_REMOTE_METADATA_CONFIG_KEY} or metadata)"
        )
    if isinstance(metadata, str):
        metadata = _decode_metadata_bytes(metadata)

    return {
        "agent_name": None if agent_name is None else str(agent_name),
        "metadata": metadata,
        "sources": sources,
    }


def _decode_metadata_bytes(value: str) -> bytes:
    try:
        return base64.b64decode(value, validate=True)
    except Exception:
        return value.encode("latin1")
