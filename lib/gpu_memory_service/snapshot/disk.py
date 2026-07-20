# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import base64
import json
import os
from typing import Any, Dict, Optional, Sequence, Tuple

from gpu_memory_service.common.snapshot_profile import SnapshotProfile
from gpu_memory_service.common.protocol.messages import GetAllocationResponse
from gpu_memory_service.common.vmm import get_vmm
from gpu_memory_service.snapshot.backends.pinned_host import (
    PINNED_COPY_CHUNK_SIZE,
    close_pinned_copy_slots,
    make_pinned_copy_slots,
)
from gpu_memory_service.snapshot.model import AllocationEntry, SaveManifest

_SAVE_COPY_BUFFERS = 1


class _NullLogger:
    def warning(self, *_: Any, **__: Any) -> None:
        return None


_NULL_LOGGER = _NullLogger()


def _write_all_from_view(
    fd: int,
    view: memoryview,
    file_path: str,
    profile: SnapshotProfile | None = None,
    **profile_fields: Any,
) -> None:
    """Write a memoryview to a file descriptor, retrying partial writes."""
    total = len(view)
    done = 0
    if profile is None or not profile.enabled:
        while done < total:
            written = os.write(fd, view[done:])
            if written == 0:
                raise RuntimeError(
                    f"Short write to {file_path}: expected "
                    f"{total - done} more bytes, wrote 0"
                )
            done += written
        return

    with profile.aggregate(
        "file_write_cpu",
        byte_count=total,
        **profile_fields,
    ):
        while done < total:
            written = os.write(fd, view[done:])
            if written == 0:
                raise RuntimeError(
                    f"Short write to {file_path}: expected "
                    f"{total - done} more bytes, wrote 0"
                )
            done += written


class DeviceToFileWriter:
    """Stream bytes from CUDA device pointers into a raw shard file.

    The writer stages through reusable page-aligned, pinned host buffers.  This
    keeps the save path independent from PyTorch/NumPy while preserving the raw
    shard layout consumed by the restore backends.
    """

    def __init__(
        self,
        file_path: str,
        *,
        device: Optional[int] = None,
        buffers: int = _SAVE_COPY_BUFFERS,
        chunk_size: int = PINNED_COPY_CHUNK_SIZE,
        profile: SnapshotProfile | None = None,
        **profile_fields: Any,
    ) -> None:
        self._file_path = file_path
        self._profile = profile
        self._profile_fields = profile_fields
        buffers = int(buffers)
        chunk_size = int(chunk_size)
        if buffers <= 0:
            raise ValueError("buffers must be positive")
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        self._vmm = get_vmm()
        if device is not None:
            if profile is None:
                self._vmm.runtime_set_device(device)
            else:
                with profile.aggregate(
                    "save_worker_cuda_set_device",
                    **profile_fields,
                ):
                    self._vmm.runtime_set_device(device)
        self._slots = make_pinned_copy_slots(
            self._vmm,
            buffers,
            profile=profile,
            **profile_fields,
        )
        self._slot_index = 0
        self._closed = False
        try:
            self._fd = os.open(
                file_path,
                os.O_CREAT | os.O_TRUNC | os.O_WRONLY,
                0o666,
            )
        except Exception:
            close_pinned_copy_slots(
                self._slots,
                _NULL_LOGGER,
                "failed to close pinned save slot for %s",
                file_path,
            )
            raise
        self._chunk_size = chunk_size

    def write_device(self, src_ptr: int, byte_count: int) -> None:
        """Copy ``byte_count`` bytes from ``src_ptr`` and append them to the file."""
        done = 0
        while done < byte_count:
            chunk_size = min(self._chunk_size, byte_count - done)
            slot = self._slots[self._slot_index]
            slot.wait()
            slot.copy_from_device_async(src_ptr + done, chunk_size)
            slot.wait()
            chunk_view = slot.view[:chunk_size]
            try:
                _write_all_from_view(
                    self._fd,
                    chunk_view,
                    self._file_path,
                    self._profile,
                    **self._profile_fields,
                )
            finally:
                chunk_view.release()
            done += chunk_size
            self._slot_index = (self._slot_index + 1) % len(self._slots)

    def close(self) -> None:
        if self._closed:
            return
        error = None
        try:
            for slot in self._slots:
                slot.wait()
        except Exception as exc:  # noqa: BLE001
            error = exc
        try:
            os.close(self._fd)
        except OSError as exc:
            if error is None:
                error = exc
        close_pinned_copy_slots(
            self._slots,
            _NULL_LOGGER,
            "failed to close pinned save slot for %s",
            self._file_path,
        )
        self._closed = True
        if error is not None:
            raise error

    def __enter__(self) -> "DeviceToFileWriter":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()


def plan_shard_layout(
    allocations_info: Sequence[GetAllocationResponse],
    shard_size_bytes: int,
) -> list[Tuple[int, int]]:
    result: list[Tuple[int, int]] = []
    shard_idx = -1
    current_offset = 0
    started = False
    for alloc in allocations_info:
        size = int(alloc.aligned_size)
        if not started or (
            current_offset > 0 and current_offset + size > shard_size_bytes
        ):
            shard_idx += 1
            current_offset = 0
            started = True
        result.append((shard_idx, current_offset))
        current_offset += size
    return result


def load_manifest_and_metadata(
    input_dir: str,
    *,
    profile: SnapshotProfile | None = None,
) -> Tuple[SaveManifest, Dict[str, Dict[str, Any]]]:
    manifest_path = os.path.join(input_dir, "manifest.json")
    profile = profile or SnapshotProfile("loader", enabled=False)
    with profile.phase(
        "manifest_file_read",
        byte_count=os.path.getsize(manifest_path),
    ):
        with open(manifest_path, encoding="utf-8") as handle:
            manifest_text = handle.read()
    with profile.phase("manifest_parse"):
        manifest_payload = json.loads(manifest_text)
        manifest = SaveManifest(
            timestamp=manifest_payload["timestamp"],
            layout_hash=manifest_payload["layout_hash"],
            device=manifest_payload["device"],
            allocations=[
                AllocationEntry(**allocation)
                for allocation in manifest_payload.get("allocations", [])
            ],
        )

    metadata_path = os.path.join(input_dir, "gms_metadata.json")
    raw_meta: Dict[str, Any] = {}
    if os.path.exists(metadata_path):
        with profile.phase(
            "metadata_file_read",
            byte_count=os.path.getsize(metadata_path),
        ):
            with open(metadata_path, encoding="utf-8") as handle:
                metadata_text = handle.read()
        with profile.phase("metadata_parse"):
            raw_meta = json.loads(metadata_text)

    with profile.phase("metadata_decode", count=len(raw_meta)):
        metadata = {
            key: {
                "allocation_id": entry["allocation_id"],
                "offset_bytes": int(entry["offset_bytes"]),
                "value": base64.b64decode(entry["value"]),
            }
            for key, entry in raw_meta.items()
        }
    return manifest, metadata
