# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import base64
import errno
import json
import os
import queue
import threading
from collections import defaultdict
from concurrent.futures import CancelledError
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

if TYPE_CHECKING:
    import torch

from gpu_memory_service.client._gms_storage_model import AllocationEntry, SaveManifest


class ShardWriter:
    """Packs allocation bytes sequentially into large binary shard files.

    This is a single-threaded utility for streaming writes.  The parallel save
    path in GMSStorageClient._write_shards assigns allocations to shards via
    plan_shard_layout and writes each shard file concurrently, so it does not
    use ShardWriter directly.  ShardWriter is kept as a public utility for
    callers that want a simple sequential writer.
    """

    def __init__(self, shards_dir: str, shard_size_bytes: int = 4 * 1024**3) -> None:
        self._shards_dir = shards_dir
        self._shard_size = shard_size_bytes
        self._shard_idx = -1
        self._current_offset = 0
        self._current_file: Optional[Any] = None
        self._current_rel_path: str = ""
        os.makedirs(shards_dir, exist_ok=True)

    def _roll_shard(self) -> None:
        if self._current_file is not None:
            self._current_file.close()
        self._shard_idx += 1
        filename = f"shard_{self._shard_idx:04d}.bin"
        abs_path = os.path.join(self._shards_dir, filename)
        self._current_file = open(abs_path, "wb")
        self._current_rel_path = os.path.join("shards", filename)
        self._current_offset = 0

    def write(self, tensor: torch.Tensor) -> Tuple[str, int]:
        cpu = tensor.cpu() if hasattr(tensor, "is_cuda") and tensor.is_cuda else tensor
        if hasattr(cpu, "is_contiguous") and not cpu.is_contiguous():
            cpu = cpu.contiguous()
        arr = cpu.numpy()
        size = arr.nbytes
        if self._current_file is None or (
            self._current_offset > 0 and self._current_offset + size > self._shard_size
        ):
            self._roll_shard()

        offset = self._current_offset
        arr.tofile(self._current_file)
        self._current_offset += size
        return self._current_rel_path, offset

    def close(self) -> None:
        if self._current_file is not None:
            self._current_file.close()
            self._current_file = None

    def __enter__(self) -> "ShardWriter":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()


def read_shard_sequential(
    abs_path: str,
    sorted_entries: List[AllocationEntry],
    device: int,
    *,
    pin_memory: bool = False,
    os_module=os,
    np_module=None,
    torch_module=None,
    logger=None,
) -> Dict[str, torch.Tensor]:
    """Read one shard file front-to-back without seeking."""
    if np_module is None or torch_module is None:
        raise RuntimeError("numpy and torch modules are required to read shards")

    result: Dict[str, torch.Tensor] = {}
    device_str = f"cuda:{device}" if device >= 0 else "cpu"

    if abs_path.endswith(".pt"):
        if len(sorted_entries) != 1:
            raise RuntimeError(
                f"Expected exactly 1 entry for legacy .pt file, got "
                f"{len(sorted_entries)}: {abs_path}"
            )
        entry = sorted_entries[0]
        result[entry.allocation_id] = torch_module.load(
            abs_path,
            weights_only=True,
            map_location=device_str,
        )
        return result

    odirect_flag = getattr(os_module, "O_DIRECT", None)
    if odirect_flag is not None:
        fd: Optional[int] = None
        done = 0
        try:
            total_size = sum(entry.aligned_size for entry in sorted_entries)
            if pin_memory and torch_module.cuda.is_available():
                shard_t = torch_module.empty(
                    total_size,
                    dtype=torch_module.uint8,
                    pin_memory=True,
                )
                arr = shard_t.numpy()
            else:
                shard_t = None
                arr = np_module.empty(total_size, dtype=np_module.uint8)

            fd = os_module.open(abs_path, os_module.O_RDONLY | odirect_flag)
            try:
                mv = memoryview(arr)
                try:
                    while done < total_size:
                        read = os_module.readv(fd, [mv[done:]])
                        if read == 0:
                            raise RuntimeError(
                                f"Unexpected EOF in O_DIRECT read from {abs_path}: "
                                f"got {done} of {total_size} bytes"
                            )
                        done += read
                finally:
                    mv.release()
            finally:
                os_module.close(fd)

            offset = 0
            for entry in sorted_entries:
                size = entry.aligned_size
                if shard_t is not None:
                    tensor = shard_t[offset : offset + size]
                else:
                    tensor = torch_module.from_numpy(arr[offset : offset + size])
                if device >= 0:
                    tensor = tensor.to(device_str)
                result[entry.allocation_id] = tensor
                offset += size
            return result
        except OSError as exc:
            fallback_errnos = {errno.EINVAL, errno.EOPNOTSUPP}
            if fd is not None and exc.errno not in fallback_errnos:
                raise
            result.clear()
            if logger is not None:
                if fd is None:
                    logger.debug(
                        "O_DIRECT unsupported on %s (errno %s); using buffered reads",
                        abs_path,
                        exc.errno,
                    )
                else:
                    logger.debug(
                        "O_DIRECT read on %s hit EINVAL after %d/%d bytes; using buffered reads",
                        abs_path,
                        done,
                        total_size,
                    )

    if sorted_entries and sorted_entries[0].tensor_offset != 0:
        raise RuntimeError(
            f"Buffered shard read requires entries starting at offset 0, "
            f"got {sorted_entries[0].tensor_offset} in {abs_path}"
        )
    with open(abs_path, "rb") as handle:
        for entry in sorted_entries:
            raw = handle.read(entry.aligned_size)
            if len(raw) != entry.aligned_size:
                raise RuntimeError(
                    f"Short read from {abs_path} at offset {entry.tensor_offset}: "
                    f"expected {entry.aligned_size} bytes, got {len(raw)}"
                )
            arr = np_module.frombuffer(raw, dtype=np_module.uint8).copy()
            tensor = torch_module.from_numpy(arr)
            if device >= 0:
                tensor = tensor.to(device_str)
            result[entry.allocation_id] = tensor
    return result


def decode_metadata(raw_meta: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    return {
        key: {
            "allocation_id": entry["allocation_id"],
            "offset_bytes": int(entry["offset_bytes"]),
            "value": base64.b64decode(entry["value"]),
        }
        for key, entry in raw_meta.items()
    }


def group_entries_by_shard(
    allocations: List[AllocationEntry],
) -> Dict[str, List[AllocationEntry]]:
    groups: Dict[str, List[AllocationEntry]] = defaultdict(list)
    for entry in allocations:
        groups[entry.tensor_file].append(entry)
    for entries in groups.values():
        entries.sort(key=lambda entry: entry.tensor_offset)
    return dict(groups)


def plan_shard_layout(
    allocations_info: List[Dict[str, Any]],
    shard_size_bytes: int,
) -> List[Tuple[int, int]]:
    result: List[Tuple[int, int]] = []
    shard_idx = -1
    current_offset = 0
    started = False
    for alloc in allocations_info:
        size = int(alloc["aligned_size"])
        if not started or (
            current_offset > 0 and current_offset + size > shard_size_bytes
        ):
            shard_idx += 1
            current_offset = 0
            started = True
        result.append((shard_idx, current_offset))
        current_offset += size
    return result


def _put_entry(
    work_q: queue.Queue[Optional[Tuple[AllocationEntry, "torch.Tensor"]]],
    entry: AllocationEntry,
    tensor: "torch.Tensor",
    cancel_event: Optional[threading.Event],
    abs_path: str,
) -> None:
    """Put one entry into the work queue, respecting cancellation."""
    while True:
        if cancel_event is not None and cancel_event.is_set():
            raise CancelledError(f"shard read cancelled: {abs_path}")
        try:
            work_q.put((entry, tensor), timeout=0.1)
            return
        except queue.Full:
            pass


def read_shard_streaming_to_queue(
    abs_path: str,
    sorted_entries: List[AllocationEntry],
    work_q: queue.Queue[Optional[Tuple[AllocationEntry, "torch.Tensor"]]],
    *,
    pin_memory: bool,
    cancel_event: Optional[threading.Event] = None,
    os_module=os,
    np_module=None,
    torch_module=None,
    logger=None,
) -> int:
    """Read a shard via O_DIRECT and stream entries to the queue as they
    become readable, overlapping disk I/O with downstream GPU copies.

    Falls back to the non-streaming ``read_shard_to_queue`` when O_DIRECT
    is unavailable or the filesystem rejects it.
    """
    if not sorted_entries:
        return 0
    if np_module is None or torch_module is None:
        raise RuntimeError("numpy and torch modules are required")

    total_size = sum(e.aligned_size for e in sorted_entries)

    # Allocate a single pinned buffer for the whole shard.
    if pin_memory and torch_module.cuda.is_available():
        shard_t = torch_module.empty(
            total_size, dtype=torch_module.uint8, pin_memory=True
        )
        shard_arr = shard_t.numpy()
    else:
        shard_t = None
        shard_arr = np_module.empty(total_size, dtype=np_module.uint8)

    odirect_flag = getattr(os_module, "O_DIRECT", None)
    if odirect_flag is not None:
        fd: Optional[int] = None
        try:
            fd = os_module.open(abs_path, os_module.O_RDONLY | odirect_flag)
            mv = memoryview(shard_arr)
            bytes_read = 0
            entry_idx = 0
            next_end = (
                sorted_entries[0].tensor_offset + sorted_entries[0].aligned_size
            )

            while bytes_read < total_size:
                if cancel_event is not None and cancel_event.is_set():
                    raise CancelledError(f"shard read cancelled: {abs_path}")
                n = os_module.readv(fd, [mv[bytes_read:]])
                if n == 0:
                    raise RuntimeError(
                        f"Unexpected EOF in streaming read from {abs_path}: "
                        f"got {bytes_read} of {total_size} bytes"
                    )
                bytes_read += n

                # Queue every entry whose data is now fully in the buffer.
                while entry_idx < len(sorted_entries) and next_end <= bytes_read:
                    entry = sorted_entries[entry_idx]
                    off = entry.tensor_offset
                    if shard_t is not None:
                        tensor = shard_t[off : off + entry.aligned_size]
                    else:
                        tensor = torch_module.from_numpy(
                            shard_arr[off : off + entry.aligned_size]
                        )
                    _put_entry(work_q, entry, tensor, cancel_event, abs_path)
                    entry_idx += 1
                    if entry_idx < len(sorted_entries):
                        next_end = (
                            sorted_entries[entry_idx].tensor_offset
                            + sorted_entries[entry_idx].aligned_size
                        )

            mv.release()
            os_module.close(fd)
            fd = None
            return len(sorted_entries)
        except OSError as exc:
            if fd is not None:
                os_module.close(fd)
                fd = None
            fallback_errnos = {errno.EINVAL, errno.EOPNOTSUPP}
            if exc.errno not in fallback_errnos:
                raise
            if logger is not None:
                logger.debug(
                    "O_DIRECT streaming failed on %s (errno %s); "
                    "falling back to buffered read",
                    abs_path,
                    exc.errno,
                )

    # Fallback: buffered full-shard read, then queue all entries.
    with open(abs_path, "rb") as handle:
        raw = handle.read()
    arr = np_module.frombuffer(raw, dtype=np_module.uint8).copy()
    for entry in sorted_entries:
        off = entry.tensor_offset
        tensor = torch_module.from_numpy(arr[off : off + entry.aligned_size])
        _put_entry(work_q, entry, tensor, cancel_event, abs_path)
    return len(sorted_entries)


def read_shard_to_queue(
    abs_path: str,
    sorted_entries: List[AllocationEntry],
    work_q: queue.Queue[Optional[Tuple[AllocationEntry, torch.Tensor]]],
    *,
    pin_memory: bool,
    read_shard,
    cancel_event: Optional[threading.Event] = None,
) -> int:
    shard_result = read_shard(
        abs_path,
        sorted_entries,
        -1,
        pin_memory=pin_memory,
    )
    for entry in sorted_entries:
        _put_entry(work_q, entry, shard_result[entry.allocation_id], cancel_event, abs_path)
    return len(sorted_entries)


def load_manifest_and_metadata(
    input_dir: str,
) -> Tuple[SaveManifest, Dict[str, Dict[str, Any]]]:
    manifest_path = os.path.join(input_dir, "manifest.json")
    with open(manifest_path, encoding="utf-8") as handle:
        manifest = SaveManifest.from_dict(json.load(handle))

    metadata_path = os.path.join(input_dir, "gms_metadata.json")
    raw_meta: Dict[str, Any] = {}
    if os.path.exists(metadata_path):
        with open(metadata_path, encoding="utf-8") as handle:
            raw_meta = json.load(handle)

    return manifest, decode_metadata(raw_meta)
