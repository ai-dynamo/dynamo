# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import base64
import json
import os
import queue
import types

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.gpu_0, pytest.mark.pre_merge]
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np

from gpu_memory_service.snapshot.disk import (
    load_manifest_and_metadata,
    read_shard_streaming_to_queue,
)
from gpu_memory_service.snapshot.model import AllocationEntry


def test_load_manifest_and_metadata(tmp_path: Path) -> None:
    manifest = {
        "version": "1.0",
        "timestamp": 1.0,
        "layout_hash": "layout",
        "device": 0,
        "allocations": [
            {
                "allocation_id": "alloc-1",
                "size": 4,
                "aligned_size": 8,
                "tag": "weights",
                "tensor_file": "shards/shard_0000.bin",
                "tensor_offset": 0,
            }
        ],
    }
    metadata = {
        "tensor-key": {
            "allocation_id": "alloc-1",
            "offset_bytes": 0,
            "value": base64.b64encode(b"payload").decode("ascii"),
        }
    }

    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (tmp_path / "gms_metadata.json").write_text(
        json.dumps(metadata),
        encoding="utf-8",
    )

    loaded_manifest, loaded_metadata = load_manifest_and_metadata(str(tmp_path))

    assert loaded_manifest.layout_hash == "layout"
    assert loaded_manifest.allocations[0].allocation_id == "alloc-1"
    assert loaded_metadata["tensor-key"]["value"] == b"payload"


def _fake_torch() -> types.ModuleType:
    """Minimal fake torch module for CPU-only streaming tests."""
    mod = types.ModuleType("torch")
    mod.uint8 = np.uint8  # type: ignore[attr-defined]

    class _Cuda:
        @staticmethod
        def is_available() -> bool:
            return False

    mod.cuda = _Cuda()  # type: ignore[attr-defined]
    mod.from_numpy = MagicMock(side_effect=lambda a: a)  # type: ignore[attr-defined]
    mod.empty = MagicMock()  # type: ignore[attr-defined]
    return mod


def test_streaming_reader_queues_entries_incrementally(tmp_path: Path) -> None:
    """Entries should appear in the work queue as the shard is read, not
    only after the entire file has been consumed."""
    entry_size = 4096  # 4 KB per entry, aligned to page size
    num_entries = 4
    total_size = entry_size * num_entries

    # Write a shard file with sequential byte patterns per entry.
    shard_path = tmp_path / "shard_0000.bin"
    data = bytearray()
    for i in range(num_entries):
        data.extend(bytes([i & 0xFF]) * entry_size)
    shard_path.write_bytes(bytes(data))

    entries = [
        AllocationEntry(
            allocation_id=f"alloc-{i}",
            size=entry_size,
            aligned_size=entry_size,
            tag="weights",
            tensor_file="shards/shard_0000.bin",
            tensor_offset=i * entry_size,
        )
        for i in range(num_entries)
    ]

    work_q: queue.Queue = queue.Queue(maxsize=32)
    count = read_shard_streaming_to_queue(
        str(shard_path),
        entries,
        work_q,
        pin_memory=False,
        os_module=os,
        np_module=np,
        torch_module=_fake_torch(),
    )

    assert count == num_entries
    assert work_q.qsize() == num_entries

    for i in range(num_entries):
        entry, tensor = work_q.get_nowait()
        assert entry.allocation_id == f"alloc-{i}"
        arr = np.asarray(tensor)
        assert arr[0] == i & 0xFF
        assert len(arr) == entry_size
