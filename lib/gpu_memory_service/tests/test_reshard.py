# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from gpu_memory_service.client.gms_storage_client import (
    _CURRENT_VERSION,
    AllocationEntry,
    SaveManifest,
    _group_entries_by_shard,
    _read_shard_sequential,
    _ShardWriter,
)
from gpu_memory_service.tests.e2e.reshard import reshard


def _build_source_save(source_dir: Path) -> None:
    shards_dir = source_dir / "shards"
    shards_dir.mkdir(parents=True)

    allocations = []
    with _ShardWriter(str(shards_dir), shard_size_bytes=1024) as writer:
        for idx, fill in enumerate((1, 2)):
            data = torch.full((64,), fill, dtype=torch.uint8)
            rel_path, offset = writer.write(data)
            allocations.append(
                AllocationEntry(
                    allocation_id=f"alloc-{idx}",
                    size=64,
                    aligned_size=64,
                    tag=f"tag-{idx}",
                    tensor_file=rel_path,
                    tensor_offset=offset,
                )
            )

    manifest = SaveManifest(
        version=_CURRENT_VERSION,
        timestamp=1.0,
        layout_hash="hash",
        device=0,
        allocations=allocations,
    )
    (source_dir / "manifest.json").write_text(json.dumps(manifest.to_dict()))
    (source_dir / "gms_metadata.json").write_text(
        json.dumps(
            {
                "key0": {
                    "allocation_id": "alloc-0",
                    "offset_bytes": 0,
                    "value": "djA=",
                }
            }
        )
    )


def test_reshard_rewrites_manifest_and_metadata(tmp_path: Path) -> None:
    source_dir = tmp_path / "source"
    dest_dir = tmp_path / "dest"
    _build_source_save(source_dir)

    reshard(source_dir, dest_dir, shard_size_bytes=64)

    manifest = SaveManifest.from_dict(
        json.loads((dest_dir / "manifest.json").read_text())
    )
    assert len(manifest.allocations) == 2
    assert {entry.tensor_file for entry in manifest.allocations} == {
        "shards/shard_0000.bin",
        "shards/shard_0001.bin",
    }
    assert (dest_dir / "gms_metadata.json").exists()

    # Verify actual tensor bytes are preserved correctly.
    groups = _group_entries_by_shard(manifest.allocations)
    tensors: dict = {}
    for rel_path, sorted_entries in groups.items():
        abs_path = str(dest_dir / rel_path)
        tensors.update(_read_shard_sequential(abs_path, sorted_entries, device=-1))

    alloc_by_tag = {e.tag: e for e in manifest.allocations}
    assert tensors[alloc_by_tag["tag-0"].allocation_id].tolist() == [1] * 64
    assert tensors[alloc_by_tag["tag-1"].allocation_id].tolist() == [2] * 64


def test_reshard_rejects_non_empty_destination(tmp_path: Path) -> None:
    source_dir = tmp_path / "source"
    dest_dir = tmp_path / "dest"
    _build_source_save(source_dir)
    dest_dir.mkdir()
    (dest_dir / "stale.bin").write_bytes(b"stale")

    with pytest.raises(FileExistsError, match="must not already contain files"):
        reshard(source_dir, dest_dir, shard_size_bytes=64)
