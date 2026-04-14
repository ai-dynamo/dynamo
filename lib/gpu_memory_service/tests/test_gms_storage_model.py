# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from gpu_memory_service.client._gms_storage_model import AllocationEntry, SaveManifest


def test_manifest_round_trip() -> None:
    manifest = SaveManifest(
        version="1.0",
        timestamp=123.0,
        layout_hash="abc",
        device=2,
        allocations=[
            AllocationEntry(
                allocation_id="alloc-1",
                size=16,
                aligned_size=32,
                tag="weights",
                tensor_file="shards/shard_0000.bin",
                tensor_offset=64,
            )
        ],
    )

    restored = SaveManifest.from_dict(manifest.to_dict())

    assert restored.version == "1.0"
    assert restored.layout_hash == "abc"
    assert restored.device == 2
    assert len(restored.allocations) == 1
    assert restored.allocations[0].tensor_offset == 64
