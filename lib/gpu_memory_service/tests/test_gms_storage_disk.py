# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import base64
import json
from pathlib import Path

from gpu_memory_service.client._gms_storage_disk import load_manifest_and_metadata


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
