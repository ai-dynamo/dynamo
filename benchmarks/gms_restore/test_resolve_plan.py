# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import hashlib
import json

import pytest
from resolve_plan import resolve


def fixture(tmp_path):
    ids = [
        "GPU-00000000-0000-0000-0000-000000000001",
        "GPU-00000000-0000-0000-0000-000000000002",
    ]
    ranks = []
    for rank in range(2):
        root = tmp_path / str(rank)
        root.mkdir()
        allocs = [
            {
                "allocation_id": f"capture-rank-{rank}",
                "aligned_size": 2097152,
                "shard": "weights.bin",
                "offset": 0,
            }
        ]
        raw = json.dumps({"version": 1, "allocations": allocs}).encode()
        (root / "manifest.json").write_bytes(raw)
        ranks.append(
            {
                "rank": rank,
                "captured_ordinal": rank,
                "source_uuid": ids[rank],
                "artifact": str(root),
                "socket_device": rank,
                "allocations": allocs,
                "manifest_sha256": hashlib.sha256(raw).hexdigest(),
            }
        )
    capture = {"capture_id": "one-capture", "layout": {"tp": 2}, "ranks": ranks}
    claim = {
        "metadata": {"uid": "claim-uid"},
        "status": {
            "allocation": {
                "devices": {
                    "results": [
                        {
                            "request": f"tp-{rank}",
                            "driver": "gpu.nvidia.com",
                            "pool": "node",
                            "device": f"gpu-{1 - rank}",
                        }
                        for rank in (1, 0)
                    ]
                }
            }
        },
    }
    slices = {
        "items": [
            {
                "spec": {
                    "driver": "gpu.nvidia.com",
                    "pool": {"name": "node"},
                    "devices": [
                        {
                            "name": f"gpu-{rank}",
                            "attributes": {"uuid": {"string": ids[rank]}},
                        }
                        for rank in range(2)
                    ],
                }
            }
        ]
    }
    return capture, claim, slices, ids


def test_same_node_permutation_and_scrambled_request_order(tmp_path):
    capture, claim, slices, ids = fixture(tmp_path)
    plan = resolve(capture, claim, slices)
    assert plan["cuda_device_map"] == f"{ids[0]}={ids[1]},{ids[1]}={ids[0]}"
    assert [
        (r["rank"], r["local_device"], r["socket_device"]) for r in plan["ranks"]
    ] == [(0, 0, 0), (1, 0, 1)]


def test_wrong_capture_artifact_refused(tmp_path):
    capture, claim, slices, _ = fixture(tmp_path)
    capture["ranks"][0]["manifest_sha256"] = "bad"
    with pytest.raises(ValueError, match="not from this capture"):
        resolve(capture, claim, slices)


def test_missing_rank_request_refused(tmp_path):
    capture, claim, slices, _ = fixture(tmp_path)
    claim["status"]["allocation"]["devices"]["results"].pop()
    with pytest.raises(ValueError, match="requests"):
        resolve(capture, claim, slices)


def test_duplicate_target_refused(tmp_path):
    capture, claim, slices, _ = fixture(tmp_path)
    for r in claim["status"]["allocation"]["devices"]["results"]:
        r["device"] = "gpu-0"
    with pytest.raises(ValueError, match="duplicate destination_uuid"):
        resolve(capture, claim, slices)
