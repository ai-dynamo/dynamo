# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from gms_kv_ring.daemon.directory_server import DirectoryState
from gms_kv_ring.daemon.rpc_directory import (
    handle_directory_pool_binding,
    handle_directory_pool_geometry,
    handle_directory_promote,
    handle_directory_register_pool,
    handle_directory_register_pool_rank,
)


def promote(daemon: DirectoryState) -> int:
    response = handle_directory_promote(
        daemon,
        {
            "writer_id": "engine-0",
            "expected_epoch": daemon._content_directory_epoch,
        },
    )
    assert response["promoted"]
    return int(response["directory_epoch"])


def test_pool_geometry_is_immutable_and_idempotent():
    daemon = DirectoryState()
    epoch = promote(daemon)
    request = {
        "manifest_id": "model-layout",
        "writer_id": "engine-0",
        "engine_id": "pool-0",
        "total_blocks": 4096,
        "layout_digest": "layout",
        "expected_epoch": epoch,
    }

    assert handle_directory_register_pool(daemon, request)["registered"]
    assert handle_directory_register_pool(daemon, request)["registered"]

    response = handle_directory_pool_geometry(
        daemon,
        {"manifest_id": "model-layout", "engine_id": "pool-0"},
    )
    assert response["geometry"] == {
        "total_blocks": 4096,
        "layout_digest": "layout",
    }


def test_pool_geometry_rejects_stale_writer_and_layout_change():
    daemon = DirectoryState()
    epoch = promote(daemon)
    request = {
        "manifest_id": "model-layout",
        "writer_id": "engine-0",
        "engine_id": "pool-0",
        "total_blocks": 4096,
        "expected_epoch": epoch,
    }
    assert handle_directory_register_pool(daemon, request)["registered"]

    stale = handle_directory_register_pool(
        daemon,
        {**request, "writer_id": "engine-1"},
    )
    assert stale["rejected_stale_writer"]

    changed = handle_directory_register_pool(
        daemon,
        {**request, "total_blocks": 2048},
    )
    assert changed == {
        "ok": False,
        "error": "persistent KV pool geometry changed for this manifest",
    }


def test_pool_geometry_is_scoped_by_manifest_and_pool():
    daemon = DirectoryState()
    promote(daemon)

    missing = handle_directory_pool_geometry(
        daemon,
        {"manifest_id": "other-layout", "engine_id": "pool-0"},
    )
    assert missing["geometry"] is None


def test_pool_binding_collects_immutable_rank_allocations():
    daemon = DirectoryState()
    epoch = promote(daemon)
    base = {
        "manifest_id": "model-layout",
        "writer_id": "engine-0",
        "engine_id": "pool-0",
        "expected_tp_size": 2,
        "layout_digest": "layout",
        "expected_epoch": epoch,
    }
    rank_zero = {
        **base,
        "rank": 0,
        "allocations": [
            {
                "engine_id": "device-0",
                "tag": "kv",
                "allocation_id": "allocation-0",
                "aligned_size": 4096,
            }
        ],
    }
    rank_one = {
        **base,
        "rank": 1,
        "allocations": [
            {
                "engine_id": "device-1",
                "tag": "kv",
                "allocation_id": "allocation-1",
                "aligned_size": 4096,
            }
        ],
    }

    assert handle_directory_register_pool_rank(daemon, rank_zero)["registered"]
    assert handle_directory_register_pool_rank(daemon, rank_zero)["registered"]
    assert handle_directory_register_pool_rank(daemon, rank_one)["registered"]

    response = handle_directory_pool_binding(
        daemon, {"manifest_id": "model-layout", "engine_id": "pool-0"}
    )
    assert response["binding"] == {
        "expected_tp_size": 2,
        "layout_digest": "layout",
        "ranks": {
            "0": {
                "attached_epoch": epoch,
                "allocations": rank_zero["allocations"],
            },
            "1": {
                "attached_epoch": epoch,
                "allocations": rank_one["allocations"],
            },
        },
    }

    changed = handle_directory_register_pool_rank(
        daemon,
        {
            **rank_zero,
            "allocations": [
                {**rank_zero["allocations"][0], "allocation_id": "replacement"}
            ],
        },
    )
    assert changed == {
        "ok": False,
        "error": "persistent KV rank allocation binding changed",
    }
