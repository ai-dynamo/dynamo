# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import threading
from collections import deque
from types import SimpleNamespace

import pytest
from gms_kv_ring.daemon.rpc_directory import handle_directory_promote

pytestmark = pytest.mark.pre_merge


def test_forced_same_writer_promotion_starts_a_new_epoch():
    ready_key = ("manifest", b"r" * 32)
    active_key = ("manifest", b"a" * 32)
    ready = {
        "engine_id": "writer",
        "slot_ids": [1],
        "tier": "hbm",
        "state": "ready",
        "_owner_writer": "writer",
    }
    active = {
        "engine_id": "writer",
        "slot_ids": [2],
        "tier": "hbm",
        "state": "active",
        "_owner_writer": "writer",
    }
    daemon = SimpleNamespace(
        _content_hash_lock=threading.Condition(),
        _content_directory_epoch=4,
        _content_directory_writer_id="writer",
        _content_directory={ready_key: ready, active_key: active},
        _content_directory_by_slot={
            ("manifest", "writer", 1): ready_key[1],
            ("manifest", "writer", 2): active_key[1],
        },
        _content_directory_claims={},
        _content_directory_revision=0,
        _content_directory_changes=deque(),
    )

    response = handle_directory_promote(
        daemon,
        {
            "writer_id": "writer",
            "expected_epoch": 4,
            "force_new_epoch": True,
        },
    )

    assert response == {
        "ok": True,
        "promoted": True,
        "directory_epoch": 5,
        "writer_id": "writer",
    }
    assert ready_key in daemon._content_directory
    assert active_key not in daemon._content_directory
    assert ("manifest", "writer", 2) not in daemon._content_directory_by_slot


def test_directory_promotion_rejects_non_boolean_force_flag():
    response = handle_directory_promote(
        object(),
        {
            "writer_id": "writer",
            "expected_epoch": 4,
            "force_new_epoch": "false",
        },
    )

    assert response == {
        "ok": False,
        "error": "force_new_epoch must be a boolean",
    }


def test_repeated_writer_promotion_preserves_live_claims():
    key = ("manifest", b"h" * 32)
    entry = {"_claim_count": 1}
    daemon = SimpleNamespace(
        _content_hash_lock=threading.Condition(),
        _content_directory_epoch=4,
        _content_directory_writer_id="writer",
        _content_directory={key: entry},
        _content_directory_claims={
            "claim": {
                "writer_id": "writer",
                "epoch": 4,
                "entries": [(key, ())],
            }
        },
    )

    response = handle_directory_promote(
        daemon, {"writer_id": "writer", "expected_epoch": 4}
    )

    assert response == {
        "ok": True,
        "promoted": True,
        "directory_epoch": 4,
        "writer_id": "writer",
    }
    assert "claim" in daemon._content_directory_claims
    assert entry["_claim_count"] == 1
