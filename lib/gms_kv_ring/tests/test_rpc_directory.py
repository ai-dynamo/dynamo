# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import threading
from types import SimpleNamespace

import pytest
from gms_kv_ring.daemon.rpc_directory import handle_directory_promote

pytestmark = pytest.mark.pre_merge


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
