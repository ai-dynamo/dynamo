# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from collections import deque
from threading import Condition
from types import SimpleNamespace

import pytest
from gms_kv_ring.daemon.rpc_directory import handle_directory_changes

pytestmark = [pytest.mark.pre_merge, pytest.mark.unit, pytest.mark.gpu_0]


class History(deque):
    visited = 0

    def __iter__(self):
        raise AssertionError("do not scan old history on a tip read")

    def __reversed__(self):
        for item in super().__reversed__():
            self.visited += 1
            yield item


def daemon():
    history = History(
        {
            "revision": i,
            "manifest_id": "m",
            "scope": "engine",
            "content_hash": str(i),
            "entry": None,
        }
        for i in range(1, 1001)
    )
    return SimpleNamespace(
        _content_hash_lock=Condition(),
        _content_directory_revision=1000,
        _content_directory_changes=history,
        _content_directory_epoch=2,
        _content_directory_writer_id="writer",
    )


def test_tip_read_visits_only_delta_and_cursor():
    state = daemon()
    result = handle_directory_changes(
        state, {"manifest_id": "m", "after_revision": 998}
    )
    assert [v["revision"] for v in result["changes"]] == [999, 1000]
    assert state._content_directory_changes.visited == 3
    assert result["next_revision"] == 1000
    assert not result["has_more"]


def test_limited_read_keeps_oldest_first_and_continuation():
    state = daemon()
    result = handle_directory_changes(
        state, {"manifest_id": "m", "after_revision": 990, "limit": 2}
    )
    assert [v["revision"] for v in result["changes"]] == [991, 992]
    assert result["next_revision"] == 992
    assert result["has_more"]


def test_filtered_read_advances_past_other_scopes():
    result = handle_directory_changes(
        daemon(), {"manifest_id": "other", "after_revision": 998}
    )
    assert result["changes"] == []
    assert result["next_revision"] == 1000
    assert not result["has_more"]
