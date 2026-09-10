# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The async directory publish worker must not die on a single mutation failure.

A transient daemon error -- or this engine being fenced/demoted mid-drain --
previously set a fatal error, cleared the queue and stopped the worker thread,
silently ending ALL future directory publications for the process. That is a
permanent prefix-cache-publishing cliff. The worker must instead skip the failed
mutation (a safe cache miss) and keep serving the queue.
"""

from __future__ import annotations

import threading
from types import SimpleNamespace

import pytest
from gms_kv_ring.common.content_directory import ContentDirectory
from gms_kv_ring.daemon.rpc_directory import (
    handle_directory_ensure_hbm_capacity,
    handle_directory_lookup_claim,
)

pytestmark = pytest.mark.pre_merge


def test_publish_worker_survives_a_failing_mutation():
    directory = ContentDirectory(
        "/tmp/gms-directory-writer-resilience.sock",
        engine="test",
        block_size=16,
        mode="shadow",
    )
    try:
        calls: list[list[dict]] = []

        def flaky_publish(items):
            calls.append(items)
            if len(calls) == 1:
                raise RuntimeError("simulated transient publish failure")
            return len(items)

        directory.publish = flaky_publish  # type: ignore[method-assign]
        directory._defer_mutation("publish", [{"a": 1}])
        directory._defer_mutation("publish", [{"b": 2}])

        assert directory.flush_deferred(timeout=5.0) is True
        assert len(calls) == 2, "worker died instead of continuing past the failure"
        assert directory._mutation_failed == 1
        assert (
            directory._mutation_error is None
        ), "per-mutation failure must not be fatal"
        assert (
            directory._mutation_thread is not None
            and directory._mutation_thread.is_alive()
        )
    finally:
        directory.close()


def test_zero_capacity_request_preserves_ready_hbm_entry():
    content_hash = b"h" * 32
    key = ("manifest", content_hash)
    entry = {
        "tier": "hbm",
        "state": "ready",
        "engine_id": "engine",
        "slot_ids": [7],
        "generations": [3],
        "_claim_count": 0,
    }
    daemon = SimpleNamespace(
        _content_hash_lock=threading.Condition(),
        _content_directory_writer_id="writer",
        _content_directory_epoch=4,
        _content_directory={key: entry},
    )

    response = handle_directory_ensure_hbm_capacity(
        daemon,
        {
            "manifest_id": "manifest",
            "writer_id": "writer",
            "expected_epoch": 4,
            "required_blocks": 0,
        },
    )

    assert response == {
        "ok": True,
        "victims": [],
        "freed_blocks": 0,
        "rejected_stale_writer": False,
    }
    assert daemon._content_directory == {key: entry}


@pytest.mark.parametrize(
    ("pending_generations", "expected_hit"),
    [(None, False), ([4], True)],
)
def test_active_hbm_is_claimable_only_during_adoption(
    pending_generations, expected_hit
):
    content_hash = b"h" * 32
    key = ("manifest", content_hash)
    entry = {
        "tier": "hbm",
        "state": "active",
        "engine_id": "engine",
        "slot_ids": [7],
        "generations": [3],
        "_claim_count": 0,
        "_owner_writer": "writer",
    }
    if pending_generations is not None:
        entry["_pending_generations"] = pending_generations
    daemon = SimpleNamespace(
        _content_hash_lock=threading.Condition(),
        _content_directory_writer_id="writer",
        _content_directory_epoch=4,
        _content_directory={key: entry},
        _content_directory_claims={},
        _content_directory_access_seq=0,
    )

    response = handle_directory_lookup_claim(
        daemon,
        {
            "manifest_id": "manifest",
            "writer_id": "writer",
            "expected_epoch": 4,
            "hashes": [content_hash.hex()],
        },
    )

    assert (response["entries"][0] is not None) is expected_hit
    assert (response["claim_token"] is not None) is expected_hit
    assert entry["_claim_count"] == int(expected_hit)
