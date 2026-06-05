# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import threading
from contextlib import contextmanager

from dynamo.trtllm.kv_p2p import target_rpc_local


class _FakeTargetRpcClient:
    def __init__(self, resolve_response):
        self.resolve_response = resolve_response
        self.released = []

    async def resolve_and_lease(self, plan, source_worker_id):
        return self.resolve_response

    async def release_lease(self, lease_id, source_worker_id, reason="ack"):
        self.released.append((lease_id, source_worker_id, reason))
        return {"ok": True, "result": True}


@contextmanager
def _running_loop():
    loop = asyncio.new_event_loop()

    def run():
        asyncio.set_event_loop(loop)
        loop.run_forever()

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    try:
        yield loop
    finally:
        loop.call_soon_threadsafe(loop.stop)
        thread.join(timeout=5)
        loop.close()


def test_promoted_primary_status_is_detected_from_dict_entries():
    assert target_rpc_local._has_promoted_primary_status(
        [
            {"block_hash": 1, "status": "resolved_secondary"},
            {"block_hash": 2, "status": "promoted_primary"},
        ]
    )


def test_promoted_primary_status_is_detected_from_string_entries():
    assert target_rpc_local._has_promoted_primary_status(
        ["resolved_secondary", "promoted_primary"]
    )


def test_dispatch_resolve_rejects_promoted_primary_and_releases_lease():
    client = _FakeTargetRpcClient(
        {
            "ok": True,
            "result": {
                "lease_id": "lease-1",
                "descriptors": [{"block_hash": 10}],
                "per_block_status": [
                    {"block_hash": 10, "status": "resolved_secondary"},
                    {"block_hash": 11, "status": "promoted_primary"},
                ],
            },
        }
    )
    lease_map = target_rpc_local._LeaseSourceMap()

    with _running_loop() as loop:
        response = target_rpc_local._dispatch_resolve(
            client,
            {"plan": {"plan_id": "p1"}, "source_worker_id": 7},
            loop,
            lease_map,
            timeout_s=5,
        )

    assert response["ok"] is False
    assert response["error"] == "promoted_primary"
    assert client.released == [("lease-1", 7, "promoted_primary")]
    assert lease_map.pop("lease-1") is None


def test_dispatch_resolve_tracks_lease_for_secondary_only_response():
    client = _FakeTargetRpcClient(
        {
            "ok": True,
            "result": {
                "lease_id": "lease-2",
                "descriptors": [{"block_hash": 10}],
                "per_block_status": [
                    {"block_hash": 10, "status": "resolved_secondary"},
                ],
            },
        }
    )
    lease_map = target_rpc_local._LeaseSourceMap()

    with _running_loop() as loop:
        response = target_rpc_local._dispatch_resolve(
            client,
            {"plan": {"plan_id": "p2"}, "source_worker_id": 8},
            loop,
            lease_map,
            timeout_s=5,
        )

    assert response["ok"] is True
    assert client.released == []
    assert lease_map.pop("lease-2") == 8
