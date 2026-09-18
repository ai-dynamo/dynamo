# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Async waiting must never delay the commit that wakes a reader."""

import asyncio

import pytest
from gms_kv_ring.daemon.directory_server import DirectoryDaemon

pytestmark = pytest.mark.pre_merge


def poll_request(**updates):
    return dict(
        op="directory_changes",
        manifest_id="m",
        after_revision=0,
        wait_ms=1000,
        **updates,
    )


async def promote(daemon, writer, epoch):
    return await daemon._dispatch_async(
        {"op": "directory_promote", "writer_id": writer, "expected_epoch": epoch}
    )


async def publish(daemon, slot=1):
    return await daemon._dispatch_async(
        {
            "op": "directory_publish_batch",
            "manifest_id": "m",
            "writer_id": "primary",
            "expected_epoch": 2,
            "items": [
                {
                    "content_hash": bytes([slot]).hex(),
                    "engine_id": "0",
                    "slot_ids": [slot],
                    "generations": [1],
                    "tier": "hbm",
                }
            ],
        }
    )


@pytest.mark.asyncio
async def test_waiting_readers_do_not_block_atomic_publication():
    daemon = DirectoryDaemon("/unused")
    assert (await promote(daemon, "primary", 1))["promoted"]
    polls = [
        asyncio.create_task(daemon._dispatch_async(poll_request())) for _ in range(4)
    ]
    await asyncio.sleep(0)
    assert not any(poll.done() for poll in polls)
    assert (await publish(daemon))["published"] == 1
    responses = await asyncio.wait_for(asyncio.gather(*polls), timeout=0.5)
    assert all(len(response["changes"]) == 1 for response in responses)
    assert all(response["next_revision"] == 1 for response in responses)
    # A reader arriving after commit must not wait on the next event.
    response = await asyncio.wait_for(
        daemon._dispatch_async(poll_request()), timeout=0.5
    )
    assert response["next_revision"] == 1


@pytest.mark.asyncio
async def test_epoch_only_promotion_wakes_readers():
    daemon = DirectoryDaemon("/unused")
    pending = asyncio.create_task(daemon._dispatch_async(poll_request()))
    await asyncio.sleep(0)
    assert not pending.done()
    assert (await promote(daemon, "primary", 1))["promoted"]
    response = await asyncio.wait_for(pending, timeout=0.5)
    assert response["directory_epoch"] == 2
    assert response["next_revision"] == 0
    assert response["changes"] == []


@pytest.mark.asyncio
async def test_cancelled_waiter_does_not_cancel_other_waiters_or_commits():
    daemon = DirectoryDaemon("/unused")
    await promote(daemon, "primary", 1)
    cancelled = asyncio.create_task(daemon._dispatch_async(poll_request()))
    survivor = asyncio.create_task(daemon._dispatch_async(poll_request()))
    await asyncio.sleep(0)
    cancelled.cancel()
    with pytest.raises(asyncio.CancelledError):
        await cancelled
    assert (await publish(daemon))["published"] == 1
    assert (await asyncio.wait_for(survivor, timeout=0.5))["next_revision"] == 1


@pytest.mark.asyncio
async def test_empty_poll_times_out_without_changing_cursor():
    daemon = DirectoryDaemon("/unused")
    request = poll_request()
    request["wait_ms"] = 1
    response = await asyncio.wait_for(daemon._dispatch_async(request), timeout=0.5)
    assert response["next_revision"] == 0
    assert response["changes"] == []


@pytest.mark.asyncio
@pytest.mark.parametrize("field", ["after_revision", "wait_ms", "limit"])
async def test_invalid_cursor_preserves_validation(field):
    daemon = DirectoryDaemon("/unused")
    request = poll_request()
    request[field] = "invalid"
    result = await daemon._dispatch_async(request)
    assert not result["ok"]
    assert "malformed change cursor" in result["error"]
