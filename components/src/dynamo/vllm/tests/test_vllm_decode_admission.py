# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for remote-prefill decode admission."""

import asyncio

import pytest

from dynamo.vllm.decode_admission import DecodeRemotePrefillAdmission

pytestmark = [
    pytest.mark.pre_merge,
    pytest.mark.unit,
    pytest.mark.vllm,
    pytest.mark.gpu_0,
    pytest.mark.core,
    pytest.mark.timeout(5),
]


async def _wait_for_waiters(
    admission: DecodeRemotePrefillAdmission,
    dp_rank: int,
    expected: int,
) -> None:
    for _ in range(100):
        if admission.snapshot(dp_rank).waiting == expected:
            return
        await asyncio.sleep(0)
    raise AssertionError(f"waiter count did not reach {expected}")


@pytest.mark.asyncio
async def test_zero_limit_disables_admission():
    admission = DecodeRemotePrefillAdmission(limit=0)

    lease = await admission.acquire(dp_rank=0)

    assert lease is None
    assert admission.snapshot(0).admissions == 0


@pytest.mark.asyncio
async def test_same_dp_waits_until_release():
    admission = DecodeRemotePrefillAdmission(limit=1)
    first = await admission.acquire(dp_rank=0)
    assert first is not None

    second_task = asyncio.create_task(admission.acquire(dp_rank=0))
    await _wait_for_waiters(admission, dp_rank=0, expected=1)

    snapshot = admission.snapshot(0)
    assert snapshot.active == 1
    assert snapshot.limit_hits == 1
    assert not second_task.done()

    assert first.release("first_output") is True
    second = await asyncio.wait_for(second_task, timeout=1)
    assert second is not None
    assert second.release("first_output") is True

    snapshot = admission.snapshot(0)
    assert snapshot.active == 0
    assert snapshot.waiting == 0
    assert snapshot.admissions == 2
    assert snapshot.releases == 2


@pytest.mark.asyncio
async def test_different_dp_ranks_have_independent_limits():
    admission = DecodeRemotePrefillAdmission(limit=1)
    rank_zero = await admission.acquire(dp_rank=0)
    rank_one = await admission.acquire(dp_rank=1)

    assert rank_zero is not None
    assert rank_one is not None
    assert admission.snapshot(0).active == 1
    assert admission.snapshot(1).active == 1

    rank_zero.release("first_output")
    rank_one.release("first_output")


@pytest.mark.asyncio
async def test_waiter_cancellation_cleans_state():
    admission = DecodeRemotePrefillAdmission(limit=1)
    first = await admission.acquire(dp_rank=0)
    assert first is not None

    waiter = asyncio.create_task(admission.acquire(dp_rank=0))
    await _wait_for_waiters(admission, dp_rank=0, expected=1)
    waiter.cancel()

    with pytest.raises(asyncio.CancelledError):
        await waiter

    snapshot = admission.snapshot(0)
    assert snapshot.active == 1
    assert snapshot.waiting == 0
    assert snapshot.cancelled_waiters == 1

    first.release("terminal_before_first_output")
    assert admission.snapshot(0).active == 0


@pytest.mark.asyncio
async def test_release_is_idempotent():
    admission = DecodeRemotePrefillAdmission(limit=1)
    lease = await admission.acquire(dp_rank=0)
    assert lease is not None

    assert lease.release("first_output") is True
    assert lease.release("terminal_before_first_output") is False

    snapshot = admission.snapshot(0)
    assert snapshot.active == 0
    assert snapshot.releases == 1
