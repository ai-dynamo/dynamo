# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the shared `DeferredAbort` base."""

from __future__ import annotations

import asyncio

import pytest

from dynamo.common.backend.abort import DeferredAbort

pytestmark = [pytest.mark.unit, pytest.mark.gpu_0, pytest.mark.pre_merge, pytest.mark.asyncio]


class _CountingAbort(DeferredAbort):
    def __init__(self) -> None:
        super().__init__()
        self.fired = 0

    async def _do_abort_now(self) -> None:
        self.fired += 1


async def test_abort_before_first_token_parks_until_signal():
    guard = _CountingAbort()
    guard.abort()
    await asyncio.sleep(0)
    assert guard.fired == 0
    guard.signal_first_token()
    await asyncio.sleep(0)
    assert guard.fired == 1


async def test_abort_is_idempotent():
    guard = _CountingAbort()
    guard.signal_first_token()
    guard.abort()
    guard.abort()
    guard.abort()
    await asyncio.sleep(0)
    assert guard.fired == 1


async def test_close_cancels_parked_wait_when_first_token_never_arrives():
    guard = _CountingAbort()
    guard.abort()
    await guard.close()
    assert guard.fired == 0
