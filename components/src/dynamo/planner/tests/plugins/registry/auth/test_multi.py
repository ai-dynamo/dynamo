# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for MultiSourceAuth."""

from __future__ import annotations

import pytest

from dynamo.planner.plugins.registry.auth import (
    AuthIdentity,
    AuthValidator,
    MultiSourceAuth,
)
from dynamo.planner.plugins.registry.errors import AuthError

pytestmark = [
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
    pytest.mark.unit,
    pytest.mark.planner,
]


class _Stub(AuthValidator):
    def __init__(self, name, accept_tokens=(), raise_msg=None):
        self.name = name
        self.accept_tokens = set(accept_tokens)
        self.raise_msg = raise_msg
        self.calls = 0

    async def validate(self, token):
        self.calls += 1
        if self.raise_msg:
            raise AuthError(self.raise_msg)
        if token in self.accept_tokens:
            return AuthIdentity(source="static_secret", subject=f"from_{self.name}")
        raise AuthError(f"{self.name}: unknown token")


@pytest.mark.asyncio
async def test_first_source_accepts_short_circuits():
    a = _Stub("a", accept_tokens=["tok"])
    b = _Stub("b", accept_tokens=["tok"])
    multi = MultiSourceAuth([a, b])
    identity = await multi.validate("tok")
    assert identity.subject == "from_a"
    assert a.calls == 1
    assert b.calls == 0  # short-circuited


@pytest.mark.asyncio
async def test_second_source_accepts_when_first_rejects():
    a = _Stub("a")
    b = _Stub("b", accept_tokens=["tok"])
    multi = MultiSourceAuth([a, b])
    identity = await multi.validate("tok")
    assert identity.subject == "from_b"
    assert a.calls == 1
    assert b.calls == 1


@pytest.mark.asyncio
async def test_all_sources_reject_raises_chained_last_error():
    a = _Stub("a", raise_msg="A_FAILED")
    b = _Stub("b", raise_msg="B_FAILED")
    multi = MultiSourceAuth([a, b])
    with pytest.raises(AuthError) as excinfo:
        await multi.validate("tok")
    # All sources consulted; last error surfaces.
    assert "B_FAILED" in str(excinfo.value)
    assert a.calls == 1
    assert b.calls == 1


def test_empty_source_list_rejected_at_construction():
    with pytest.raises(ValueError, match="at least one source"):
        MultiSourceAuth([])
