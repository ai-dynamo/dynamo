# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for AllowUnauthenticatedAuth.

The validator is dev-only. Construction must log a WARNING so operators
see it even if no Register call ever arrives.
"""

from __future__ import annotations

import logging

import pytest

from dynamo.planner.plugins.registry.auth import AllowUnauthenticatedAuth

pytestmark = [
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
    pytest.mark.unit,
    pytest.mark.planner,
]


def test_construction_emits_warning(caplog):
    with caplog.at_level(logging.WARNING, logger="dynamo.planner.plugins.registry.auth.base"):
        AllowUnauthenticatedAuth()
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert any("DEV ONLY" in r.message for r in warnings)


@pytest.mark.asyncio
async def test_accepts_any_token_including_empty():
    auth = AllowUnauthenticatedAuth()
    identity = await auth.validate("anything")
    assert identity.source == "allow_unauthenticated"
    assert identity.subject == "anonymous"

    identity_empty = await auth.validate("")
    assert identity_empty.source == "allow_unauthenticated"
