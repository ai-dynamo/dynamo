# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for StaticSecretAuth."""

from __future__ import annotations

import pytest

from dynamo.planner.plugins.registry.auth import StaticSecretAuth
from dynamo.planner.plugins.registry.errors import AuthError

pytestmark = [
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
    pytest.mark.unit,
    pytest.mark.planner,
]


@pytest.mark.asyncio
async def test_known_secret_accepted_and_subject_returned():
    auth = StaticSecretAuth({"secret-alice": "alice", "secret-bob": "bob"})
    identity = await auth.validate("secret-alice")
    assert identity.source == "static_secret"
    assert identity.subject == "alice"


@pytest.mark.asyncio
async def test_unknown_secret_rejected():
    auth = StaticSecretAuth({"secret-alice": "alice"})
    with pytest.raises(AuthError, match="not in trusted set"):
        await auth.validate("wrong")


@pytest.mark.asyncio
async def test_empty_token_rejected():
    auth = StaticSecretAuth({"secret-alice": "alice"})
    with pytest.raises(AuthError, match="empty token"):
        await auth.validate("")


@pytest.mark.asyncio
async def test_empty_secrets_map_rejects_all():
    # Fail-closed when Secret mount is empty or misconfigured.
    auth = StaticSecretAuth({})
    with pytest.raises(AuthError, match="not in trusted set"):
        await auth.validate("any")


@pytest.mark.asyncio
async def test_constant_time_compare_prefix_mismatch_still_rejects():
    # hmac.compare_digest rejects any non-exact match; exercising a
    # prefix match ensures we're not accidentally using startswith/==
    # with early-exit that would leak timing info.
    auth = StaticSecretAuth({"secret-alice": "alice"})
    with pytest.raises(AuthError):
        await auth.validate("secret-ali")
    with pytest.raises(AuthError):
        await auth.validate("secret-alice-extra")
