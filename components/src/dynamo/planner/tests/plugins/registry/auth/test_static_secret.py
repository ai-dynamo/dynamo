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


def test_construction_rejects_empty_subject():
    """Empty subject is rejected at construction time so the gateway's
    ``authenticated_unregister`` subject-match check cannot be bypassed
    against in-process plugins (which default to ``auth_subject=""``).
    See ``static_secret.py`` constructor + ``server.authenticated_unregister``.
    """
    with pytest.raises(ValueError, match="empty subject"):
        StaticSecretAuth({"some-token": ""})


def test_construction_rejects_empty_subject_among_valid_entries():
    """A mixed mapping where only one entry has an empty subject still
    fails fast — fail-closed at config validation."""
    with pytest.raises(ValueError, match="empty subject"):
        StaticSecretAuth({"good-token": "ext-plugins", "bad-token": ""})


def test_construction_accepts_all_distinguishing_subjects():
    """Valid mapping (every secret → non-empty subject) constructs cleanly."""
    auth = StaticSecretAuth({"t1": "subj-a", "t2": "subj-b"})
    assert auth is not None


def test_empty_subject_error_does_not_leak_secret_bytes():
    """The empty-subject ValueError must not include any prefix of the
    secret token. Startup logs and config-validation surfaces can
    surface this message, so even a 4-char prefix is an unnecessary
    secret leak."""
    secret = "super-secret-token-please-do-not-log-me"
    with pytest.raises(ValueError, match="empty subject") as exc_info:
        StaticSecretAuth({secret: ""})
    msg = str(exc_info.value)
    # No substring of the token may appear in the error message. We
    # check at least the first 4 chars (the previous bug) plus a few
    # longer windows just to be sure.
    for window in (secret[:4], secret[:6], secret[:8], secret):
        assert window not in msg, f"error message leaks {window!r}: {msg!r}"
