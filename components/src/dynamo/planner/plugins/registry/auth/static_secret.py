# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``StaticSecretAuth`` — constant-time lookup against a configured secrets map.

The v1 must-have validator: a shared-secret scheme backed by a K8s Secret
mount (or equivalent). Keys are secret values; values are caller labels
(e.g. ``"shared-team-a"``) returned via ``AuthIdentity.subject`` for
audit. Uses ``hmac.compare_digest`` to avoid timing side channels when
ruling out non-matches.
"""

from __future__ import annotations

import hmac
from typing import Mapping

from dynamo.planner.plugins.registry.auth.base import (
    AuthIdentity,
    AuthValidator,
)
from dynamo.planner.plugins.registry.errors import AuthError


class StaticSecretAuth(AuthValidator):
    """Validate tokens by exact-match against a pre-shared secrets map.

    Args:
        secrets: mapping of ``secret_value -> subject_label``. An empty
            mapping is accepted at construction time (so an empty Secret
            mount doesn't crash startup) but every ``validate`` call will
            then raise ``AuthError`` — the registry effectively rejects
            all tokens, which is the correct fail-closed behaviour.
    """

    def __init__(self, secrets: Mapping[str, str]) -> None:
        self._secrets: dict[str, str] = dict(secrets)

    async def validate(self, token: str) -> AuthIdentity:
        if not token:
            raise AuthError("static_secret: empty token")
        # Constant-time comparison avoids leaking "first N chars matched"
        # via timing. Python dict lookup is fast-path but not constant;
        # for N small secrets, iterating + compare_digest is fine.
        for secret, subject in self._secrets.items():
            if hmac.compare_digest(token, secret):
                return AuthIdentity(
                    source="static_secret", subject=subject
                )
        raise AuthError("static_secret: token not in trusted set")


__all__ = ["StaticSecretAuth"]
