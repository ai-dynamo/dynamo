# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``MultiSourceAuth`` — fan-out composition of AuthValidators.

Tries each configured source in order; the first to return an
``AuthIdentity`` wins. Short-circuit on first success — subsequent
sources are not consulted, which is critical when a later source would
incur network I/O (e.g. K8s TokenReview).

On total failure, raises ``AuthError`` with the *last* underlying
failure message. Callers (the RPC layer) log the chained message for
server-side audit but only surface ``reject_reason="auth_failed"`` to
the client.
"""

from __future__ import annotations

from typing import Sequence

from dynamo.planner.plugins.registry.auth.base import (
    AuthIdentity,
    AuthValidator,
)
from dynamo.planner.plugins.registry.errors import AuthError


class MultiSourceAuth(AuthValidator):
    def __init__(self, sources: Sequence[AuthValidator]) -> None:
        if not sources:
            raise ValueError(
                "MultiSourceAuth requires at least one source; "
                "empty list would silently reject every token."
            )
        self._sources: list[AuthValidator] = list(sources)

    async def validate(self, token: str) -> AuthIdentity:
        last_err: Exception | None = None
        for source in self._sources:
            try:
                return await source.validate(token)
            except AuthError as exc:
                last_err = exc
                continue
        # All sources rejected — raise with chained context (server log only).
        raise AuthError(
            f"all {len(self._sources)} auth source(s) rejected token; "
            f"last error: {last_err}"
        )


__all__ = ["MultiSourceAuth"]
