# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``AuthValidator`` ABC + ``AuthIdentity`` record + dev-only open validator.

On validation failure, validators MUST raise ``AuthError``; the RPC layer
converts that to a generic ``RegisterResponse(accepted=False,
reject_reason="auth_failed")`` — never propagate the specific failure to
the client, to avoid giving a token oracle.
"""

from __future__ import annotations

import abc
import logging
from dataclasses import dataclass, field
from typing import Literal

from dynamo.planner.plugins.registry.errors import AuthError

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class AuthIdentity:
    """Outcome of a successful ``AuthValidator.validate`` call.

    ``source`` names the validator that accepted the token (useful for
    audit logs + metrics). ``subject`` is the caller identity (static
    label / K8s ``namespace/serviceaccount`` / SPIFFE ID).
    ``metadata`` may carry validator-specific extras (e.g. ``audience``).
    """

    source: Literal["static_secret", "allow_unauthenticated"]
    subject: str
    metadata: dict[str, str] = field(default_factory=dict)


class AuthValidator(abc.ABC):
    """Validate a plugin-supplied auth token.

    Implementations MUST be async to permit out-of-process validation
    (e.g. K8s TokenReview API); pure in-memory validators (static_secret)
    still satisfy this by returning from a coroutine without awaiting.
    """

    @abc.abstractmethod
    async def validate(self, token: str) -> AuthIdentity:
        """Return an ``AuthIdentity`` on success; raise ``AuthError`` on
        any validation failure."""
        raise NotImplementedError


class AllowUnauthenticatedAuth(AuthValidator):
    """Dev-only bypass validator: accepts any token.

    Emits a WARNING on construction so operators see it in logs even if
    the registry never receives a real Register call. Production startup
    scripts are expected to grep for this log line and refuse to bring up
    the planner Pod when the validator is enabled outside dev.
    """

    def __init__(self) -> None:
        log.warning(
            "AllowUnauthenticatedAuth enabled — ALL Register requests will "
            "be accepted without auth. DEV ONLY; MUST NOT run in production."
        )

    async def validate(self, token: str) -> AuthIdentity:
        return AuthIdentity(source="allow_unauthenticated", subject="anonymous")


__all__ = [
    "AuthValidator",
    "AuthIdentity",
    "AllowUnauthenticatedAuth",
    "AuthError",
]
