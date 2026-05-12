# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Registry error hierarchy.

All errors raised by ``PluginRegistryServer`` and its auth validators
inherit ``RegistryError``. Orchestrator callers catch the base class for
generic handling or a specific subclass when the reason matters for
audit / metric labels.

``AuthError`` deliberately uses a generic ``reject_reason="auth_failed"``
at the RPC boundary — the *specific* failure mode is captured in the
exception message for server-side audit only, to avoid giving token
oracles to clients.
"""

from __future__ import annotations


class RegistryError(Exception):
    """Base class for all PluginRegistry errors."""


class AuthError(RegistryError):
    """Auth validation failed.

    Raised by ``AuthValidator.validate``; the RPC layer converts this into
    ``RegisterResponse(accepted=False, reject_reason="auth_failed")``
    without leaking the specific reason to the client.
    """


class ProtocolVersionError(RegistryError):
    """Requested ``protocol_version`` outside the server's supported range."""


class DuplicatePluginIdError(RegistryError):
    """``plugin_id`` already registered.

    The server rejects duplicates rather than upserting — clients must
    explicitly ``Unregister`` + ``Register`` for version upgrades so
    cache invalidation is deterministic.
    """
