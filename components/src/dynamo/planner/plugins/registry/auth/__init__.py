# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pluggable auth validators for PluginRegistry.

Validator hierarchy::

    AuthValidator (ABC)
    ├── StaticSecretAuth        # shared-secret map
    ├── MultiSourceAuth         # fan-out across the above
    └── AllowUnauthenticatedAuth  # DEV ONLY; emits WARNING on init

Wired in ``registry/config.py``'s ``build_auth_validator``. Selection is
per-deployment config; PR #1 ships ``static_secret`` + the dev bypass.
K8s ServiceAccount tokens and SPIFFE JWT-SVIDs land in a follow-up PR.
"""

from dynamo.planner.plugins.registry.auth.base import (
    AllowUnauthenticatedAuth,
    AuthIdentity,
    AuthValidator,
)
from dynamo.planner.plugins.registry.auth.multi import MultiSourceAuth
from dynamo.planner.plugins.registry.auth.static_secret import StaticSecretAuth

__all__ = [
    "AuthValidator",
    "AuthIdentity",
    "StaticSecretAuth",
    "MultiSourceAuth",
    "AllowUnauthenticatedAuth",
]
