# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pluggable auth validators for PluginRegistry.

Validator hierarchy::

    AuthValidator (ABC)
    ├── StaticSecretAuth        # shared-secret map
    ├── MultiSourceAuth         # fan-out across the above
    ├── AllowUnauthenticatedAuth  # DEV ONLY; emits WARNING on init
    ├── K8sSATokenAuth          # Kubernetes ServiceAccount TokenReview
    └── SpiffeJwtAuth           # SPIFFE JWT-SVID via SPIRE JWKS

All five are wired in ``registry/config.py``'s ``build_auth_validator``.
Selection is per-deployment config — single- and multi-tenant clusters
typically compose ``static_secret`` + one of ``k8s_sa`` / ``spiffe_jwt``
via ``MultiSourceAuth``.
"""

from dynamo.planner.plugins.registry.auth.base import (
    AllowUnauthenticatedAuth,
    AuthIdentity,
    AuthValidator,
)
from dynamo.planner.plugins.registry.auth.k8s_sa_token import K8sSATokenAuth
from dynamo.planner.plugins.registry.auth.multi import MultiSourceAuth
from dynamo.planner.plugins.registry.auth.spiffe_jwt import SpiffeJwtAuth
from dynamo.planner.plugins.registry.auth.static_secret import StaticSecretAuth

__all__ = [
    "AuthValidator",
    "AuthIdentity",
    "StaticSecretAuth",
    "MultiSourceAuth",
    "AllowUnauthenticatedAuth",
    "K8sSATokenAuth",
    "SpiffeJwtAuth",
]
