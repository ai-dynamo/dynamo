# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for registry config + factories."""

from __future__ import annotations

import logging

import pytest
from pydantic import ValidationError

from dynamo.planner.plugins.clock import VirtualClock
from dynamo.planner.plugins.registry.auth import (
    AllowUnauthenticatedAuth,
    MultiSourceAuth,
    StaticSecretAuth,
)
from dynamo.planner.plugins.registry.config import (
    AuthConfig,
    InProcessPluginSpec,
    PluginRegistrationConfig,
    build_auth_validator,
    build_registry_from_config,
)
from dynamo.planner.plugins.registry.server import PluginRegistryServer

pytestmark = [
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
    pytest.mark.unit,
    pytest.mark.planner,
]


# ---------------------------------------------------------------------------
# build_auth_validator
# ---------------------------------------------------------------------------


def test_empty_trusted_sources_rejected():
    with pytest.raises(ValueError, match="trusted_sources"):
        build_auth_validator(AuthConfig())


def test_static_secret_only_builds_multi_with_one_source():
    validator = build_auth_validator(
        AuthConfig(trusted_sources=["static_secret"], static_secrets={"t": "a"})
    )
    assert isinstance(validator, MultiSourceAuth)


def test_static_secret_empty_secrets_logs_warning(caplog):
    with caplog.at_level(logging.WARNING, logger="dynamo.planner.plugins.registry.config"):
        build_auth_validator(AuthConfig(trusted_sources=["static_secret"], static_secrets={}))
    assert any("static_secrets is empty" in r.message for r in caplog.records)


def test_allow_unauthenticated_source_supported():
    validator = build_auth_validator(
        AuthConfig(trusted_sources=["allow_unauthenticated"])
    )
    assert isinstance(validator, MultiSourceAuth)


@pytest.mark.asyncio
async def test_multi_source_preserves_order():
    validator = build_auth_validator(
        AuthConfig(
            trusted_sources=["static_secret", "allow_unauthenticated"],
            static_secrets={"good": "alice"},
        )
    )
    # Unknown token falls through to allow_unauthenticated (anonymous).
    identity = await validator.validate("unknown")
    assert identity.source == "allow_unauthenticated"
    # Known token is accepted by the first source.
    identity2 = await validator.validate("good")
    assert identity2.source == "static_secret"


def test_unknown_source_rejected_by_pydantic():
    """``AuthSource`` is a Literal restricted to the sources PR #1 ships
    (``static_secret`` / ``allow_unauthenticated``). Anything else —
    including the follow-up ``k8s_sa`` / ``spiffe_jwt`` — fails Pydantic
    validation before reaching ``build_auth_validator``."""
    with pytest.raises(ValidationError):
        AuthConfig(trusted_sources=["k8s_sa"])  # type: ignore[list-item]


# ---------------------------------------------------------------------------
# build_registry_from_config
# ---------------------------------------------------------------------------


def test_build_registry_from_config_returns_server_and_breaker():
    config = PluginRegistrationConfig(
        auth=AuthConfig(trusted_sources=["static_secret"], static_secrets={"t": "a"}),
    )
    clock = VirtualClock()
    server, cb = build_registry_from_config(config, clock)
    assert isinstance(server, PluginRegistryServer)
    # Circuit breaker returned so orchestrator can hand it to scheduler / monitor.
    assert cb is not None


@pytest.mark.asyncio
async def test_build_registry_propagates_protocol_versions():
    from dynamo.planner.plugins.types import RegisterRequest

    from dynamo.planner.plugins.transport.config import TransportConfig

    config = PluginRegistrationConfig(
        auth=AuthConfig(trusted_sources=["allow_unauthenticated"]),
        protocol_version_min="1.0",
        protocol_version_max="1.2",
        # PR #1 dropped unix:// transport; tests now use grpc:// stub
        # endpoints, which require allow_insecure_grpc=True.
        transport=TransportConfig(allow_insecure_grpc=True),
    )
    clock = VirtualClock()
    server, _ = build_registry_from_config(config, clock)
    # v1.1 is in range [1.0, 1.2] — accepted.
    resp = await server.register(
        RegisterRequest(
            plugin_id="p",
            plugin_type="propose",
            endpoint="grpc://127.0.0.1:9000",
            protocol_version="1.1",
        )
    )
    assert resp.accepted, resp.reject_reason


# ---------------------------------------------------------------------------
# InProcessPluginSpec
# ---------------------------------------------------------------------------


def test_in_process_plugin_spec_rejects_unknown_field_protocol_version():
    # In-process plugins are compile-time bound; protocol_version is
    # nonsensical and should be rejected by extra='forbid'.
    with pytest.raises(ValidationError):
        InProcessPluginSpec(
            module="x",
            **{"class": "Y"},
            plugin_id="p",
            plugin_type="propose",
            priority=1,
            protocol_version="1.0",  # type: ignore[call-arg]
        )


def test_in_process_plugin_spec_class_alias_works():
    spec = InProcessPluginSpec.model_validate({
        "module": "dynamo.example",
        "class": "MyPlugin",
        "plugin_id": "mp",
        "plugin_type": "predict",
        "priority": 5,
    })
    assert spec.class_ == "MyPlugin"
    assert spec.module == "dynamo.example"


def test_in_process_plugin_spec_defaults_reasonable():
    spec = InProcessPluginSpec.model_validate({
        "module": "x",
        "class": "Y",
        "plugin_id": "p",
        "plugin_type": "propose",
        "priority": 1,
    })
    assert spec.hold_policy == "ACCEPT_WHEN_IDLE"
    assert spec.execution_interval_seconds == 0.0
    assert spec.kwargs == {}
