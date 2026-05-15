# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for transport config + factories."""

from __future__ import annotations

import os

import pytest

from dynamo.planner.plugins.clock import VirtualClock, WallClock
from dynamo.planner.plugins.transport import (
    GrpcTransport,
    InProcessTransport,
)
from dynamo.planner.plugins.transport.config import (
    ClockConfig,
    GrpcMtlsConfig,
    TransportConfig,
    make_clock,
    make_transport_for_endpoint,
)

pytestmark = [
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
    pytest.mark.unit,
    pytest.mark.planner,
]


class _StubPlugin:
    async def Predict(self, req):
        return req


# ----- TransportConfig defaults -----


def test_transport_config_defaults():
    c = TransportConfig()
    assert c.allow_insecure_grpc is False
    assert c.grpc_mtls is None
    assert c.request_timeout_seconds == 5.0


def test_transport_config_extra_forbid():
    """v11 M-5 spirit: unknown fields rejected."""
    with pytest.raises(Exception, match="extra"):
        TransportConfig(unknown_field="x")  # type: ignore[call-arg]


# ----- make_transport_for_endpoint dispatch -----


def test_factory_inproc_with_instance():
    t = make_transport_for_endpoint("p1", "inproc://p1", TransportConfig(), in_process_instance=_StubPlugin())
    assert isinstance(t, InProcessTransport)
    assert t.plugin_id == "p1"
    assert t.endpoint == "inproc://p1"


def test_factory_inproc_without_instance_rejected():
    with pytest.raises(ValueError, match="in_process_instance required"):
        make_transport_for_endpoint("p1", "inproc://p1", TransportConfig())


def test_factory_unix_rejected_unknown_scheme():
    """``unix://`` was dropped from PR #1; only ``inproc://`` + ``grpc://``
    are accepted. Lock the new rejection contract."""
    with pytest.raises(ValueError, match="unknown endpoint scheme"):
        make_transport_for_endpoint("p2", "unix:///tmp/x.sock", TransportConfig())


def test_factory_grpc_default_refuses_insecure():
    with pytest.raises(ValueError, match="requires mTLS or"):
        make_transport_for_endpoint("p3", "grpc://host:9090", TransportConfig())


def test_factory_grpc_with_allow_insecure():
    cfg = TransportConfig(allow_insecure_grpc=True)
    t = make_transport_for_endpoint("p3", "grpc://host:9090", cfg)
    assert isinstance(t, GrpcTransport)


def test_factory_unknown_scheme():
    with pytest.raises(ValueError, match="unknown endpoint scheme"):
        make_transport_for_endpoint("p", "tcp://nope", TransportConfig())


def test_factory_propagates_request_timeout():
    cfg = TransportConfig(request_timeout_seconds=12.5)
    t = make_transport_for_endpoint("p", "inproc://p", cfg, in_process_instance=_StubPlugin())
    assert t.timeout_seconds == 12.5


# ----- Clock factory + production safety -----


def test_make_clock_wall():
    c = make_clock(ClockConfig())
    assert isinstance(c, WallClock)


def test_make_clock_virtual_rejected_in_production(monkeypatch):
    monkeypatch.delenv("DYNAMO_PLANNER_TEST", raising=False)
    with pytest.raises(ValueError, match="DYNAMO_PLANNER_TEST=1"):
        make_clock(ClockConfig(type="virtual"))


def test_make_clock_virtual_allowed_in_test_mode(monkeypatch):
    monkeypatch.setenv("DYNAMO_PLANNER_TEST", "1")
    c = make_clock(ClockConfig(type="virtual", virtual_start_now=42.0))
    assert isinstance(c, VirtualClock)
    assert c.now() == 42.0


def test_make_clock_unknown_type():
    """Pydantic Literal validates type field at construction; ValueError early."""
    with pytest.raises(Exception):  # Pydantic ValidationError
        ClockConfig(type="invalid")  # type: ignore[arg-type]
