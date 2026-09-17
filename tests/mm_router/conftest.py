# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared fixtures for multimodal router integration tests."""

import pytest

from tests.conftest import EtcdServer, NatsServer


@pytest.fixture(scope="module")
def mm_runtime_services(request):
    """Run isolated discovery services and restore the caller's environment."""
    with (
        NatsServer(request, port=0) as nats,
        EtcdServer(request, port=0) as etcd,
        pytest.MonkeyPatch.context() as monkeypatch,
    ):
        monkeypatch.setenv("NATS_SERVER", f"nats://localhost:{nats.port}")
        monkeypatch.setenv("ETCD_ENDPOINTS", f"http://localhost:{etcd.port}")
        yield
