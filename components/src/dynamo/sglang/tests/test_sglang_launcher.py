# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
from unittest.mock import Mock

import pytest

from dynamo.sglang import sidecar

pytestmark = [
    pytest.mark.unit,
    pytest.mark.gpu_0,
    pytest.mark.sglang,
    pytest.mark.sidecar,
    pytest.mark.core,
    pytest.mark.pre_merge,
]


@pytest.mark.parametrize("configured_endpoint", [None, "http://127.0.0.1:31002"])
def test_managed_endpoint_fallback_preserves_explicit_configuration(
    monkeypatch, configured_endpoint
):
    injected_endpoint = "http://127.0.0.1:31001"
    monkeypatch.setenv("SGLANG_GRPC_ENDPOINT", injected_endpoint)
    monkeypatch.delenv("DYN_SIDECAR_GRPC_ENDPOINT", raising=False)
    if configured_endpoint is not None:
        monkeypatch.setenv("DYN_SIDECAR_GRPC_ENDPOINT", configured_endpoint)
    calls = []
    monkeypatch.setattr(sidecar, "configure_dynamo_logging", Mock())
    monkeypatch.setattr(
        sidecar._backend,
        "_run_sglang_sidecar",
        lambda argv: calls.append((argv, os.environ.get("DYN_SIDECAR_GRPC_ENDPOINT"))),
    )
    argv = ["--grpc-endpoint", "http://127.0.0.1:31003"] if configured_endpoint else []

    sidecar.main(argv)

    assert calls == [(argv, configured_endpoint or injected_endpoint)]
