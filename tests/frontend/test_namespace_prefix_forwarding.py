# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise CLI parsing and startup through the Rust argument boundary."""

import asyncio
import os
import sys
from unittest.mock import AsyncMock, Mock

import pytest

import dynamo.frontend.main as frontend_main

pytestmark = [pytest.mark.pre_merge, pytest.mark.unit, pytest.mark.gpu_0]


@pytest.mark.asyncio
@pytest.mark.timeout(30)
@pytest.mark.parametrize("prefix", [None, "", "ns"])
async def test_namespace_prefix_reaches_entrypoint(monkeypatch, prefix):
    for name in list(os.environ):
        if name.startswith("DYN_"):
            monkeypatch.delenv(name)
    monkeypatch.setenv(frontend_main.MIN_INITIAL_WORKERS_ENV, "1")
    argv = ["dynamo.frontend", "--namespace", "ns"]
    if prefix is not None:
        argv.extend(["--namespace-prefix", prefix])
    monkeypatch.setattr(sys, "argv", argv)

    monkeypatch.setattr(frontend_main, "DistributedRuntime", Mock())
    monkeypatch.setattr(frontend_main, "_export_transport_tls_env", Mock())
    monkeypatch.setattr(
        frontend_main, "warn_if_frontend_cpu_affinity_spans_numa_nodes", Mock()
    )
    monkeypatch.setattr(asyncio.get_running_loop(), "add_signal_handler", Mock())
    entrypoint = Mock()
    monkeypatch.setattr(frontend_main, "EntrypointArgs", entrypoint)
    monkeypatch.setattr(frontend_main, "make_engine", AsyncMock())
    monkeypatch.setattr(frontend_main, "run_input", AsyncMock())

    await frontend_main.async_main()

    entrypoint.assert_called_once()
    kwargs = entrypoint.call_args.kwargs
    assert kwargs["namespace"] == "ns"
    if prefix is None:
        assert "namespace_prefix" not in kwargs
    else:
        assert kwargs["namespace_prefix"] == prefix
