# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
import asyncio
import signal
from unittest.mock import AsyncMock, Mock

import pytest

from dynamo.frontend import main as frontend
from dynamo.frontend.frontend_args import FrontendArgGroup, FrontendConfig

pytestmark = [
    pytest.mark.unit,
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
    pytest.mark.timeout(10),
]


@pytest.mark.parametrize("serving_failed", [False, True])
def test_shutdown_failure_preserves_serving_error(monkeypatch, serving_failed):
    # Regression: joining failed shutdown in finally replaces the serving error,
    # obscuring the failure that caused the frontend to stop.
    parser = argparse.ArgumentParser()
    FrontendArgGroup().add_arguments(parser)
    config = FrontendConfig.from_cli_args(parser.parse_args([]))
    monkeypatch.setattr(frontend, "parse_args", lambda: (config, None, None))
    for name in (
        "dump_config",
        "_export_transport_tls_env",
        "warn_if_frontend_cpu_affinity_spans_numa_nodes",
        "DistributedRuntime",
        "EntrypointArgs",
    ):
        monkeypatch.setattr(frontend, name, Mock())
    monkeypatch.setattr(frontend, "make_engine", AsyncMock())
    monkeypatch.setenv("DYN_ROUTER_MIN_INITIAL_WORKERS", "0")
    monkeypatch.delenv("DYN_SYSTEM_PORT", raising=False)
    serving_error = ValueError("serving failed")
    shutdown_error = RuntimeError("shutdown failed")
    monkeypatch.setattr(
        frontend, "graceful_shutdown", AsyncMock(side_effect=shutdown_error)
    )

    async def scenario():
        handlers = {}
        monkeypatch.setattr(
            asyncio.get_running_loop(),
            "add_signal_handler",
            lambda sig, cb: handlers.update({sig: cb}),
        )

        async def serve(*args):
            handlers[signal.SIGTERM]()
            if serving_failed:
                raise serving_error

        monkeypatch.setattr(frontend, "run_input", serve)
        expected = serving_error if serving_failed else shutdown_error
        with pytest.raises(type(expected)) as caught:
            await frontend.async_main()
        assert caught.value is expected

    asyncio.run(scenario())
