# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise the frontend entrypoint's explicit parsed-config binding transport."""

import argparse
import asyncio
import os
from unittest.mock import AsyncMock, Mock

import pytest

from dynamo.frontend import main as frontend
from dynamo.frontend.frontend_args import FrontendArgGroup, FrontendConfig
from dynamo.llm import EngineType, EntrypointArgs

pytestmark = [pytest.mark.pre_merge, pytest.mark.unit, pytest.mark.gpu_0]


@pytest.mark.parametrize("gateway_url", [None, "http://configured-batch"])
def test_native_entrypoint_accepts_explicit_batch_gateway(
    monkeypatch: pytest.MonkeyPatch, gateway_url: str | None
) -> None:
    monkeypatch.setenv("DYN_BATCH_GATEWAY_URL", "http://later-environment")
    assert isinstance(
        EntrypointArgs(EngineType.Dynamic, batch_gateway_url=gateway_url),
        EntrypointArgs,
    )


@pytest.mark.parametrize("gateway_url", [None, "http://configured-batch"])
def test_frontend_passes_batch_gateway_explicitly(
    monkeypatch: pytest.MonkeyPatch, gateway_url: str | None
) -> None:
    for name in (
        "DYN_BATCH_GATEWAY_URL",
        "DYN_INTERACTIVE",
        "DYN_KSERVE_GRPC_SERVER",
        "DYN_FRONTEND_ROUTE_EXTENSIONS",
        "DYN_ACTIVE_DECODE_BLOCKS_THRESHOLD",
        "DYN_ACTIVE_PREFILL_TOKENS_THRESHOLD",
        "DYN_ACTIVE_PREFILL_TOKENS_THRESHOLD_FRAC",
        "DYN_SYSTEM_PORT",
        "DYN_ROUTER_MIN_INITIAL_WORKERS",
        "NATS_TLS_INSECURE",
    ):
        monkeypatch.delenv(name, raising=False)
    parser = argparse.ArgumentParser()
    FrontendArgGroup().add_arguments(parser)
    config = FrontendConfig.from_cli_args(parser.parse_args([]))
    config.batch_gateway_url = gateway_url
    config.validate()

    # Change the environment AFTER parsing: the binding must receive the parsed
    # value (including explicit None), not re-read a different environment value.
    monkeypatch.setenv("DYN_BATCH_GATEWAY_URL", "http://later-environment")
    monkeypatch.setattr(frontend, "parse_args", lambda: (config, None, None))
    monkeypatch.setattr(frontend, "_export_transport_tls_env", Mock())
    monkeypatch.setattr(frontend, "dump_config", Mock())
    monkeypatch.setattr(
        frontend, "warn_if_frontend_cpu_affinity_spans_numa_nodes", Mock()
    )
    monkeypatch.setattr(frontend, "DistributedRuntime", Mock())
    monkeypatch.setattr(frontend, "build_router_config", Mock(return_value=None))
    entrypoint_args = Mock()
    monkeypatch.setattr(frontend, "EntrypointArgs", entrypoint_args)
    monkeypatch.setattr(frontend, "make_engine", AsyncMock())
    run_input = AsyncMock()
    monkeypatch.setattr(frontend, "run_input", run_input)

    async def run_frontend() -> None:
        loop = asyncio.get_running_loop()
        monkeypatch.setattr(loop, "add_signal_handler", Mock())
        await frontend.async_main()

    asyncio.run(run_frontend())

    assert entrypoint_args.call_args.kwargs["batch_gateway_url"] == gateway_url
    assert os.environ["DYN_BATCH_GATEWAY_URL"] == "http://later-environment"
    assert run_input.call_args.args[1] == "http"
