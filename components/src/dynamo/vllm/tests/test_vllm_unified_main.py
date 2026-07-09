# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

pytest.importorskip("vllm.v1.engine.async_llm")
pytest.importorskip("vllm.usage.usage_lib")

import dynamo.vllm.unified_main as unified_main  # noqa: E402
from dynamo.common.constants import DisaggregationMode  # noqa: E402
from dynamo.vllm.llm_engine import VllmLLMEngine  # noqa: E402

pytestmark = [
    pytest.mark.unit,
    pytest.mark.vllm,
    pytest.mark.core,
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
]


def test_main_routes_headless_to_run_dynamo_headless():
    """A headless secondary node must bypass the Worker/engine path entirely
    and hand off to vLLM's run_headless via run_dynamo_headless."""
    config = SimpleNamespace(headless=True)
    with (
        patch.object(unified_main, "parse_args", return_value=config) as parse_args,
        patch.object(unified_main, "run_dynamo_headless") as run_headless,
        patch.object(unified_main, "run") as run,
    ):
        unified_main.main()

    parse_args.assert_called_once_with(fpm_trace_relay_supported=False)
    run_headless.assert_called_once_with(config)
    run.assert_not_called()


@pytest.mark.asyncio
async def test_main_routes_normal_node_to_run():
    """A non-headless node drives the unified Worker via run(VllmLLMEngine)."""
    config = SimpleNamespace(
        headless=False, disaggregation_mode=DisaggregationMode.AGGREGATED
    )
    with (
        patch.object(unified_main, "parse_args", return_value=config) as parse_args,
        patch.object(unified_main, "run_dynamo_headless") as run_headless,
        patch.object(unified_main, "run") as run,
    ):
        unified_main.main()

    parse_args.assert_called_once_with(fpm_trace_relay_supported=False)
    run_headless.assert_not_called()
    run.assert_called_once()
    assert run.call_args.args == (unified_main.VllmLLMEngine,)

    engine_factory = run.call_args.kwargs["engine_factory"]
    expected = (MagicMock(), MagicMock())
    with patch.object(
        unified_main.VllmLLMEngine,
        "from_args",
        new_callable=AsyncMock,
        return_value=expected,
    ) as from_args:
        assert await engine_factory(["--model", "test-model"]) == expected

    from_args.assert_awaited_once_with(["--model", "test-model"], config=config)


@pytest.mark.asyncio
async def test_main_routes_encode_node_to_dedicated_engine():
    config = SimpleNamespace(
        headless=False, disaggregation_mode=DisaggregationMode.ENCODE
    )
    with (
        patch.object(unified_main, "parse_args", return_value=config),
        patch.object(unified_main, "run") as run,
    ):
        unified_main.main()

    assert run.call_args.args == (unified_main.VllmEncodeEngine,)
    engine_factory = run.call_args.kwargs["engine_factory"]
    expected = (MagicMock(), MagicMock())
    with patch.object(
        unified_main.VllmEncodeEngine,
        "from_args",
        new_callable=AsyncMock,
        return_value=expected,
    ) as from_args:
        assert await engine_factory([]) == expected
    from_args.assert_awaited_once_with([], config=config)


@pytest.mark.asyncio
async def test_from_args_rejects_headless_reaching_engine_path():
    """Defense in depth: if a headless config reaches from_args (run() driven
    directly, bypassing unified_main), fail loud instead of booting a full
    backend on a workers-only node."""
    config = SimpleNamespace(
        headless=True,
        disaggregation_mode=DisaggregationMode.AGGREGATED,
    )
    with patch("dynamo.vllm.llm_engine.parse_args", return_value=config) as parse_args:
        with pytest.raises(NotImplementedError, match="headless"):
            await VllmLLMEngine.from_args([])

    parse_args.assert_called_once_with([], fpm_trace_relay_supported=False)
