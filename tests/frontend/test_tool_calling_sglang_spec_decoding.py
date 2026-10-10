# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tool calling against an SGLang worker running EAGLE3 speculative decoding.

Reruns the tool-calling suite from ``test_tool_calling_sglang.py`` against a
worker with an EAGLE3 draft model, then checks the worker's speculative
decoding counters to confirm speculation was active.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Generator

import pytest
import requests

from tests.frontend.test_tool_calling_sglang import (  # noqa: F401
    MODEL_NAME,
    TOOLS_WEATHER,
    OpenAI,
    TestToolCallingMultiTurn,
    TestToolCallingProtocol,
    ToolCallingFrontendProcess,
    WorkerProcess,
    _cleanup_sglang_stragglers,
    assert_finish_reason,
    model,
    parse_and_validate_tool_call,
    runtime_services,
    stream_chat,
    tool_schema_map,
)
from tests.utils.constants import DynamoPortRange
from tests.utils.payloads import SGLangSpecDecodeMetricsPayload
from tests.utils.port_utils import allocate_port, deallocate_ports

DRAFT_MODEL_NAME = "GavinLucky/SGLang-EAGLE3-Qwen3-0.6B-SpecForge"

pytestmark = [
    pytest.mark.sglang,
    pytest.mark.core,
    pytest.mark.e2e,
    pytest.mark.gpu_1,
    pytest.mark.integration,
    pytest.mark.model(MODEL_NAME),
    pytest.mark.model(DRAFT_MODEL_NAME),
    pytest.mark.timeout(300),
]

EAGLE3_ARGS = (
    "--page-size",
    "16",
    "--speculative-algorithm",
    "EAGLE3",
    "--speculative-draft-model-path",
    DRAFT_MODEL_NAME,
    "--speculative-num-steps",
    "3",
    "--speculative-eagle-topk",
    "1",
    "--speculative-num-draft-tokens",
    "4",
    # Exposes the sglang:spec_* counters on the worker system port.
    "--enable-metrics",
)


@dataclass(frozen=True)
class SpecDecodingStack:
    frontend_port: int
    system_port: int


@pytest.fixture(scope="module")
def spec_decoding_stack(
    request, runtime_services, predownload_models  # noqa: F811
) -> Generator[SpecDecodingStack, None, None]:
    """Start an EAGLE3 SGLang worker with worker-side parsers behind a plain frontend."""
    allocated_ports: list[int] = []
    try:
        system_port = allocate_port(DynamoPortRange.SERVE.value)
        allocated_ports.append(system_port)
        fpm_port = allocate_port(DynamoPortRange.FPM.value)
        allocated_ports.append(fpm_port)

        with WorkerProcess(
            request,
            system_port=system_port,
            fpm_port=fpm_port,
            topology="rust_parsers",
            extra_args=EAGLE3_ARGS,
        ):
            time.sleep(2)
            frontend_port = allocate_port(DynamoPortRange.FRONTEND.value)
            allocated_ports.append(frontend_port)
            with ToolCallingFrontendProcess(
                request, frontend_port=frontend_port, topology="rust_parsers"
            ):
                yield SpecDecodingStack(
                    frontend_port=frontend_port, system_port=system_port
                )
    finally:
        try:
            _cleanup_sglang_stragglers()
            time.sleep(3)
        finally:
            deallocate_ports(allocated_ports)


@pytest.fixture(scope="module")
def client(spec_decoding_stack: SpecDecodingStack) -> OpenAI:
    return OpenAI(
        api_key="EMPTY",
        base_url=f"http://localhost:{spec_decoding_stack.frontend_port}/v1",
    )


class TestToolCallingSpecDecodingMetrics:
    def test_speculation_active_during_tool_calls(
        self, spec_decoding_stack: SpecDecodingStack, client: OpenAI
    ):
        # Drive a streaming and a non-streaming tool call here so the metric
        # check does not depend on which other tests ran first.
        schema = tool_schema_map(TOOLS_WEATHER)
        streamed = stream_chat(
            client,
            MODEL_NAME,
            messages=[{"role": "user", "content": "What's the weather in Seoul?"}],
            tools=TOOLS_WEATHER,
            temperature=0,
            seed=0,
        )
        assert_finish_reason(streamed, {"tool_calls"})
        parse_and_validate_tool_call(
            streamed.tool_calls[0], schema, expected_name="get_weather"
        )

        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": "What's the weather in Madrid?"}],
            tools=TOOLS_WEATHER,
            max_tokens=4096,
            temperature=0,
            seed=0,
        )
        assert response.choices[0].finish_reason == "tool_calls"
        parse_and_validate_tool_call(
            response.choices[0].message.tool_calls[0].model_dump(),
            schema,
            expected_name="get_weather",
        )

        metrics = requests.get(
            f"http://localhost:{spec_decoding_stack.system_port}/metrics",
            timeout=10,
        )
        metrics.raise_for_status()
        SGLangSpecDecodeMetricsPayload(
            body={},
            expected_response=[],
            expected_log=[],
            port=spec_decoding_stack.system_port,
            min_num_requests=2,
        ).validate(metrics, metrics.text)
