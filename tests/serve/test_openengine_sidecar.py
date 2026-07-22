# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Common frontend smoke coverage for every OpenEngine-backed engine server."""

import dataclasses
import os

import pytest

from tests.serve.common import (
    WORKSPACE_DIR,
    params_with_model_mark,
    run_serve_deployment,
)
from tests.utils.engine_process import EngineConfig
from tests.utils.payloads import ChatPayload

MODEL = "Qwen/Qwen3-0.6B"


class OpenEngineChatPayload(ChatPayload):
    """Require content, a terminal reason, and nonempty terminal usage."""

    def validate(self, response, content: str) -> None:
        super().validate(response, content)
        result = response.json()
        choice = result["choices"][0]
        assert choice.get("finish_reason"), f"missing terminal finish reason: {result}"
        usage = result.get("usage")
        assert (
            usage and usage.get("prompt_tokens", 0) > 0
        ), f"missing terminal prompt usage: {result}"
        assert (
            usage.get("completion_tokens", 0) > 0
        ), f"missing terminal completion usage: {result}"


def _payload() -> OpenEngineChatPayload:
    return OpenEngineChatPayload(
        body={
            "messages": [
                {"role": "user", "content": "Reply with one short color name."}
            ],
            "max_tokens": 8,
            "temperature": 0.0,
        },
        repeat_count=1,
        expected_response=[],
        expected_log=[],
        timeout=90,
    )


def _config(engine: str, topology: str) -> EngineConfig:
    gpu_count = 1 if topology == "aggregate" else 2
    directory = os.path.join(WORKSPACE_DIR, "examples", "backends", engine, "launch")
    framework_mark = {
        "vllm": pytest.mark.vllm,
        "sglang": pytest.mark.sglang,
        "trtllm": pytest.mark.trtllm,
    }[engine]
    return EngineConfig(
        name=f"openengine-{engine}-{topology}",
        directory=directory,
        script_name=(
            "openengine_agg.sh" if topology == "aggregate" else "openengine_disagg.sh"
        ),
        marks=[
            pytest.mark.integration,
            pytest.mark.nightly,
            pytest.mark.h100,
            pytest.mark.gpu_1 if gpu_count == 1 else pytest.mark.gpu_2,
            pytest.mark.timeout(480),
            framework_mark,
        ],
        model=MODEL,
        env={"MODEL_PATH": MODEL, "DYN_LORA_ENABLED": "false"},
        request_payloads=[_payload()],
        timeout=480,
    )


OPENENGINE_CONFIGS = {
    f"{engine}-{topology}": _config(engine, topology)
    for engine in ("vllm", "sglang", "trtllm")
    for topology in ("aggregate", "disaggregated")
}


@pytest.fixture(params=params_with_model_mark(OPENENGINE_CONFIGS))
def openengine_config(request):
    return OPENENGINE_CONFIGS[request.param]


@pytest.mark.e2e
@pytest.mark.parametrize("num_system_ports", [2], indirect=True)
def test_openengine_sidecar(
    openengine_config,
    request,
    runtime_services_dynamic_ports,
    dynamo_dynamic_ports,
    num_system_ports,
    predownload_models,
):
    """Run the identical sidecar/frontend acceptance path for every engine."""

    assert num_system_ports == 2
    config = dataclasses.replace(
        openengine_config,
        frontend_port=dynamo_dynamic_ports.frontend_port,
    )
    run_serve_deployment(config, request, ports=dynamo_dynamic_ports)
