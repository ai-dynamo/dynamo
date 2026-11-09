# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""`logit_bias` through the SGLang chat processor into the Rust request, with a mocker."""

from __future__ import annotations

from typing import Generator

import pytest

from tests.frontend.conftest import MockerWorkerProcess, wait_for_http_completions_ready
from tests.utils.client import send_request
from tests.utils.constants import QWEN
from tests.utils.managed_process import DynamoFrontendProcess
from tests.utils.port_utils import ServicePorts

TEST_MODEL = QWEN

pytestmark = [
    pytest.mark.pre_merge,
    pytest.mark.e2e,
    pytest.mark.core,
    pytest.mark.gpu_0,
    pytest.mark.parallel,
    pytest.mark.sglang,
    pytest.mark.model(TEST_MODEL),
]


@pytest.fixture(scope="function")
def sglang_processor_mocker(
    request: pytest.FixtureRequest,
    runtime_services_dynamic_ports: object,
    dynamo_dynamic_ports: ServicePorts,
    predownload_tokenizers: object,
) -> Generator[int, None, None]:
    _ = runtime_services_dynamic_ports, predownload_tokenizers
    frontend_port = dynamo_dynamic_ports.frontend_port
    with DynamoFrontendProcess(
        request,
        frontend_port=frontend_port,
        extra_args=["--dyn-chat-processor", "sglang"],
        terminate_all_matching_process_names=False,
    ):
        with MockerWorkerProcess(
            request,
            TEST_MODEL,
            frontend_port,
            dynamo_dynamic_ports.system_ports[0],
            extra_args=["--engine-type", "sglang"],
        ):
            wait_for_http_completions_ready(
                frontend_port=frontend_port, model=TEST_MODEL
            )
            yield frontend_port


@pytest.mark.timeout(120)
def test_chat_logit_bias_reaches_worker(sglang_processor_mocker: int) -> None:
    # The mocker ignores the bias; the request only succeeds if the processor's
    # preproc dict converts into PreprocessedRequest (HashMap<u32, f32>).
    response = send_request(
        f"http://localhost:{sglang_processor_mocker}/v1/chat/completions",
        {
            "model": TEST_MODEL,
            "messages": [{"role": "user", "content": "Name a fruit."}],
            "max_tokens": 4,
            "logit_bias": {"59604": 100},
        },
        timeout=60,
    )
    assert response.status_code == 200, response.text
    assert response.json()["choices"]
