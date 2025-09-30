# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path

import pytest
import requests

from tests.serve.common import (
    SERVE_TEST_DIR,
    download_from_nats,
    get_mdc_from_etcd,
    params_with_model_mark,
    run_serve_deployment,
)
from tests.utils.engine_process import EngineConfig, EngineProcess
from tests.utils.payload_builder import chat_payload_default, completion_payload_default

logger = logging.getLogger(__name__)


@dataclass
class SGLangConfig(EngineConfig):
    """Configuration for SGLang test scenarios"""

    stragglers: list[str] = field(default_factory=lambda: ["SGLANG:EngineCore"])


sglang_dir = os.environ.get("SGLANG_DIR", "/workspace/components/backends/sglang")

sglang_configs = {
    "aggregated": SGLangConfig(
        name="aggregated",
        directory=SERVE_TEST_DIR,
        script_name="sglang_agg.sh",
        marks=[pytest.mark.gpu_1],
        model="deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
        env={},
        models_port=8000,
        request_payloads=[chat_payload_default(), completion_payload_default()],
    ),
    "custom_chat_template": SGLangConfig(
        name="custom_chat_template",
        directory=SERVE_TEST_DIR,
        script_name="sglang_custom_template.sh",
        marks=[pytest.mark.gpu_1],
        model="Qwen/Qwen3-0.6B",
        env={},
        models_port=8000,
        request_payloads=[chat_payload_default()],
    ),
    "disaggregated": SGLangConfig(
        name="disaggregated",
        directory=sglang_dir,
        script_name="disagg.sh",
        marks=[pytest.mark.gpu_2],
        model="Qwen/Qwen3-0.6B",
        env={},
        models_port=8000,
        request_payloads=[chat_payload_default(), completion_payload_default()],
    ),
    "kv_events": SGLangConfig(
        name="kv_events",
        directory=sglang_dir,
        script_name="agg_router.sh",
        marks=[pytest.mark.gpu_2],
        model="Qwen/Qwen3-0.6B",
        env={
            "DYN_LOG": "dynamo_llm::kv_router::publisher=trace,dynamo_llm::kv_router::scheduler=info",
        },
        models_port=8000,
        request_payloads=[
            chat_payload_default(
                expected_log=[
                    r"ZMQ listener .* received batch with \d+ events \(seq=\d+\)",
                    r"Event processor for worker_id \d+ processing event: Stored\(",
                    r"Selected worker: \d+, logit: ",
                ]
            )
        ],
    ),
    "template_verification": SGLangConfig(
        name="template_verification",
        directory=SERVE_TEST_DIR,
        script_name="template_verifier.sh",
        marks=[pytest.mark.gpu_1],
        model="Qwen/Qwen3-0.6B",
        env={},
        models_port=8000,
        request_payloads=[],
    ),
}


@pytest.fixture(params=params_with_model_mark(sglang_configs))
def sglang_config_test(request):
    """Fixture that provides different SGLang test configurations"""
    return sglang_configs[request.param]


@pytest.mark.e2e
@pytest.mark.sglang
def test_sglang_deployment(
    sglang_config_test, request, runtime_services, predownload_models
):
    """Test SGLang deployment scenarios using common helpers"""
    config = sglang_config_test
    run_serve_deployment(config, request)


@pytest.mark.skip(
    reason="Requires 4 GPUs - enable when hardware is consistently available"
)
def test_sglang_disagg_dp_attention(request, runtime_services, predownload_models):
    """Test sglang disaggregated with DP attention (requires 4 GPUs)"""

    # Kept for reference; this test uses a different launch path and is skipped


@pytest.mark.e2e
@pytest.mark.sglang
@pytest.mark.gpu_1
def test_custom_jinja_template_mdc(request, runtime_services, predownload_models):
    """Test that custom jinja template is correctly stored in MDC (etcd/NATS).
    This test
    """

    config = sglang_configs["custom_chat_template"]

    # For model_name = "Qwen/Qwen3-0.6B" => mdc_key = "mdc/qwen_qwen_3-0_6"
    # Replicate Dynamo's slugification: lowercase + replace invalid characters with an underscore
    mdc_key = f"mdc/{re.sub(r'[^a-z0-9_-]', '_', config.model.lower()).lstrip('_')}"

    expected_template_path = Path(SERVE_TEST_DIR) / "fixtures" / "custom_template.jinja"

    with open(expected_template_path, "r") as f:
        expected_template = f.read()

    # Launch SGLang with custom template
    with EngineProcess.from_script(config, request) as _:
        mdc_data = get_mdc_from_etcd(mdc_key)
        assert (
            "chat_template_file" in mdc_data
        ), "MDC missing 'chat_template_file' field"

        if "hf_chat_template" in mdc_data["chat_template_file"]:
            template_info = mdc_data["chat_template_file"]["hf_chat_template"]
        else:
            assert (
                False
            ), f"No custom chat template found in MDC. Available keys: {list(mdc_data['chat_template_file'].keys())}"

        assert (
            "path" in template_info
        ), f"Template info missing 'path' field. Template info: {template_info}"

        nats_url = template_info["path"]

        # Download template from NATS
        downloaded_template = download_from_nats(nats_url)
        print(downloaded_template)
        # Compare templates
        if expected_template != downloaded_template:
            # Print both templates for inspection
            print("\n=== EXPECTED TEMPLATE ===")
            print(expected_template)
            print("\n=== DOWNLOADED TEMPLATE ===")
            print(downloaded_template)
            print("\n")

            assert False, "Expected and Downloaded Template do not match"


@pytest.mark.e2e
@pytest.mark.sglang
@pytest.mark.gpu_1
def test_custom_template_verification(request, runtime_services, predownload_models):
    """Test that custom jinja template is applied during preprocessing.

    This test verifies that:
    1. The custom template is applied by the frontend
    2. The template marker appears in the tokenized input
    3. The backend correctly identifies the marker
    """

    config = sglang_configs["template_verification"]

    with EngineProcess.from_script(config, request) as _:
        # Send test request
        payload = {
            "model": config.model,
            "messages": [{"role": "user", "content": "test message"}],
            "stream": False,
        }

        response = requests.post(
            f"http://localhost:{config.models_port}/v1/chat/completions",
            json=payload,
            timeout=10,
        )

        # Verify response
        assert (
            response.status_code == 200
        ), f"Request failed with status {response.status_code}"
        result = response.json()

        # Check that we got the success message
        assert "choices" in result, "Response missing 'choices' field"
        assert len(result["choices"]) > 0, "No choices in response"

        message_content = ""
        if "message" in result["choices"][0]:
            message_content = result["choices"][0]["message"].get("content", "")
        elif "text" in result["choices"][0]:
            message_content = result["choices"][0].get("text", "")

        assert (
            "Successfully Applied Chat Template" in message_content
        ), f"Template verification failed. Response: {message_content}"
