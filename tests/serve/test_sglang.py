# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
import os
from dataclasses import dataclass
from typing import Any, List

import pytest
import requests

from tests.utils.managed_process import ManagedProcess

logger = logging.getLogger(__name__)


@dataclass
class SGLangConfig:
    """Configuration for SGLang test scenarios"""

    script_name: str
    marks: List[Any]
    name: str


class SGLangProcess(ManagedProcess):
    """Simple process manager for sglang shell scripts"""

    def __init__(self, script_name, request):
        self.port = 8000
        sglang_dir = "/workspace/components/backends/sglang"
        script_path = os.path.join(sglang_dir, "launch", script_name)

        # Verify script exists
        if not os.path.exists(script_path):
            raise FileNotFoundError(f"SGLang script not found: {script_path}")

        # Make script executable and run it
        command = ["bash", script_path]

        super().__init__(
            command=command,
            timeout=900,
            display_output=True,
            working_dir=sglang_dir,
            health_check_ports=[],  # Disable port health check
            health_check_urls=[
                (f"http://localhost:{self.port}/v1/models", self._check_models_api)
            ],
            delayed_start=60,  # Give SGLang more time to fully start
            terminate_existing=False,
            stragglers=[],  # Don't kill any stragglers automatically
            log_dir=request.node.name,
        )

    def _check_models_api(self, response):
        """Check if models API is working and returns models"""
        try:
            if response.status_code != 200:
                return False
            data = response.json()
            return data.get("data") and len(data["data"]) > 0
        except Exception:
            return False


# SGLang test configurations
sglang_configs = {
    "aggregated": SGLangConfig(
        script_name="agg.sh", marks=[pytest.mark.gpu_1], name="aggregated"
    ),
    "disaggregated": SGLangConfig(
        script_name="disagg.sh", marks=[pytest.mark.gpu_2], name="disaggregated"
    ),
}


@pytest.fixture(
    params=[
        pytest.param("aggregated", marks=[pytest.mark.gpu_1]),
        pytest.param("disaggregated", marks=[pytest.mark.gpu_2]),
    ]
)
def sglang_config_test(request):
    """Fixture that provides different SGLang test configurations"""
    return sglang_configs[request.param]


@pytest.mark.e2e
@pytest.mark.slow
@pytest.mark.sglang
def test_sglang_deployment(request, runtime_services, sglang_config_test):
    """Test SGLang deployment scenarios"""

    # First check if sglang is available
    try:
        import sglang

        logger.info(f"SGLang version: {sglang.__version__}")
    except ImportError:
        pytest.skip("SGLang not available")

    config = sglang_config_test

    with SGLangProcess(config.script_name, request) as server:
        # Test chat completions
        response = requests.post(
            f"http://localhost:{server.port}/v1/chat/completions",
            json={
                "model": "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
                "messages": [
                    {
                        "role": "user",
                        "content": "Why is Roger Federer the best tennis player of all time?",
                    }
                ],
                "max_tokens": 50,
            },
            timeout=120,
        )

        assert response.status_code == 200
        result = response.json()
        assert "choices" in result
        assert len(result["choices"]) > 0
        content = result["choices"][0]["message"]["content"]
        assert len(content) > 0
        logger.info(f"SGLang {config.name} response: {content}")

        # Test completions endpoint for disaggregated only
        if config.name == "disaggregated":
            response = requests.post(
                f"http://localhost:{server.port}/v1/completions",
                json={
                    "model": "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
                    "prompt": "Roger Federer is the greatest tennis player of all time",
                    "max_tokens": 30,
                },
                timeout=120,
            )

            assert response.status_code == 200
            result = response.json()
            assert "choices" in result
            assert len(result["choices"]) > 0
            text = result["choices"][0]["text"]
            assert len(text) > 0
            logger.info(f"SGLang completions response: {text}")


@pytest.mark.skip(
    reason="Requires 4 GPUs - enable when hardware is consistently available"
)
def test_sglang_disagg_dp_attention(request, runtime_services):
    """Test sglang disaggregated with DP attention (requires 4 GPUs)"""

    with SGLangProcess("disagg_dp_attn.sh", request) as server:
        # Test chat completions with the DP attention model
        response = requests.post(
            f"http://localhost:{server.port}/v1/chat/completions",
            json={
                "model": "silence09/DeepSeek-R1-Small-2layers",  # DP attention model
                "messages": [{"role": "user", "content": "Tell me about MoE models"}],
                "max_tokens": 50,
            },
            timeout=120,
        )

        # TODO: Once this is enabled, we can test out the rest of the HTTP endpoints around
        # flush_cache and expert distribution recording

        assert response.status_code == 200
        result = response.json()
        assert "choices" in result
        assert len(result["choices"]) > 0
        content = result["choices"][0]["message"]["content"]
        assert len(content) > 0
        logger.info(f"SGLang DP attention response: {content}")
