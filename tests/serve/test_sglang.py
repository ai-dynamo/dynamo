# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
import os
import time

import pytest
import requests

from tests.utils.managed_process import ManagedProcess

logger = logging.getLogger(__name__)


class SGLangProcess(ManagedProcess):
    """Simple process manager for sglang shell scripts"""
    
    def __init__(self, script_name, request, port=8000):
        sglang_dir = "/workspace/examples/sglang"
        script_path = os.path.join(sglang_dir, "launch", script_name)
        
        # Make script executable and run it
        os.chmod(script_path, 0o755)
        command = ["bash", script_path]
        
        self.port = port
        
        super().__init__(
            command=command,
            timeout=600,
            display_output=True,
            working_dir=sglang_dir,
            health_check_ports=[port],
            delayed_start=30,  # Give sglang time to start
            log_dir=request.node.name,
        )


@pytest.mark.e2e
@pytest.mark.slow 
@pytest.mark.sglang
@pytest.mark.gpu_1
def test_sglang_aggregated(request, runtime_services):
    """Test sglang aggregated deployment"""
    
    with SGLangProcess("agg.sh", request) as server:
        # Test chat completions
        response = requests.post(
            f"http://localhost:{server.port}/v1/chat/completions",
            json={
                "model": "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
                "messages": [{"role": "user", "content": "Tell me a joke about AI"}],
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
        logger.info(f"SGLang aggregated response: {content}")


@pytest.mark.e2e
@pytest.mark.slow
@pytest.mark.sglang
@pytest.mark.gpu_2
def test_sglang_disaggregated(request, runtime_services):
    """Test sglang disaggregated deployment (requires 2 GPUs)"""
    
    with SGLangProcess("disagg.sh", request) as server:
        # Test chat completions
        response = requests.post(
            f"http://localhost:{server.port}/v1/chat/completions",
            json={
                "model": "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
                "messages": [{"role": "user", "content": "Tell me a joke about AI"}],
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
        logger.info(f"SGLang disaggregated response: {content}")
        
        # Test completions endpoint too
        response = requests.post(
            f"http://localhost:{server.port}/v1/completions",
            json={
                "model": "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
                "prompt": "The future of AI is",
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


@pytest.mark.skip(reason="Requires 4 GPUs - enable when hardware available")
@pytest.mark.e2e
@pytest.mark.slow
@pytest.mark.sglang
@pytest.mark.gpu_4
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
        
        assert response.status_code == 200
        result = response.json()
        assert "choices" in result
        assert len(result["choices"]) > 0
        content = result["choices"][0]["message"]["content"]
        assert len(content) > 0
        logger.info(f"SGLang DP attention response: {content}")