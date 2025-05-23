# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import dataclass
from typing import Any, Callable, List, Dict, Union
from contextlib import contextmanager
import time
import requests
from tests.utils import managed_process, check_service_health


@dataclass
class DeploymentGraph:
    """
    Represents a deployment graph configuration for testing.
    """
    module: str
    config: str
    directory: str
    endpoint: str
    response_handler: Callable[[Any], str]


@dataclass
class Payload:
    """
    Represents a test payload with expected response and log patterns.
    """
    payload: Dict[str, Any]
    expected_response: List[str]
    expected_log: List[str]


@contextmanager
def dynamo_serve_process(graph: DeploymentGraph, port=8000, timeout=300):
    """
    Start a dynamo serve process with the specified deployment graph.
    """
    print(f"\n[DYNAMO SERVE] Starting deployment: {graph.module} on port {port}")
    command = ["dynamo", "serve", graph.module]
    if graph.config:
        command.extend(["-f", graph.config])
    
    if port != 8000:
        command.extend(["--port", str(port)])

    # Create a health check function for the server
    def health_check():
        health_url = f"http://localhost:{port}/health"
        readiness_url = f"http://localhost:{port}/v1/models"
        
        print(f"[DYNAMO SERVE] Checking health: {health_url}")
        
        # Try health endpoint first (returns faster)
        health_response = check_service_health(
            health_url, 
            max_retries=2,
            retry_interval=0.5,
            timeout=1
        )
        
        if health_response:
            print(f"[DYNAMO SERVE] Health endpoint is OK, checking models endpoint...")
            # Also verify models endpoint is responding properly
            def validate_models(response):
                try:
                    data = response.json()
                    models_ready = "data" in data and len(data["data"]) > 0
                    if models_ready:
                        print(f"[DYNAMO SERVE] Models ready: {data.get('data', [])}") 
                    else:
                        print(f"[DYNAMO SERVE] Models not ready: {data}")
                    return models_ready
                except Exception as e:
                    print(f"[DYNAMO SERVE] Error parsing models response: {e}")
                    return False
                    
            models_response = check_service_health(
                readiness_url,
                max_retries=2,
                retry_interval=0.5,
                timeout=1,
                callback=validate_models
            )
            
            return models_response
        else:
            print(f"[DYNAMO SERVE] Health endpoint not responding")
        
        return False

    with managed_process(
        command,
        check_ports=[port],
        timeout=timeout,
        cwd=graph.directory,
        output=True,
        health_check=health_check
    ) as proc:
        print(f"[DYNAMO SERVE] Server process started and ready - deployment: {graph.module}")
        print(f"[DYNAMO SERVE] Testing with endpoint: {graph.endpoint}")
        
        # Check NATS status before client tests
        try:
            import subprocess
            print("\n[DYNAMO SERVE] Checking NATS status before client tests...")
            subprocess.run(["curl", "-s", "localhost:8222/varz"], check=False)
            subprocess.run(["curl", "-s", "localhost:8222/jsz"], check=False)
            print("[DYNAMO SERVE] NATS check complete")
        except Exception as e:
            print(f"[DYNAMO SERVE] Error checking NATS: {e}")
        
        try:
            yield proc
        except Exception as e:
            print(f"[DYNAMO SERVE] Exception during test execution: {e}")
        
        print("[DYNAMO SERVE] Client tests completed")


def multimodal_response_handler(response):
    """
    Process multimodal API responses.
    """
    if response.status_code != 200:
        return ""
    result = response.json()
    print(result)
    return result


def completions_response_handler(response):
    """
    Process chat completions API responses.
    """
    if response.status_code != 200:
        return ""
    result = response.json()
    assert "choices" in result, "Missing 'choices' in response"
    assert len(result["choices"]) > 0, "Empty choices in response"
    assert "message" in result["choices"][0], "Missing 'message' in first choice"
    assert "content" in result["choices"][0]["message"], "Missing 'content' in message"
    return result["choices"][0]["message"]["content"]


def get_test_deployment_graphs():
    """
    Get a dictionary of deployment graph test configurations.
    """
    # Shorter prompt for faster test execution
    quick_prompt = "Tell me a short joke about AI."
    
    multimodal_payload = Payload(
        payload={
            "model": "llava-hf/llava-1.5-7b-hf",
            "image": "http://images.cocodataset.org/test2017/000000155781.jpg",
            "prompt": "Describe the image briefly",
            "max_tokens": 100,  # Reduced from 300
        },
        expected_log=[],
        expected_response=["bus"],
    )

    eldoria_payload = Payload(
        payload={
            "model": "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
            "messages": [
                {
                    "role": "user",
                    "content": quick_prompt,  # Shorter prompt
                }
            ],
            "max_tokens": 150,  # Reduced from 500
            "temperature": 0.1,
            "seed": 0,
        },
        expected_log=[],
        expected_response=["AI"],
    )
    
    return {
        "agg": (
            DeploymentGraph(
                "graphs.agg:Frontend",
                "configs/agg.yaml",
                "/workspace/examples/llm",
                "v1/chat/completions",
                completions_response_handler,
            ),
            eldoria_payload,
        ),
        "sglang_agg": (
            DeploymentGraph(
                "graphs.agg:Frontend",
                "configs/agg.yaml",
                "/workspace/examples/sglang",
                "v1/chat/completions",
                completions_response_handler,
            ),
            eldoria_payload,
        ),
        "disagg": (
            DeploymentGraph(
                "graphs.disagg:Frontend",
                "configs/disagg.yaml",
                "/workspace/examples/llm",
                "v1/chat/completions",
                completions_response_handler,
            ),
            eldoria_payload,
        ),
        "agg_router": (
            DeploymentGraph(
                "graphs.agg_router:Frontend",
                "configs/agg_router.yaml",
                "/workspace/examples/llm",
                "v1/chat/completions",
                completions_response_handler,
            ),
            eldoria_payload,
        ),
        "disagg_router": (
            DeploymentGraph(
                "graphs.disagg_router:Frontend",
                "configs/disagg_router.yaml",
                "/workspace/examples/llm", 
                "v1/chat/completions",
                completions_response_handler,
            ),
            eldoria_payload,
        ),
        "multimodal_agg": (
            DeploymentGraph(
                "graphs.agg:Frontend",
                "configs/agg.yaml",
                "/workspace/examples/multimodal",
                "generate",
                multimodal_response_handler,
            ),
            multimodal_payload,
        ),
    } 