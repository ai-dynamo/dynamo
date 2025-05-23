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


import os
import threading
import time
from io import StringIO
from typing import Optional

import pytest
import requests
from tests.e2e.dynamo_client import DynamoRunProcess, send_chat_completion_request
from tests.utils import find_free_port, check_service_health


@pytest.fixture()
def dynamo_run(backend, model, input_type, timeout):
    """
    Create and start a DynamoRunProcess for testing.
    """
    port = find_free_port()
    
    with DynamoRunProcess(
        model=model, backend=backend, port=port, input_type=input_type, timeout=timeout
    ) as process:
        yield process


@pytest.mark.parametrize(
    ["backend", "timeout"],
    [
        pytest.param("vllm", 200, marks=pytest.mark.vllm),  # Reduced timeout from 300
    ],
    ids=["vllm"],
)
@pytest.mark.parametrize("model", ["deepseek-ai/DeepSeek-R1-Distill-Llama-8B"])
@pytest.mark.parametrize(
    ["input_type", "stream"],
    [("text", False), ("http", True), ("http", False)],
    ids=["text", "http-streaming", "http"],
)
@pytest.mark.parametrize("prompt", [("Hello!", "Hello! How can I assist you today?")])
@pytest.mark.gpu
@pytest.mark.e2e
@pytest.mark.vllm
@pytest.mark.slow
def test_run(dynamo_run, backend, model, input_type, prompt, stream, timeout):
    """
    Test dynamo-run in various configurations.
    """
    # For HTTP endpoint, verify it's responding before proceeding
    if input_type == "http":
        # Create an event to signal when the service is ready
        service_ready = threading.Event()
        
        # Start a background thread to monitor API health
        def health_monitor():
            readiness_url = f"{dynamo_run.url}/v1/models"
            
            start_time = time.time()
            max_time = 60  # seconds
            
            while time.time() - start_time < max_time:
                try:
                    response = requests.get(readiness_url, timeout=1)
                    if response.status_code == 200:
                        data = response.json()
                        if "data" in data and len(data["data"]) > 0:
                            print("API is ready and returning models")
                            service_ready.set()
                            return
                except Exception:
                    pass
                time.sleep(1)
        
        # Start health monitor in background
        health_thread = threading.Thread(target=health_monitor)
        health_thread.daemon = True
        health_thread.start()
        
        # Wait for service to be ready with timeout
        if not service_ready.wait(timeout=timeout):
            pytest.fail(f"API health check timed out after {timeout} seconds")
        
        # Now make the request
        content = send_chat_completion_request(
            url=dynamo_run.url,
            model=model,
            prompt=prompt[0],
            stream=stream,
            max_tokens=100  # Reduced from 200 for faster tests
        )
    else:
        # Test text-based CLI interface
        dynamo_run.send_input(prompt[0])
        
        # Wait for the prompt to be ready, which indicates the response is complete
        if not dynamo_run._prompt_ready.wait(timeout):
            raise TimeoutError("Text interface prompt not detected")
            
        content = dynamo_run.output
    
    # Verify response
    assert content is not None, "No response content received"
    assert prompt[1] in content, f"Expected response '{prompt[1]}' not found in '{content}'"
