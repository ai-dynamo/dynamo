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
import time
import threading
from typing import Any, Callable, Optional

import pytest
import requests
from tests.utils import find_free_port, cleanup_directory, check_service_health
from tests.e2e.testutils import dynamo_serve_process, get_test_deployment_graphs, DeploymentGraph, Payload


@pytest.fixture(params=["agg", "agg_router", "multimodal_agg", "sglang_agg"])
def deployment_graph_test(request):
    """
    Fixture that provides different deployment graph test configurations.
    """
    deployment_graphs = get_test_deployment_graphs()
    return deployment_graphs[request.param]


@pytest.mark.gpu
@pytest.mark.e2e
@pytest.mark.vllm
@pytest.mark.slow
def test_serve_deployment(deployment_graph_test):
    """
    Test dynamo serve deployments with different graph configurations.
    """
    print("\n[TEST] Starting test_deployment")
    deployment_graph, payload = deployment_graph_test
    response = None
    port = find_free_port()
    
    print(f"[TEST] Testing deployment: {deployment_graph.module} on port {port}")
    print(f"[TEST] Payload: {payload.payload}")
    
    # Use an event to signal when the service is ready
    service_ready = threading.Event()
    request_allowed = threading.Event()
    
    # Start a background thread to monitor service health
    def health_monitor():
        url = f"http://localhost:{port}/{deployment_graph.endpoint}"
        readiness_url = f"http://localhost:{port}/v1/models"
        
        print(f"[TEST] Health monitor starting - checking {readiness_url}")
        start_time = time.time()
        timeout = 300  # seconds
        
        # Wait for ports to be open first
        while time.time() - start_time < timeout:
            if check_service_health(readiness_url, max_retries=1, retry_interval=0.1):
                print("[TEST] Service API is ready!")
                service_ready.set()
                request_allowed.set()
                return
            time.sleep(1)
        
        print("[TEST] Service health check timed out")
    
    # Start health monitor in background
    health_thread = threading.Thread(target=health_monitor)
    health_thread.daemon = True
    health_thread.start()
    
    # Check NATS status before starting test
    try:
        import subprocess
        result = subprocess.run(["curl", "-s", "localhost:8222/varz"], capture_output=True, text=True)
        print(f"[TEST] NATS status before test: {'OK' if 'server_id' in result.stdout else 'NOT READY'}")
    except Exception as e:
        print(f"[TEST] Error checking NATS: {e}")
    
    print("[TEST] Starting dynamo serve process")
    with dynamo_serve_process(deployment_graph, port=port, timeout=300):
        # Wait for service to be ready
        print("[TEST] Waiting for service to be ready")
        if not service_ready.wait(timeout=180):
            print("[TEST] Service health check timed out after 180 seconds")
            pytest.fail("Service health check timed out after 180 seconds")
            
        url = f"http://localhost:{port}/{deployment_graph.endpoint}"
        
        # Send the request
        print(f"[TEST] Sending request to {url}")
        try:
            response = requests.post(url, json=payload.payload, timeout=300)
            print(f"[TEST] Response status: {response.status_code}")
        except Exception as e:
            print(f"[TEST] Request failed: {str(e)}")
            pytest.fail(f"Request failed: {str(e)}")
            
        # Process the response
        if response.status_code != 200:
            print(f"[TEST] Error response: {response.text}")
            pytest.fail(f"Service returned status code {response.status_code}: {response.text}")
            
        print("[TEST] Processing response")
        content = deployment_graph.response_handler(response)
        
        # Check for expected responses
        assert content, "Empty response content"
        for expected in payload.expected_response:
            assert expected in content, f"Expected '{expected}' not found in response"
        
        print("[TEST] Test completed successfully")
