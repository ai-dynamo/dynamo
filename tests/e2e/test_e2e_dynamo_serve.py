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
from tests.utils import find_free_port, cleanup_directory, check_service_health, wait_for_service_health
from tests.e2e.testutils import dynamo_serve_process, get_test_deployment_graphs, DeploymentGraph, Payload


@pytest.fixture(params=["agg", "agg_router", "disagg"])
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
def test_serve_deployment(deployment_graph_test, etcd_server, nats_server, model_loader, service_ports):
    """
    Test dynamo serve deployments with different graph configurations.
    
    Uses model_loader to preload models for improved performance and dynamic port allocation.
    """
    print("\n[TEST] Starting test_deployment")
    deployment_graph, payload = deployment_graph_test
    response = None
    
    # Use dynamic port allocation to avoid conflicts
    port = find_free_port()
    print(f"[TEST] Allocated dynamic port: {port}")
    
    print(f"[TEST] Testing deployment: {deployment_graph.module} on port {port}")
    print(f"[TEST] Payload: {payload.payload}")
    print(f"[TEST] Service ports: ETCD={service_ports['etcd_client']}, NATS={service_ports['nats_client']}")
    
    # Extract the model name from the payload
    model_name = None
    if "model" in payload.payload:
        model_name = payload.payload["model"]
    
    # Try to preload the model if possible
    preloaded_model = None
    if model_name:
        try:
            preloaded_model = model_loader(model_name, "vllm")
            print(f"[TEST] Successfully preloaded model: {model_name}")
        except Exception as e:
            print(f"[TEST] Unable to preload model {model_name}: {e}")
    
    # Use an event to signal when the service is ready
    service_ready = threading.Event()
    
    # Check NATS status before starting test
    try:
        import subprocess
        # Using configured NATS monitoring port
        nats_monitor_port = service_ports['nats_http']
        result = subprocess.run(["curl", "-s", f"localhost:{nats_monitor_port}/varz"], capture_output=True, text=True)
        print(f"[TEST] NATS status before test: {'OK' if 'server_id' in result.stdout else 'NOT READY'}")
    except Exception as e:
        print(f"[TEST] Error checking NATS: {e}")
    
    print("[TEST] Starting dynamo serve process")
    # Pass preloaded model to dynamo_serve_process if available
    with dynamo_serve_process(deployment_graph, port=port, timeout=300, preloaded_model=preloaded_model):
        # Wait for service to be ready using the utility function
        readiness_url = f"http://localhost:{port}/v1/models"
        print(f"[TEST] Waiting for service to be ready at {readiness_url}")
        
        # Use the wait_for_service_health utility instead of a custom thread
        if wait_for_service_health(readiness_url, timeout=180, check_interval=1.0):
            print("[TEST] Service API is ready!")
            service_ready.set()
        else:
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
