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
import json
from typing import Any, Callable, Optional

import pytest
import requests
from tests.utils import find_free_port, cleanup_directory, check_service_health
from tests.e2e.testutils import dynamo_serve_process, get_test_deployment_graphs, DeploymentGraph, Payload


@pytest.fixture(params=[
    pytest.param("agg", marks=pytest.mark.gpu),
    pytest.param("agg_router", marks=pytest.mark.gpu),
])
def deployment_graph_test(request):
    """
    Fixture that provides different deployment graph test configurations with appropriate markers.
    """
    deployment_graphs = get_test_deployment_graphs()
    return deployment_graphs[request.param]


def check_etcd_health(etcd_server: str, timeout: int = 10) -> bool:
    """Check ETCD health with detailed logging."""
    print(f"[HEALTH] Checking ETCD server: {etcd_server}")
    try:
        import subprocess
        health_cmd = ["etcdctl", "--endpoints", etcd_server, "endpoint", "health"]
        result = subprocess.run(health_cmd, capture_output=True, text=True, timeout=timeout)
        
        if result.returncode == 0 and "healthy" in result.stdout.lower():
            print(f"[HEALTH] ETCD status: READY - {result.stdout.strip()}")
            return True
        else:
            print(f"[HEALTH] ETCD status: NOT READY - stdout: {result.stdout}, stderr: {result.stderr}")
            return False
    except subprocess.TimeoutExpired:
        print(f"[HEALTH] ETCD health check timed out after {timeout}s")
        return False
    except Exception as e:
        print(f"[HEALTH] Error checking ETCD: {e}")
        return False


def check_nats_health(nats_server: dict, timeout: int = 10) -> bool:
    """Check NATS health with detailed logging."""
    print(f"[HEALTH] Checking NATS server: {nats_server}")
    try:
        monitor_url = f"{nats_server['http_monitor']}/varz"
        response = requests.get(monitor_url, timeout=timeout)
        
        if response.status_code == 200:
            data = response.json()
            print(f"[HEALTH] NATS status: READY")
            print(f"[HEALTH] NATS version: {data.get('version', 'unknown')}")
            print(f"[HEALTH] NATS jetstream: {data.get('jetstream', False)}")
            print(f"[HEALTH] NATS connections: {data.get('connections', 0)}")
            return True
        else:
            print(f"[HEALTH] NATS status: NOT READY - HTTP {response.status_code}")
            return False
    except Exception as e:
        print(f"[HEALTH] Error checking NATS: {e}")
        return False


def wait_for_server_ready(base_url: str, test_payload: dict, max_wait: int = 300, process=None) -> bool:
    """Wait for the server to be fully ready by testing the chat/completions endpoint."""
    print(f"[HEALTH] Waiting for server to be ready at {base_url}")
    print(f"[HEALTH] Using chat/completions endpoint as readiness indicator")
    start_time = time.time()
    
    completions_url = f"{base_url}/v1/chat/completions"
    expected_model = test_payload.get('model', 'unknown')
    print(f"[HEALTH] Testing model: {expected_model}")
    
    # Create a minimal version of the test payload for readiness check
    readiness_payload = {
        "model": test_payload.get("model"),
        "messages": test_payload.get("messages", [{"role": "user", "content": "Hello"}]),
        "max_tokens": min(test_payload.get("max_tokens", 10), 10),  # Use small token count for readiness check
        "temperature": test_payload.get("temperature", 0.1)
    }
    
    print(f"[HEALTH] Readiness check payload: {json.dumps(readiness_payload, indent=2)}")
    
    attempt_count = 0
    consecutive_connection_failures = 0
    max_consecutive_failures = 20  # If we can't connect 20 times in a row, server likely failed to start
    
    while time.time() - start_time < max_wait:
        attempt_count += 1
        elapsed = int(time.time() - start_time)
        
        # Check if the server process is still running
        if process and process.poll() is not None:
            exit_code = process.returncode
            print(f"[HEALTH] ERROR: Server process exited with code {exit_code} during health check")
            print(f"[HEALTH] Process died after {elapsed}s, cannot continue health checks")
            return False
        
        print(f"[HEALTH] Attempt {attempt_count} (elapsed: {elapsed}s) - Testing chat/completions endpoint...")
        
        try:
            # Use a shorter timeout and add connection timeout
            response = requests.post(
                completions_url, 
                json=readiness_payload, 
                timeout=(3, 8),  # (connection_timeout, read_timeout) - shorter timeouts
                headers={'Content-Type': 'application/json'}
            )
            
            # Reset consecutive failure counter on successful connection
            consecutive_connection_failures = 0
            print(f"[HEALTH] Response received with status: {response.status_code}")
            
            if response.status_code == 200:
                try:
                    data = response.json()
                    if "choices" in data and len(data["choices"]) > 0:
                        choice = data["choices"][0]
                        if "message" in choice and "content" in choice["message"]:
                            content = choice["message"]["content"]
                            print(f"[HEALTH] Chat/completions endpoint is ready!")
                            print(f"[HEALTH] Sample response: {content[:50]}..." if len(content) > 50 else f"[HEALTH] Sample response: {content}")
                            print(f"[HEALTH] Server is fully ready after {elapsed}s!")
                            return True
                        else:
                            print(f"[HEALTH] Response missing expected structure: {data}")
                    else:
                        print(f"[HEALTH] Response missing choices: {data}")
                except json.JSONDecodeError as e:
                    print(f"[HEALTH] Invalid JSON response: {e}")
                    print(f"[HEALTH] Raw response: {response.text[:200]}...")
                    
            elif response.status_code == 503:
                print(f"[HEALTH] Service unavailable (503) - server still starting up...")
                
            elif response.status_code == 404:
                print(f"[HEALTH] Endpoint not found (404) - check configuration")
                print(f"[HEALTH] URL: {completions_url}")
                
            elif response.status_code == 422:
                print(f"[HEALTH] Invalid request (422) - check payload format")
                try:
                    error_data = response.json()
                    print(f"[HEALTH] Error details: {json.dumps(error_data, indent=2)}")
                except:
                    print(f"[HEALTH] Error response: {response.text}")
                    
            elif response.status_code == 500:
                print(f"[HEALTH] Internal server error (500) - backend issue")
                try:
                    error_data = response.json()
                    print(f"[HEALTH] Error details: {json.dumps(error_data, indent=2)}")
                except:
                    print(f"[HEALTH] Error response: {response.text[:200]}...")
                    
            else:
                print(f"[HEALTH] Unexpected status {response.status_code}: {response.text[:200]}...")
                
        except requests.exceptions.ConnectTimeout:
            consecutive_connection_failures += 1
            print(f"[HEALTH] Connection timeout - server not accepting connections yet ({consecutive_connection_failures}/{max_consecutive_failures})")
            
        except requests.exceptions.ReadTimeout:
            print(f"[HEALTH] Read timeout - server accepted connection but didn't respond in time")
            
        except requests.exceptions.ConnectionError as e:
            consecutive_connection_failures += 1
            print(f"[HEALTH] Connection error ({consecutive_connection_failures}/{max_consecutive_failures}): {str(e)[:100]}...")
            
        except requests.exceptions.Timeout as e:
            consecutive_connection_failures += 1
            print(f"[HEALTH] Request timeout ({consecutive_connection_failures}/{max_consecutive_failures}): {str(e)[:100]}...")
            
        except Exception as e:
            print(f"[HEALTH] Unexpected error: {str(e)[:100]}...")
        
        # Check if we've had too many consecutive connection failures
        if consecutive_connection_failures >= max_consecutive_failures:
            print(f"[HEALTH] ERROR: {consecutive_connection_failures} consecutive connection failures")
            print(f"[HEALTH] Server likely failed to start HTTP service properly")
            print(f"[HEALTH] This suggests the server process started but workers didn't initialize")
            return False
        
        # Wait before next attempt
        if time.time() - start_time < max_wait:
            print(f"[HEALTH] Waiting 3 seconds before next attempt...")
            time.sleep(3)  # Wait 3 seconds between attempts
    
    print(f"[HEALTH] Chat/completions endpoint failed to become ready within {max_wait}s")
    return False


def validate_config_file(deployment_graph: DeploymentGraph) -> bool:
    """Validate that the config file exists and is properly formatted."""
    if not deployment_graph.config:
        print(f"[CONFIG] No config file specified")
        return True
    
    config_path = os.path.join(deployment_graph.directory, deployment_graph.config)
    if not os.path.exists(config_path):
        print(f"[CONFIG] ERROR: Config file not found: {config_path}")
        return False
    
    print(f"[CONFIG] Config file found: {config_path}")
    
    # Try to parse the config file
    try:
        import yaml
        with open(config_path, 'r') as f:
            config_data = yaml.safe_load(f)
        
        # Check for required sections
        if 'Frontend' not in config_data:
            print(f"[CONFIG] WARNING: No Frontend section in config")
        else:
            frontend_config = config_data['Frontend']
            print(f"[CONFIG] Frontend port: {frontend_config.get('port', 'not specified')}")
            print(f"[CONFIG] Frontend endpoint: {frontend_config.get('endpoint', 'not specified')}")
            print(f"[CONFIG] Served model: {frontend_config.get('served_model_name', 'not specified')}")
        
        print(f"[CONFIG] Config file validation passed")
        return True
        
    except Exception as e:
        print(f"[CONFIG] Error parsing config file: {e}")
        return False


@pytest.mark.gpu
@pytest.mark.e2e
@pytest.mark.vllm
@pytest.mark.slow
def test_serve_deployment(deployment_graph_test, nats_server, etcd_server, model_cache):
    """
    Test dynamo serve deployments with comprehensive health checks.
    """
    print("\n" + "="*80)
    print("[TEST] Starting test_serve_deployment with improved health checks")
    print("="*80)
    
    deployment_graph, payload = deployment_graph_test
    port = 8000  # Use the actual port where server runs (config overrides command line)
    
    print(f"[TEST] Testing deployment: {deployment_graph.module}")
    print(f"[TEST] Config file: {deployment_graph.config}")
    print(f"[TEST] Directory: {deployment_graph.directory}")
    print(f"[TEST] Endpoint: {deployment_graph.endpoint}")
    print(f"[TEST] Port: {port}")
    print(f"[TEST] Payload model: {payload.payload.get('model', 'unknown')}")
    
    # Step 1: Validate configuration
    print(f"\n[TEST] Step 1: Validating configuration...")
    if not validate_config_file(deployment_graph):
        pytest.fail("Configuration validation failed")
    
    # Step 2: Check infrastructure dependencies
    print(f"\n[TEST] Step 2: Checking infrastructure dependencies...")
    
    # Check ETCD
    if not check_etcd_health(etcd_server):
        pytest.fail(f"ETCD is not ready: {etcd_server}")
    
    # Check NATS
    if not check_nats_health(nats_server):
        pytest.fail(f"NATS is not ready: {nats_server}")
    
    print(f"[TEST] Infrastructure dependencies are ready")
    
    # Step 3: Start dynamo serve process
    print(f"\n[TEST] Step 3: Starting dynamo serve process...")
    
    with dynamo_serve_process(deployment_graph, port=port, timeout=300) as proc:
        # Get the actual port the server is running on
        actual_port = getattr(proc, 'actual_port', port)
        base_url = f"http://localhost:{actual_port}"
        
        print(f"[TEST] Server process started (PID: {proc.pid})")
        print(f"[TEST] Server URL: {base_url}")
        
        # Step 4: Wait for server to be fully ready
        print(f"\n[TEST] Step 4: Waiting for server to be fully ready...")
        print(f"[TEST] Using chat/completions endpoint as the primary readiness indicator")
        
        if not wait_for_server_ready(base_url, payload.payload, max_wait=180, process=proc):
            # Check if the process is still running
            if proc.poll() is not None:
                exit_code = proc.returncode
                print(f"[TEST] ERROR: Server process exited with code {exit_code} during startup")
                pytest.fail(f"Server process exited unexpectedly with code {exit_code}")
            else:
                print(f"[TEST] ERROR: Server process is running but failed to become ready")
                pytest.fail("Server failed to become ready within timeout - process running but not responding")
        
        # Step 5: Send the actual test request
        print(f"\n[TEST] Step 5: Sending test request...")
        url = f"{base_url}/{deployment_graph.endpoint}"
        
        print(f"[TEST] Request URL: {url}")
        print(f"[TEST] Request payload: {json.dumps(payload.payload, indent=2)}")
        
        try:
            response = requests.post(url, json=payload.payload, timeout=300)
            print(f"[TEST] Response status: {response.status_code}")
            
            if response.status_code != 200:
                print(f"[TEST] Error response headers: {dict(response.headers)}")
                print(f"[TEST] Error response body: {response.text}")
                
                # Try to get more diagnostic information
                try:
                    error_data = response.json()
                    print(f"[TEST] Parsed error data: {json.dumps(error_data, indent=2)}")
                except:
                    pass
                
                pytest.fail(f"Service returned status code {response.status_code}: {response.text}")
            
        except Exception as e:
            print(f"[TEST] Request failed: {str(e)}")
            pytest.fail(f"Request failed: {str(e)}")
        
        # Step 6: Process and validate response
        print(f"\n[TEST] Step 6: Processing response...")
        
        try:
            response_data = response.json()
            print(f"[TEST] Response data keys: {list(response_data.keys())}")
            
            content = deployment_graph.response_handler(response)
            print(f"[TEST] Extracted content: {content[:100]}..." if len(content) > 100 else f"[TEST] Extracted content: {content}")
            
            # Check for expected responses
            if not content:
                pytest.fail("Empty response content")
            
            for expected in payload.expected_response:
                if expected not in content:
                    print(f"[TEST] Expected '{expected}' not found in response")
                    print(f"[TEST] Full response content: {content}")
                    pytest.fail(f"Expected '{expected}' not found in response")
            
            print(f"[TEST] All expected responses found in content")
            
        except Exception as e:
            print(f"[TEST] Error processing response: {e}")
            print(f"[TEST] Raw response: {response.text}")
            pytest.fail(f"Error processing response: {e}")
        
        print(f"\n[TEST] Test completed successfully!")
        print("="*80)
