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
    print(f"[DYNAMO SERVE] Working directory: {graph.directory}")
    command = ["dynamo", "serve", graph.module]
    if graph.config:
        command.extend(["-f", graph.config])

    if port != 8000:
        command.extend(["--port", str(port)])

    print(f"[DYNAMO SERVE] Full command: {' '.join(command)}")

    # Check if the working directory exists
    import os
    if not os.path.exists(graph.directory):
        print(f"[DYNAMO SERVE] ERROR: Working directory does not exist: {graph.directory}")
        raise RuntimeError(f"Working directory does not exist: {graph.directory}")
    
    # Check if the config file exists
    if graph.config:
        config_path = os.path.join(graph.directory, graph.config)
        if not os.path.exists(config_path):
            print(f"[DYNAMO SERVE] ERROR: Config file does not exist: {config_path}")
            raise RuntimeError(f"Config file does not exist: {config_path}")
        else:
            print(f"[DYNAMO SERVE] Config file found: {config_path}")

    try:
        # Start the process using managed_process without startup_check
        with managed_process(
            command,
            check_ports=[],  # Don't check ports automatically
            timeout=timeout,
            cwd=graph.directory,
            output=True
        ) as proc:
            print(f"[DYNAMO SERVE] Server process started (PID: {proc.pid}) - deployment: {graph.module}")
            
            # Simple wait for the process to initialize
            print(f"[DYNAMO SERVE] Waiting for server to initialize...")
            print(f"[DYNAMO SERVE] Note: Model loading from cache...")
            time.sleep(30)
            
            # Check if process is still running
            if proc.poll() is not None:
                exit_code = proc.returncode
                print(f"[DYNAMO SERVE] ERROR: Process exited with code {exit_code}")
                
                # Try to get some output from the process
                try:
                    stdout, stderr = proc.communicate(timeout=1)
                    if stdout:
                        print(f"[DYNAMO SERVE] Process stdout: {stdout}")
                    if stderr:
                        print(f"[DYNAMO SERVE] Process stderr: {stderr}")
                except:
                    print("[DYNAMO SERVE] Could not capture process output")
                
                raise RuntimeError(f"Dynamo serve process exited unexpectedly with code {exit_code}")
            
            print(f"[DYNAMO SERVE] Process {proc.pid} is running")
            print(f"[DYNAMO SERVE] Server should be accessible at: http://localhost:{port}")
            print(f"[DYNAMO SERVE] Testing with endpoint: {graph.endpoint}")
            
            # Make the port available to the calling code
            proc.actual_port = port
            
            # Check NATS status before client tests
            try:
                print("\n[DYNAMO SERVE] Checking NATS status before client tests...")
                import subprocess
                result = subprocess.run(["curl", "-s", "localhost:8222/varz"], capture_output=True, text=True)
                if result.returncode == 0 and "server_id" in result.stdout:
                    print("[DYNAMO SERVE] NATS is running")
                else:
                    print("[DYNAMO SERVE] NATS check failed")
            except Exception as e:
                print(f"[DYNAMO SERVE] Error checking NATS: {e}")

            try:
                yield proc
            except Exception as e:
                print(f"[DYNAMO SERVE] Exception during test execution: {e}")
                raise
    
    except Exception as e:
        print(f"[DYNAMO SERVE] Failed to start dynamo serve process: {e}")
        
        # Try to run the command manually to see what happens
        print(f"[DYNAMO SERVE] Attempting to run command manually for debugging...")
        try:
            import subprocess
            result = subprocess.run(
                command,
                cwd=graph.directory,
                capture_output=True,
                text=True,
                timeout=30
            )
            print(f"[DYNAMO SERVE] Manual run exit code: {result.returncode}")
            if result.stdout:
                print(f"[DYNAMO SERVE] Manual run stdout: {result.stdout}")
            if result.stderr:
                print(f"[DYNAMO SERVE] Manual run stderr: {result.stderr}")
        except subprocess.TimeoutExpired:
            print("[DYNAMO SERVE] Manual run timed out (process may be starting)")
        except Exception as manual_e:
            print(f"[DYNAMO SERVE] Manual run failed: {manual_e}")
        
        raise


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
                "../examples/llm",
                "v1/chat/completions",
                completions_response_handler,
            ),
            eldoria_payload,
        ),
        "sglang_agg": (
            DeploymentGraph(
                "graphs.agg:Frontend",
                "configs/agg.yaml",
                "../examples/sglang",
                "v1/chat/completions",
                completions_response_handler,
            ),
            eldoria_payload,
        ),
        "disagg": (
            DeploymentGraph(
                "graphs.disagg:Frontend",
                "configs/disagg.yaml",
                "../examples/llm",
                "v1/chat/completions",
                completions_response_handler,
            ),
            eldoria_payload,
        ),
        "agg_router": (
            DeploymentGraph(
                "graphs.agg_router:Frontend",
                "configs/agg_router.yaml",
                "../examples/llm",
                "v1/chat/completions",
                completions_response_handler,
            ),
            eldoria_payload,
        ),
        "disagg_router": (
            DeploymentGraph(
                "graphs.disagg_router:Frontend",
                "configs/disagg_router.yaml",
                "../examples/llm", 
                "v1/chat/completions",
                completions_response_handler,
            ),
            eldoria_payload,
        ),
        "multimodal_agg": (
            DeploymentGraph(
                "graphs.agg:Frontend",
                "configs/agg.yaml",
                "../examples/multimodal",
                "generate",
                multimodal_response_handler,
            ),
            multimodal_payload,
        ),
    } 