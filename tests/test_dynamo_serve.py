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
import shutil
import subprocess
import time
from contextlib import contextmanager
from typing import Optional

import pytest
import requests

pytestmark = pytest.mark.e2e


@contextmanager
def managed_process(
    command, env=None, check_ports=[], timeout=600, cwd=None, output=False
):
    print(" ".join(command))
    _input: Optional[int] = subprocess.DEVNULL
    _output: Optional[int] = subprocess.DEVNULL
    if output:
        _input = None
        _output = None
    proc = subprocess.Popen(
        command,
        env=env or os.environ.copy(),
        cwd=cwd,
        stdin=_input,
        stdout=_output,
        stderr=_output,
    )
    start_time = time.time()

    # Wait for ports to become available
    while time.time() - start_time < timeout:
        if all(is_port_open(p) for p in check_ports):
            break
        time.sleep(0.1)
    else:
        proc.terminate()
        raise TimeoutError(f"Ports {check_ports} not ready in time")

    try:
        yield proc
    finally:
        proc.terminate()
        proc.wait()


def is_port_open(port):
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(("localhost", port)) == 0


@pytest.fixture()
def etcd_server():
    etcd_env = os.environ.copy()
    etcd_env["ALLOW_NONE_AUTHENTICATION"] = "yes"
    shutil.rmtree("/tmp/etcd-test-data", ignore_errors=True)
    with managed_process(
        [
            "etcd",
            "--listen-client-urls",
            "http://0.0.0.0:2379",
            "--advertise-client-urls",
            "http://0.0.0.0:2379",
            "--data-dir",
            "/tmp/etcd-test-data",
        ],
        env=etcd_env,
        check_ports=[2379],
    ) as proc:
        yield proc


@pytest.fixture()
def nats_server():
    shutil.rmtree("/tmp/nats/jetstream", ignore_errors=True)
    with managed_process(
        ["nats-server", "-js", "--trace", "--store_dir", "/tmp/nats/jetstream"],
        check_ports=[4222],
    ) as proc:
        yield proc


@pytest.fixture(params=["multimodal_agg", "multimodal_disagg"])
def deployment_config(request):
    config_map = {
        "multimodal_agg": (
            "graphs.agg:Frontend",
            "configs/agg.yaml",
            "/workspace/examples/multimodal",
        ),
        "multimodal_disagg": (
            "graphs.disagg:Frontend",
            "configs/disagg.yaml",
            "/workspace/examples/multimodal",
        ),
    }
    return config_map[request.param]


@contextmanager
def dynamo_serve_process(graph_module, config_path, cwd):
    command = ["dynamo", "serve", graph_module]
    if config_path:
        command.extend(["-f", config_path])

    # command.extend(["--VllmWorker.model", "mock"])
    with managed_process(
        command,
        check_ports=[8000],  # Frontend port
        timeout=600,
        cwd=cwd,
        output=True,
    ) as proc:
        yield proc


def test_deployment(etcd_server, nats_server, deployment_config):
    graph_module, config_path, cwd = deployment_config

    with dynamo_serve_process(graph_module, config_path, cwd):
        if "multimodal" in cwd:
            url = "http://localhost:8000/generate"
            payload = {
                "model": "llava-hf/llava-1.5-7b-hf",
                "image": "http://images.cocodataset.org/test2017/000000155781.jpg",
                "prompt": "Describe the image",
                "max_tokens": 300,
            }
        else:
            # Test OpenAI-compatible endpoint
            url = "http://localhost:8000/v1/chat/completions"
            payload = {
                "model": "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
                "messages": [
                    {
                        "role": "user",
                        "content": "In the heart of Eldoria, an ancient land of boundless magic and mysterious creatures, lies the long-forgotten city of Aeloria. Once a beacon of knowledge and power, Aeloria was buried beneath the shifting sands of time, lost to the world for centuries. You are an intrepid explorer, known for your unparalleled curiosity and courage, who has stumbled upon an ancient map hinting at ests that Aeloria holds a secret so profound that it has the potential to reshape the very fabric of reality. Your journey will take you through treacherous deserts, enchanted forests, and across perilous mountain ranges. Your Task: Character Background: Develop a detailed background for your character. Describe their motivations for seeking out Aeloria, their skills and weaknesses, and any personal connections to the ancient city or its legends. Are they driven by a quest for knowledge, a search for lost familt clue is hidden.",
                    }
                ],
                "max_tokens": 500,
                "temperature": 0.1,
            }

        # Retry to account for service startup time
        for _ in range(10):
            try:
                response = requests.post(url, json=payload, timeout=600)
                if response.status_code == 200:
                    time.sleep(0.5)
                    break
                time.sleep(5)
            except Exception as e:
                print(f"Exception: {e}")
                time.sleep(5)
        else:
            if response.status_code != 200:
                pytest.fail("Service failed to start within timeout")

        if "multimodal" in cwd:
            assert response.status_code == 200
            assert response.json() is not None
            print("Response: ", response.json())
            # Expect a "bus" keyword in the response
            assert "bus" in response.json()
        else:
            assert response.status_code == 200
            result = response.json()
            assert "choices" in result
            assert len(result["choices"]) > 0
            assert "message" in result["choices"][0]
            assert len(result["choices"][0]["message"]["content"]) > 10
            print(result["choices"][0]["message"]["content"])


@pytest.fixture(scope="function", autouse=True)
def cleanup_processes():
    yield
    subprocess.run(["pkill", "-f", "nats-server|etcd"], check=False)
