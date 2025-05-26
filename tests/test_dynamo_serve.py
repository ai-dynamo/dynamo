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
from dataclasses import dataclass
from typing import Any, Callable, Optional

import pytest
import requests


@contextmanager
def managed_process(
    command, env=None, check_ports=[], timeout=600, cwd=None, output=False
):
    cmd_string = " ".join(command)
    cmd_name = command[0]
    print()
    print("*" * 80)
    print(f"* Starting: {cmd_name}")
    print(f"* \t {cmd_string}")
    print("*" * 80)
    print()

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
        raise TimeoutError(
            f"{cmd_name} failed to start. Ports {check_ports} not ready in time"
        )

    try:
        yield proc
    finally:
        print(f"Terminating {cmd_name}")
        proc.terminate()
        proc.wait()


def is_port_open(port):
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(("localhost", port)) == 0


@pytest.fixture()
def etcd_server(port=2379):
    port_string = str(port)
    etcd_env = os.environ.copy()
    etcd_env["ALLOW_NONE_AUTHENTICATION"] = "yes"
    shutil.rmtree("/tmp/etcd-test-data", ignore_errors=True)
    with managed_process(
        [
            "etcd",
            "--listen-client-urls",
            f"http://0.0.0.0:{port_string}",
            "--advertise-client-urls",
            f"http://0.0.0.0:{port_string}",
            "--data-dir",
            "/tmp/etcd-test-data",
        ],
        env=etcd_env,
        check_ports=[port],
    ) as proc:
        yield proc


@pytest.fixture()
def nats_server(port=4222):
    shutil.rmtree("/tmp/nats/jetstream", ignore_errors=True)
    with managed_process(
        ["nats-server", "-js", "--trace", "--store_dir", "/tmp/nats/jetstream"],
        check_ports=[port],
    ) as proc:
        yield proc


@pytest.fixture(scope="function", autouse=True)
def runtime_processes(etcd_server, nats_server):
    yield
    subprocess.run(["pkill", "-f", "nats-server|etcd|.*http.*"], check=False)


@dataclass
class DeploymentGraph:
    module: str
    config: str
    directory: str
    endpoint: str
    response_handler: Callable[[Any], str]


@dataclass
class Payload:
    payload: dict
    expected_response: list[str]
    expected_log: list[str]


@contextmanager
def dynamo_serve_process(graph: DeploymentGraph, port=8000, timeout=600):
    command = ["dynamo", "serve", graph.module]
    if graph.config:
        command.extend(["-f", graph.config])

    with managed_process(
        command,
        check_ports=[port],  # Frontend port
        timeout=timeout,
        cwd=graph.directory,
        output=True,
    ) as proc:
        yield proc


def multimodal_response_handler(response):
    if response.status_code != 200:
        return ""
    result = response.json()
    print(result)
    return result


def completions_response_handler(response):
    if response.status_code != 200:
        return ""
    result = response.json()
    assert "choices" in result
    assert len(result["choices"]) > 0
    assert "message" in result["choices"][0]
    assert "content" in result["choices"][0]["message"]
    return result["choices"][0]["message"]["content"]


@pytest.fixture(
    params=["agg", "agg_router", "multimodal_agg", "sglang_agg"]
)  #'disagg_router','disagg'])
def deployment_graph_test(request):
    multimodal_payload = Payload(
        payload={
            "model": "llava-hf/llava-1.5-7b-hf",
            "image": "http://images.cocodataset.org/test2017/000000155781.jpg",
            "prompt": "Describe the image",
            "max_tokens": 300,
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
                    "content": "In the heart of Eldoria, an ancient land of boundless magic and mysterious creatures, lies the long-forgotten city of Aeloria. Once a beacon of knowledge and power, Aeloria was buried beneath the shifting sands of time, lost to the world for centuries. You are an intrepid explorer, known for your unparalleled curiosity and courage, who has stumbled upon an ancient map hinting at ests that Aeloria holds a secret so profound that it has the potential to reshape the very fabric of reality. Your journey will take you through treacherous deserts, enchanted forests, and across perilous mountain ranges. Your Task: Character Background: Develop a detailed background for your character. Describe their motivations for seeking out Aeloria, their skills and weaknesses, and any personal connections to the ancient city or its legends. Are they driven by a quest for knowledge, a search for lost familt clue is hidden.",
                }
            ],
            "max_tokens": 500,
            "temperature": 0.1,
            "seed": 0,
        },
        expected_log=[],
        expected_response=["Eldoria"],
    )
    graphs = {
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
                "/workspae/examples/llm",
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
    return graphs[request.param]


@pytest.mark.gpu
def test_deployment(deployment_graph_test):
    deployment_graph, payload = deployment_graph_test
    response = None
    with dynamo_serve_process(deployment_graph):
        url = f"http://localhost:8000/{deployment_graph.endpoint}"

        # Retry to account for service startup time
        for _ in range(60):
            try:
                response = requests.post(url, json=payload.payload, timeout=600)
                if response.status_code == 200:
                    break
                    time.sleep(0.5)
                    continue
                time.sleep(5)
            except Exception:
                time.sleep(1)
        else:
            if not response or response.status_code != 200:
                pytest.fail("Service failed to start within timeout")
        content = deployment_graph.response_handler(response)

        for response in payload.expected_response:
            assert response in content
