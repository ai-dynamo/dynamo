# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Paired vLLM/Dynamo HTTP fixtures for focused protocol regressions.

Expose protocol_server in the test module by assigning vllm_protocol.protocol_server.
The fixture yields a base URL and evidence directory for each server variant.
Use generate() to capture raw unary/SSE responses and assert terminal outcomes.
The installed Dynamo and vLLM packages must match the source under test.
"""

import json
import os
import sys
from contextlib import ExitStack
from pathlib import Path
from uuid import uuid4

import pytest
import requests

from tests.utils.managed_process import ManagedProcess
from tests.utils.port_utils import reserved_ports


@pytest.fixture(params=["vllm", "dynamo"])
def protocol_server(request, tmp_path):
    """Yield one real server URL and evidence directory; stop only owned processes.

    Both variants use the same model revision, engine limits, qwen3 reasoner and
    hermes tool parser. Automatic tool choice is enabled for composition probes.
    File discovery
    isolates Dynamo without shared etcd/NATS; reserved ports avoid fixed listeners.
    Startup errors propagate through ManagedProcess's cleanup. KV allocation honors
    the repository profiler override; no process changes CUDA device selection.
    """
    source = Path(__file__).resolve().parents[2]
    environment = {
        **os.environ,
        "TOKENIZERS_PARALLELISM": "false",
        "DYN_IGNORE_OPENAI_FE_UNSUPPORTED_FIELDS": "0",
    }
    engine = [
        "--model",
        "Qwen/Qwen3-0.6B",
        "--revision",
        "c1899de289a04d12100db370d81485cdf75e47ca",
        "--served-model-name",
        "structured-output-test",
        "--max-model-len",
        "512",
        "--max-num-seqs",
        "2",
        "--dtype",
        "bfloat16",
        "--seed",
        "0",
        "--enforce-eager",
        "--kv-cache-memory-bytes",
        environment.get("_PROFILE_OVERRIDE_VLLM_KV_CACHE_BYTES", "1073741824"),
    ]
    with reserved_ports(2, 19000) as ports, ExitStack() as stack:
        http_port, worker_port = ports
        base = f"http://127.0.0.1:{http_port}"
        commands = []
        common = {
            "working_dir": str(source),
            "log_dir": str(tmp_path),
            "terminate_all_matching_process_names": False,
            "timeout": 300,
        }
        if request.param == "vllm":
            command = [
                sys.executable,
                "-m",
                "vllm.entrypoints.openai.api_server",
                *engine,
                "--host",
                "127.0.0.1",
                "--port",
                str(http_port),
                "--reasoning-parser",
                "qwen3",
                "--enable-auto-tool-choice",
                "--tool-call-parser",
                "hermes",
            ]
            commands.append(command)
            stack.enter_context(
                ManagedProcess(
                    command=command,
                    env=environment,
                    display_name="direct-vllm",
                    health_check_urls=[base + "/health"],
                    **common,
                )
            )
        else:
            namespace = "structured-" + uuid4().hex
            environment.update(
                {
                    "DYN_DISCOVERY_BACKEND": "file",
                    "DYN_FILE_KV": str(tmp_path / "discovery"),
                    "DYN_REQUEST_PLANE": "tcp",
                    "DYN_EVENT_PLANE": "zmq",
                    "DYN_NAMESPACE": namespace,
                    "DYN_SYSTEM_PORT": str(worker_port),
                }
            )
            command = [
                sys.executable,
                "-m",
                "dynamo.vllm",
                *engine,
                "--dyn-reasoning-parser",
                "qwen3",
                "--dyn-tool-call-parser",
                "hermes",
            ]
            commands.append(command)
            stack.enter_context(
                ManagedProcess(
                    command=command,
                    env=environment,
                    display_name="dynamo-worker",
                    health_check_urls=[f"http://127.0.0.1:{worker_port}/health"],
                    **common,
                )
            )
            command = [
                sys.executable,
                "-m",
                "dynamo.frontend",
                "--http-port",
                str(http_port),
                "--namespace",
                namespace,
            ]
            commands.append(command)
            stack.enter_context(
                ManagedProcess(
                    command=command,
                    env=environment,
                    display_name="dynamo-frontend",
                    health_check_urls=[
                        (
                            base + "/v1/models",
                            lambda response: any(
                                model["id"] == "structured-output-test"
                                for model in response.json().get("data", [])
                            ),
                        )
                    ],
                    **common,
                )
            )
        (tmp_path / "launch.json").write_text(json.dumps(commands, indent=2))
        with requests.get(base + "/openapi.json", timeout=30) as response:
            response.raise_for_status()
            (tmp_path / "openapi.json").write_bytes(response.content)
        yield base, tmp_path


def generate(
    base, path, payload, evidence, expected_status=200, expected_finish="stop"
):
    """Save raw HTTP evidence and return content, or None for an expected HTTP error.

    Streaming must end with [DONE] and expected_finish (normally stop; use length
    when deliberately exhausting max_tokens). Malformed chunks,
    error events, truncation and non-200 statuses fail rather than being normalized
    away. Only content fragments are assembled, not IDs, timestamps or tool calls.
    For pre-header rejection checks, pass expected_status=400: require JSON rather
    than treating HTTP 200 with an SSE error as equivalent to an HTTP error.
    """
    evidence.mkdir()
    (evidence / "request.json").write_text(json.dumps(payload, indent=2))
    with requests.post(base + path, json=payload, timeout=90) as response:
        (evidence / "response.body").write_bytes(response.content)
        (evidence / "response.json").write_text(
            json.dumps(
                {"status": response.status_code, "headers": dict(response.headers)},
                indent=2,
            )
        )
        assert response.status_code == expected_status, response.text
        if expected_status != 200:
            assert response.headers["Content-Type"].startswith("application/json")
            return None
        chat = path == "/v1/chat/completions"
        if not payload["stream"]:
            choice = response.json()["choices"][0]
            assert choice["finish_reason"] == expected_finish, choice
            return choice["message"]["content"] if chat else choice["text"]
        assert response.headers["Content-Type"].startswith("text/event-stream")
        events = [
            line[5:].strip()
            for line in response.text.splitlines()
            if line.startswith("data:")
        ]
        assert events and events[-1] == "[DONE]", response.text
        chunks = [json.loads(event) for event in events[:-1]]
        assert all("error" not in chunk for chunk in chunks), chunks
        choices = [choice for chunk in chunks for choice in chunk.get("choices", [])]
        assert [
            choice["finish_reason"]
            for choice in choices
            if choice.get("finish_reason") is not None
        ] == [expected_finish]
        return "".join(
            (
                (choice.get("delta", {}).get("content") or "")
                if chat
                else choice.get("text", "")
            )
            for choice in choices
        )
