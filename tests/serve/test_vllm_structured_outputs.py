# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Real HTTP regression for vLLM-style structured_outputs on both endpoints.

Run the same constraints on direct vLLM and Dynamo backed by that installed vLLM.
The constrained answer differs from the prompt, so HTTP 200 or ordinary model
obedience cannot pass as constraint enforcement. Raw requests/responses and launch
arguments stay in tmp_path for diagnosis and external evidence archival. Use
``-o tmp_path_retention_policy=all`` to retain evidence from passing tests too.

Per-request whitespace enforcement is deliberately not asserted: the pinned
xgrammar implementation uses its server configuration (vLLM issue #48765).
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

pytestmark = [
    pytest.mark.e2e,
    pytest.mark.vllm,
    pytest.mark.core,
    pytest.mark.gpu_1,
    pytest.mark.profiled_vram_gib(3.4),
    pytest.mark.post_merge,
    pytest.mark.model("Qwen/Qwen3-0.6B"),
    pytest.mark.requested_vllm_kv_cache_bytes(1_073_741_824),
    pytest.mark.timeout(235),
]


@pytest.fixture(params=["vllm", "dynamo"])
def structured_server(request, tmp_path):
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


def _generate(base, path, payload, evidence):
    """Save raw HTTP evidence and return assembled content from a successful request.

    Streaming must end with [DONE] and a stop finish reason; malformed chunks,
    error events, truncation and non-200 statuses fail rather than being normalized
    away. Only content fragments are assembled, not IDs, timestamps or tool calls.
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
        assert response.status_code == 200, response.text
        chat = path == "/v1/chat/completions"
        if not payload["stream"]:
            choice = response.json()["choices"][0]
            assert choice["finish_reason"] == "stop", choice
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
        ] == ["stop"]
        return "".join(
            (
                (choice.get("delta", {}).get("content") or "")
                if chat
                else choice.get("text", "")
            )
            for choice in choices
        )


def test_structured_outputs_constrain_chat_and_completion(structured_server):
    """Guard shared admission and semantic delivery, in unary and streaming modes.

    The parent Dynamo returns 400 for the first nested constraint; unchanged direct
    vLLM must satisfy every case. Completion uses an already-finished reasoning
    prefix, because this reasoner activates constraints only after </think>.
    Whitespace inside valid JSON is allowed; exact string constraints are not trimmed.
    """
    base, evidence = structured_server
    schema = {
        "type": "object",
        "properties": {"answer": {"const": "violet"}},
        "required": ["answer"],
        "additionalProperties": False,
    }
    constraints = [
        ("json", {"json": schema}),
        ("json-string", {"json": json.dumps(schema)}),
        ("json-object", {"json_object": True}),
        (
            "structural-tag",
            {
                "structural_tag": json.dumps(
                    {
                        "type": "structural_tag",
                        "format": {"type": "json_schema", "json_schema": schema},
                    }
                )
            },
        ),
        ("choice", {"choice": ["violet"]}),
        ("regex", {"regex": "violet"}),
        ("grammar", {"grammar": 'root ::= "violet"'}),
    ]
    for endpoint in ("chat", "completion"):
        path = "/v1/chat/completions" if endpoint == "chat" else "/v1/completions"
        payload = {
            "model": "structured-output-test",
            "temperature": 0,
            "max_tokens": 64,
            "stream": False,
        }
        if endpoint == "chat":
            payload.update(
                messages=[{"role": "user", "content": "Answer with the value seven."}],
                chat_template_kwargs={"enable_thinking": False},
            )
        else:
            payload["prompt"] = (
                "<|im_start|>user\nAnswer with the value seven.<|im_end|>\n"
                "<|im_start|>assistant\n<think>\n\n</think>\n\n"
            )
        baseline = _generate(base, path, payload, evidence / (endpoint + "-baseline"))
        assert (
            "violet" not in baseline
        ), "The control must distinguish constrained output"
        for stream in (False, True):
            for name, constraint in constraints:
                result = _generate(
                    base,
                    path,
                    {**payload, "stream": stream, "structured_outputs": constraint},
                    evidence / f"{endpoint}-{name}-{stream}",
                )
                if name == "json-object":
                    assert isinstance(json.loads(result), dict), result
                elif name in ("json", "json-string", "structural-tag"):
                    assert json.loads(result) == {"answer": "violet"}, result
                else:
                    assert result == "violet", result
        if endpoint == "chat":
            # response_format replaces a same-kind constraint before its content
            # is validated. Text format instead leaves the explicit regex intact.
            overrides = [
                (
                    "replace-json",
                    {"json": "{"},
                    {
                        "type": "json_schema",
                        "json_schema": {"name": "answer", "schema": schema},
                    },
                ),
                ("replace-object", {"json_object": False}, {"type": "json_object"}),
                ("text-regex", {"regex": "violet"}, {"type": "text"}),
            ]
            for stream in (False, True):
                for name, constraint, response_format in overrides:
                    result = _generate(
                        base,
                        path,
                        {
                            **payload,
                            "stream": stream,
                            "structured_outputs": constraint,
                            "response_format": response_format,
                        },
                        evidence / f"chat-{name}-{stream}",
                    )
                    if name == "replace-json":
                        assert json.loads(result) == {"answer": "violet"}, result
                    elif name == "replace-object":
                        assert isinstance(json.loads(result), dict), result
                    else:
                        assert result == "violet", result
