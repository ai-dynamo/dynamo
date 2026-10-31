# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Legacy completions retain vLLM's resolved generation defaults."""

import json
import os
import sys
from itertools import product
from uuid import uuid4

import pytest
import requests

from tests.utils.constants import QWEN
from tests.utils.http_checks import check_health_ready, check_models_api
from tests.utils.managed_process import DynamoFrontendProcess, ManagedProcess


def _assert_completion(url, payload, expected_tokens, expected_choices):
    with requests.post(
        url, json=payload, stream=payload["stream"], timeout=60
    ) as response:
        assert response.status_code == 200, response.text
        if not payload["stream"]:
            body = response.json()
            choices, usage = body["choices"], body["usage"]
        else:
            choices, usage, finished = [], None, False
            for line in response.iter_lines():
                if not line or not line.startswith(b"data: "):
                    continue
                data = line.removeprefix(b"data: ")
                if data == b"[DONE]":
                    finished = True
                    break
                chunk = json.loads(data)
                choices.extend(
                    choice for choice in chunk["choices"] if choice.get("finish_reason")
                )
                if chunk.get("usage") is not None:
                    usage = chunk["usage"]
            assert finished, "Completion stream ended without [DONE]"
        assert len(choices) == expected_choices, payload
        assert {choice["index"] for choice in choices} == set(range(expected_choices))
        assert all(choice["finish_reason"] == "length" for choice in choices), payload
        assert usage is not None, payload
        assert usage["completion_tokens"] == expected_tokens * expected_choices, payload
        assert (
            usage["total_tokens"] == usage["prompt_tokens"] + usage["completion_tokens"]
        )


@pytest.mark.post_merge
@pytest.mark.e2e
@pytest.mark.vllm
@pytest.mark.core
@pytest.mark.gpu_1
@pytest.mark.model(QWEN)
@pytest.mark.profiled_vram_gib(3.8)
@pytest.mark.requested_vllm_kv_cache_bytes(536_870_912)
@pytest.mark.timeout(360)
@pytest.mark.parametrize("configured_default", [None, 8, 32], ids=["native", "8", "32"])
def test_vllm_completion_defaults(
    configured_default,
    request,
    tmp_path,
    monkeypatch,
    dynamo_dynamic_ports,
    predownload_models,
):
    from vllm import SamplingParams

    ports = dynamo_dynamic_ports
    namespace = f"completion-default-{uuid4().hex}"
    monkeypatch.delenv("DYN_NAMESPACE_PREFIX", raising=False)
    for key, value in {
        "DYN_NAMESPACE": namespace,
        "DYN_DISCOVERY_BACKEND": "file",
        "DYN_FILE_KV": str(tmp_path / "discovery"),
        "DYN_REQUEST_PLANE": "tcp",
        "DYN_EVENT_PLANE": "zmq",
        "DYN_SYSTEM_PORT": str(ports.system_ports[0]),
    }.items():
        monkeypatch.setenv(key, value)
    command = [
        sys.executable,
        "-m",
        "dynamo.vllm",
        "--model",
        QWEN,
        "--endpoint",
        f"dyn://{namespace}.vllm.generate",
        "--generation-config",
        "vllm",
        "--max-model-len",
        "256",
        "--max-num-seqs",
        "2",
        "--enforce-eager",
        "--kv-cache-memory-bytes",
        "536870912",
    ]
    if configured_default is not None:
        command.extend(
            [
                "--override-generation-config",
                json.dumps({"max_new_tokens": configured_default}),
            ]
        )
    default_limit = (
        SamplingParams().max_tokens
        if configured_default is None
        else configured_default
    )
    base_url = f"http://127.0.0.1:{ports.frontend_port}"
    with (
        DynamoFrontendProcess(request, frontend_port=ports.frontend_port),
        ManagedProcess(
            command=command,
            env=os.environ.copy(),
            health_check_urls=[
                (f"{base_url}/v1/models", check_models_api),
                (
                    f"http://127.0.0.1:{ports.system_ports[0]}/health",
                    check_health_ready,
                ),
            ],
            timeout=240,
            terminate_all_matching_process_names=False,
            log_dir=str(tmp_path / "worker"),
            display_name="vllm",
        ),
    ):
        prompts = (
            ("Hello", 1),
            ([1, 2, 3], 1),
            (["Hello", "Hi"], 2),
            ([[1, 2], [3, 4]], 2),
        )
        limits = (
            ({}, default_limit),
            ({"max_tokens": None}, default_limit),
            ({"max_tokens": 4}, 4),
            ({"max_tokens": 64}, 64),
        )
        for (prompt, count), (limit, expected), stream in product(
            prompts, limits, (False, True)
        ):
            payload = {
                "model": QWEN,
                "prompt": prompt,
                "stream": stream,
                "ignore_eos": True,
                **limit,
            }
            if stream:
                payload["stream_options"] = {"include_usage": True}
            _assert_completion(f"{base_url}/v1/completions", payload, expected, count)
