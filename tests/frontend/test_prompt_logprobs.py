# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import math
import os

import pytest
import requests

from tests.utils.constants import QWEN
from tests.utils.gpu_args import build_gpu_mem_args
from tests.utils.managed_process import DynamoFrontendProcess, ManagedProcess
from tests.utils.payloads import check_models_api

pytestmark = [
    pytest.mark.post_merge,
    pytest.mark.gpu_1,
    pytest.mark.e2e,
    pytest.mark.core,
    pytest.mark.model(QWEN),
    pytest.mark.timeout(300),
]


# Peaks measured with tests/utils/profile_pytest.py.
@pytest.mark.parametrize(
    "backend,worker_args",
    [
        pytest.param(
            "vllm",
            ["--model", QWEN, "--enforce-eager", "--max-model-len", "1024"],
            marks=[
                pytest.mark.vllm,
                pytest.mark.profiled_vram_gib(3.3),
                pytest.mark.requested_vllm_kv_cache_bytes(536870912),
            ],
            id="vllm",
        ),
        pytest.param(
            "sglang",
            [
                "--model-path",
                QWEN,
                "--context-length",
                "1024",
                "--disable-cuda-graph",
            ],
            marks=[
                pytest.mark.sglang,
                pytest.mark.profiled_vram_gib(2.8),
                pytest.mark.requested_sglang_kv_tokens(2048),
            ],
            id="sglang",
        ),
    ],
)
@pytest.mark.parametrize("request_plane", ["tcp"], indirect=True)
@pytest.mark.parametrize("event_plane", ["zmq"], indirect=True)
def test_chat_prompt_logprobs(
    request,
    backend,
    worker_args,
    runtime_services_dynamic_ports,
    dynamo_dynamic_ports,
    predownload_models,
    monkeypatch,
    tmp_path,
):
    monkeypatch.setenv("DYN_DISCOVERY_BACKEND", "etcd")
    monkeypatch.setenv("DYN_REQUEST_PLANE", "tcp")
    monkeypatch.setenv("DYN_EVENT_PLANE", "zmq")
    env = os.environ.copy()
    env["DYN_SYSTEM_PORT"] = str(dynamo_dynamic_ports.system_ports[0])
    env.setdefault("_PROFILE_OVERRIDE_VLLM_KV_CACHE_BYTES", "536870912")
    env.setdefault("_PROFILE_OVERRIDE_SGLANG_MAX_TOTAL_TOKENS", "2048")
    memory_args = build_gpu_mem_args(f"build_{backend}_gpu_mem_args", env=env)
    frontend_port = dynamo_dynamic_ports.frontend_port
    with DynamoFrontendProcess(
        request,
        frontend_port=frontend_port,
        extra_args=["--dyn-chat-processor", backend],
    ), ManagedProcess(
        command=["python3", "-m", f"dynamo.{backend}", *worker_args, *memory_args],
        env=env,
        health_check_urls=[
            (f"http://localhost:{frontend_port}/v1/models", check_models_api)
        ],
        timeout=240,
        terminate_all_matching_process_names=False,
        log_dir=str(tmp_path / "worker"),
    ):
        payload = {
            "model": QWEN,
            "messages": [{"role": "user", "content": "1 + 1 = ?"}],
            "stream": False,
            "temperature": 0,
            "max_tokens": 4,
            "chat_template_kwargs": {"enable_thinking": False},
            "prompt_logprobs": 0,
        }
        response = requests.post(
            f"http://localhost:{frontend_port}/v1/chat/completions",
            json=payload,
            timeout=60,
        )
        response.raise_for_status()
        body = response.json()
        assert body["choices"][0].get("logprobs") is None
        logprobs = body["prompt_logprobs"]
        assert len(logprobs) == body["usage"]["prompt_tokens"] > 1
        assert logprobs[0] is None
        for position in logprobs[1:]:
            assert isinstance(position, dict) and len(position) == 1
            token_id, entry = next(iter(position.items()))
            assert token_id.isdecimal() and 0 <= int(token_id) <= 2**32 - 1
            assert isinstance(entry["logprob"], (int, float))
            assert math.isfinite(entry["logprob"])

        if backend == "vllm":
            stop = body["choices"][0]["message"]["content"][:1]
            assert stop
            for stream in (False, True):
                with requests.post(
                    f"http://localhost:{frontend_port}/v1/chat/completions",
                    json={
                        **payload,
                        "stop": [stop],
                        "stream": stream,
                        "nvext": {"extra_fields": ["prompt_logprobs"]},
                    },
                    stream=stream,
                    timeout=60,
                ) as stopped_response:
                    stopped_response.raise_for_status()
                    if stream:
                        data = [
                            line.removeprefix("data: ")
                            for line in stopped_response.iter_lines(decode_unicode=True)
                            if line.startswith("data: ")
                        ]
                        assert data[-1] == "[DONE]"
                        chunks = [json.loads(item) for item in data[:-1]]
                        assert all("prompt_logprobs" not in chunk for chunk in chunks)
                    else:
                        chunks = [stopped_response.json()]
                        assert len(chunks[0]["prompt_logprobs"]) == len(logprobs)

                metadata = [
                    chunk["nvext"]["prompt_logprobs"]
                    for chunk in chunks
                    if "prompt_logprobs" in chunk.get("nvext", {})
                ]
                assert [len(value) for value in metadata] == [len(logprobs)]
                choices = [choice for chunk in chunks for choice in chunk["choices"]]
                assert [c["finish_reason"] for c in choices if c["finish_reason"]] == [
                    "stop"
                ]
                assert all(
                    not choice.get("delta" if stream else "message", {}).get("content")
                    for choice in choices
                )
