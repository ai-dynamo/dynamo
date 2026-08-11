# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU-only response-catalog coverage through the real Rust frontend pipeline."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import pytest
import requests

from tests.frontend.conftest import MockerWorkerProcess
from tests.utils.managed_process import DynamoFrontendProcess

MODEL_NAME = "synthetic-qwen-response-catalog"
REPO_ROOT = Path(__file__).resolve().parents[2]
TINY_MODEL = REPO_ROOT / "lib/llm/tests/data/sample-models/TinyLlama_v1.1"

WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the weather for a city.",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    },
}

pytestmark = [
    pytest.mark.pre_merge,
    pytest.mark.gpu_0,
    pytest.mark.e2e,
    pytest.mark.parallel,
]


def _write_synthetic_model(tmp_path: Path) -> Path:
    model_path = tmp_path / "synthetic-model"
    shutil.copytree(TINY_MODEL, model_path)
    tokenizer_config_path = model_path / "tokenizer_config.json"
    tokenizer_config = json.loads(tokenizer_config_path.read_text(encoding="utf-8"))
    tokenizer_config["chat_template"] = (
        "{% for message in messages %}"
        "{{ message['role'] + ': ' + message['content'] + '\\n' }}"
        "{% endfor %}assistant: <think>"
    )
    tokenizer_config_path.write_text(
        json.dumps(tokenizer_config),
        encoding="utf-8",
    )
    return model_path


def _write_catalog(tmp_path: Path) -> Path:
    catalog_path = tmp_path / "responses.json"
    catalog_path.write_text(
        json.dumps(
            {
                "version": 1,
                "cases": [
                    {
                        "id": "weather",
                        "response": {
                            "reasoning": "I should call the weather tool.",
                            "content": None,
                            "tool_calls": [
                                {
                                    "name": "get_weather",
                                    "arguments": {"city": "Seattle"},
                                }
                            ],
                        },
                        "chunk_size": 3,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return catalog_path


def _payload(
    *,
    stream: bool,
    model: str = MODEL_NAME,
    case_id: str = "weather",
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": "What is the weather?"}],
        "tools": [WEATHER_TOOL],
        "tool_choice": "auto",
        "max_tokens": 128,
        "stream": stream,
        "nvext": {"annotations": [f"output_replay_id:{case_id}"]},
    }
    if stream:
        payload["stream_options"] = {"include_usage": True}
    return payload


def _stream_semantics(response: requests.Response) -> tuple[dict[str, Any], int, int]:
    response.raise_for_status()
    reasoning_parts: list[str] = []
    content_parts: list[str] = []
    tool_calls: dict[int, dict[str, Any]] = {}
    terminal_count = 0
    usage_count = 0
    saw_done = False

    for line in response.iter_lines(decode_unicode=True):
        if not line:
            continue
        assert line.startswith("data: "), line
        payload = line.removeprefix("data: ")
        if payload == "[DONE]":
            saw_done = True
            break
        chunk = json.loads(payload)
        if chunk.get("usage") is not None:
            usage_count += 1
        for choice in chunk.get("choices", []):
            if choice.get("finish_reason") is not None:
                terminal_count += 1
                assert choice["finish_reason"] == "tool_calls"
            delta = choice.get("delta") or {}
            if delta.get("reasoning_content") is not None:
                reasoning_parts.append(delta["reasoning_content"])
            if delta.get("content") is not None:
                content_parts.append(delta["content"])
            for call in delta.get("tool_calls") or []:
                index = call["index"]
                merged = tool_calls.setdefault(
                    index,
                    {"name": "", "arguments": ""},
                )
                function = call.get("function") or {}
                merged["name"] += function.get("name") or ""
                merged["arguments"] += function.get("arguments") or ""

    assert saw_done
    return (
        {
            "reasoning": "".join(reasoning_parts),
            "content": "".join(content_parts) or None,
            "tool_calls": [tool_calls[index] for index in sorted(tool_calls)],
        },
        terminal_count,
        usage_count,
    )


@pytest.mark.timeout(180)
def test_response_catalog_stream_and_unary_match_through_rust_frontend(
    request,
    runtime_services_dynamic_ports,
    dynamo_dynamic_ports,
    tmp_path: Path,
) -> None:
    _ = runtime_services_dynamic_ports
    model_path = _write_synthetic_model(tmp_path)
    catalog_path = _write_catalog(tmp_path)
    frontend_port = dynamo_dynamic_ports.frontend_port
    system_port = dynamo_dynamic_ports.system_ports[0]

    with DynamoFrontendProcess(
        request,
        frontend_port=frontend_port,
        terminate_all_matching_process_names=False,
    ):
        with MockerWorkerProcess(
            request,
            str(model_path),
            frontend_port,
            system_port,
            extra_args=[
                "--model-name",
                MODEL_NAME,
                "--response-catalog-path",
                str(catalog_path),
                "--model-output-profile",
                "qwen3_5",
            ],
        ):
            unary_response = requests.post(
                f"http://localhost:{frontend_port}/v1/chat/completions",
                json=_payload(stream=False),
                timeout=60,
            )
            unary_response.raise_for_status()
            unary = unary_response.json()
            choice = unary["choices"][0]
            message = choice["message"]
            unary_semantics = {
                "reasoning": message["reasoning_content"],
                "content": message.get("content"),
                "tool_calls": [
                    {
                        "name": call["function"]["name"],
                        "arguments": call["function"]["arguments"],
                    }
                    for call in message["tool_calls"]
                ],
            }
            assert choice["finish_reason"] == "tool_calls"

            streaming_response = requests.post(
                f"http://localhost:{frontend_port}/v1/chat/completions",
                json=_payload(stream=True),
                timeout=60,
                stream=True,
            )
            streamed_semantics, terminal_count, usage_count = _stream_semantics(
                streaming_response
            )

    assert streamed_semantics == unary_semantics
    assert unary_semantics == {
        "reasoning": "I should call the weather tool.",
        "content": None,
        "tool_calls": [
            {
                "name": "get_weather",
                "arguments": '{"city":"Seattle"}',
            }
        ],
    }
    assert terminal_count == 1
    assert usage_count == 1
    serialized = json.dumps(unary_semantics)
    for marker in ("<think>", "</think>", "<tool_call>", "<parameter="):
        assert marker not in serialized
