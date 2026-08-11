# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU-only response-catalog coverage through the real Rust frontend pipeline."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, NamedTuple

import pytest
import requests
from huggingface_hub import snapshot_download

from tests.frontend.conftest import MockerWorkerProcess
from tests.utils.managed_process import DynamoFrontendProcess

MODEL_NAME = "synthetic-qwen-response-catalog"
REPO_ROOT = Path(__file__).resolve().parents[2]
TINY_MODEL = REPO_ROOT / "lib/llm/tests/data/sample-models/TinyLlama_v1.1"
GLM_TRUNCATED_OUTPUT = "<tool_call>get_weather<arg_key>city</arg_key><arg_value>Sea"

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
    pytest.mark.gpu_0,
    pytest.mark.e2e,
    pytest.mark.parallel,
]


class OfficialProfile(NamedTuple):
    name: str
    repo_id: str
    revision: str
    required_files: tuple[str, ...]


OFFICIAL_METADATA_PATTERNS = (
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "tiktoken.model",
    "*.tiktoken",
    "chat_template*.jinja",
    "encoding/**",
    "encoding_k3.py",
    "tokenization_kimi.py",
)

OFFICIAL_PROFILES = (
    pytest.param(
        OfficialProfile(
            "kimi_k3",
            "moonshotai/Kimi-K3",
            "9f62e4e9fffbd0a83ddd60e1c209d828994b3569",
            (
                "config.json",
                "tokenizer_config.json",
                "tiktoken.model",
                "encoding_k3.py",
            ),
        ),
        id="kimi_k3",
        marks=pytest.mark.model("moonshotai/Kimi-K3"),
    ),
    pytest.param(
        OfficialProfile(
            "deepseek_v4",
            "deepseek-ai/DeepSeek-V4-Pro",
            "b5968e9190ef611bbf34a7229255be88a0e937c1",
            (
                "config.json",
                "tokenizer.json",
                "tokenizer_config.json",
                "encoding/encoding_dsv4.py",
            ),
        ),
        id="deepseek_v4",
        marks=pytest.mark.model("deepseek-ai/DeepSeek-V4-Pro"),
    ),
    pytest.param(
        OfficialProfile(
            "qwen3_5",
            "Qwen/Qwen3.5-4B",
            "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a",
            (
                "config.json",
                "tokenizer.json",
                "tokenizer_config.json",
                "chat_template.jinja",
            ),
        ),
        id="qwen3_5",
        marks=pytest.mark.model("Qwen/Qwen3.5-4B"),
    ),
    pytest.param(
        OfficialProfile(
            "glm_5_2",
            "zai-org/GLM-5",
            "c183ef8c61faee82855eca1ed9bb3a9a7ce3b0b2",
            (
                "config.json",
                "tokenizer.json",
                "tokenizer_config.json",
                "chat_template.jinja",
            ),
        ),
        id="glm_5_2",
        marks=pytest.mark.model("zai-org/GLM-5"),
    ),
    pytest.param(
        OfficialProfile(
            "gpt_oss",
            "openai/gpt-oss-20b",
            "6cee5e81ee83917806bbde320786a8fb61efebee",
            (
                "config.json",
                "tokenizer.json",
                "tokenizer_config.json",
                "chat_template.jinja",
            ),
        ),
        id="gpt_oss",
        marks=pytest.mark.model("openai/gpt-oss-20b"),
    ),
)


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
                    },
                    {
                        "id": "glm-truncated",
                        "raw_output": f"</think>{GLM_TRUNCATED_OUTPUT}",
                        "finish_reason": "length",
                        "chunk_size": 3,
                    },
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


def _stream_semantics(
    response: requests.Response,
    expected_finish_reason: str = "tool_calls",
) -> tuple[dict[str, Any], int, int, list[dict[str, Any]]]:
    response.raise_for_status()
    reasoning_parts: list[str] = []
    content_parts: list[str] = []
    tool_calls: dict[int, dict[str, Any]] = {}
    terminal_count = 0
    usage_count = 0
    saw_done = False
    chunks: list[dict[str, Any]] = []

    for line in response.iter_lines(decode_unicode=True):
        if not line:
            continue
        assert line.startswith("data: "), line
        payload = line.removeprefix("data: ")
        if payload == "[DONE]":
            saw_done = True
            break
        chunk = json.loads(payload)
        chunks.append(chunk)
        if chunk.get("usage") is not None:
            usage_count += 1
        for choice in chunk.get("choices", []):
            if choice.get("finish_reason") is not None:
                terminal_count += 1
                assert choice["finish_reason"] == expected_finish_reason
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
        chunks,
    )


def _assert_response_catalog_e2e(
    request,
    dynamo_dynamic_ports,
    model_path: Path,
    catalog_path: Path,
    model_output_profile: str,
) -> None:
    frontend_port = dynamo_dynamic_ports.frontend_port
    system_port = dynamo_dynamic_ports.system_ports[0]
    glm_truncated = None

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
                model_output_profile,
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
            streamed_semantics, terminal_count, usage_count, chunks = _stream_semantics(
                streaming_response
            )

            if model_output_profile == "glm_5_2":
                truncated_unary_response = requests.post(
                    f"http://localhost:{frontend_port}/v1/chat/completions",
                    json=_payload(stream=False, case_id="glm-truncated"),
                    timeout=60,
                )
                truncated_unary_response.raise_for_status()
                truncated_unary = truncated_unary_response.json()
                truncated_choice = truncated_unary["choices"][0]
                truncated_message = truncated_choice["message"]
                truncated_stream_response = requests.post(
                    f"http://localhost:{frontend_port}/v1/chat/completions",
                    json=_payload(stream=True, case_id="glm-truncated"),
                    timeout=60,
                    stream=True,
                )
                (
                    truncated_streamed_semantics,
                    truncated_terminal_count,
                    truncated_usage_count,
                    truncated_chunks,
                ) = _stream_semantics(
                    truncated_stream_response,
                    expected_finish_reason="length",
                )
                glm_truncated = (
                    {
                        "reasoning": truncated_message.get("reasoning_content") or "",
                        "content": truncated_message.get("content"),
                        "tool_calls": truncated_message.get("tool_calls") or [],
                    },
                    truncated_choice["finish_reason"],
                    truncated_streamed_semantics,
                    truncated_terminal_count,
                    truncated_usage_count,
                    truncated_chunks,
                )

    assert streamed_semantics == unary_semantics, json.dumps(chunks, indent=2)
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

    if glm_truncated is not None:
        (
            truncated_unary_semantics,
            truncated_finish_reason,
            truncated_streamed_semantics,
            truncated_terminal_count,
            truncated_usage_count,
            truncated_chunks,
        ) = glm_truncated
        assert truncated_streamed_semantics == truncated_unary_semantics, json.dumps(
            truncated_chunks, indent=2
        )
        assert truncated_finish_reason == "length"
        assert truncated_terminal_count == 1
        assert truncated_usage_count == 1
        assert truncated_unary_semantics["reasoning"] == ""
        assert truncated_unary_semantics["tool_calls"] == []
        assert truncated_unary_semantics["content"].count(GLM_TRUNCATED_OUTPUT) == 1


@pytest.mark.pre_merge
@pytest.mark.timeout(180)
def test_response_catalog_stream_and_unary_match_through_rust_frontend(
    request,
    runtime_services_dynamic_ports,
    dynamo_dynamic_ports,
    tmp_path: Path,
) -> None:
    _ = runtime_services_dynamic_ports
    _assert_response_catalog_e2e(
        request,
        dynamo_dynamic_ports,
        _write_synthetic_model(tmp_path),
        _write_catalog(tmp_path),
        "qwen3_5",
    )


@pytest.mark.nightly
@pytest.mark.timeout(180)
@pytest.mark.parametrize("official_profile", OFFICIAL_PROFILES)
def test_response_catalog_against_pinned_official_tokenizer(
    request,
    runtime_services_dynamic_ports,
    dynamo_dynamic_ports,
    tmp_path: Path,
    official_profile: OfficialProfile,
) -> None:
    _ = runtime_services_dynamic_ports
    model_path = Path(
        snapshot_download(
            repo_id=official_profile.repo_id,
            revision=official_profile.revision,
            allow_patterns=OFFICIAL_METADATA_PATTERNS,
        )
    )
    for relative_path in official_profile.required_files:
        assert (model_path / relative_path).is_file(), relative_path

    _assert_response_catalog_e2e(
        request,
        dynamo_dynamic_ports,
        model_path,
        _write_catalog(tmp_path),
        official_profile.name,
    )
