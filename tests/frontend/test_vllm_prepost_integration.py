# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Generator

import pytest
import requests

from tests.utils.constants import QWEN
from tests.utils.managed_process import DynamoFrontendProcess, ManagedProcess
from tests.utils.port_utils import ServicePorts

logger = logging.getLogger(__name__)

TEST_MODEL = QWEN
CAPTURE_PATH_ENV = "DYN_VLLM_PREPOST_CAPTURE_PATH"

SEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": "search_gutenberg_books",
        "description": "Search for books in the Project Gutenberg library",
        "parameters": {
            "type": "object",
            "properties": {
                "search_terms": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of search terms to find books",
                }
            },
            "required": ["search_terms"],
        },
    },
}

pytestmark = [
    pytest.mark.vllm,
    # vllm frontend doesn't need or use the GPU, but in CI pytorch seems to look for the Device
    pytest.mark.gpu_1,
    pytest.mark.pre_merge,
    pytest.mark.integration,
    pytest.mark.parallel,
    pytest.mark.model(TEST_MODEL),
]


class MockVllmPrepostWorkerProcess(ManagedProcess):
    """Test worker that captures frontend tokenized requests."""

    def __init__(
        self,
        request,
        *,
        frontend_port: int,
        capture_path: Path,
        worker_id: str = "vllm-prepost-worker",
    ) -> None:
        env = os.environ.copy()
        env[CAPTURE_PATH_ENV] = str(capture_path)

        super().__init__(
            command=["python3", "-m", "tests.frontend.vllm_prepost_worker"],
            env=env,
            health_check_urls=[
                (
                    f"http://localhost:{frontend_port}/v1/models",
                    self._check_models_api,
                )
            ],
            timeout=300,
            display_output=True,
            terminate_all_matching_process_names=False,
            straggler_commands=["-m tests.frontend.vllm_prepost_worker"],
            log_dir=f"{request.node.name}_{worker_id}",
        )

    @staticmethod
    def _check_models_api(response: requests.Response) -> bool:
        try:
            if response.status_code != 200:
                return False
            data = response.json()
        except (ValueError, KeyError):
            return False

        for model in data.get("data", []):
            if model.get("id") == TEST_MODEL:
                return True
        return False


def _read_captured_request(path: Path, timeout_s: float = 20.0) -> dict[str, Any]:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
        time.sleep(0.1)
    raise AssertionError(f"Timed out waiting for captured request at {path}")


def _collect_stream_chunks(response: requests.Response) -> list[dict[str, Any]]:
    response.raise_for_status()

    chunks: list[dict[str, Any]] = []
    saw_done = False
    for line in response.iter_lines(decode_unicode=True):
        if not line:
            continue
        assert line.startswith("data: "), f"Unexpected SSE line: {line!r}"
        payload = line[len("data: ") :]
        if payload == "[DONE]":
            saw_done = True
            break
        chunks.append(json.loads(payload))

    assert saw_done, "Missing [DONE] marker in SSE stream"
    assert chunks, "Expected streamed chunks but got none"
    return chunks


def _collect_reasoning(chunks: list[dict[str, Any]]) -> str:
    parts: list[str] = []
    for chunk in chunks:
        for choice in chunk.get("choices", []):
            reasoning = (choice.get("delta") or {}).get("reasoning_content")
            if reasoning is not None:
                parts.append(reasoning)
    return "".join(parts)


def _collect_tool_calls(chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged: dict[int, dict[str, Any]] = {}

    for chunk in chunks:
        for choice in chunk.get("choices", []):
            for tool_call in (choice.get("delta") or {}).get("tool_calls") or []:
                idx = tool_call["index"]
                if idx not in merged:
                    merged[idx] = {
                        "id": tool_call.get("id"),
                        "type": tool_call.get("type"),
                        "function": {
                            "name": tool_call.get("function", {}).get("name"),
                            "arguments": tool_call.get("function", {}).get(
                                "arguments", ""
                            ),
                        },
                    }
                    continue

                existing = merged[idx]
                if tool_call.get("id") and not existing["id"]:
                    existing["id"] = tool_call["id"]
                if tool_call.get("type") and not existing["type"]:
                    existing["type"] = tool_call["type"]

                incoming_fn = tool_call.get("function", {})
                if incoming_fn.get("name") and not existing["function"]["name"]:
                    existing["function"]["name"] = incoming_fn["name"]
                if incoming_fn.get("arguments"):
                    existing["function"]["arguments"] += incoming_fn["arguments"]

    return [merged[idx] for idx in sorted(merged)]


@pytest.fixture(scope="function")
def start_services(
    request,
    runtime_services_dynamic_ports,
    dynamo_dynamic_ports: ServicePorts,
    tmp_path: Path,
) -> Generator[tuple[int, Path], None, None]:
    _ = runtime_services_dynamic_ports

    frontend_port = dynamo_dynamic_ports.frontend_port
    capture_path = tmp_path / "captured_request.json"

    with DynamoFrontendProcess(
        request,
        frontend_port=frontend_port,
        extra_args=[
            "--dyn-chat-processor",
            "vllm",
            "--discovery-backend",
            "etcd",  # Started by the fixture
            "--request-plane",
            "tcp",
            "--enable-auto-tool-choice",
            "--tool-call-parser",
            "hermes",
            "--reasoning-parser",
            "qwen3",
        ],
        extra_env={"DYN_VLLM_STREAM_INTERVAL": "20"},
        terminate_all_matching_process_names=False,
    ):
        logger.info("Frontend started on port %s", frontend_port)
        with MockVllmPrepostWorkerProcess(
            request,
            frontend_port=frontend_port,
            capture_path=capture_path,
        ):
            logger.info("vLLM pre/post test worker registered model %s", TEST_MODEL)
            yield frontend_port, capture_path


@pytest.mark.timeout(120)
def test_vllm_chat_processor_tokenizes_and_streams_tool_calls(
    start_services: tuple[int, Path],
) -> None:
    frontend_port, capture_path = start_services

    payload = {
        "model": TEST_MODEL,
        "messages": [
            {
                "role": "user",
                "content": "What are the titles of some James Joyce books? Use the tool to search.",
            }
        ],
        "tools": [SEARCH_TOOL],
        "tool_choice": "auto",
        "stream": True,
        "max_tokens": 128,
    }

    response = requests.post(
        f"http://localhost:{frontend_port}/v1/chat/completions",
        json=payload,
        timeout=60,
        stream=True,
    )
    chunks = _collect_stream_chunks(response)
    captured = _read_captured_request(capture_path)

    assert captured["model"] == TEST_MODEL
    assert isinstance(captured["token_ids"], list) and captured["token_ids"]

    decoded_prompt = captured["decoded_prompt"]
    assert "What are the titles of some James Joyce books?" in decoded_prompt
    assert "search_gutenberg_books" in decoded_prompt

    reasoning = _collect_reasoning(chunks)
    assert "titles of some James Joyce books" in reasoning

    tool_calls = _collect_tool_calls(chunks)
    assert len(tool_calls) == 1
    tool_call = tool_calls[0]
    assert tool_call["function"]["name"] == "search_gutenberg_books"
    assert json.loads(tool_call["function"]["arguments"]) == {
        "search_terms": ["James Joyce", "Project Gutenberg"],
    }

    content = "".join(
        (choice.get("delta") or {}).get("content") or ""
        for chunk in chunks
        for choice in chunk.get("choices", [])
    )
    assert "<tool_call>" not in content
    assert "</tool_call>" not in content

    finish_reasons = [
        choice.get("finish_reason")
        for chunk in chunks
        for choice in chunk.get("choices", [])
        if choice.get("finish_reason")
    ]
    assert finish_reasons, "Expected at least one finish_reason"
    assert set(finish_reasons) <= {"stop", "tool_calls"}
