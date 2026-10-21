# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import contextlib
import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Generator, Optional

import pytest
import requests

from tests.conftest import EtcdServer, NatsServer
from tests.utils.constants import QWEN, DynamoPortRange
from tests.utils.managed_process import DynamoFrontendProcess, ManagedProcess
from tests.utils.port_utils import allocate_port, deallocate_port

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
    pytest.mark.core,
    # gpu_1 not gpu_0: vLLM DeviceConfig(device='auto') fails on CPU-only arm64
    # runners with "Failed to infer device type" even for mock tests.
    pytest.mark.gpu_1,
    pytest.mark.xpu_1,
    pytest.mark.profiled_vram_gib(0),
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
            timeout=60,
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


# Key of one service group: (discovery_backend, request_plane, event_plane).
_ServiceConfig = tuple[str, str, Optional[str]]


class _PrepostServiceGroup:
    """NATS/etcd, frontend and mock worker for one runtime configuration.

    Owns its runtime-service ports, frontend port and capture directory, so it
    does not depend on any function-scoped fixture and can outlive one case.
    """

    def __init__(self, request, config: _ServiceConfig, capture_dir: Path) -> None:
        self.config = config
        self.capture_path = capture_dir / "captured_request.json"
        self.frontend_port = 0
        # Same NATS_SERVER / ETCD_ENDPOINTS values that runtime_services_dynamic_ports
        # would set for this configuration; None means "unset".
        self.env: dict[str, Optional[str]] = {}
        self._processes: list[ManagedProcess] = []
        self._stack = contextlib.ExitStack()
        try:
            self._start(request)
        except BaseException:
            self._stack.close()
            raise

    def _start(self, request) -> None:
        discovery_backend, request_plane, event_plane = self.config
        # Mirrors the branch logic of runtime_services_dynamic_ports in tests/conftest.py.
        nats_required = request_plane == "nats" or event_plane == "nats"
        nats_free = event_plane == "zmq" and not nats_required
        start_nats = nats_required or (discovery_backend == "etcd" and not nats_free)
        start_etcd = discovery_backend == "etcd"

        if nats_free:
            self.env["NATS_SERVER"] = None
        if start_nats:
            nats = self._stack.enter_context(
                NatsServer(request, port=0, disable_jetstream=True)
            )
            self._processes.append(nats)
            self.env["NATS_SERVER"] = f"nats://localhost:{nats.port}"
        if start_etcd:
            etcd = self._stack.enter_context(EtcdServer(request, port=0))
            self._processes.append(etcd)
            self.env["ETCD_ENDPOINTS"] = f"http://localhost:{etcd.port}"

        self.frontend_port = allocate_port(DynamoPortRange.FRONTEND.value)
        self._stack.callback(deallocate_port, self.frontend_port)

        # The children copy os.environ when they are constructed, so the group's
        # environment only has to be in place while they are spawned. Restoring it
        # right away leaves nothing for the function-scoped monkeypatch to unwind.
        with pytest.MonkeyPatch.context() as mp:
            for name, value in self.env.items():
                if value is None:
                    mp.delenv(name, raising=False)
                else:
                    mp.setenv(name, value)

            frontend = self._stack.enter_context(
                DynamoFrontendProcess(
                    request,
                    frontend_port=self.frontend_port,
                    extra_args=[
                        "--dyn-chat-processor",
                        "vllm",
                        "--discovery-backend",
                        "etcd",  # Started by the group
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
                )
            )
            self._processes.append(frontend)
            logger.info("Frontend started on port %s", self.frontend_port)

            worker = self._stack.enter_context(
                MockVllmPrepostWorkerProcess(
                    request,
                    frontend_port=self.frontend_port,
                    capture_path=self.capture_path,
                )
            )
            self._processes.append(worker)
            logger.info("vLLM pre/post test worker registered model %s", TEST_MODEL)

    def is_healthy(self) -> bool:
        return all(process.is_running() for process in self._processes)

    def close(self) -> None:
        # Worker, frontend, etcd, NATS, then the frontend port reservation.
        self._stack.close()


class _PrepostServiceManager:
    """Module-owned holder of at most one live _PrepostServiceGroup.

    Cases with the same runtime configuration share a group. A case with a
    different configuration (for example request_plane=tcp, event_plane=zmq)
    closes the live group and gets its own, so it never reuses services that
    were started for another transport.
    """

    def __init__(self, request, tmp_path_factory: pytest.TempPathFactory) -> None:
        self._request = request
        self._tmp_path_factory = tmp_path_factory
        self._group: Optional[_PrepostServiceGroup] = None

    def get(self, config: _ServiceConfig) -> _PrepostServiceGroup:
        group = self._group
        if group is not None and group.config == config and group.is_healthy():
            logger.info("Reusing vLLM pre/post services for config %s", config)
            return group
        if group is not None:
            if group.config == config:
                logger.warning(
                    "vLLM pre/post services for config %s exited; restarting", config
                )
            self.close()
        logger.info("Starting vLLM pre/post services for config %s", config)
        capture_dir = self._tmp_path_factory.mktemp("vllm_prepost_capture")
        self._group = _PrepostServiceGroup(self._request, config, capture_dir)
        return self._group

    def close(self) -> None:
        group, self._group = self._group, None
        if group is not None:
            group.close()


@pytest.fixture(scope="module")
def prepost_service_manager(
    request, tmp_path_factory: pytest.TempPathFactory
) -> Generator[_PrepostServiceManager, None, None]:
    manager = _PrepostServiceManager(request, tmp_path_factory)
    try:
        yield manager
    finally:
        manager.close()


@pytest.fixture(scope="function")
def start_services(
    discovery_backend,
    request_plane,
    event_plane,
    monkeypatch,
    prepost_service_manager: _PrepostServiceManager,
) -> Generator[tuple[int, Path], None, None]:
    # event_plane has already set DYN_EVENT_PLANE through this case's monkeypatch,
    # so a newly started group inherits it.
    group = prepost_service_manager.get((discovery_backend, request_plane, event_plane))

    # The worker writes <capture>.tmp and then renames it onto the capture path.
    # Remove both so this case cannot read the previous case's request.
    group.capture_path.unlink(missing_ok=True)
    group.capture_path.with_suffix(group.capture_path.suffix + ".tmp").unlink(
        missing_ok=True
    )

    # Same per-case environment that runtime_services_dynamic_ports provides.
    for name, value in group.env.items():
        if value is None:
            monkeypatch.delenv(name, raising=False)
        else:
            monkeypatch.setenv(name, value)

    yield group.frontend_port, group.capture_path


@pytest.mark.timeout(180)  # 0-GiB unit test, floor 180s (3x observed 43s)
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

    guided_decoding = captured["sampling_options"].get("guided_decoding")
    assert isinstance(guided_decoding, dict)
    structural_tag = guided_decoding.get("structural_tag")
    assert isinstance(structural_tag, dict)
    serialized_tag = json.dumps(structural_tag)
    assert "search_gutenberg_books" in serialized_tag
    assert "search_terms" in serialized_tag

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


@pytest.mark.timeout(180)  # 0-GiB unit test, floor 180s (3x observed 43s)
def test_vllm_chat_processor_forwards_max_thinking_tokens(
    start_services: tuple[int, Path],
) -> None:
    """nvext.max_thinking_tokens reaches the worker as
    stop_conditions.max_thinking_tokens after the Python frontend processes it."""
    frontend_port, capture_path = start_services

    payload = {
        "model": TEST_MODEL,
        "messages": [{"role": "user", "content": "Solve: 1+1."}],
        "max_tokens": 32,
        "nvext": {"max_thinking_tokens": 16},
    }

    response = requests.post(
        f"http://localhost:{frontend_port}/v1/chat/completions",
        json=payload,
        timeout=60,
    )
    response.raise_for_status()
    captured = _read_captured_request(capture_path)

    assert captured["stop_conditions"]["max_thinking_tokens"] == 16


@pytest.mark.timeout(180)
def test_vllm_chat_processor_forwards_thinking_token_budget(
    start_services: tuple[int, Path],
) -> None:
    """Root-level `thinking_token_budget` reaches the worker as
    stop_conditions.max_thinking_tokens after the Python frontend processes it."""
    frontend_port, capture_path = start_services

    payload = {
        "model": TEST_MODEL,
        "messages": [{"role": "user", "content": "Solve: 1+1."}],
        "max_tokens": 32,
        "thinking_token_budget": 32,
    }

    response = requests.post(
        f"http://localhost:{frontend_port}/v1/chat/completions",
        json=payload,
        timeout=60,
    )
    response.raise_for_status()
    captured = _read_captured_request(capture_path)

    assert captured["stop_conditions"]["max_thinking_tokens"] == 32


@pytest.mark.timeout(180)
def test_vllm_chat_processor_thinking_token_budget_overrides_nvext(
    start_services: tuple[int, Path],
) -> None:
    """Root-level `thinking_token_budget` takes precedence over the legacy
    nvext.max_thinking_tokens field."""
    frontend_port, capture_path = start_services

    payload = {
        "model": TEST_MODEL,
        "messages": [{"role": "user", "content": "Solve: 1+1."}],
        "max_tokens": 32,
        "thinking_token_budget": 32,
        "nvext": {"max_thinking_tokens": 16},
    }

    response = requests.post(
        f"http://localhost:{frontend_port}/v1/chat/completions",
        json=payload,
        timeout=60,
    )
    response.raise_for_status()
    captured = _read_captured_request(capture_path)

    assert captured["stop_conditions"]["max_thinking_tokens"] == 32


@pytest.mark.timeout(180)
def test_vllm_responses_processor_forwards_thinking_token_budget(
    start_services: tuple[int, Path],
) -> None:
    """Root-level `thinking_token_budget` on a request reaches the
    worker as stop_conditions.max_thinking_tokens after JSON deserialization."""
    frontend_port, capture_path = start_services

    payload = {
        "model": TEST_MODEL,
        "input": "Solve: 1+1.",
        "max_output_tokens": 32,
        "stream": False,
        "thinking_token_budget": 32,
    }

    response = requests.post(
        f"http://localhost:{frontend_port}/v1/responses",
        json=payload,
        timeout=60,
    )
    response.raise_for_status()
    captured = _read_captured_request(capture_path)

    assert captured["stop_conditions"]["max_thinking_tokens"] == 32


@pytest.mark.timeout(180)
def test_vllm_responses_processor_thinking_token_budget_overrides_nvext(
    start_services: tuple[int, Path],
) -> None:
    """Root-level `thinking_token_budget` takes precedence over the legacy
    nvext.max_thinking_tokens field."""
    frontend_port, capture_path = start_services

    payload = {
        "model": TEST_MODEL,
        "input": "Solve: 1+1.",
        "max_output_tokens": 32,
        "stream": False,
        "thinking_token_budget": 32,
        "nvext": {"max_thinking_tokens": 16},
    }

    response = requests.post(
        f"http://localhost:{frontend_port}/v1/responses",
        json=payload,
        timeout=60,
    )
    response.raise_for_status()
    captured = _read_captured_request(capture_path)

    assert captured["stop_conditions"]["max_thinking_tokens"] == 32


@pytest.mark.timeout(180)
def test_vllm_responses_processor_forwards_max_thinking_tokens(
    start_services: tuple[int, Path],
) -> None:
    """Legacy nvext.max_thinking_tokens alone on request still
    reaches the worker as stop_conditions.max_thinking_tokens."""
    frontend_port, capture_path = start_services

    payload = {
        "model": TEST_MODEL,
        "input": "Solve: 1+1.",
        "max_output_tokens": 32,
        "stream": False,
        "nvext": {"max_thinking_tokens": 16},
    }

    response = requests.post(
        f"http://localhost:{frontend_port}/v1/responses",
        json=payload,
        timeout=60,
    )
    response.raise_for_status()
    captured = _read_captured_request(capture_path)

    assert captured["stop_conditions"]["max_thinking_tokens"] == 16
