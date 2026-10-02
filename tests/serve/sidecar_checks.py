# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import time
import uuid
from pathlib import Path

import requests

from tests.fault_tolerance.cancellation.utils import (
    CancellableRequest,
    read_streaming_responses,
)
from tests.utils.client import send_request
from tests.utils.payloads import ChatPayload, StreamingChatPayload
from tests.utils.prometheus import find_metric_samples


def _metrics(port: int) -> str:
    response = requests.get(f"http://127.0.0.1:{port}/metrics", timeout=2)
    response.raise_for_status()
    return response.text


def _wait_for_scheduler(backend: str, port: int, *, is_active: bool = False) -> None:
    names = {
        "vllm": ("vllm:num_requests_running", "vllm:num_requests_waiting"),
    }[backend]
    deadline = time.monotonic() + 10
    while True:
        body = _metrics(port)
        samples = [find_metric_samples(body, name) for name in names]
        assert all(samples), f"Missing scheduler metrics {names}: {body}"
        running, waiting = map(sum, samples)
        if running > 0 if is_active else running == waiting == 0:
            return
        assert (
            time.monotonic() < deadline
        ), f"{backend} scheduler is_active={is_active}: {body}"
        time.sleep(0.05)


def assert_cancellation_and_recovery(
    *, backend: str, model: str, frontend_port: int, engine_http_port: int
) -> None:
    """Disconnect active generation, check engine cleanup, then generate again."""
    _wait_for_scheduler(backend, engine_http_port)
    body = {
        "model": model,
        "messages": [{"role": "user", "content": "Count from one to a thousand."}],
        "max_tokens": 2048,
        "ignore_eos": True,
        "temperature": 0.0,
        "chat_template_kwargs": {"enable_thinking": False},
        "stream": True,
    }
    request = CancellableRequest()
    try:
        request.post(
            f"http://127.0.0.1:{frontend_port}/v1/chat/completions",
            json=body,
            stream=True,
            timeout=30,
        )
        deadline = time.monotonic() + 10
        while request.get_response() is None:
            request.raise_for_early_failure()
            assert (
                time.monotonic() < deadline
            ), "No streaming response before cancellation"
            time.sleep(0.05)
        read_streaming_responses(
            request, expected_count=1, deadline_s=10, require_content=True
        )
        _wait_for_scheduler(backend, engine_http_port, is_active=True)
    finally:
        request.cancel()
    _wait_for_scheduler(backend, engine_http_port)
    payload = StreamingChatPayload(
        body={**body, "max_tokens": 4, "stream_options": {"include_usage": True}},
        expected_response=[],
        expected_log=[],
        expected_finish_reason="length",
        expected_completion_tokens=4,
        port=frontend_port,
    )
    with send_request(payload.url(), payload.body, stream=True) as response:
        payload.process_response(response)
    _wait_for_scheduler(backend, engine_http_port)


def assert_kv_transfer(
    *,
    backend: str,
    payload: ChatPayload,
    prefill_http_port: int,
    decode_http_port: int,
    probe_path: Path,
) -> None:
    """Require a fresh completed transfer as well as a successful response."""

    def transferred() -> float:
        return sum(
            json.loads(line)["bytes"] for line in probe_path.read_text().splitlines()
        )

    before = transferred()
    payload.body["messages"][0]["content"] = (
        f"Request {uuid.uuid4()}. " + payload.body["messages"][0]["content"]
    )
    with send_request(payload.url(), payload.body) as response:
        payload.process_response(response)
    deadline = time.monotonic() + 10
    while transferred() <= before:
        assert time.monotonic() < deadline, f"{backend}: no completed KV transfer"
        time.sleep(0.05)
    _wait_for_scheduler(backend, decode_http_port)
    _wait_for_scheduler(backend, prefill_http_port)
