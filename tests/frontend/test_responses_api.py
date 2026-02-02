# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Basic validation of the Responses API (/v1/responses) endpoint.
# For full compliance testing, use the OpenResponses bun CLI:
#   cd ~/openresponses && bun run test:compliance --base-url http://localhost:9000/v1 --api-key test --model <model>
# See https://www.openresponses.org/compliance

from __future__ import annotations

import json
import logging
from typing import Any, Dict

import pytest
import requests

from tests.utils.constants import QWEN

logger = logging.getLogger(__name__)

TEST_MODEL = QWEN

pytestmark = [
    pytest.mark.e2e,
    pytest.mark.gpu_0,  # Mocker is CPU-only (no GPU required)
    pytest.mark.post_merge,
    pytest.mark.parallel,
    pytest.mark.model(TEST_MODEL),
]


def _post_responses(
    payload: Dict[str, Any],
    frontend_port: int,
    timeout: int = 180,
    stream: bool = False,
) -> requests.Response:
    """Send a request to the Responses API endpoint."""
    headers = {"Content-Type": "application/json"}
    return requests.post(
        f"http://localhost:{frontend_port}/v1/responses",
        headers=headers,
        json=payload,
        timeout=timeout,
        stream=stream,
    )


# -- Non-streaming tests --


def test_basic_text_response(start_services_with_mocker) -> None:
    """Test basic text input returns a valid response object."""
    frontend_port = start_services_with_mocker
    payload = {
        "model": TEST_MODEL,
        "input": "What is the capital of France?",
    }

    resp = _post_responses(payload, frontend_port)

    assert resp.status_code == 200, (
        f"Expected 200, got {resp.status_code}: {resp.text}"
    )

    data = resp.json()
    assert data["object"] == "response"
    assert data["id"].startswith("resp_")
    assert data["status"] == "completed"
    assert data["model"] == TEST_MODEL
    assert len(data["output"]) > 0

    # First output item should be a message with text content
    msg = data["output"][0]
    assert msg["type"] == "message"
    assert msg["role"] == "assistant"
    assert len(msg["content"]) > 0
    assert msg["content"][0]["type"] == "output_text"
    assert len(msg["content"][0]["text"]) > 0


def test_system_prompt_via_instructions(start_services_with_mocker) -> None:
    """Test that instructions field is accepted and produces a response."""
    frontend_port = start_services_with_mocker
    payload = {
        "model": TEST_MODEL,
        "instructions": "You are a helpful assistant.",
        "input": "Hello",
    }

    resp = _post_responses(payload, frontend_port)

    assert resp.status_code == 200, (
        f"Expected 200, got {resp.status_code}: {resp.text}"
    )
    data = resp.json()
    assert data["status"] == "completed"
    assert len(data["output"]) > 0


def test_input_items_multi_turn(start_services_with_mocker) -> None:
    """Test Input::Items with user and assistant message history."""
    frontend_port = start_services_with_mocker
    payload = {
        "model": TEST_MODEL,
        "input": [
            {"type": "message", "role": "user", "content": "My name is Alice."},
            {
                "type": "message",
                "role": "assistant",
                "content": "Hello Alice! How can I help you?",
            },
            {
                "type": "message",
                "role": "user",
                "content": "What is my name?",
            },
        ],
    }

    resp = _post_responses(payload, frontend_port)

    assert resp.status_code == 200, (
        f"Expected 200, got {resp.status_code}: {resp.text}"
    )
    data = resp.json()
    assert data["status"] == "completed"
    assert len(data["output"]) > 0


# -- Streaming tests --


def test_streaming_response(start_services_with_mocker) -> None:
    """Test streaming returns valid SSE events in the correct order."""
    frontend_port = start_services_with_mocker
    payload = {
        "model": TEST_MODEL,
        "input": "Count to five.",
        "stream": True,
    }

    resp = _post_responses(payload, frontend_port, stream=True)

    assert resp.status_code == 200, (
        f"Expected 200, got {resp.status_code}: {resp.text}"
    )
    assert "text/event-stream" in resp.headers.get("content-type", "")

    events = []
    for line in resp.iter_lines(decode_unicode=True):
        if not line:
            continue
        if line.startswith("event: "):
            event_type = line[len("event: "):]
        elif line.startswith("data: "):
            data_str = line[len("data: "):]
            if data_str == "[DONE]":
                events.append(("done", None))
            else:
                events.append((event_type, json.loads(data_str)))

    event_types = [e[0] for e in events]

    # Verify required lifecycle events are present and in order
    assert event_types[0] == "response.created"
    assert event_types[1] == "response.in_progress"
    assert "response.output_item.added" in event_types
    assert "response.content_part.added" in event_types
    assert "response.output_text.delta" in event_types
    assert "response.output_text.done" in event_types
    assert "response.content_part.done" in event_types
    assert "response.output_item.done" in event_types

    # Last real event before [DONE] should be response.completed
    non_done = [e for e in event_types if e != "done"]
    assert non_done[-1] == "response.completed"

    # Verify text deltas concatenate to the final text
    deltas = [
        e[1]["delta"] for e in events if e[0] == "response.output_text.delta"
    ]
    done_events = [e for e in events if e[0] == "response.output_text.done"]
    assert len(done_events) == 1
    assert done_events[0][1]["text"] == "".join(deltas)
