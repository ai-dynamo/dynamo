# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from typing import Any
from unittest.mock import Mock

import pytest
import requests

from tests.utils.payloads import ResponsesPayload, ResponsesStreamPayload

pytestmark = [pytest.mark.unit, pytest.mark.pre_merge, pytest.mark.gpu_0]


def _message(item_id: str, text: str, phase: str | None = None) -> dict[str, Any]:
    return {
        "id": item_id,
        "type": "message",
        "role": "assistant",
        "phase": phase,
        "content": [{"type": "output_text", "text": text}],
    }


@pytest.mark.parametrize(
    "phases,expected",
    [
        (["final_answer", "final_answer"], "later"),
        ([None, None], "later"),
        (["final_answer", "commentary"], "earlier"),
    ],
)
def test_unary_selects_last_final_answer(
    phases: list[str | None], expected: str
) -> None:
    response = Mock(spec=requests.Response)
    response.json.return_value = {
        "object": "response",
        "id": "resp_test",
        "status": "completed",
        "output": [
            {"type": "reasoning", "content": []},
            _message("msg_1", "earlier", phases[0]),
            _message("msg_2", "later", phases[1]),
        ],
    }

    assert ResponsesPayload.extract_content(response) == expected


def _stream_response(texts: dict[str, str], done_ids: list[str]) -> Mock:
    events: list[tuple[str, dict[str, Any]]] = [
        ("response.created", {}),
        ("response.in_progress", {}),
    ]
    for item_id, text in texts.items():
        events.extend(
            [
                ("response.output_item.added", {"item": _message(item_id, "")}),
                ("response.content_part.added", {"item_id": item_id}),
            ]
        )
        if text:
            events.append(
                ("response.output_text.delta", {"item_id": item_id, "delta": text})
            )
        events.extend(
            ("response.output_text.done", {"item_id": item_id, "text": text})
            for _ in range(done_ids.count(item_id))
        )
        events.extend(
            [
                ("response.content_part.done", {"item_id": item_id}),
                ("response.output_item.done", {"item": _message(item_id, text)}),
            ]
        )
    events.append(("response.completed", {}))
    response = Mock(spec=requests.Response)
    response.iter_lines.return_value = iter(
        line
        for event_type, data in events
        for line in (f"event: {event_type}", f"data: {json.dumps(data)}", "")
    )
    return response


@pytest.mark.parametrize(
    "done_ids,error",
    [
        (["msg_1", "msg_2"], None),
        (["msg_2"], "Missing output_text.done for items"),
        (["msg_1", "msg_2", "msg_2"], "Duplicate output_text.done"),
    ],
)
def test_stream_requires_one_text_done_per_item(
    done_ids: list[str], error: str | None
) -> None:
    response = _stream_response({"msg_1": "first", "msg_2": "second"}, done_ids)
    if error is None:
        assert ResponsesStreamPayload.extract_content(response) == "firstsecond"
    else:
        with pytest.raises(AssertionError, match=error):
            ResponsesStreamPayload.extract_content(response)


def test_stream_accepts_empty_done_without_text_deltas() -> None:
    response = _stream_response(
        {"msg_empty": "", "msg_final": "answer"}, ["msg_empty", "msg_final"]
    )

    assert ResponsesStreamPayload.extract_content(response) == "answer"
