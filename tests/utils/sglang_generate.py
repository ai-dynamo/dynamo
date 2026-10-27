# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Client and contract checks for SGLang's native streaming `/generate` API.

Shared by the real-worker and mocker tests so both are held to one definition of
what a native stream owes its caller.
"""

from __future__ import annotations

import json
import math
from typing import Any

from tests.utils.client import send_request


def stream_generate(
    *,
    frontend_port: int,
    body: dict[str, Any],
    timeout: float = 120,
) -> list[dict[str, Any]]:
    """POST a streaming `/generate` request and return its SSE data events."""

    with send_request(
        f"http://localhost:{frontend_port}/generate",
        {**body, "stream": True},
        timeout=timeout,
        stream=True,
    ) as response:
        assert response.status_code == 200, response.text
        events: list[dict[str, Any]] = []
        stream_finished = False
        for line in response.iter_lines(decode_unicode=True):
            if not line or not line.startswith("data:"):
                continue
            data = line.removeprefix("data:").strip()
            if data == "[DONE]":
                stream_finished = True
                continue
            event = json.loads(data)
            assert isinstance(event, dict), event
            assert not event.get("error"), event
            events.append(event)

    assert stream_finished, events
    assert events, "SGLang /generate returned no SSE data events"
    return events


def assert_native_stream(
    events: list[dict[str, Any]],
    *,
    prompt_tokens: int,
) -> list[int]:
    """Assert the invariants of a stream requested with `return_logprob`, and
    return its concatenated output IDs.

    Chunks carry only their own new tokens, each with an aligned logprob entry;
    the terminal response reports the prompt size and the cumulative output size.
    """

    output_ids: list[int] = []
    for event in events:
        event_ids = event.get("output_ids")
        assert isinstance(event_ids, list) and all(
            isinstance(token_id, int) for token_id in event_ids
        ), event

        meta_info = event.get("meta_info")
        assert isinstance(meta_info, dict), event
        if not event_ids:
            continue

        logprobs = meta_info.get("output_token_logprobs")
        assert isinstance(logprobs, list), event
        assert len(logprobs) == len(event_ids), event
        for token_id, entry in zip(event_ids, logprobs, strict=True):
            assert isinstance(entry, list) and len(entry) >= 2, event
            logprob, logprob_token_id = entry[:2]
            assert isinstance(logprob, (int, float)) and math.isfinite(logprob), event
            assert logprob_token_id == token_id, event

        output_ids.extend(event_ids)

    assert output_ids, events
    meta_info = events[-1]["meta_info"]
    assert meta_info.get("prompt_tokens") == prompt_tokens, events[-1]
    assert meta_info.get("completion_tokens") == len(output_ids), {
        "completion_tokens": meta_info.get("completion_tokens"),
        "output_ids": output_ids,
    }
    return output_ids
