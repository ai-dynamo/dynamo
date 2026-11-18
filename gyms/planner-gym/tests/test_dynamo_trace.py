# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the generic Dynamo request-trace reader."""

from __future__ import annotations

import gzip
import json

import pytest
from autoscaling_arena.datasets import (
    TRACE_SCHEMA,
    discover_trace_shards,
    parse_trace_event,
)


def _request_event(
    *,
    input_length: int = 65,
    hashes: list[int] | None = None,
    session_id: str | None = None,
) -> bytes:
    agent_context = None if session_id is None else {"session_id": session_id}
    return json.dumps(
        {
            "event": {
                "schema": TRACE_SCHEMA,
                "event_type": "request_end",
                "event_time_unix_ms": 1200,
                **({"agent_context": agent_context} if agent_context else {}),
                "request": {
                    "request_id": "request-1",
                    "request_received_ms": 1000,
                    "output_tokens": 4,
                    "total_time_ms": 200.0,
                    "replay": {
                        "input_length": input_length,
                        "trace_block_size": 64,
                        "input_sequence_hashes": hashes or [11, 12],
                    },
                },
            }
        }
    ).encode()


def test_parse_request_event():
    event = parse_trace_event(_request_event(), "trace.jsonl:1")

    assert event is not None
    assert event.is_request
    assert event.timestamp_ms == 1000
    assert event.request is not None
    assert event.request.request_id == "request-1"
    assert event.request.end_ms == 1200
    assert event.request.input_sequence_hashes == (11, 12)


def test_parse_request_event_preserves_optional_session_context():
    event = parse_trace_event(
        _request_event(session_id="conversation-7"), "trace.jsonl:1"
    )

    assert event is not None and event.request is not None
    assert event.request.session_id == "conversation-7"


def test_parse_request_event_validates_hash_count():
    with pytest.raises(ValueError, match="requires 2 hashes, got 1"):
        parse_trace_event(_request_event(hashes=[11]), "trace.jsonl:1")


def test_discover_trace_shards_deduplicates_plain_and_gzip(tmp_path):
    plain = tmp_path / "trace-1.jsonl"
    compressed = tmp_path / "trace-1.jsonl.gz"
    plain.write_bytes(_request_event() + b"\n")
    with gzip.open(compressed, "wb") as handle:
        handle.write(_request_event() + b"\n")

    assert discover_trace_shards(tmp_path) == [compressed]
    assert discover_trace_shards(tmp_path, prefer_uncompressed=True) == [plain]
