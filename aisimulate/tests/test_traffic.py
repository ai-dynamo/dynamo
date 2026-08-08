# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import math

import pytest

from aisimulate.traffic import materialize_configured_traffic

pytestmark = [
    pytest.mark.pre_merge,
    pytest.mark.planner,
    pytest.mark.gpu_0,
    pytest.mark.unit,
]


def _write_jsonl(path, rows):
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    return path


def test_materializes_canonical_inline_requests_without_trace_io():
    assert materialize_configured_traffic(
        {
            "requests": [
                {
                    "id": "exact",
                    "arrival_time_ms": 1.5,
                    "input_token_ids": [10, 20, 30],
                    "output_tokens": 2,
                    "session_id": "session-a",
                    "metadata": {"source": "inline"},
                },
                {
                    "arrival_time_ms": 4,
                    "input_tokens": 7,
                    "output_tokens": 1,
                },
            ]
        }
    ) == [
        {
            "id": "exact",
            "arrival_time_ms": 1.5,
            "input_tokens": 3,
            "input_token_ids": [10, 20, 30],
            "output_tokens": 2,
            "session_id": "session-a",
            "metadata": {"source": "inline"},
        },
        {
            "id": "request-1",
            "arrival_time_ms": 4.0,
            "input_tokens": 7,
            "output_tokens": 1,
            "metadata": None,
        },
    ]


@pytest.mark.parametrize("selector", ["trace", "trace_path"])
def test_materializes_static_mooncake_with_global_hash_interning(tmp_path, selector):
    large_hash = 2**40 + 7
    trace = _write_jsonl(
        tmp_path / "requests.jsonl",
        [
            {
                "request_id": "later",
                "input_length": 2000,
                "output_length": 3,
                "hash_ids": [large_hash, 17],
                "timestamp": 104,
                "priority": -2,
                "strict_priority": 3,
                "policy_class": "gold",
            },
            {
                "request_id": "first",
                "session_id": "single-turn-session",
                "input_tokens": 513,
                "output_tokens": 1,
                "hash_ids": [17, large_hash],
                "created_time": 100,
            },
        ],
    )

    requests = materialize_configured_traffic(
        {
            selector: {"path": str(trace)},
            "trace_block_size": 512,
            "speedup": 2,
        }
    )

    assert [request["id"] for request in requests] == ["first", "later"]
    assert [request["arrival_time_ms"] for request in requests] == [0.0, 2.0]
    assert requests[0]["input_tokens"] == 513
    assert requests[0]["input_token_ids"] == [1] * 512 + [0]
    assert requests[0]["session_id"] == "single-turn-session"
    assert requests[0]["metadata"] == {"priority": 0, "strict_priority": 0}
    assert requests[1]["input_tokens"] == 1024
    assert requests[1]["input_token_ids"] == [0] * 512 + [1] * 512
    assert requests[1]["metadata"] == {
        "priority": -2,
        "strict_priority": 3,
        "policy_class": "gold",
    }


def test_generated_ids_do_not_collide_with_authored_ids(tmp_path):
    trace = _write_jsonl(
        tmp_path / "requests.jsonl",
        [
            {
                "input_length": 1,
                "output_length": 1,
                "hash_ids": [1],
                "timestamp": 0,
            },
            {
                "request_id": "trace-1",
                "input_length": 1,
                "output_length": 1,
                "hash_ids": [2],
                "timestamp": 1,
            },
        ],
    )

    requests = materialize_configured_traffic({"trace": str(trace)})

    assert [request["id"] for request in requests] == ["trace-1-2", "trace-1"]


@pytest.mark.parametrize(
    ("config", "message"),
    [
        ({"trace": "x", "trace_path": "y"}, "exactly one"),
        ({"trace": "x", "format": "dynamo"}, "only format='mooncake'"),
        ({"trace": "x", "trace_block_size": 0}, "positive integer"),
        ({"trace": "x", "speedup": 0}, "must be positive"),
        ({"trace": "x", "speedup": math.inf}, "finite number"),
        ({"trace": {"path": "x", "paths": ["y"]}}, "unsupported field"),
        ({"trace": "x", "replay": {}}, "unsupported field"),
    ],
)
def test_rejects_unsupported_trace_configuration(config, message):
    with pytest.raises(ValueError, match=message):
        materialize_configured_traffic(config)


@pytest.mark.parametrize(
    ("row", "message"),
    [
        (
            {
                "input_length": 1,
                "output_length": 1,
                "output_token_ids": [4],
                "hash_ids": [1],
                "timestamp": 0,
            },
            "authored output_token_ids",
        ),
        (
            {
                "session_id": "s",
                "input_length": 1,
                "output_length": 1,
                "hash_ids": [1],
                "timestamp": 0,
                "delay": 1,
            },
            "session follow-up",
        ),
        (
            {
                "request_id": "r",
                "input_length": 1,
                "output_length": 1,
                "hash_ids": [1],
                "timestamp": 0,
                "wait_for": ["other"],
            },
            "unsupported field",
        ),
        (
            {
                "input_length": 1,
                "output_length": 1,
                "hash_ids": [2**64],
                "timestamp": 0,
            },
            "unsigned 64-bit",
        ),
        (
            {
                "input_length": 1,
                "output_length": 1,
                "timestamp": 0,
            },
            "hash_ids must be a non-empty list",
        ),
    ],
)
def test_rejects_rows_that_cannot_be_replayed_exactly(tmp_path, row, message):
    trace = _write_jsonl(tmp_path / "requests.jsonl", [row])

    with pytest.raises(ValueError, match=message):
        materialize_configured_traffic({"trace": str(trace)})


def test_rejects_session_follow_up_and_duplicate_request_ids(tmp_path):
    follow_up = _write_jsonl(
        tmp_path / "follow-up.jsonl",
        [
            {
                "request_id": "one",
                "session_id": "session",
                "input_length": 1,
                "output_length": 1,
                "hash_ids": [1],
                "timestamp": 0,
            },
            {
                "request_id": "two",
                "session_id": "session",
                "input_length": 2,
                "output_length": 1,
                "hash_ids": [1],
                "timestamp": 1,
            },
        ],
    )
    with pytest.raises(ValueError, match="multiple rows for session"):
        materialize_configured_traffic({"trace": str(follow_up)})

    duplicate = _write_jsonl(
        tmp_path / "duplicate.jsonl",
        [
            {
                "request_id": "same",
                "input_length": 1,
                "output_length": 1,
                "hash_ids": [1],
                "timestamp": 0,
            },
            {
                "request_id": "same",
                "input_length": 1,
                "output_length": 1,
                "hash_ids": [2],
                "timestamp": 1,
            },
        ],
    )
    with pytest.raises(ValueError, match="duplicate request_id"):
        materialize_configured_traffic({"trace": str(duplicate)})


def test_missing_timestamp_and_explicit_zero_delay_are_burst_arrivals(tmp_path):
    trace = _write_jsonl(
        tmp_path / "burst.jsonl",
        [
            {
                "input_length": 1,
                "output_length": 1,
                "hash_ids": [1],
                "delay_ms": 0,
            }
        ],
    )

    requests = materialize_configured_traffic({"trace": str(trace)})

    assert requests[0]["arrival_time_ms"] == 0.0


def test_reports_path_and_line_for_invalid_json(tmp_path):
    trace = tmp_path / "invalid.jsonl"
    trace.write_text("\n{not-json}\n")

    with pytest.raises(ValueError, match=r"invalid\.jsonl line 2 as JSON"):
        materialize_configured_traffic({"trace": str(trace)})
