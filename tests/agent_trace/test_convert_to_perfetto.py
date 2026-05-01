# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from benchmarks.agent_trace.convert_to_perfetto import convert_records

pytestmark = [pytest.mark.pre_merge, pytest.mark.unit, pytest.mark.gpu_0]


def test_convert_records_emits_request_stages_and_metadata():
    trace, converted = convert_records(
        [
            {
                "timestamp": 10,
                "event": {
                    "schema": "dynamo.agent.trace.v1",
                    "event_type": "request_end",
                    "event_time_unix_ms": 2000,
                    "event_source": "dynamo",
                    "agent_context": {
                        "workflow_type_id": "ms_agent",
                        "workflow_id": "workflow-1",
                        "program_id": "workflow-1:researcher",
                    },
                    "request": {
                        "request_id": "req-1",
                        "x_request_id": "caller-1",
                        "model": "test-model",
                        "input_tokens": 100,
                        "output_tokens": 10,
                        "cached_tokens": 80,
                        "request_received_ms": 1000,
                        "prefill_wait_time_ms": 5,
                        "prefill_time_ms": 7,
                        "ttft_ms": 12,
                        "total_time_ms": 50,
                        "avg_itl_ms": 4.2,
                        "kv_hit_rate": 0.8,
                        "queue_depth": 2,
                        "worker": {
                            "prefill_worker_id": 1,
                            "prefill_dp_rank": 0,
                            "decode_worker_id": 2,
                            "decode_dp_rank": 0,
                        },
                    },
                },
            }
        ],
        include_stages=True,
        include_markers=True,
    )

    assert converted == 1
    events = trace["traceEvents"]
    assert [event for event in events if event["ph"] == "M"]

    request_events = [event for event in events if event.get("cat") == "dynamo.llm"]
    assert len(request_events) == 1
    request = request_events[0]
    assert request["name"] == "LLM request: test-model"
    assert request["ts"] == 1_000_000
    assert request["dur"] == 50_000
    assert request["args"]["x_request_id"] == "caller-1"
    assert request["args"]["worker.decode_worker_id"] == 2

    stage_events = [event for event in events if event.get("cat") == "dynamo.llm.stage"]
    assert {event["tid"] for event in stage_events} == {request["tid"]}

    stage_names = {event["name"] for event in stage_events}
    assert stage_names == {"prefill wait", "prefill", "decode"}

    markers = [event for event in events if event.get("cat") == "dynamo.llm.marker"]
    assert len(markers) == 1
    assert markers[0]["name"] == "first token"
    assert markers[0]["ts"] == 1_012_000


def test_convert_records_can_emit_stages_on_separate_tracks():
    trace, _ = convert_records(
        [
            {
                "event": {
                    "schema": "dynamo.agent.trace.v1",
                    "event_type": "request_end",
                    "event_time_unix_ms": 1050,
                    "agent_context": {
                        "workflow_id": "workflow-1",
                        "program_id": "workflow-1:researcher",
                    },
                    "request": {
                        "request_id": "req-1",
                        "model": "test-model",
                        "request_received_ms": 1000,
                        "prefill_wait_time_ms": 5,
                        "prefill_time_ms": 7,
                        "ttft_ms": 12,
                        "total_time_ms": 50,
                    },
                },
            }
        ],
        include_stages=True,
        include_markers=False,
        separate_stage_tracks=True,
    )

    request = next(
        event for event in trace["traceEvents"] if event.get("cat") == "dynamo.llm"
    )
    stage_tids = {
        event["tid"]
        for event in trace["traceEvents"]
        if event.get("cat") == "dynamo.llm.stage"
    }
    assert stage_tids != {request["tid"]}

    thread_names = [
        event["args"]["name"]
        for event in trace["traceEvents"]
        if event.get("name") == "thread_name"
    ]
    assert thread_names == [
        "workflow-1:researcher",
        "workflow-1:researcher stages",
    ]


def test_convert_records_clamps_stage_rounding_overlap():
    trace, _ = convert_records(
        [
            {
                "event": {
                    "schema": "dynamo.agent.trace.v1",
                    "event_type": "request_end",
                    "event_time_unix_ms": 49_743.776002,
                    "agent_context": {
                        "workflow_id": "workflow-1",
                        "program_id": "workflow-1:searcher",
                    },
                    "request": {
                        "request_id": "req-1",
                        "model": "test-model",
                        "request_received_ms": 0,
                        "prefill_wait_time_ms": 230.700717,
                        "prefill_time_ms": 2144.397696,
                        "ttft_ms": 2375.098413,
                        "total_time_ms": 49743.776002,
                    },
                },
            }
        ],
        include_stages=True,
        include_markers=False,
    )

    stages = sorted(
        [
            event
            for event in trace["traceEvents"]
            if event.get("cat") == "dynamo.llm.stage"
        ],
        key=lambda event: event["ts"],
    )
    assert [event["name"] for event in stages] == [
        "prefill wait",
        "prefill",
        "decode",
    ]
    for current, next_event in zip(stages, stages[1:]):
        assert current["ts"] + current["dur"] <= next_event["ts"]

    assert stages[1]["ts"] + stages[1]["dur"] == stages[2]["ts"]


def test_convert_records_splits_overlapping_program_requests_into_lanes():
    def record(request_id: str, start_ms: int, total_ms: int):
        return {
            "event": {
                "schema": "dynamo.agent.trace.v1",
                "event_type": "request_end",
                "event_time_unix_ms": start_ms + total_ms,
                "agent_context": {
                    "workflow_type_id": "ms_agent",
                    "workflow_id": "workflow-1",
                    "program_id": "workflow-1:searcher",
                },
                "request": {
                    "request_id": request_id,
                    "model": "test-model",
                    "request_received_ms": start_ms,
                    "ttft_ms": 10,
                    "total_time_ms": total_ms,
                },
            }
        }

    trace, converted = convert_records(
        [record("req-1", 1000, 100), record("req-2", 1050, 100)],
        include_stages=False,
        include_markers=False,
    )

    assert converted == 2
    assert not [
        event
        for event in trace["traceEvents"]
        if event.get("cat") == "dynamo.llm.marker"
    ]

    thread_names = [
        event["args"]["name"]
        for event in trace["traceEvents"]
        if event.get("name") == "thread_name"
    ]
    assert thread_names == [
        "workflow-1:searcher [lane 1]",
        "workflow-1:searcher [lane 2]",
    ]

    request_tids = {
        event["args"]["request_id"]: event["tid"]
        for event in trace["traceEvents"]
        if event.get("cat") == "dynamo.llm"
    }
    assert request_tids["req-1"] != request_tids["req-2"]
