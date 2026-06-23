# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.request_trace.convert_to_perfetto import convert_records  # noqa: E402


def check_convert_records_emits_request_stages_and_metadata():
    trace, converted = convert_records(
        [
            {
                "timestamp": 10,
                "event": {
                    "schema": "dynamo.request.trace.v1",
                    "event_type": "request_end",
                    "event_time_unix_ms": 2000,
                    "event_source": "dynamo",
                    "agent_context": {
                        "trajectory_id": "session-1:researcher",
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
                        "finish_reason_metadata": {
                            "finish_reason": "tool_calls",
                            "backend_finish_reason": "stop",
                            "stop_reason": "END",
                            "tool_calls": [
                                {
                                    "choice_index": 0,
                                    "tool_call_index": 0,
                                    "id": "call-1",
                                    "name": "web_search",
                                }
                            ],
                            "choices": [
                                {
                                    "choice_index": 0,
                                    "finish_reason": "tool_calls",
                                    "backend_finish_reason": "stop",
                                    "stop_reason": "END",
                                }
                            ],
                        },
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

    assert converted == 2
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
    assert request["args"]["finish.finish_reason"] == "tool_calls"
    assert request["args"]["finish.backend_finish_reason"] == "stop"
    assert request["args"]["finish.stop_reason"] == "END"
    assert request["args"]["finish.tool_call_count"] == 1
    assert request["args"]["finish.tool_call_names"] == "web_search"
    assert request["args"]["finish.choice_count"] == 1
    assert request["args"]["finish.choice_finish_reasons"] == "0:tool_calls"

    stage_events = [event for event in events if event.get("cat") == "dynamo.llm.stage"]
    assert {event["tid"] for event in stage_events} == {request["tid"]}

    stage_names = {event["name"] for event in stage_events}
    assert stage_names == {"prefill wait", "prefill", "decode"}

    markers = [event for event in events if event.get("cat") == "dynamo.llm.marker"]
    assert len(markers) == 1
    assert markers[0]["name"] == "first token"
    assert markers[0]["ts"] == 1_012_000


def check_convert_records_accepts_enriched_request_trace_schema():
    trace, converted = convert_records(
        [
            {
                "event": {
                    "schema": "dynamo.request.trace.v1",
                    "event_type": "request_end",
                    "event_time_unix_ms": 1050,
                    "event_source": "dynamo",
                    "agent_context": {
                        "trajectory_id": "session-1:researcher",
                    },
                    "request": {
                        "request_id": "req-1",
                        "model": "test-model",
                        "request_received_ms": 1000,
                        "total_time_ms": 50,
                        "output_tokens": 4,
                    },
                },
            },
            {
                "event": {
                    "schema": "dynamo.request.trace.v1",
                    "event_type": "tool_end",
                    "event_time_unix_ms": 1100,
                    "event_source": "harness",
                    "agent_context": {
                        "trajectory_id": "session-1:researcher",
                    },
                    "tool": {
                        "tool_call_id": "call-1",
                        "tool_class": "shell",
                        "started_at_unix_ms": 1060,
                        "ended_at_unix_ms": 1100,
                        "status": "succeeded",
                    },
                },
            },
        ],
        include_stages=False,
        include_markers=False,
    )

    assert converted == 2
    categories = {event.get("cat") for event in trace["traceEvents"]}
    assert "dynamo.llm" in categories
    assert "dynamo.agent.tool" in categories


def check_convert_records_accepts_context_free_request_trace_schema():
    trace, converted = convert_records(
        [
            {
                "event": {
                    "schema": "dynamo.request.trace.v1",
                    "event_type": "request_end",
                    "event_time_unix_ms": 1050,
                    "request": {
                        "request_id": "req-plain",
                        "model": "test-model",
                        "request_received_ms": 1000,
                        "total_time_ms": 50,
                        "output_tokens": 4,
                    },
                },
            }
        ],
        include_stages=False,
        include_markers=False,
    )

    assert converted == 1
    request = next(
        event for event in trace["traceEvents"] if event.get("cat") == "dynamo.llm"
    )
    assert request["name"] == "LLM request: test-model"
    assert request["args"]["request_id"] == "req-plain"
    thread_names = [
        event["args"]["name"]
        for event in trace["traceEvents"]
        if event.get("name") == "thread_name"
    ]
    assert thread_names == ["request-only"]


def check_convert_records_clamps_stage_rounding_overlap():
    trace, _ = convert_records(
        [
            {
                "event": {
                    "schema": "dynamo.request.trace.v1",
                    "event_type": "request_end",
                    "event_time_unix_ms": 49_743.776002,
                    "agent_context": {
                        "trajectory_id": "session-1:searcher",
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


def check_convert_records_splits_overlapping_trajectory_requests_into_lanes():
    def record(request_id: str, start_ms: int, total_ms: int):
        return {
            "event": {
                "schema": "dynamo.request.trace.v1",
                "event_type": "request_end",
                "event_time_unix_ms": start_ms + total_ms,
                "agent_context": {
                    "trajectory_id": "session-1:searcher",
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
        "session-1:searcher [lane 1]",
        "session-1:searcher [lane 2]",
    ]

    request_tids = {
        event["args"]["request_id"]: event["tid"]
        for event in trace["traceEvents"]
        if event.get("cat") == "dynamo.llm"
    }
    assert request_tids["req-1"] != request_tids["req-2"]


def check_convert_records_uses_one_process_for_all_trajectories():
    def record(trajectory_id: str):
        return {
            "event": {
                "schema": "dynamo.request.trace.v1",
                "event_type": "request_end",
                "event_time_unix_ms": 1050,
                "agent_context": {
                    "trajectory_id": trajectory_id,
                },
                "request": {
                    "request_id": trajectory_id,
                    "model": "test-model",
                    "request_received_ms": 1000,
                    "total_time_ms": 50,
                },
            }
        }

    trace, converted = convert_records(
        [record("session-1"), record("session-2")],
        include_stages=False,
        include_markers=False,
    )

    assert converted == 2
    pids = {
        event["pid"]
        for event in trace["traceEvents"]
        if event.get("cat") == "dynamo.llm"
    }
    assert pids == {1}


def check_convert_records_emits_tool_duration_slices():
    trace, converted = convert_records(
        [
            {
                "event": {
                    "schema": "dynamo.request.trace.v1",
                    "event_type": "tool_end",
                    "event_time_unix_ms": 1300,
                    "event_source": "harness",
                    "agent_context": {
                        "trajectory_id": "session-1:searcher",
                    },
                    "tool": {
                        "tool_call_id": "call-1",
                        "tool_class": "web_search",
                        "status": "succeeded",
                        "started_at_unix_ms": 1000,
                        "ended_at_unix_ms": 1250,
                        "duration_ms": 250,
                        "output_bytes": 2048,
                    },
                },
            }
        ],
        include_stages=True,
        include_markers=False,
    )

    assert converted == 1
    tool_events = [
        event
        for event in trace["traceEvents"]
        if event.get("cat") == "dynamo.agent.tool"
    ]
    assert len(tool_events) == 1
    tool_event = tool_events[0]
    assert tool_event["name"] == "Tool: web_search"
    assert tool_event["ts"] == 1_000_000
    assert tool_event["dur"] == 250_000
    assert tool_event["args"]["tool_call_id"] == "call-1"
    assert tool_event["args"]["started_at_unix_ms"] == 1000
    assert tool_event["args"]["ended_at_unix_ms"] == 1250
    assert tool_event["args"]["output_bytes"] == 2048

    thread_names = [
        event["args"]["name"]
        for event in trace["traceEvents"]
        if event.get("name") == "thread_name"
    ]
    assert thread_names == ["session-1:searcher tools"]


def check_convert_records_infers_tool_slices_between_requests():
    def request(
        request_id: str,
        start_ms: int,
        total_ms: int,
        finish_reason_metadata: dict | None = None,
    ):
        return {
            "event": {
                "schema": "dynamo.request.trace.v1",
                "event_type": "request_end",
                "event_time_unix_ms": start_ms + total_ms,
                "event_source": "dynamo",
                "agent_context": {
                    "trajectory_id": "codex-session",
                    "parent_trajectory_id": "parent-session",
                },
                "request": {
                    "request_id": request_id,
                    "model": "test-model",
                    "request_received_ms": start_ms,
                    "total_time_ms": total_ms,
                    **(
                        {"finish_reason_metadata": finish_reason_metadata}
                        if finish_reason_metadata
                        else {}
                    ),
                },
            }
        }

    trace, converted = convert_records(
        [
            request(
                "req-tool-call",
                1000,
                50,
                {
                    "finish_reason": "tool_calls",
                    "tool_calls": [
                        {
                            "choice_index": 0,
                            "tool_call_index": 0,
                            "id": "call-1",
                            "name": "exec_command",
                        }
                    ],
                },
            ),
            request("req-after-tool", 1200, 40),
        ],
        include_stages=False,
        include_markers=False,
    )

    assert converted == 3
    tool_event = next(
        event
        for event in trace["traceEvents"]
        if event.get("cat") == "dynamo.agent.tool"
    )
    assert tool_event["name"] == "Tool: exec_command"
    assert tool_event["ts"] == 1_050_000
    assert tool_event["dur"] == 150_000
    assert tool_event["args"]["event_type"] == "inferred_tool_call"
    assert tool_event["args"]["inferred"] is True
    assert tool_event["args"]["host_execution"] is False
    assert tool_event["args"]["source_request_id"] == "req-tool-call"
    assert tool_event["args"]["next_request_id"] == "req-after-tool"
    assert tool_event["args"]["tool_call_id"] == "call-1"
    assert tool_event["args"]["parent_trajectory_id"] == "parent-session"

    thread_names = [
        event["args"]["name"]
        for event in trace["traceEvents"]
        if event.get("name") == "thread_name"
    ]
    assert thread_names == ["codex-session", "codex-session tools"]

    trace, converted = convert_records(
        [
            request(
                "req-tool-call",
                1000,
                100,
                {
                    "finish_reason": "tool_calls",
                    "tool_calls": [{"id": "call-1", "name": "exec_command"}],
                },
            ),
            request("req-overlap", 1050, 40),
            request("req-later", 1200, 40),
        ],
        include_stages=False,
        include_markers=False,
    )

    assert converted == 4
    tool_event = next(
        event
        for event in trace["traceEvents"]
        if event.get("cat") == "dynamo.agent.tool"
    )
    assert tool_event["ts"] == 1_100_000
    assert tool_event["dur"] == 1_000
    assert tool_event["args"]["duration_unknown"] is True
    assert tool_event["args"]["synthetic_duration"] is True
    assert tool_event["args"]["visual_duration_ms"] == 1.0
    assert tool_event["args"]["overlapping_request_id"] == "req-overlap"
    assert "next_request_id" not in tool_event["args"]
    assert "ended_at_unix_ms" not in tool_event["args"]
    assert "duration_ms" not in tool_event["args"]

    trace, converted = convert_records(
        [
            request(
                "req-terminal-tool-call",
                1000,
                100,
                {
                    "finish_reason": "tool_calls",
                    "tool_calls": [{"id": "call-1", "name": "exec_command"}],
                },
            ),
        ],
        include_stages=False,
        include_markers=False,
    )

    assert converted == 2
    tool_event = next(
        event
        for event in trace["traceEvents"]
        if event.get("cat") == "dynamo.agent.tool"
    )
    assert tool_event["ts"] == 1_100_000
    assert tool_event["dur"] == 1_000
    assert tool_event["args"]["duration_unknown"] is True
    assert tool_event["args"]["synthetic_duration"] is True
    assert "next_request_id" not in tool_event["args"]
    assert "overlapping_request_id" not in tool_event["args"]

    trace, converted = convert_records(
        [
            request(
                "req-tool-call",
                1000,
                50,
                {
                    "finish_reason": "tool_calls",
                    "tool_calls": [{"id": "call-1", "name": "exec_command"}],
                },
            ),
            request("req-after-tool", 1200, 40),
            {
                "event": {
                    "schema": "dynamo.request.trace.v1",
                    "event_type": "tool_end",
                    "event_time_unix_ms": 1175,
                    "event_source": "harness",
                    "agent_context": {
                        "trajectory_id": "codex-session",
                    },
                    "tool": {
                        "tool_call_id": "call-1",
                        "tool_class": "exec_command",
                        "started_at_unix_ms": 1060,
                        "ended_at_unix_ms": 1175,
                    },
                }
            },
        ],
        include_stages=False,
        include_markers=False,
    )

    assert converted == 3
    tool_events = [
        event
        for event in trace["traceEvents"]
        if event.get("cat") == "dynamo.agent.tool"
    ]
    assert len(tool_events) == 1
    assert "inferred" not in tool_events[0]["args"]
    assert tool_events[0]["ts"] == 1_060_000
    assert tool_events[0]["dur"] == 115_000


def check_convert_records_pairs_tool_start_and_end_without_duration():
    trace, converted = convert_records(
        [
            {
                "event": {
                    "schema": "dynamo.request.trace.v1",
                    "event_type": "tool_start",
                    "event_time_unix_ms": 1000,
                    "event_source": "harness",
                    "agent_context": {
                        "trajectory_id": "session-1:searcher",
                    },
                    "tool": {
                        "tool_call_id": "call-1",
                        "tool_class": "web_search",
                        "status": "running",
                    },
                },
            },
            {
                "event": {
                    "schema": "dynamo.request.trace.v1",
                    "event_type": "tool_end",
                    "event_time_unix_ms": 1250,
                    "event_source": "harness",
                    "agent_context": {
                        "trajectory_id": "session-1:searcher",
                    },
                    "tool": {
                        "tool_call_id": "call-1",
                        "tool_class": "web_search",
                        "status": "succeeded",
                    },
                },
            },
        ],
        include_stages=True,
        include_markers=False,
    )

    assert converted == 1
    assert not [
        event
        for event in trace["traceEvents"]
        if event.get("cat") == "dynamo.agent.tool.marker"
    ]
    tool_event = next(
        event
        for event in trace["traceEvents"]
        if event.get("cat") == "dynamo.agent.tool"
    )
    assert tool_event["ts"] == 1_000_000
    assert tool_event["dur"] == 250_000


def check_convert_records_renders_zero_duration_tool_as_synthetic_span():
    trace, converted = convert_records(
        [
            {
                "event": {
                    "schema": "dynamo.request.trace.v1",
                    "event_type": "tool_start",
                    "event_time_unix_ms": 1000,
                    "event_source": "harness",
                    "agent_context": {
                        "trajectory_id": "session-1:searcher",
                    },
                    "tool": {
                        "tool_call_id": "call-1",
                        "tool_class": "file_system",
                        "status": "running",
                    },
                },
            },
            {
                "event": {
                    "schema": "dynamo.request.trace.v1",
                    "event_type": "tool_end",
                    "event_time_unix_ms": 1000,
                    "event_source": "harness",
                    "agent_context": {
                        "trajectory_id": "session-1:searcher",
                    },
                    "tool": {
                        "tool_call_id": "call-1",
                        "tool_class": "file_system",
                        "status": "succeeded",
                        "duration_ms": 0.0,
                    },
                },
            },
        ],
        include_stages=True,
        include_markers=False,
    )

    assert converted == 1
    assert not [
        event
        for event in trace["traceEvents"]
        if event.get("cat") == "dynamo.agent.tool.marker"
    ]
    tool_event = next(
        event
        for event in trace["traceEvents"]
        if event.get("cat") == "dynamo.agent.tool"
    )
    assert tool_event["name"] == "Tool: file_system"
    assert tool_event["ts"] == 1_000_000
    assert tool_event["dur"] == 1_000
    assert tool_event["args"]["duration_ms"] == 0.0
    assert tool_event["args"]["synthetic_duration"] is True
    assert tool_event["args"]["visual_duration_ms"] == 1.0


CHECKS = [
    check_convert_records_emits_request_stages_and_metadata,
    check_convert_records_accepts_enriched_request_trace_schema,
    check_convert_records_accepts_context_free_request_trace_schema,
    check_convert_records_clamps_stage_rounding_overlap,
    check_convert_records_splits_overlapping_trajectory_requests_into_lanes,
    check_convert_records_uses_one_process_for_all_trajectories,
    check_convert_records_emits_tool_duration_slices,
    check_convert_records_infers_tool_slices_between_requests,
    check_convert_records_pairs_tool_start_and_end_without_duration,
    check_convert_records_renders_zero_duration_tool_as_synthetic_span,
]


def main() -> int:
    for check in CHECKS:
        check()
        print(f"PASS {check.__name__}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
