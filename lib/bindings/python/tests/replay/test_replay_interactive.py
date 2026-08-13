# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from dynamo.mocker import MockEngineArgs
from dynamo.replay import (
    OfflineReplaySession,
    PoolSpec,
    ReplayAgenticRequest,
    ReplayAgenticWorkflow,
    ReplayReport,
    ReplayRequestSpec,
    ReplayRoutingConstraints,
    WorkerSpec,
    WorkerTarget,
)

pytestmark = [
    pytest.mark.gpu_0,
    pytest.mark.parallel,
    pytest.mark.pre_merge,
    pytest.mark.unit,
    pytest.mark.timeout(120),
]


def _engine_args(*, max_model_len: int | None = None) -> MockEngineArgs:
    return MockEngineArgs(
        block_size=4,
        num_gpu_blocks=64,
        max_num_seqs=4,
        max_num_batched_tokens=64,
        speedup_ratio=1000.0,
        g1_backend="native",
        max_model_len=max_model_len,
    )


def _pool_engine_args(*, speedup_ratio: float, num_gpu_blocks: int) -> MockEngineArgs:
    return MockEngineArgs(
        block_size=4,
        num_gpu_blocks=num_gpu_blocks,
        max_num_seqs=4,
        max_num_batched_tokens=64,
        speedup_ratio=speedup_ratio,
        g1_backend="native",
    )


def _request(
    logical_request_id: str,
    authored_turn_index: int,
    *,
    target: WorkerTarget | None = None,
) -> ReplayRequestSpec:
    return ReplayRequestSpec(
        logical_request_id=logical_request_id,
        attempt_id=f"attempt-{logical_request_id}",
        group_id="group-0",
        session_id="session-0",
        authored_turn_index=authored_turn_index,
        input_length=4,
        hash_ids=[10 + authored_turn_index],
        trace_block_size=4,
        output_length=1,
        output_token_ids=[100 + authored_turn_index],
        target=target,
    )


def _drive_external_to_terminals(
    session: OfflineReplaySession,
    expected_request_ids: set[str],
) -> list[dict]:
    events: list[dict] = []
    terminal_ids: set[str] = set()
    for _ in range(100):
        session.settle_current_time()
        batch = session.drain_events()
        events.extend(batch)

        assigned = False
        for event in batch:
            data = event["event"]
            if event["event_type"] == "placement_needed":
                session.assign(data["logical_request_id"], WorkerTarget(worker_id=0))
                assigned = True
            elif event["event_type"] == "terminal":
                terminal_ids.add(data["logical_request_id"])

        if terminal_ids == expected_request_ids:
            return events
        if assigned:
            continue

        next_event_ms = session.next_event_time_ms()
        if next_event_ms is None:
            pytest.fail(
                "interactive replay became quiescent before all terminal events: "
                f"expected={expected_request_ids}, observed={terminal_ids}, "
                f"snapshot={session.snapshot()}"
            )
        session.advance_next()

    pytest.fail("interactive replay did not terminate within 100 virtual timestamps")


def test_interactive_lifecycle_correlation_and_final_report_conversion():
    session = OfflineReplaySession(
        _engine_args(),
        trace_block_size=4,
        num_workers=1,
        router="external",
    )

    assert session.settle_current_time() == {"status": "quiescent", "now_ms": 0.0}
    session.submit(_request("request-0", 0))
    events = _drive_external_to_terminals(session, {"request-0"})

    event_types = [event["event_type"] for event in events]
    assert event_types.count("placement_needed") == 1
    assert event_types.count("routed") == 1
    assert event_types.count("queued") == 1
    assert event_types.count("admitted") == 1
    assert event_types.count("first_token") == 1
    assert event_types.count("terminal") == 1
    assert {
        (event["event"]["attempt_id"], event["event"]["group_id"])
        for event in events
    } == {("attempt-request-0", "group-0")}

    placement = next(
        event["event"] for event in events if event["event_type"] == "placement_needed"
    )
    assert placement["requested_output_length"] is None
    assert placement["attempt_id"] == "attempt-request-0"
    assert placement["group_id"] == "group-0"
    assert placement["emitted_output_count"] == 0
    assert placement["ttft_ms"] is None
    assert placement["e2e_latency_ms"] is None

    terminal = next(event["event"] for event in events if event["event_type"] == "terminal")
    assert terminal["logical_request_id"] == "request-0"
    assert terminal["attempt_id"] == "attempt-request-0"
    assert terminal["group_id"] == "group-0"
    assert terminal["session_id"] == "session-0"
    assert terminal["authored_turn_index"] == 0
    assert terminal["worker_id"] == 0
    assert terminal["dp_rank"] == 0
    assert terminal["terminal_status"] == "completed"
    assert terminal["emitted_output_count"] == 1

    assert session.pending_placements() == []
    worker = session.snapshot()["workers"][0]
    assert worker["in_flight_requests"] == 0
    assert worker["queued_requests"] == 0
    assert worker["running_requests"] == 0
    assert worker["queued_tokens"] == 0
    assert worker["running_tokens"] == 0
    assert worker["max_num_seqs"] == 4
    assert worker["preemption_count"] == 0
    assert worker["kv_capacity_blocks"] == 64
    assert worker["kv_occupied_blocks"] is not None
    assert worker["kv_free_blocks"] is not None
    assert worker["kv_occupied_blocks"] + worker["kv_free_blocks"] == 64
    session.close_admission()
    assert session.settle_current_time()["status"] == "drained"

    report = session.finalize()
    assert isinstance(report, ReplayReport)
    assert report.summary["completed_requests"] == 1
    assert report.per_request is not None
    assert len(report.per_request) == 1
    assert report.per_request[0]["logical_request_id"] == "request-0"
    assert report.per_request[0]["attempt_id"] == "attempt-request-0"
    assert report.per_request[0]["group_id"] == "group-0"
    assert report.per_request[0]["uuid"] == terminal["internal_uuid"]
    assert report.coverage["capture_per_request"] is True

    with pytest.raises(Exception, match="already finalized"):
        session.now_ms()


@pytest.mark.parametrize(
    ("target", "error"),
    [
        ({"pool_id": "default", "dp_rank": 0}, "missing field.*worker_id"),
        (
            {"worker_id": 0, "recorded_queue_ms": 12},
            "unknown field.*recorded_queue_ms",
        ),
        ({"pool_id": 7, "worker_id": 0}, "invalid type.*integer.*string"),
        ({"worker_id": "0"}, "invalid type.*string.*usize"),
        ({"worker_id": 0, "dp_rank": "0"}, "invalid type.*string.*usize"),
    ],
)
def test_interactive_assign_target_direct_parser_preserves_schema_errors(
    target: dict, error: str
):
    session = OfflineReplaySession(
        _engine_args(),
        trace_block_size=4,
        num_workers=1,
        router="external",
    )
    session.submit(_request("target-schema", 0))
    session.settle_current_time()
    assert [event["event_type"] for event in session.drain_events()] == [
        "placement_needed"
    ]

    with pytest.raises(
        ValueError,
        match=rf"invalid interactive replay worker target: .*{error}",
    ):
        session.assign("target-schema", target)

    assert [item["logical_request_id"] for item in session.pending_placements()] == [
        "target-schema"
    ]


@pytest.mark.parametrize(
    "target",
    [
        {"worker_id": 0},
        {"worker_id": 0, "pool_id": "default"},
        {"worker_id": 0, "dp_rank": 0},
        # Preserve pythonize/Serde's historical PyLong behavior: bool is a
        # valid integer input and maps to worker 0 here.
        {"worker_id": False, "dp_rank": False},
    ],
)
def test_interactive_assign_target_direct_parser_preserves_defaults_and_bool(target: dict):
    session = OfflineReplaySession(
        _engine_args(),
        trace_block_size=4,
        num_workers=1,
        router="external",
    )
    session.submit(_request("target-defaults", 0))
    session.settle_current_time()
    assert [event["event_type"] for event in session.drain_events()] == [
        "placement_needed"
    ]

    session.assign("target-defaults", target)

    routed = session.drain_events()
    assert next(
        event["event"] for event in routed if event["event_type"] == "routed"
    )["pool_id"] == "default"


def test_interactive_step_status_direct_converter_schema_is_exact_for_all_variants():
    session = OfflineReplaySession(
        _engine_args(),
        trace_block_size=4,
        num_workers=1,
        router="external",
    )

    advanced = session.advance_to(5.0)
    quiescent = session.settle_current_time()
    session.close_admission()
    drained = session.settle_current_time()

    assert advanced == {"status": "advanced", "now_ms": 5.0}
    assert quiescent == {"status": "quiescent", "now_ms": 5.0}
    assert drained == {"status": "drained", "now_ms": 5.0}
    assert all(
        list(status) == ["status", "now_ms"]
        for status in [advanced, quiescent, drained]
    )


def test_interactive_event_and_pending_python_schema_is_exact():
    event_keys = [
        "logical_request_id",
        "attempt_id",
        "group_id",
        "internal_uuid",
        "session_id",
        "authored_turn_index",
        "timestamp_ms",
        "pool_id",
        "worker_id",
        "dp_rank",
        "terminal_status",
        "input_length",
        "requested_output_length",
        "emitted_output_count",
        "reused_input_tokens",
        "ttft_ms",
        "e2e_latency_ms",
        "priority",
        "strict_priority",
        "policy_class",
        "routing_constraints",
        "eligible_pool_ids",
        "candidates",
    ]
    pending_keys = [
        "logical_request_id",
        "attempt_id",
        "group_id",
        "internal_uuid",
        "session_id",
        "authored_turn_index",
        "ready_at_ms",
        "input_length",
        "priority",
        "strict_priority",
        "policy_class",
        "routing_constraints",
        "eligible_pool_ids",
        "candidates",
    ]
    candidate_keys = [
        "target",
        "active",
        "draining",
        "eligible",
        "constraint_reason",
        "in_flight_requests",
        "queued_requests",
        "running_requests",
        "queued_tokens",
        "running_tokens",
        "max_num_seqs",
        "preemption_count",
        "kv_prefix_overlap_tokens",
        "kv_capacity_blocks",
        "kv_occupied_blocks",
        "kv_free_blocks",
        "tags",
        "taints",
        "capabilities",
    ]
    session = OfflineReplaySession(
        pools=[
            PoolSpec(
                pool_id="pool-a",
                engine_args=_pool_engine_args(
                    speedup_ratio=1000.0,
                    num_gpu_blocks=64,
                ),
                workers=[
                    WorkerSpec(
                        worker_id=7,
                        tags=("primary",),
                        taints=("trusted",),
                        capabilities=("chat",),
                    )
                ],
            )
        ],
        trace_block_size=4,
        router="external",
    )
    request = _request("schema", 0).to_native()
    request["routing_constraints"] = ReplayRoutingConstraints(
        required_taints=("trusted",),
        preferred_taints={"trusted": 2.0},
    ).to_native()
    session.submit(request)
    session.settle_current_time()

    placement_wrapper = session.drain_events()[0]
    placement = placement_wrapper["event"]
    assert list(placement_wrapper) == ["event_type", "event"]
    assert placement_wrapper["event_type"] == "placement_needed"
    assert list(placement) == event_keys
    assert {
        key: placement[key]
        for key in (
            "pool_id",
            "worker_id",
            "dp_rank",
            "terminal_status",
            "requested_output_length",
            "reused_input_tokens",
            "ttft_ms",
            "e2e_latency_ms",
            "policy_class",
        )
    } == {
        "pool_id": None,
        "worker_id": None,
        "dp_rank": None,
        "terminal_status": None,
        "requested_output_length": None,
        "reused_input_tokens": None,
        "ttft_ms": None,
        "e2e_latency_ms": None,
        "policy_class": None,
    }
    assert list(placement["routing_constraints"]) == [
        "required_taints",
        "preferred_taints",
    ]
    assert placement["routing_constraints"] == {
        "required_taints": ["trusted"],
        "preferred_taints": {"trusted": 2.0},
    }
    candidate = placement["candidates"][0]
    assert list(candidate) == candidate_keys
    assert list(candidate["target"]) == ["pool_id", "worker_id", "dp_rank"]
    assert candidate["target"] == {
        "pool_id": "pool-a",
        "worker_id": 7,
        "dp_rank": 0,
    }
    assert candidate["constraint_reason"] is None

    pending = session.pending_placements()[0]
    assert list(pending) == pending_keys
    assert list(pending["routing_constraints"]) == [
        "required_taints",
        "preferred_taints",
    ]
    assert list(pending["candidates"][0]) == candidate_keys
    assert pending["policy_class"] is None

    session.assign("schema", WorkerTarget(pool_id="pool-a", worker_id=7))
    lifecycle = [placement_wrapper] + _drive_external_to_terminals(session, {"schema"})
    assert {event["event_type"] for event in lifecycle} == {
        "placement_needed",
        "routed",
        "queued",
        "admitted",
        "first_token",
        "terminal",
    }
    assert all(list(event) == ["event_type", "event"] for event in lifecycle)
    assert all(list(event["event"]) == event_keys for event in lifecycle)
    terminal = next(
        event["event"] for event in lifecycle if event["event_type"] == "terminal"
    )
    assert terminal["terminal_status"] == "completed"
    assert all(
        event["event"]["terminal_status"] is None
        for event in lifecycle
        if event["event_type"] != "terminal"
    )


def test_interactive_agentic_append_releases_child_from_parent_terminal():
    session = OfflineReplaySession(
        _engine_args(),
        trace_block_size=4,
        num_workers=1,
        router="external",
    )
    session.append_agentic_workflow(
        ReplayAgenticWorkflow(
            trace_block_size=4,
            requests=[
                ReplayAgenticRequest(request=_request("root", 0)),
                ReplayAgenticRequest(
                    request=_request("child", 1),
                    wait_for=["root"],
                    dependency_delay_ms=0.0,
                ),
            ],
        ),
        release_at_ms=0.0,
    )

    events = _drive_external_to_terminals(session, {"root", "child"})
    placement_order = [
        event["event"]["logical_request_id"]
        for event in events
        if event["event_type"] == "placement_needed"
    ]
    terminals = [
        event["event"] for event in events if event["event_type"] == "terminal"
    ]

    assert placement_order == ["root", "child"]
    assert [event["logical_request_id"] for event in terminals] == ["root", "child"]
    assert terminals[1]["timestamp_ms"] >= terminals[0]["timestamp_ms"]

    session.close()
    assert session.is_drained()
    report = session.finalize()
    assert report.per_request is not None
    assert {
        (record["logical_request_id"], record["authored_turn_index"])
        for record in report.per_request
    } == {("root", 0), ("child", 1)}


def test_interactive_assignment_and_admission_errors_preserve_session_state():
    session = OfflineReplaySession(
        _engine_args(),
        trace_block_size=4,
        num_workers=1,
        router="external",
    )

    invalid = _request("invalid", 0).to_native()
    invalid["hash_ids"] = []
    with pytest.raises(Exception, match=r"requires exactly .* hash IDs"):
        session.submit(invalid)

    unknown_field = _request("unknown", 0).to_native()
    unknown_field["recorded_e2e_ms"] = 123.0
    with pytest.raises(Exception, match="unknown field"):
        session.submit(unknown_field)

    missing_attempt = _request("missing-attempt", 0).to_native()
    del missing_attempt["attempt_id"]
    with pytest.raises(Exception, match="attempt_id"):
        session.submit(missing_attempt)

    missing_group = _request("missing-group", 0).to_native()
    del missing_group["group_id"]
    with pytest.raises(Exception, match="group_id"):
        session.submit(missing_group)

    unknown_constraint = _request("unknown-constraint", 0).to_native()
    unknown_constraint["routing_constraints"]["recorded_queue_ms"] = 12.0
    with pytest.raises(Exception, match="unknown field"):
        session.submit(unknown_constraint)
    assert session.snapshot()["pending_request_count"] == 0

    session.submit(_request("valid", 0))
    session.settle_current_time()
    placement_events = session.drain_events()
    assert [event["event_type"] for event in placement_events] == ["placement_needed"]

    with pytest.raises(Exception, match="DP rank 1"):
        session.assign("valid", WorkerTarget(worker_id=0, dp_rank=1))
    assert [item["logical_request_id"] for item in session.pending_placements()] == [
        "valid"
    ]

    with pytest.raises(Exception, match="unknown field"):
        session.assign(
            "valid",
            {
                "pool_id": "default",
                "worker_id": 0,
                "dp_rank": 0,
                "recorded_queue_ms": 12,
            },
        )
    assert [item["logical_request_id"] for item in session.pending_placements()] == [
        "valid"
    ]

    session.assign("valid", WorkerTarget(worker_id=0))
    events = _drive_external_to_terminals(session, {"valid"})
    assert sum(event["event_type"] == "terminal" for event in events) == 1

    session.close_admission()
    with pytest.raises(Exception, match="after admission is closed"):
        session.submit(_request("late", 1))
    session.finalize()


def test_interactive_rejected_parent_cancels_descendant_and_reports_statuses():
    session = OfflineReplaySession(
        _engine_args(max_model_len=16),
        trace_block_size=4,
        num_workers=1,
        router="external",
    )
    oversized = ReplayRequestSpec(
        logical_request_id="oversized",
        attempt_id="oversized-attempt-0",
        group_id="failed-group",
        session_id="failed-session",
        authored_turn_index=0,
        input_length=20,
        hash_ids=[1, 2, 3, 4, 5],
        trace_block_size=4,
        output_length=1,
    )
    child = ReplayRequestSpec(
        logical_request_id="blocked-child",
        attempt_id="blocked-child-attempt-0",
        group_id="failed-group",
        session_id="failed-session",
        authored_turn_index=1,
        input_length=4,
        hash_ids=[6],
        trace_block_size=4,
        output_length=1,
    )
    session.append_agentic_workflow(
        ReplayAgenticWorkflow(
            trace_block_size=4,
            requests=[
                ReplayAgenticRequest(request=oversized),
                ReplayAgenticRequest(request=child, wait_for=["oversized"]),
            ],
        ),
        release_at_ms=0.0,
    )

    events = _drive_external_to_terminals(
        session,
        {"oversized", "blocked-child"},
    )
    terminals = [
        event["event"] for event in events if event["event_type"] == "terminal"
    ]
    assert [terminal["logical_request_id"] for terminal in terminals] == [
        "oversized",
        "blocked-child",
    ]
    assert [terminal["terminal_status"] for terminal in terminals] == [
        "rejected",
        "canceled",
    ]
    assert not any(
        event["event_type"] == "routed"
        and event["event"]["logical_request_id"] == "blocked-child"
        for event in events
    )

    session.close()
    report = session.finalize()
    assert report.per_request is not None
    assert {
        record["logical_request_id"]: record["terminal_status"]
        for record in report.per_request
    } == {"oversized": "rejected", "blocked-child": "canceled"}
    assert {
        record["logical_request_id"]: (record["attempt_id"], record["group_id"])
        for record in report.per_request
    } == {
        "oversized": ("oversized-attempt-0", "failed-group"),
        "blocked-child": ("blocked-child-attempt-0", "failed-group"),
    }


def test_interactive_finalize_requires_closed_and_drained_session():
    session = OfflineReplaySession(
        _engine_args(),
        trace_block_size=4,
        num_workers=1,
        router="round_robin",
    )
    with pytest.raises(Exception, match="admission remains open"):
        session.finalize()

    session.submit(_request("native-route", 0))
    session.close_admission()
    with pytest.raises(Exception, match="work remains incomplete"):
        session.finalize()

    for _ in range(100):
        session.settle_current_time()
        if session.is_drained():
            break
        if session.next_event_time_ms() is not None:
            session.advance_next()
    else:
        pytest.fail("native round-robin interactive replay did not drain")

    report = session.finalize()
    assert report.summary["completed_requests"] == 1


def test_interactive_static_heterogeneous_pools_serialize_and_route_both_forms():
    session = OfflineReplaySession(
        pools=[
            PoolSpec(
                pool_id="fast",
                engine_args=_pool_engine_args(
                    speedup_ratio=1000.0,
                    num_gpu_blocks=64,
                ),
                workers=[
                    WorkerSpec(
                        worker_id=10,
                        max_num_seqs=4,
                        tags=("primary",),
                        taints=("fast",),
                        capabilities=("chat",),
                    )
                ],
            ),
            PoolSpec(
                pool_id="slow",
                engine_args=_pool_engine_args(
                    speedup_ratio=1.0,
                    num_gpu_blocks=32,
                ),
                workers=[WorkerSpec(worker_id=20, max_num_seqs=1)],
            ),
        ],
        trace_block_size=4,
        router="external",
    )
    workers = session.snapshot()["workers"]
    assert [
        (worker["pool_id"], worker["worker_id"], worker["max_num_seqs"])
        for worker in workers
    ] == [("fast", 10, 4), ("slow", 20, 1)]
    assert [worker["kv_capacity_blocks"] for worker in workers] == [64, 32]
    assert workers[0]["tags"] == ["primary"]
    assert workers[0]["taints"] == ["fast"]
    assert workers[0]["capabilities"] == ["chat"]

    session.submit(_request("exact-fast", 0))
    slow = _request("pool-slow", 1).to_native()
    slow["session_id"] = "session-1"
    session.submit(slow)
    session.settle_current_time()
    placement_events = session.drain_events()
    assert [event["event"]["logical_request_id"] for event in placement_events] == [
        "exact-fast",
    ]
    assert all(
        event["event"]["eligible_pool_ids"] == ["fast", "slow"]
        for event in placement_events
    )
    candidates = placement_events[0]["event"]["candidates"]
    assert [candidate["target"] for candidate in candidates] == [
        {"pool_id": "fast", "worker_id": 10, "dp_rank": 0},
        {"pool_id": "slow", "worker_id": 20, "dp_rank": 0},
    ]
    assert candidates[0]["tags"] == ["primary"]
    assert candidates[0]["taints"] == ["fast"]
    assert candidates[0]["capabilities"] == ["chat"]

    with pytest.raises(Exception, match="not awaiting placement"):
        session.assign_pool("pool-slow", "slow")
    session.assign(
        "exact-fast",
        WorkerTarget(pool_id="fast", worker_id=10),
    )
    second_boundary = session.drain_events()
    assert [
        event["event"]["logical_request_id"]
        for event in second_boundary
        if event["event_type"] == "placement_needed"
    ] == ["pool-slow"]
    placement_events += second_boundary
    session.assign_pool("pool-slow", "slow")
    events = placement_events + _drive_external_to_terminals(
        session,
        {"exact-fast", "pool-slow"},
    )
    terminals = {
        event["event"]["logical_request_id"]: event["event"]
        for event in events
        if event["event_type"] == "terminal"
    }
    assert terminals["exact-fast"]["pool_id"] == "fast"
    assert terminals["exact-fast"]["worker_id"] == 10
    assert terminals["pool-slow"]["pool_id"] == "slow"
    assert terminals["pool-slow"]["worker_id"] == 20
    assert terminals["exact-fast"]["timestamp_ms"] != terminals["pool-slow"][
        "timestamp_ms"
    ]

    session.close()
    report = session.finalize()
    assert report.per_request is not None
    assert {
        record["logical_request_id"]: (record["pool_id"], record["worker_id"])
        for record in report.per_request
    } == {"exact-fast": ("fast", 10), "pool-slow": ("slow", 20)}
    routed_targets = {
        record["logical_request_id"]: (
            record["routing_history"][-1]["pool_id"],
            record["routing_history"][-1]["worker_id"],
            record["routing_history"][-1]["dp_rank"],
        )
        for record in report.per_request
    }
    assert routed_targets == {
        "exact-fast": ("fast", 10, 0),
        "pool-slow": ("slow", 20, 0),
    }


def test_interactive_constraints_and_invalid_assignment_recovery():
    session = OfflineReplaySession(
        pools=[
            PoolSpec(
                pool_id="eligible",
                engine_args=_pool_engine_args(
                    speedup_ratio=1000.0,
                    num_gpu_blocks=64,
                ),
                workers=[
                    WorkerSpec(
                        worker_id=0,
                        taints=("trusted",),
                        tags=("primary",),
                        capabilities=("chat",),
                    )
                ],
            ),
            PoolSpec(
                pool_id="ineligible",
                engine_args=_pool_engine_args(
                    speedup_ratio=1000.0,
                    num_gpu_blocks=64,
                ),
                workers=[WorkerSpec(worker_id=0, taints=("batch",))],
            ),
        ],
        trace_block_size=4,
        router="external",
    )
    request = _request("constrained", 0).to_native()
    request["priority"] = -7
    request["strict_priority"] = 13
    request["policy_class"] = "latency-sensitive"
    request["routing_constraints"] = ReplayRoutingConstraints(
        required_taints=("trusted",),
        preferred_taints={"trusted": 2.0},
    ).to_native()
    session.submit(request)
    session.settle_current_time()
    placement = session.drain_events()[0]["event"]
    assert placement["attempt_id"] == "attempt-constrained"
    assert placement["group_id"] == "group-0"
    assert placement["routing_constraints"] == {
        "required_taints": ["trusted"],
        "preferred_taints": {"trusted": 2.0},
    }
    assert placement["eligible_pool_ids"] == ["eligible"]
    assert [candidate["eligible"] for candidate in placement["candidates"]] == [
        True,
        False,
    ]
    assert placement["candidates"][1]["constraint_reason"] is not None
    pending = session.pending_placements()
    assert len(pending) == 1
    assert pending[0]["input_length"] == 4
    assert pending[0]["priority"] == -7
    assert pending[0]["strict_priority"] == 13
    assert pending[0]["policy_class"] == "latency-sensitive"
    assert pending[0]["routing_constraints"] == placement["routing_constraints"]
    assert "requested_output_length" not in pending[0]
    assert "output_length" not in pending[0]
    assert "ttft_ms" not in pending[0]
    assert "e2e_latency_ms" not in pending[0]

    with pytest.raises(Exception, match="pool.*unavailable"):
        session.assign(
            "constrained",
            WorkerTarget(pool_id="missing", worker_id=0),
        )
    assert [item["logical_request_id"] for item in session.pending_placements()] == [
        "constrained"
    ]

    with pytest.raises(Exception, match="required taints"):
        session.assign(
            "constrained",
            WorkerTarget(pool_id="ineligible", worker_id=0),
        )
    assert [item["logical_request_id"] for item in session.pending_placements()] == [
        "constrained"
    ]

    session.assign(
        "constrained",
        WorkerTarget(pool_id="eligible", worker_id=0),
    )
    with pytest.raises(Exception, match="not awaiting placement"):
        session.assign(
            "constrained",
            WorkerTarget(pool_id="eligible", worker_id=0),
        )
    events = _drive_external_to_terminals(session, {"constrained"})
    terminal = next(
        event["event"] for event in events if event["event_type"] == "terminal"
    )
    assert terminal["attempt_id"] == "attempt-constrained"
    assert terminal["group_id"] == "group-0"
    session.close()
    report = session.finalize()
    assert report.per_request is not None
    assert report.per_request[0]["attempt_id"] == "attempt-constrained"
    assert report.per_request[0]["group_id"] == "group-0"


def test_interactive_pool_topology_mapping_rejects_unknown_fields_without_fallback():
    engine_args = _pool_engine_args(speedup_ratio=1.0, num_gpu_blocks=32)
    with pytest.raises(ValueError, match="unknown replay worker topology fields"):
        OfflineReplaySession(
            pools=[
                {
                    "pool_id": "pool",
                    "engine_args": engine_args,
                    "workers": [{"worker_id": 0, "recorded_latency_ms": 10.0}],
                }
            ],
            trace_block_size=4,
        )

    with pytest.raises(ValueError, match="unknown replay pool topology fields"):
        OfflineReplaySession(
            pools=[
                {
                    "pool_id": "pool",
                    "engine_args": engine_args,
                    "workers": [{"worker_id": 0}],
                    "performance_profile": "recorded-service-time",
                }
            ],
            trace_block_size=4,
        )

    with pytest.raises(ValueError, match="either engine_args or pools"):
        OfflineReplaySession(
            engine_args=engine_args,
            pools=[PoolSpec("pool", engine_args, [WorkerSpec(0)])],
            trace_block_size=4,
        )


@pytest.mark.parametrize(
    ("constraints", "message"),
    [
        ({"required_taints": [""], "preferred_taints": {}}, "required taint"),
        (
            {"required_taints": ["dup", "dup"], "preferred_taints": {}},
            "duplicates required taint",
        ),
        (
            {"required_taints": [], "preferred_taints": {" bad ": 1.0}},
            "preferred taint",
        ),
        (
            {"required_taints": [], "preferred_taints": {"score": float("nan")}},
            "preferred-taint weight",
        ),
    ],
)
def test_invalid_routing_constraints_roll_back_public_append(constraints, message):
    session = OfflineReplaySession(
        engine_args=_engine_args(),
        num_workers=1,
        trace_block_size=4,
        router="external",
    )
    first = _request("rollback-first", 0).to_native()
    invalid = _request("rollback-invalid", 1).to_native()
    invalid["routing_constraints"] = constraints
    workflow = {
        "trace_block_size": 4,
        "requests": [
            {"request": first},
            {"request": invalid},
        ],
    }
    with pytest.raises(Exception, match=message):
        session.append_agentic_workflow(workflow, 0.0)
    assert session.snapshot()["pending_request_count"] == 0

    # Exact identity reuse proves the rejected batch did not leak registration.
    session.submit(first)
    events = _drive_external_to_terminals(session, {"rollback-first"})
    assert sum(event["event_type"] == "terminal" for event in events) == 1
    session.close()
    session.finalize()


def test_unsatisfiable_static_taints_fail_atomically_and_session_recovers():
    session = OfflineReplaySession(
        pools=[
            PoolSpec(
                "default",
                _engine_args(),
                [WorkerSpec(0, taints=["plain"])],
            )
        ],
        trace_block_size=4,
    )
    authored = _request("no-eligible", 0).to_native()
    authored["routing_constraints"] = {
        "required_taints": ["secure"],
        "preferred_taints": {},
    }

    with pytest.raises(Exception, match="no static active worker"):
        session.submit(authored)
    assert session.snapshot()["pending_request_count"] == 0
    assert session.pending_placements() == []
    assert session.drain_events() == []

    authored["routing_constraints"] = {
        "required_taints": [],
        "preferred_taints": {},
    }
    session.submit(authored)
    events = _drive_external_to_terminals(session, {"no-eligible"})
    assert sum(event["event_type"] == "terminal" for event in events) == 1
    session.close()
    report = session.finalize()
    assert report.summary["completed_requests"] == 1


def test_prefix_reset_fails_in_python_and_native_boundaries_without_identity_leak():
    request = _request("prefix-reset", 0)
    with pytest.raises(ValueError, match="does not support prefix_reset=true"):
        ReplayAgenticRequest(request=request, prefix_reset=True).to_native()

    session = OfflineReplaySession(
        engine_args=_engine_args(),
        num_workers=1,
        trace_block_size=4,
        router="external",
    )
    raw_request = request.to_native()
    with pytest.raises(Exception, match="unsupported prefix_reset=true"):
        session.append_agentic_workflow(
            {
                "trace_block_size": 4,
                "requests": [
                    {"request": raw_request, "prefix_reset": True},
                ],
            },
            0.0,
        )
    assert session.snapshot()["pending_request_count"] == 0
    session.submit(raw_request)
    _drive_external_to_terminals(session, {"prefix-reset"})
    session.close()
    session.finalize()


def test_non_pooled_static_session_rejects_startup_time():
    with pytest.raises(Exception, match="must not configure startup_time"):
        OfflineReplaySession(
            engine_args=MockEngineArgs(
                block_size=4,
                num_gpu_blocks=64,
                max_num_seqs=4,
                max_num_batched_tokens=64,
                speedup_ratio=1000.0,
                g1_backend="native",
                startup_time=1.0,
            ),
            num_workers=1,
            trace_block_size=4,
            router="external",
        )
