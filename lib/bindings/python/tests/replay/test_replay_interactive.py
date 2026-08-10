# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from dynamo.mocker import MockEngineArgs
from dynamo.replay import (
    OfflineReplaySession,
    ReplayAgenticRequest,
    ReplayAgenticWorkflow,
    ReplayReport,
    ReplayRequestSpec,
    WorkerTarget,
)

pytestmark = [
    pytest.mark.gpu_0,
    pytest.mark.parallel,
    pytest.mark.pre_merge,
    pytest.mark.unit,
    pytest.mark.timeout(120),
]


def _engine_args() -> MockEngineArgs:
    return MockEngineArgs(
        block_size=4,
        num_gpu_blocks=64,
        max_num_seqs=4,
        max_num_batched_tokens=64,
        speedup_ratio=1000.0,
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

    terminal = next(event["event"] for event in events if event["event_type"] == "terminal")
    assert terminal["logical_request_id"] == "request-0"
    assert terminal["session_id"] == "session-0"
    assert terminal["authored_turn_index"] == 0
    assert terminal["worker_id"] == 0
    assert terminal["dp_rank"] == 0
    assert terminal["terminal_status"] == "completed"
    assert terminal["emitted_output_count"] == 1

    assert session.pending_placements() == []
    assert session.snapshot()["workers"][0]["in_flight_requests"] == 0
    session.close_admission()
    assert session.settle_current_time()["status"] == "drained"

    report = session.finalize()
    assert isinstance(report, ReplayReport)
    assert report.summary["completed_requests"] == 1
    assert report.per_request is not None
    assert len(report.per_request) == 1
    assert report.per_request[0]["logical_request_id"] == "request-0"
    assert report.per_request[0]["uuid"] == terminal["internal_uuid"]
    assert report.coverage["capture_per_request"] is True

    with pytest.raises(Exception, match="already finalized"):
        session.now_ms()


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
    with pytest.raises(Exception, match="hash capacity"):
        session.submit(invalid)

    session.submit(_request("valid", 0))
    session.settle_current_time()
    placement_events = session.drain_events()
    assert [event["event_type"] for event in placement_events] == ["placement_needed"]

    with pytest.raises(Exception, match="DP rank 1"):
        session.assign("valid", WorkerTarget(worker_id=0, dp_rank=1))
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
