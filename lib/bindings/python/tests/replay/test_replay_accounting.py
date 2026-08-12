# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from dynamo.mocker import MockEngineArgs
from dynamo.replay import (
    OfflineReplaySession,
    PoolSpec,
    ReplayRequestSpec,
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


def _engine_args() -> MockEngineArgs:
    return MockEngineArgs(
        block_size=4,
        num_gpu_blocks=64,
        max_num_seqs=4,
        max_num_batched_tokens=64,
        speedup_ratio=1000.0,
        g1_backend="native",
    )


def test_static_inactive_snapshot_and_final_accounting_are_explicit():
    session = OfflineReplaySession(
        pools=[
            PoolSpec("active-pool", _engine_args(), [WorkerSpec(worker_id=3)]),
            PoolSpec(
                "inactive-pool",
                _engine_args(),
                [WorkerSpec(worker_id=7, active=False)],
            ),
        ],
        trace_block_size=4,
        router="external",
    )

    workers = session.snapshot()["workers"]
    assert [worker["lifecycle_status"] for worker in workers] == [
        "active",
        "static_inactive",
    ]
    assert all(worker["provisioned"] for worker in workers)
    assert workers[1]["active"] is False
    assert workers[1]["draining"] is False

    session.submit(
        ReplayRequestSpec(
            logical_request_id="request-0",
            attempt_id="attempt-0",
            group_id="group-0",
            session_id="session-0",
            authored_turn_index=0,
            input_length=4,
            hash_ids=[10],
            trace_block_size=4,
            output_length=1,
            output_token_ids=[100],
        )
    )

    terminal = False
    for _ in range(100):
        session.settle_current_time()
        for event in session.drain_events():
            data = event["event"]
            if event["event_type"] == "placement_needed":
                session.assign(
                    data["logical_request_id"],
                    WorkerTarget(pool_id="active-pool", worker_id=3),
                )
            elif event["event_type"] == "terminal":
                terminal = True
        if terminal:
            break
        assert session.next_event_time_ms() is not None
        session.advance_next()
    else:
        pytest.fail("request did not reach a terminal state")

    session.close()
    report = session.finalize()
    accounting = report.summary["topology_accounting"]
    assert [
        (worker["pool_id"], worker["worker_id"])
        for worker in accounting["workers"]
    ] == [("active-pool", 3), ("inactive-pool", 7)]
    assert accounting["workers"][0]["request_counts"]["completed_requests"] == 1
    assert accounting["workers"][1]["request_counts"]["num_requests"] == 0
    assert accounting["workers"][1]["lifecycle_status"] == "static_inactive"
    assert accounting["workers"][1]["worker_seconds"] > 0.0
    worker_reuse = sum(
        worker["reused_input_tokens_by_status"]["completed"]
        for worker in accounting["workers"]
    )
    pool_reuse = sum(
        pool_["reused_input_tokens_by_status"]["completed"]
        for pool_ in accounting["pools"]
    )
    assert worker_reuse == pool_reuse
    assert pool_reuse == accounting["global"]["reused_input_tokens_by_status"][
        "completed"
    ]
    assert sum(
        pool_["terminal_counts"]["completed"] for pool_ in accounting["pools"]
    ) == accounting["global"]["terminal_counts"]["completed"]
    assert accounting["reconciliation"] == {
        "global_request_counts_match": True,
        "global_topology_counts_match": True,
        "pool_request_counts_match": True,
        "terminal_counts_match": True,
        "pool_terminal_counts_match": True,
        "pool_token_counts_match": True,
        "reused_input_tokens_match": True,
        "worker_seconds_match": True,
        "pool_worker_seconds_match": True,
        "gpu_hours_match": True,
        "pool_gpu_hours_match": True,
    }
