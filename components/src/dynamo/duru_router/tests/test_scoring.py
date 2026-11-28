# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Score calculation, router-owned accounting, and lifecycle regressions."""

import gc
import json
import tempfile
import threading
import time
import weakref
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from dynamo.duru_router import scoring
from dynamo.duru_router.scoring import Ingress, LiveU, priority_key, run
from dynamo.duru_router.tests.helpers import (
    Tail,
    arrival_record,
    configuration,
    event_feeds,
)
from dynamo.duru_router.tests.helpers import memory_mode as memory_mode
from dynamo.duru_router.tests.helpers import replay_accounting, work_step
from dynamo.duru_router.transport import InProcessChannel

pytestmark = [pytest.mark.gpu_0]


def make_book(capacity=4):
    book = LiveU(capacity, {})
    book.cost = lambda phase, p, d, k: 2.0 if phase == "prefill" else 1.0
    return book


@pytest.mark.pre_merge
@pytest.mark.unit
@pytest.mark.parametrize("capacity", [0, -1, True, 1.5, "1"])
def test_invalid_capacity_is_rejected(capacity):
    with pytest.raises(ValueError, match="Capacity must be a positive integer"):
        LiveU(capacity, {})


@pytest.mark.pre_merge
@pytest.mark.unit
@pytest.mark.parametrize(
    "request_id,root,agent,prompt_tokens,message",
    [
        ("q", "root", "agent", 8, "Request ID already active"),
        ("new", "other", "agent", 8, "Agent root changed"),
        ("new", "root", "agent", 0, "Prompt tokens"),
        ("new", "root", "agent", -1, "Prompt tokens"),
        ("new", "root", "agent", True, "Prompt tokens"),
        ("new", "root", "agent", 1.5, "Prompt tokens"),
    ],
)
def test_invalid_arrival_does_not_advance_or_mutate_accounting(
    request_id, root, agent, prompt_tokens, message
):
    book = make_book()
    book.submit("q", "root", "agent", 8, 10)
    before = deepcopy(vars(book))
    with pytest.raises(ValueError, match=message):
        book.submit(request_id, root, agent, prompt_tokens, 20)
    assert vars(book) == before


@pytest.mark.pre_merge
@pytest.mark.unit
def test_regressing_clock_does_not_rewind_entitlement():
    book = make_book()
    book.submit("q", "root", "agent", 8, 10)
    book.advance(20)
    with pytest.raises(ValueError, match="Accounting time regressed"):
        book.advance(19)
    assert book.last == 20
    assert book.entitlement == {"root": 10}
    book.advance(21)
    assert book.entitlement == {"root": 11}


@pytest.mark.pre_merge
@pytest.mark.unit
def test_children_split_root_ps_share_and_tool_gaps():
    book = make_book(1)
    book.submit("a", "root", "parent", 100, 0)
    book.submit("b", "root", "child", 100, 0)
    book.submit("c", "other", "other", 100, 0)
    book.finish("a", 2)
    book.finish("b", 4)
    book.finish("c", 6)
    book.advance(100)
    assert dict(book.entitlement) == {"root": 2.0, "other": 4.0}


@pytest.mark.pre_merge
@pytest.mark.unit
@pytest.mark.parametrize("agents", [2, 5], ids=["parallel", "fanout-capped"])
def test_parallel_children_sum_time_and_work(agents):
    book = make_book(2)
    book.cost = lambda *args: 10.0
    for agent in range(agents):
        book.submit(agent, "root", agent, 100, 0)
        book.accepted(work_step(agent=agent))
    book.advance(10)
    assert book.entitlement["root"] == 20.0
    assert book.work["root"] == agents * 10
    assert book.measured("root") == 2 / agents


@pytest.mark.pre_merge
@pytest.mark.unit
def test_same_agent_overlap_duplicate_finish_and_return():
    book = make_book(1)
    book.submit("a", "root", "agent", 100, 0)
    book.submit("b", "root", "agent", 100, 2)
    book.finish("a", 4)
    book.finish("a", 5)  # Duplicate completion must not release b.
    book.advance(6)  # Cutoff while b remains outstanding.
    assert book.entitlement["root"] == 6.0
    assert book.root_active["root"] == 1
    book.finish("b", 8)
    book.advance(20)  # Tool gap.
    assert book.entitlement["root"] == 8.0
    assert not book.root_active
    book.submit("c", "root", "agent", 100, 20)
    book.finish("c", 22)
    assert book.entitlement["root"] == 10.0


class NoGlobalIteration(dict):
    """Permit keyed operations, but fail if cleanup scans retained history."""

    def __iter__(self):
        raise AssertionError("Cleanup must not iterate unrelated root histories")

    keys = items = values = __iter__


class NoGlobalSetIteration(set):
    __iter__ = NoGlobalIteration.__iter__


@pytest.mark.pre_merge
@pytest.mark.unit
@pytest.mark.parametrize("other_roots", [0, 100])
def test_root_cleanup_uses_only_its_indexed_members(other_roots):
    index = scoring.LifecycleIndex()
    book = make_book()
    for i in range(other_roots):
        root = f"other-{i}"
        index.arrive(root, root, root, root)
        index.dispatch(root)
        book.submit(root, root, root, 8, 0)
        book.finish(root, 0)
    for rid, agent in [("a", "parent"), ("b", "child")]:
        index.arrive(rid, rid, "target", agent)
        index.dispatch(rid)
        book.submit(rid, "target", agent, 8, 0)
        book.finish(rid, 0)
        index.finish(rid)
        index.undispatch(rid)
    for name in (
        "request_identities",
        "request_programs",
        "program_identities",
        "root_requests",
        "root_programs",
    ):
        setattr(index, name, NoGlobalIteration(getattr(index, name)))
    book.agent_root = NoGlobalIteration(book.agent_root)
    index.worker_requests = NoGlobalSetIteration(index.worker_requests)
    book.finish_root("target", 0)
    index.end_program("a")
    index.release_root("target")
    assert len(index.request_identities) == other_roots
    assert len(index.program_identities) == other_roots
    assert len(index.worker_requests) == other_roots
    assert len(book.agent_root) == other_roots
    assert "target" not in index.root_requests
    assert "target" not in index.root_programs
    assert "target" not in book.root_agents
    for i in range(other_roots):
        root = f"other-{i}"
        assert index.request_identities[root] == (root, root)
        assert book.agent_root[root] == root


@pytest.mark.pre_merge
@pytest.mark.unit
def test_ended_program_reuse_is_not_removed_with_its_previous_root():
    index = scoring.LifecycleIndex()
    index.arrive("old", "program", "old-root", "old-agent")
    index.dispatch("old")
    index.dispatch("old")
    assert index.root_workers["old-root"] == 1
    with pytest.raises(ValueError, match="outstanding requests"):
        index.end_program("program")
    index.finish("old")
    index.finish("old")
    index.end_program("program")
    index.arrive("new", "program", "new-root", "new-agent")
    with pytest.raises(ValueError, match="outstanding worker requests"):
        index.release_root("old-root")
    index.undispatch("old")
    index.release_root("old-root")
    assert index.program_identities == {"program": ("new-root", "new-agent")}
    assert index.request_identities == {"new": ("new-root", "new-agent")}
    assert index.request_programs == {"new": "program"}
    assert index.program_active == {"program": 1}
    assert index.root_programs == {"new-root": {"program"}}
    assert not index.root_workers


@pytest.mark.pre_merge
@pytest.mark.unit
def test_unused_root_share_is_not_redistributed():
    book = make_book(4)
    book.submit("a", "X", "a", 100, 0)
    book.submit("b", "X", "b", 100, 0)
    book.submit("c", "Y", "c", 100, 0)
    book.advance(10)
    assert dict(book.entitlement) == {"X": 20.0, "Y": 10.0}
    book.submit("d", "X", "d", 100, 10)
    book.advance(20)
    assert dict(book.entitlement) == {"X": 40.0, "Y": 20.0}


@pytest.mark.pre_merge
@pytest.mark.unit
def test_zero_work_root_keeps_prediction_until_acceptance():
    book = make_book()
    book.accepted(dict(accepted=False))
    assert not book.work
    book.submit("a", "root", "agent", 100, 0)
    book.advance(6)
    assert book.score("a")["U"] is None
    assert book.score("a")["lookahead_s"] == 6.0
    assert book.score("a")["score"] == 1.0
    assert book.score("a")["kind"] == "zero_work_root_next_five"


@pytest.mark.pre_merge
@pytest.mark.unit
def test_new_child_uses_root_measured_work():
    book = make_book()
    book.submit("a", "root", "parent", 100, 0)
    book.accepted(work_step(agent="parent"))
    book.finish("a", 2)
    book.submit("b", "root", "child", 100, 3)
    book.advance(7)
    assert book.score("b")["score"] == 3
    assert book.score("b")["lookahead_s"] is None
    assert book.score("b")["kind"] == "measured"
    assert book.score("b")["U"] == 3


@pytest.mark.pre_merge
@pytest.mark.unit
def test_valid_subunit_scores_and_invalid_fallback():
    values = [None, 0.2, float("nan"), 0.8, 0.8, 0.0]
    order = sorted(range(len(values)), key=lambda i: priority_key(values[i], i, i))
    assert order == [3, 4, 1, 5, 0, 2]


@pytest.mark.pre_merge
@pytest.mark.unit
def test_unknown_calibration_disables_score():
    book = make_book()
    book.cost = lambda *args: None
    book.submit("a", "root", "agent", 100, 0)
    book.advance(1)
    assert book.score("a")["score"] is None


def make_grid_book():
    # Prefill = 10 + 2P + D milliseconds; decode = 10 + K milliseconds.
    return LiveU(
        1,
        {
            "prefill": {"P": [1, 3], "D": [0, 4], "wall_ms": [[12, 16], [16, 20]]},
            "decode": {"grid": [{"K": 1, "wall_ms": 11}, {"K": 5, "wall_ms": 15}]},
        },
    )


@pytest.mark.pre_merge
@pytest.mark.unit
def test_cost_cache_does_not_retain_finished_accounting_session():
    book = make_grid_book()
    book.cost("decode", 1, 4, 5)
    reference = weakref.ref(book)
    del book
    gc.collect()
    assert reference() is None


@pytest.mark.pre_merge
@pytest.mark.unit
def test_grid_edges_and_extrapolated_prefill_decode():
    book = make_grid_book()
    for p, d in ((1, 0), (2, 2), (3, 4), (5, 2), (2, 8), (5, 8)):
        assert book.cost("prefill", p, d, p + d) == pytest.approx(
            (10 + 2 * p + d) / 1000, rel=0, abs=5e-8
        )
    for k in (0, 1, 3, 5, 9):
        assert book.cost("decode", 1, k - 1, k) == pytest.approx(
            (10 + k) / 1000, rel=0, abs=5e-8
        )
    assert book.extrapolated_cost_evaluations == 5
    book.cost("decode", 1, 8, 9)  # Cached lookup, no new predictor evaluation.
    assert book.extrapolated_cost_evaluations == 5


@pytest.mark.pre_merge
@pytest.mark.unit
def test_extrapolation_keeps_accepted_root_u_defined():
    book = make_grid_book()
    book.submit("req", "root", "agent", 5, 0)
    book.advance(2)
    book.accepted(work_step(p=5, d=8))
    assert book.invalid["root"] == 0
    assert book.work["root"] == pytest.approx(0.028, rel=0, abs=5e-8)
    assert book.score("req")["U"] == pytest.approx(2 / 0.028, rel=0, abs=5e-8)


@pytest.mark.pre_merge
@pytest.mark.unit
def test_extrapolated_next_five_lookahead():
    book = make_grid_book()
    book.submit("req", "root", "agent", 5, 0)
    book.advance(1)
    # P5/D0 prefill (20 ms), followed by K6..9 decode (16..19 ms).
    assert book.next_five(5) == pytest.approx(0.09, rel=0, abs=5e-8)
    assert book.score("req")["score"] == pytest.approx(1 / 0.09, rel=0, abs=5e-8)
    assert book.score("req")["U"] is None


@pytest.mark.pre_merge
@pytest.mark.unit
@pytest.mark.parametrize("outside", [False, True], ids=["inside-grid", "outside-grid"])
@pytest.mark.parametrize(
    "ms",
    [0, -1, float("nan"), float("inf"), -float("inf")],
    ids=["zero", "negative", "nan", "positive-infinity", "negative-infinity"],
)
def test_nonpositive_and_nonfinite_predictions_still_invalid(monkeypatch, outside, ms):
    book = make_grid_book()
    monkeypatch.setattr(
        "dynamo.duru_router.scoring.predict", lambda *args: (ms, outside)
    )
    book.submit("req", "root", "agent", 5, 0)
    book.accepted(work_step(p=5, d=8))
    assert book.invalid["root"] == 1
    assert book.measured("root") is None
    assert book.score("req")["score"] is None
    assert book.extrapolated_cost_evaluations == 0


@pytest.mark.pre_merge
@pytest.mark.unit
def test_lookahead_chunk_shape():
    book = make_book()
    seen = []
    book.cost = lambda *args: seen.append(args) or 1.0
    assert book.next_five(20000) == 5.0
    assert seen == [
        ("prefill", 8192, 0, 8192),
        ("prefill", 8192, 8192, 16384),
        ("prefill", 3616, 16384, 20000),
        ("decode", 1, 20000, 20001),
        ("decode", 1, 20001, 20002),
    ]


def write_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            json.dumps(dict(row, sequence=i)) + "\n" for i, row in enumerate(rows, 1)
        )
    )


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


@pytest.mark.pre_merge
@pytest.mark.unit
def test_tail_retries_partial_record_and_rejects_truncation(tmp_path):
    path = tmp_path / "events.jsonl"
    row = dict(sequence=1, time=10, event="u_clock")
    path.write_text(json.dumps(row))
    tail = Tail(path)
    assert tail.read() == []
    assert tail.offset == 0
    path.write_text(json.dumps(row) + "\n")
    assert tail.read() == [row]
    path.write_text("")
    with pytest.raises(ValueError, match="Journal truncated"):
        tail.read()


@pytest.mark.pre_merge
@pytest.mark.unit
def test_tail_rejects_sequence_gap_before_updating_progress(tmp_path):
    path = tmp_path / "events.jsonl"
    path.write_text(json.dumps(dict(sequence=2, time=10, event="footer")) + "\n")
    tail = Tail(path)
    with pytest.raises(ValueError, match="Journal gap"):
        tail.read()
    assert tail.sequence == 0
    assert tail.received_through == 0
    assert not tail.closed


@pytest.mark.pre_merge
@pytest.mark.unit
@pytest.mark.parametrize("same_batch", [False, True])
def test_event_reader_rejects_records_after_footer(tmp_path, same_batch):
    channel = InProcessChannel()
    reader = scoring.EventStreamReader(tmp_path / "events", receiver=channel)
    footer = dict(sequence=1, time=10, event="footer")
    channel.submit("record", footer)
    try:
        if not same_batch:
            assert reader.read() == [footer]
            assert reader.read() == []
        channel.submit("record", dict(sequence=2, time=11, event="u_clock"))
        with pytest.raises(ValueError, match="event after footer"):
            reader.read()
        assert reader.sequence == 1
        assert reader.received_through == 10
        assert reader.closed
    finally:
        reader.close()


@pytest.fixture
def config(tmp_path):
    return configuration(tmp_path)


@pytest.fixture
def records(config):
    _, cfg = config
    ingress = [
        arrival_record(),
        dict(time=100.4, event="request_completed", request_id="req"),
        dict(time=100.5, event="footer"),
    ]
    step = work_step(p=8, client_request_id="req", scheduled_at=100.1, end=100.2)
    worker = [
        dict(time=100.2, event="accepted_output", steps=[dict(step)]),
        # Closed workers must not hold back later ingress events.
        dict(time=100.3, event="footer"),
    ]
    write_rows(Path(cfg["ingress"]), ingress)
    write_rows(Path(cfg["ledgers"][0]), worker)
    return ingress, worker


def audit_rows(cfg):
    return read_rows(Path(cfg["audit_dir"]) / "duru-control.jsonl")


def run_once(config):
    path, cfg = config
    stop = threading.Event()
    stop.set()
    clock = SimpleNamespace(now=time.monotonic())

    def wait(delay):
        clock.now += delay

    with (
        patch.object(scoring, "time", SimpleNamespace(monotonic=lambda: clock.now)),
        patch.object(
            scoring,
            "threading",
            SimpleNamespace(Event=lambda: SimpleNamespace(wait=wait)),
        ),
    ):
        replay_accounting(path, stop)
    return audit_rows(cfg)[-1]


@pytest.mark.usefixtures("journal_validation_mode")
@pytest.mark.post_merge
@pytest.mark.integration
@pytest.mark.timeout(30)
def test_shutdown_with_missing_inputs_is_incomplete(config):
    footer = run_once(config)
    assert footer["last_snapshot"] is None
    assert not footer["complete"]
    assert not Path(config[1]["scores"]).exists()


@pytest.mark.usefixtures("journal_validation_mode")
@pytest.mark.post_merge
@pytest.mark.integration
@pytest.mark.timeout(30)
@pytest.mark.parametrize("closed", [False, True])
def test_final_pass_respects_input_watermarks(config, records, closed):
    if not closed:
        Path(config[1]["ledgers"][0]).write_text("")
    footer = run_once(config)
    assert footer["complete"] is closed
    assert footer["inputs_closed"] is closed
    assert footer["pending_events"] == (0 if closed else 2)
    if closed:
        assert footer["last_snapshot"]["accepted_steps"] == 1


@pytest.mark.usefixtures("journal_validation_mode")
@pytest.mark.post_merge
@pytest.mark.integration
@pytest.mark.timeout(30)
def test_invalid_worker_configuration_is_rejected(config, monkeypatch):
    path, cfg = config
    monkeypatch.setenv("DURU_ACCOUNTING_TRANSPORT", "memory")
    cfg["ledgers"] = []
    path.write_text(json.dumps(cfg))
    with pytest.raises(ValueError, match="worker ledger"):
        run(path)
    assert not Path(cfg["scores"]).exists()


@pytest.mark.usefixtures("journal_validation_mode")
@pytest.mark.post_merge
@pytest.mark.integration
@pytest.mark.timeout(30)
def test_live_accounting_rejects_journal_transport(config):
    with pytest.raises(ValueError, match="requires IPC"):
        run(config[0])


@pytest.fixture
def cleanup_participants(config, monkeypatch):
    path, cfg = config
    cfg["ledgers"].append(str(path.parent / "second-worker.jsonl"))
    path.write_text(json.dumps(cfg))
    monkeypatch.setenv("DURU_RECORD_EVENTS", "1")
    sources = [
        SimpleNamespace(read=lambda: [], close=Mock(), closed=True, received_through=0)
        for _ in range(3)
    ]
    audit = SimpleNamespace(submit=Mock(), close=Mock())
    publisher = SimpleNamespace(publish=Mock(), close=Mock())
    source_factory = Mock(side_effect=sources)
    audit_factory = Mock(return_value=audit)
    publisher_factory = Mock(return_value=publisher)
    monkeypatch.setattr(scoring, "EventStreamReader", source_factory)
    monkeypatch.setattr(scoring, "open_journal", audit_factory)
    monkeypatch.setattr(scoring, "ScorePublisher", publisher_factory)
    stop = threading.Event()
    stop.set()
    return SimpleNamespace(
        path=path,
        stop=stop,
        sources=sources,
        audit=audit,
        publisher=publisher,
        source_factory=source_factory,
        audit_factory=audit_factory,
        publisher_factory=publisher_factory,
        router_publish=Mock(),
    )


@pytest.fixture
def timeout_accounting(cleanup_participants, monkeypatch):
    """Advance source coverage and the deadline clock independently, without sleeps."""
    participants = cleanup_participants
    clock = SimpleNamespace(now=100.0, done=False, current=None)
    monkeypatch.setattr(scoring, "time", SimpleNamespace(monotonic=lambda: clock.now))
    participants.stop = SimpleNamespace(
        is_set=lambda: clock.done, wait=lambda timeout: False
    )

    def execute(batches, timeout=300):
        pending = iter(batches)

        def ingress_read():
            batch = next(pending, None)
            if batch is None:
                clock.done = True
                clock.current = (clock.now, clock.now, [], [])
                for source in participants.sources:
                    source.closed = True
                    source.received_through = clock.now
                return []
            clock.current = batch
            clock.now, cutoff, ingress, _ = batch
            for source in participants.sources:
                source.closed = False
                source.received_through = cutoff
                source.sequence = 1
            return ingress

        participants.sources[0].read = ingress_read
        participants.sources[1].read = lambda: clock.current[3]
        run(
            participants.path,
            participants.stop,
            router_publish=participants.router_publish,
            dispatch_ack_timeout_seconds=timeout,
        )
        return [
            call.args[0]
            for call in participants.router_publish.call_args_list
            if call.args[0]
        ]

    return execute, participants


@pytest.fixture
def shutdown_accounting(cleanup_participants, monkeypatch):
    participants = cleanup_participants
    clock = SimpleNamespace(now=100.0, waits=[])

    def wait(delay):
        clock.waits.append(delay)
        clock.now += delay

    monkeypatch.setattr(scoring, "time", SimpleNamespace(monotonic=lambda: clock.now))
    monkeypatch.setattr(
        scoring, "threading", SimpleNamespace(Event=lambda: SimpleNamespace(wait=wait))
    )

    def execute(batches):
        calls = []
        for index, source in enumerate(participants.sources):
            source.closed = False
            source.sequence = 0
            pending = iter(batches)
            count = SimpleNamespace(value=0)
            calls.append(count)

            def read(source=source, index=index, pending=pending, count=count):
                count.value += 1
                rows = next(pending, ([], [], []))[index]
                for row in rows:
                    source.sequence += 1
                    source.received_through = max(source.received_through, row["time"])
                    source.closed = row["event"] == "footer"
                return rows

            source.read = read
        run(participants.path, participants.stop)
        return participants.audit.submit.call_args.args[1], calls

    return execute, clock


@pytest.mark.pre_merge
@pytest.mark.unit
def test_shutdown_drains_work_and_footers_after_empty_reads(shutdown_accounting):
    execute, clock = shutdown_accounting
    first_clock = dict(time=100, event="u_clock")
    next_clock = dict(time=102, event="u_clock")
    footer = dict(time=103, event="footer")
    empty = ([], [], [])
    final, calls = execute(
        [
            ([arrival_record()], [first_clock], [first_clock]),
            empty,
            (
                [dict(time=102, event="request_completed", request_id="req")],
                [
                    dict(
                        time=101,
                        event="accepted_output",
                        steps=[
                            work_step(
                                client_request_id="req", scheduled_at=100.5, end=101
                            )
                        ],
                    ),
                    next_clock,
                ],
                [next_clock],
            ),
            empty,
            ([footer], [footer], [footer]),
        ]
    )
    assert final["complete"] and final["inputs_closed"]
    assert final["pending_events"] == 0
    assert final["last_snapshot"]["accepted_steps"] == 1
    assert final["last_snapshot"]["output_tokens"] == 1
    assert final["last_snapshot"]["roots"]["root"]["E"] == 2
    assert final["last_snapshot"]["roots"]["root"]["W"] == pytest.approx(0.01)
    assert [call.value for call in calls] == [6, 6, 6]
    assert clock.waits == [0.05, 0.05]


@pytest.mark.pre_merge
@pytest.mark.unit
@pytest.mark.parametrize("pending", [False, True])
def test_shutdown_deadline_reports_incomplete_inputs_without_busy_polling(
    shutdown_accounting, pending
):
    execute, clock = shutdown_accounting
    batches = [([arrival_record()] if pending else [], [], [])]
    final, calls = execute(batches)
    assert not final["complete"] and not final["inputs_closed"]
    assert final["pending_events"] == int(pending)
    assert sum(clock.waits) == pytest.approx(5)
    assert all(0 < delay <= 0.05 for delay in clock.waits)
    assert calls[0].value == len(clock.waits) + 1 + int(pending)


def dispatch_events():
    return [
        arrival_record(),
        dict(time=100, event="request_dispatched", request_id="req"),
    ]


def terminal_ack(at, *, root="root"):
    return dict(
        time=at,
        event="worker_request_finished",
        requests=[dict(client_request_id="req", root=root, agent="agent")],
    )


@pytest.mark.pre_merge
@pytest.mark.unit
@pytest.mark.parametrize("outcome", ["request_failed", "request_cancelled"])
def test_dispatch_timeout_uses_failure_time_despite_frozen_cutoff_and_keeps_late_work(
    timeout_accounting, outcome
):
    execute, participants = timeout_accounting
    snapshots = execute(
        [
            (100, 100, dispatch_events(), []),
            (
                101,
                100,
                [
                    dict(time=101, event=outcome, request_id="req"),
                    dict(time=101.1, event="agent_group_end", root="root"),
                ],
                [],
            ),
            (400.999, 100, [], []),
            (401, 100, [], []),
            (
                402,
                402,
                [],
                [
                    dict(
                        time=102,
                        event="accepted_output",
                        steps=[
                            work_step(
                                client_request_id="req", scheduled_at=100.5, end=102
                            )
                        ],
                    ),
                    terminal_ack(402),
                ],
            ),
        ]
    )
    assert all("dispatch_timeout" not in snapshot for snapshot in snapshots[:3])
    alarm = dict(request_id="req", timeout_seconds=300, failed_at=101)
    assert snapshots[3]["dispatch_timeout"] == alarm
    assert snapshots[3]["observation_time"] == 100
    assert snapshots[3]["accounting_state"]["worker_requests"] == 1
    assert snapshots[3]["accounting_state"]["request_identities"] == 1
    late = snapshots[4]
    assert late["dispatch_timeout"] == alarm
    assert late["accepted_steps"] == 1
    assert late["output_tokens"] == 1
    assert late["accounting_state"]["worker_requests"] == 0
    assert late["accounting_state"]["request_identities"] == 0
    assert late["roots"] == {}
    # Accounting still publishes worker snapshots after the admission gate latches.
    assert participants.publisher.publish.call_args_list[4].args[0] == late


@pytest.mark.pre_merge
@pytest.mark.unit
@pytest.mark.parametrize("ack_at", [200, 401, 500])
def test_received_ack_cancels_timeout_before_ordered_accounting_catches_up(
    timeout_accounting, ack_at
):
    execute, _ = timeout_accounting
    snapshots = execute(
        [
            (100, 100, dispatch_events(), []),
            (101, 100, [dict(time=101, event="request_failed", request_id="req")], []),
            (ack_at, 100, [], [terminal_ack(ack_at)]),
            (500, 100, [], []),
            (501, 501, [], []),
        ]
    )
    assert all("dispatch_timeout" not in snapshot for snapshot in snapshots)
    assert snapshots[3]["accounting_state"]["worker_requests"] == 1
    assert snapshots[4]["accounting_state"]["worker_requests"] == 0


@pytest.mark.pre_merge
@pytest.mark.unit
@pytest.mark.parametrize("resolution", [None, "rejected", "acknowledged"])
def test_watchdog_observes_dispatch_and_resolution_before_ordered_cutoff(
    timeout_accounting, resolution
):
    execute, _ = timeout_accounting
    ingress = [dict(time=101, event="request_dispatched", request_id="req")]
    if resolution == "rejected":
        ingress.append(
            dict(time=101.5, event="request_dispatch_rejected", request_id="req")
        )
    ingress.append(dict(time=102, event="request_failed", request_id="req"))
    snapshots = execute(
        [
            (100, 100, [arrival_record()], []),
            (102, 100, ingress, []),
            (200, 100, [], [terminal_ack(150)] if resolution == "acknowledged" else []),
            (401.999, 100, [], []),
            (402, 100, [], []),
        ]
    )
    assert all("dispatch_timeout" not in snapshot for snapshot in snapshots[:4])
    assert snapshots[4]["observation_time"] == 100
    assert snapshots[4]["accounting_state"]["worker_requests"] == 0
    assert snapshots[4]["accounting_state"]["request_identities"] == 1
    if resolution is None:
        assert snapshots[4]["dispatch_timeout"] == dict(
            request_id="req", timeout_seconds=300, failed_at=102
        )
    else:
        assert "dispatch_timeout" not in snapshots[4]


@pytest.mark.pre_merge
@pytest.mark.unit
@pytest.mark.parametrize(
    "case", ["running", "completed", "rejected", "undispatched", "disabled"]
)
def test_only_failed_uncertain_dispatch_starts_a_deadline(timeout_accounting, case):
    execute, _ = timeout_accounting
    arrivals = dispatch_events() if case != "undispatched" else [arrival_record()]
    outcomes = []
    if case == "rejected":
        outcomes.append(
            dict(time=101, event="request_dispatch_rejected", request_id="req")
        )
    if case != "running":
        outcomes.append(
            dict(
                time=101,
                request_id="req",
                event=(
                    "request_completed" if case == "completed" else "request_failed"
                ),
            )
        )
    snapshots = execute(
        [(100, 100, arrivals, []), (101, 101, outcomes, []), (1000, 101, [], [])],
        timeout=None if case == "disabled" else 300,
    )
    assert all("dispatch_timeout" not in snapshot for snapshot in snapshots)


@pytest.mark.pre_merge
@pytest.mark.unit
@pytest.mark.parametrize("identity_known", [False, True])
@pytest.mark.parametrize("valid", [False, True])
def test_dispatch_timeout_requires_matching_ack_identity(identity_known, valid):
    index = scoring.LifecycleIndex()
    monitor = scoring.DispatchAckMonitor(index, 300)
    if identity_known:
        index.arrive("r", "p", "root", "agent")
        index.dispatch("r")
    monitor.failed("r", 100)
    monitor.acknowledged("r", ("root" if valid else "wrong-root", "agent"))
    if not identity_known:
        index.arrive("r", "p", "root", "agent")
        index.dispatch("r")
    monitor.arm("r")
    monitor.failed("r", 200)  # A duplicate failure must not extend the deadline.
    assert monitor.expire(399.999) is None
    assert bool(monitor.expire(400)) is (not valid)
    assert index.worker_requests == {"r"}


@pytest.mark.pre_merge
@pytest.mark.unit
def test_acknowledged_deadline_storage_is_bounded():
    index = scoring.LifecycleIndex()
    monitor = scoring.DispatchAckMonitor(index, 300)
    for number in range(300):
        rid = str(number)
        index.arrive(rid, rid, rid, rid)
        index.dispatch(rid)
        monitor.failed(rid, 100)
        monitor.acknowledged(rid, (rid, rid))
        monitor.forget(rid)
    assert (
        not monitor.failures and not monitor.acknowledgements and not monitor.deadlines
    )
    assert len(monitor.heap) <= 128
    assert monitor.expire(1000) is None


@pytest.mark.pre_merge
@pytest.mark.unit
@pytest.mark.parametrize("value", ["0", "-1", "nan", "inf", "-inf", "bad", ""])
def test_invalid_dispatch_ack_timeout_is_rejected(monkeypatch, value):
    monkeypatch.setenv("DURU_DISPATCH_ACK_TIMEOUT_SECONDS", value)
    with pytest.raises(ValueError, match="DURU_DISPATCH_ACK_TIMEOUT_SECONDS"):
        scoring.RouterAccounting("unused")


@pytest.mark.pre_merge
@pytest.mark.unit
@pytest.mark.parametrize("value,expected", [(None, 300), ("0.5", 0.5)])
def test_dispatch_ack_timeout_configuration_reaches_accounting(
    monkeypatch, value, expected
):
    monkeypatch.delenv("DURU_DISPATCH_ACK_TIMEOUT_SECONDS", raising=False)
    if value is not None:
        monkeypatch.setenv("DURU_DISPATCH_ACK_TIMEOUT_SECONDS", value)
    worker = Mock()
    monkeypatch.setattr(scoring, "run", worker)
    owner = scoring.RouterAccounting("unused")
    owner._run()
    assert worker.call_args.kwargs["dispatch_ack_timeout_seconds"] == expected


@pytest.mark.pre_merge
@pytest.mark.unit
def test_interleaved_roots_release_only_after_their_worker_acknowledgements(
    cleanup_participants, monkeypatch
):
    participants = cleanup_participants
    book = make_book(1)
    monkeypatch.setattr(scoring, "LiveU", lambda *args: book)
    ingress = []
    for root in ("a", "b"):
        for number in (1, 2):
            rid = f"{root}{number}"
            ingress.extend(
                [
                    arrival_record(100, rid, root, root=root, agent=root),
                    dict(time=100, event="request_dispatched", request_id=rid),
                ]
            )
    ingress.extend(
        [
            dict(time=100, event="request_dispatched", request_id="b1"),
            dict(time=100.1, event="request_dispatch_rejected", request_id="b2"),
            *[
                dict(time=102, event="request_completed", request_id=rid)
                for rid in ("a1", "a2", "b1")
            ],
            dict(time=102, event="request_failed", request_id="b2"),
            dict(time=102.1, event="agent_group_end", root="a"),
            dict(time=102.1, event="agent_group_end", root="b"),
        ]
    )

    def accepted(rid, at):
        return dict(
            event="accepted_output",
            steps=[work_step(client_request_id=rid, root=rid[0], agent=rid[0], end=at)],
        )

    def acknowledged(rid, at):
        return dict(
            time=at,
            event="worker_request_finished",
            requests=[dict(client_request_id=rid, root=rid[0], agent=rid[0])],
        )

    batches = [
        (103, ingress, [accepted("a1", 101), accepted("b1", 101)]),
        (
            104.5,
            [],
            [
                acknowledged("a1", 104),
                acknowledged("a1", 104.1),
                accepted("a2", 104.2),
                accepted("b1", 104.3),
            ],
        ),
        (105, [], [acknowledged("b1", 105)]),
        (
            107,
            [],
            [
                acknowledged("b1", 105.1),
                acknowledged("a2", 106),
                acknowledged("a2", 106.1),
            ],
        ),
        (107, [], []),
    ]
    for source, column in zip(participants.sources, (1, 2, None)):
        source.sequence = 0
        pending = iter(batches)

        def read(source=source, column=column, pending=pending):
            cutoff, arrivals, work = next(pending, batches[-1])
            source.received_through = cutoff
            source.closed = not arrivals and not work
            source.sequence += 1
            return arrivals if column == 1 else work if column == 2 else []

        source.read = read
    run(
        participants.path, participants.stop, router_publish=participants.router_publish
    )
    snapshots = [
        call.args[0]
        for call in participants.router_publish.call_args_list
        if call.args[0]
    ]
    first, partial, b_released, both_released = snapshots[:4]
    assert first["accounting_state"]["worker_requests"] == 3
    assert first["accounting_state"]["closing_roots"] == 2
    assert set(first["roots"]) == {"a", "b"}
    assert partial["accounting_state"]["worker_requests"] == 2
    assert partial["accounting_state"]["request_identities"] == 4
    assert partial["roots"]["a"]["E"] == partial["roots"]["b"]["E"] == 1
    assert partial["roots"]["a"]["W"] == partial["roots"]["b"]["W"] == 4
    assert set(b_released["roots"]) == {"a"}
    assert b_released["accounting_state"]["request_identities"] == 2
    assert b_released["accounting_state"]["program_identities"] == 1
    assert b_released["accounting_state"]["worker_requests"] == 1
    assert b_released["accounting_state"]["closing_roots"] == 1
    assert both_released["accounting_state"] == dict(
        request_identities=0,
        program_identities=0,
        agents=0,
        roots=0,
        pending_events=0,
        worker_requests=0,
        closing_roots=0,
    )
    assert both_released["accepted_steps"] == 4
    assert both_released["output_tokens"] == 4
    assert not book.root_agents


@pytest.mark.pre_merge
@pytest.mark.unit
@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("measured", [False, True])
def test_overlapping_program_uses_highest_active_request_score(
    cleanup_participants, monkeypatch, reverse, measured
):
    participants = cleanup_participants
    book = make_book(1)
    book.prefill_budget = 8
    monkeypatch.setattr(scoring, "LiveU", lambda *args: book)
    short = arrival_record(100, "short", "shared")
    long = dict(arrival_record(100, "long", "shared"), prompt_tokens=40)
    arrivals = [short, long]
    if reverse:
        arrivals.reverse()
    for source in participants.sources:
        source.received_through = 110
        source.sequence = 1
    participants.sources[0].read = Mock(side_effect=[arrivals, []])
    if measured:
        participants.sources[1].read = Mock(
            side_effect=[
                [
                    dict(
                        event="accepted_output",
                        steps=[work_step(client_request_id="short", end=105)],
                    )
                ],
                [],
            ]
        )
    run(
        participants.path, participants.stop, router_publish=participants.router_publish
    )
    snapshot = next(
        call.args[0]
        for call in participants.router_publish.call_args_list
        if call.args[0]
    )
    assert book.entitlement["root"] == 10
    assert book.root_active["root"] == 1
    assert set(snapshot["requests"]) == {"short", "long"}
    program = snapshot["programs"]["shared"]
    if measured:
        assert program["score"] == program["U"] == 5
        # Equal scores/arrival times use request identity, not insertion order.
        assert program["request_id"] == "long"
        assert book.work["root"] == 2
    else:
        assert snapshot["requests"]["short"]["score"] == pytest.approx(10 / 6)
        assert snapshot["requests"]["long"]["score"] == 1
        assert program["score"] == pytest.approx(10 / 6)
        assert program["request_id"] == "short"
        assert program["U"] is None
        assert book.work["root"] == 0


@pytest.mark.pre_merge
@pytest.mark.unit
def test_accounting_rejects_late_work_before_charging_it(
    cleanup_participants, monkeypatch
):
    participants = cleanup_participants
    book = make_book()
    book.submit("q", "root", "agent", 8, 90)
    book.advance(100)
    before = deepcopy(vars(book))
    monkeypatch.setattr(scoring, "LiveU", Mock(return_value=book))
    for source in participants.sources:
        source.received_through = 100
    participants.sources[1].read = Mock(
        return_value=[dict(event="accepted_output", steps=[work_step(end=99)])]
    )
    with pytest.raises(ValueError, match="Late compact event"):
        run(participants.path, participants.stop)
    assert vars(book) == before


@pytest.mark.pre_merge
@pytest.mark.unit
@pytest.mark.parametrize("failure", ["audit", "source", "router_publish", "publisher"])
def test_accounting_attempts_every_cleanup_after_failure(cleanup_participants, failure):
    participants = cleanup_participants
    failing_close = {
        "audit": participants.audit.close,
        "source": participants.sources[1].close,
        "router_publish": participants.router_publish,
        "publisher": participants.publisher.close,
    }[failure]
    failing_close.side_effect = RuntimeError("injected cleanup failure")
    with pytest.raises(RuntimeError, match="injected cleanup failure"):
        run(
            participants.path,
            participants.stop,
            router_publish=participants.router_publish,
        )
    participants.audit.close.assert_called_once_with()
    for source in participants.sources:
        source.close.assert_called_once_with()
    participants.router_publish.assert_called_once_with({})
    participants.publisher.close.assert_called_once_with()


@pytest.mark.pre_merge
@pytest.mark.unit
@pytest.mark.parametrize("failure", ["source", "audit", "publisher"])
def test_accounting_closes_acquired_resources_after_setup_failure(
    cleanup_participants, failure
):
    participants = cleanup_participants
    error = RuntimeError("injected setup failure")
    if failure == "source":
        participants.source_factory.side_effect = [participants.sources[0], error]
    else:
        factory = {
            "audit": participants.audit_factory,
            "publisher": participants.publisher_factory,
        }[failure]
        factory.side_effect = error
    with pytest.raises(RuntimeError, match="injected setup failure"):
        run(
            participants.path,
            participants.stop,
            router_publish=participants.router_publish,
        )
    acquired = 1 if failure == "source" else 3
    for source in participants.sources[:acquired]:
        source.close.assert_called_once_with()
    for source in participants.sources[acquired:]:
        source.close.assert_not_called()
    if failure == "publisher":
        participants.audit.close.assert_called_once_with()
    else:
        participants.audit.close.assert_not_called()
    participants.publisher.close.assert_not_called()
    participants.router_publish.assert_called_once_with({})


@pytest.mark.usefixtures("journal_validation_mode")
@pytest.mark.post_merge
@pytest.mark.integration
@pytest.mark.timeout(30)
@pytest.mark.parametrize(
    "case, message",
    [
        ("missing_program", "program_id"),
        ("changed_program", "program identity changed"),
        ("reuse_active", "Request ID reused"),
        ("reuse_finished", "Request ID reused"),
    ],
)
def test_invalid_ingress_identity_invalidates_scores(config, records, case, message):
    cfg = config[1]
    ingress, _ = records
    if case == "missing_program":
        ingress[0].pop("program_id")
    elif case == "changed_program":
        ingress.insert(
            -1,
            arrival_record(100.45, "next", root="other-root", agent="other-agent"),
        )
    else:
        arrival = 100.15 if case == "reuse_active" else 100.45
        ingress.append(arrival_record(arrival))
        ingress.sort(key=lambda row: row["time"])
    write_rows(Path(cfg["ingress"]), ingress)
    with pytest.raises(ValueError, match=message):
        run_once(config)
    assert not audit_rows(cfg)[-1]["complete"]
    assert not Path(cfg["scores"]).exists()


@pytest.mark.usefixtures("journal_validation_mode")
@pytest.mark.post_merge
@pytest.mark.integration
@pytest.mark.timeout(30)
@pytest.mark.parametrize("field", ["client_request_id", "root", "agent"])
def test_worker_identity_must_join_ingress(config, records, field):
    cfg = config[1]
    _, worker = records
    worker[0]["steps"][0][field] = "unknown"
    write_rows(Path(cfg["ledgers"][0]), worker)
    message = (
        "Unjoined work"
        if field == "client_request_id"
        else "Worker request identity mismatch"
    )
    with pytest.raises(ValueError, match=message):
        run_once(config)
    assert not audit_rows(cfg)[-1]["complete"]
    assert not Path(cfg["scores"]).exists()


@pytest.mark.usefixtures("journal_validation_mode")
@pytest.mark.post_merge
@pytest.mark.integration
@pytest.mark.timeout(30)
def test_explicit_program_ids_preserve_idle_and_overlapping_agent_requests(
    config, records
):
    cfg = config[1]
    ingress, _ = records
    ingress[-1:-1] = [
        arrival_record(100.45, "next", "other-session"),
        arrival_record(100.46, "parallel", "parallel-session"),
    ]
    write_rows(Path(cfg["ingress"]), ingress)
    snapshot = run_once(config)["last_snapshot"]
    assert set(snapshot["programs"]) == {
        "native/session",
        "other-session",
        "parallel-session",
    }
    assert snapshot["programs"]["native/session"]["kind"] == "idle_measured_only"
    for request, program in [
        ("next", "other-session"),
        ("parallel", "parallel-session"),
    ]:
        assert snapshot["programs"][program]["request_id"] == request
        assert snapshot["requests"][request]["program_id"] == program
    assert snapshot["roots"]["root"]["active_agents"] == 1
    assert snapshot["roots"]["root"]["E"] == pytest.approx(0.45)


@pytest.mark.usefixtures("journal_validation_mode")
@pytest.mark.post_merge
@pytest.mark.integration
@pytest.mark.timeout(30)
@pytest.mark.parametrize("bad_id", [None, "", 7])
def test_ingress_requires_program_id_and_rejects_second_producer(tmp_path, bad_id):
    ingress = Ingress(tmp_path)
    fields = arrival_record()
    fields.pop("event")
    fields.pop("time")
    try:
        with pytest.raises(ValueError, match="program_id"):
            ingress.emit("request_arrived", 1, **dict(fields, program_id=bad_id))
        ingress.emit("request_arrived", 2, **fields, unrelated="ignored")
    finally:
        ingress.close()
    journal = tmp_path / "duru-ingress.jsonl"
    original = journal.read_bytes()
    rows = read_rows(journal)
    assert rows[0]["sequence"] == 1
    assert rows[0]["program_id"] == "native/session"
    assert "unrelated" not in rows[0]
    assert [row["event"] for row in rows] == ["request_arrived", "footer"]
    with pytest.raises(FileExistsError):
        Ingress(tmp_path)
    assert journal.read_bytes() == original


@pytest.mark.usefixtures("journal_validation_mode")
@pytest.mark.post_merge
@pytest.mark.integration
@pytest.mark.timeout(30)
def test_ingress_closes_transport_when_footer_emission_fails(tmp_path):
    ingress = Ingress(tmp_path)
    with patch.object(ingress.journal, "close", wraps=ingress.journal.close) as close:
        with patch.object(ingress, "emit", side_effect=RuntimeError("footer rejected")):
            with pytest.raises(RuntimeError, match="footer rejected"):
                ingress.close()
        close.assert_called_once()


@pytest.fixture
def journal_validation_mode(monkeypatch, memory_mode):
    monkeypatch.setenv("DURU_ACCOUNTING_TRANSPORT", "journal")
    monkeypatch.setenv("DURU_RECORD_EVENTS", "1")
    with tempfile.TemporaryDirectory(prefix="duru-replay-test-") as directory:
        monkeypatch.setenv("DURU_IPC_DIR", directory)
        yield


@pytest.mark.pre_merge
@pytest.mark.unit
@pytest.mark.timeout(3)
def test_accounting_failure_invalidates_local_scores(monkeypatch):
    published, fail_now = threading.Event(), threading.Event()
    snapshot = dict(observation_time=time.monotonic(), requests={"q": {"score": 1}})
    failure = ValueError("bad event")

    def fail(*args, **kwargs):
        kwargs["router_publish"](snapshot)
        published.set()
        assert fail_now.wait(1)
        raise failure

    monkeypatch.setattr(scoring, "run", fail)
    owner = scoring.RouterAccounting("unused").start()
    try:
        assert published.wait(1)
        assert owner.reader.refresh() is snapshot
        fail_now.set()
        with pytest.raises(ValueError) as result:
            owner._result.result(timeout=1)
        assert result.value is failure
        owner._thread.join(1)
        assert not owner._thread.is_alive()
        with pytest.raises(scoring.AccountingError, match="accounting failed") as error:
            owner.reader.refresh()
        assert error.value.__cause__ is failure
        assert owner._snapshot == {}
    finally:
        fail_now.set()
        with pytest.raises(scoring.AccountingError, match="accounting failed"):
            owner.close()


@pytest.mark.pre_merge
@pytest.mark.unit
@pytest.mark.parametrize("failure", [KeyboardInterrupt(), SystemExit()])
def test_accounting_does_not_consume_process_control_exceptions(monkeypatch, failure):
    def fail(*args, **kwargs):
        raise failure

    monkeypatch.setattr(scoring, "run", fail)
    owner = scoring.RouterAccounting("unused")
    owner._router_publish({"requests": {"q": {"score": 1}}})
    with pytest.raises(type(failure)) as error:
        owner._run()
    assert error.value is failure
    assert owner._snapshot == {}
    assert not owner._result.done()


@pytest.mark.pre_merge
@pytest.mark.integration
@pytest.mark.timeout(1)  # About 0.2s per case locally; round up the 3x allowance.
@pytest.mark.parametrize(
    "record,diagnostics", [(False, False), (False, True), (True, False)]
)
@pytest.mark.parametrize("in_process", [False, True])
def test_router_owned_accounting_matches_worker_snapshot(
    tmp_path, monkeypatch, record, diagnostics, in_process
):
    path, cfg = configuration(tmp_path)
    monkeypatch.setenv("DURU_SCORES", cfg["scores"])
    monkeypatch.setenv("DURU_RECORD_EVENTS", "1" if record else "0")
    monkeypatch.setenv("DURU_DIAGNOSTICS", "1" if diagnostics else "0")
    now = time.monotonic()
    # Keep freshness independent of IPC startup time while real test deadlines
    # continue to use this module's clock.
    monkeypatch.setattr(scoring, "time", SimpleNamespace(monotonic=lambda: now + 0.01))
    ingress_rows = [
        arrival_record(now - 0.01, "q", "p", root="r", agent="a"),
        dict(event="footer", time=now + 0.002),
    ]
    worker_rows = [
        dict(
            event="accepted_output",
            time=now + 0.001,
            steps=[
                work_step(
                    p=8,
                    root="r",
                    agent="a",
                    client_request_id="q",
                    scheduled_at=now,
                    end=now + 0.001,
                )
            ],
        ),
        dict(event="footer", time=now + 0.002),
    ]
    with event_feeds(cfg, ingress_rows, worker_rows, in_process=in_process) as ingress:
        owner = scoring.RouterAccounting(path, ingress=ingress).start()
        engine_reader = scoring.Reader()
        try:
            deadline = time.monotonic() + 5
            while time.monotonic() < deadline:
                snapshot = owner.reader.refresh()
                engine_snapshot = engine_reader.refresh()
                if (
                    snapshot.get("accepted_steps") == 1
                    and engine_snapshot.get("requests")
                    and engine_snapshot["observation_time"]
                    == snapshot["observation_time"]
                ):
                    break
                time.sleep(0.01)
            else:
                pytest.fail("Router accounting failed to publish")
            assert not Path(cfg["scores"]).exists()
            assert engine_snapshot["observation_time"] == snapshot["observation_time"]
            assert engine_snapshot["requests"] == {
                rid: {"score": row["score"]}
                for rid, row in snapshot["requests"].items()
            }
            assert ("roots" in snapshot) is (record or diagnostics)
            assert ("source_sequences" in snapshot) is record
            assert ("source_received_through" in snapshot) is record
            if not record:
                assert not Path(cfg["audit_dir"]).exists()
                assert not list(tmp_path.rglob("*.jsonl"))
            assert owner.reader.snapshot_source is not None
            assert owner._thread is not threading.current_thread()
            # File corruption cannot change the router's direct-memory decision.
            monkeypatch.setattr(
                Path, "read_text", lambda *a, **k: pytest.fail("Unexpected file read")
            )
            assert owner.reader.refresh()["programs"]["p"]["U"] is not None
        finally:
            try:
                owner.close()
            finally:
                engine_reader.close()
        assert not owner._thread.is_alive()
        assert owner._snapshot == {}
