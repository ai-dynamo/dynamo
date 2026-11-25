# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Score calculation, router-owned accounting, and lifecycle regressions."""

import json
import tempfile
import threading
import time
from pathlib import Path
from unittest.mock import patch

import pytest

from dynamo.duru_router import scoring
from dynamo.duru_router.scoring import Ingress, LiveU, priority_key, run
from dynamo.duru_router.tests.helpers import arrival_record, configuration, event_feeds
from dynamo.duru_router.tests.helpers import memory_mode as memory_mode
from dynamo.duru_router.tests.helpers import replay_accounting, work_step

pytestmark = [pytest.mark.gpu_0]


def make_book(capacity=4):
    book = LiveU(capacity, {})
    book.cost = lambda phase, p, d, k: 2.0 if phase == "prefill" else 1.0
    return book


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
def test_accounting_failure_invalidates_local_scores(monkeypatch):
    def fail(*args, **kwargs):
        raise ValueError("bad event")

    monkeypatch.setattr(scoring, "run", fail)
    owner = scoring.RouterAccounting("unused").start()
    owner._thread.join(2)
    with pytest.raises(RuntimeError, match="accounting failed"):
        owner.reader.refresh()
    assert owner._snapshot == {}
    with pytest.raises(RuntimeError, match="accounting failed"):
        owner.close()


@pytest.mark.pre_merge
@pytest.mark.unit
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
    now = time.time()
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
                if snapshot.get("accepted_steps") == 1 and engine_snapshot.get(
                    "requests"
                ):
                    break
                time.sleep(0.01)
            else:
                pytest.fail("Router accounting failed to publish")
            assert not Path(cfg["scores"]).exists()
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
            owner.close()
            engine_reader.close()
        assert not owner._thread.is_alive()
        assert owner._snapshot == {}
