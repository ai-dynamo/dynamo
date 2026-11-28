# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Duru admission rules and their integration with ThunderAgent."""

import asyncio
import json
import time
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import pytest_asyncio

from dynamo.duru_router import router as router_module
from dynamo.duru_router.admission import Candidate, DuruPolicy
from dynamo.duru_router.router import DuruScheduler, ThunderAgentConfig
from dynamo.duru_router.scoring import Reader
from dynamo.thunderagent_router import router as native_router
from dynamo.thunderagent_router.policy import ResumeChoice
from dynamo.thunderagent_router.program_state import ProgramLifecycle, ProgramStatus
from dynamo.thunderagent_router.tests.test_router import FakeCapacity

pytestmark = [
    pytest.mark.pre_merge,
    pytest.mark.unit,
    pytest.mark.gpu_0,
    pytest.mark.timeout(10),
]


class RouterCase:
    def __init__(self):
        self.snapshot = {}
        self.scores, self.events, self.tasks = {}, [], []
        self.capacity = FakeCapacity(workers={(1, 0): 1000})
        self.router = DuruScheduler(
            self.capacity,
            ThunderAgentConfig(
                buffer_per_program=0,
                pause_threshold=1.0,
                resume_hysteresis=0.0,
            ),
            reader=Reader(snapshot_source=lambda: self.snapshot),
        )

    def score(self, program, value):
        self.scores[program] = dict(
            score=value, U=value, arrival=len(self.scores), request_id=program
        )

    def save(self, age=0):
        self.snapshot = dict(
            observation_time=time.monotonic() - age, programs=deepcopy(self.scores)
        )

    def put(self, program, demand, score):
        self.score(program, score)
        obj = self.router._table.begin_request(program, demand)
        obj.lifecycle = ProgramLifecycle.PAUSED
        obj.waiting = asyncio.Event()
        self.router._table.paused[program] = None
        return obj

    async def resume(self, age=0):
        self.save(age)
        await self.router._greedy_resume(self.capacity.workers)

    async def start(self, program, demand):
        self.save()
        checked = asyncio.Event()
        original = self.router._admission_ready

        def ready(program_id):
            checked.set()
            return original(program_id)

        with patch.object(self.router, "_admission_ready", ready):
            task = asyncio.create_task(self.router.before_request(program, demand))
            self.tasks.append(task)
            await asyncio.wait_for(checked.wait(), 1)
        return task

    @property
    def admitted(self):
        return [event["program"] for event in self.events]

    def used(self, replica):
        # Read the same occupancy buckets used by the shared scheduler.
        return self.router._replica_usage_locked(self.capacity.workers)[replica].used


@pytest_asyncio.fixture
async def case(tmp_path, monkeypatch):
    monkeypatch.setenv("DURU_SCORES", str(tmp_path / "scores.json"))
    case = RouterCase()
    monkeypatch.setenv("DURU_AUDIT_DIR", str(tmp_path))
    monkeypatch.delenv("DURU_ADMISSION_ALPHA", raising=False)
    monkeypatch.delenv("DURU_DIAGNOSTICS", raising=False)
    selected = None

    def observe_selection(method):
        def select(*args, **kwargs):
            nonlocal selected
            selected = method(*args, **kwargs)
            return selected

        return select

    original_resume = case.router._resume_program

    def observe_resume(program, replica):
        # Observe the host's real decisions rather than requiring production
        # audit logging to expose scheduling behavior to tests.
        assert selected is not None and selected.replica == replica
        case.events.append(
            dict(
                program=program.program_id,
                score=case.router._policy.score_row(program.program_id).get("score"),
                forced=selected.forced,
                remaining=selected.remaining,
            )
        )
        original_resume(program, replica)

    for method in ("_select_resume_locked", "_select_resume_override_locked"):
        monkeypatch.setattr(
            case.router, method, observe_selection(getattr(case.router, method))
        )
    monkeypatch.setattr(case.router, "_resume_program", observe_resume)
    try:
        yield case
    finally:
        for task in case.tasks:
            task.cancel()
        await asyncio.wait_for(asyncio.gather(*case.tasks, return_exceptions=True), 1)


@pytest.mark.asyncio
@pytest.mark.parametrize("diagnostics", [False, True])
async def test_optional_diagnostics_preserve_admission(
    case, monkeypatch, caplog, diagnostics
):
    monkeypatch.setenv("DURU_DIAGNOSTICS", "1" if diagnostics else "0")
    monkeypatch.setenv("DURU_ADMISSION_ALPHA", "1")
    caplog.set_level("INFO", logger="dynamo.duru_router.diagnostics")
    case.put("ready", 100, 2)
    with patch(
        "dynamo.duru_router.events.open_journal",
        side_effect=AssertionError("Admission must not open an audit journal"),
    ):
        await case.resume()
    assert case.admitted == ["ready"]
    assert case.events[0]["forced"] is False
    records = [r for r in caplog.records if r.name.endswith(".diagnostics")]
    assert len(records) == int(diagnostics)
    if diagnostics:
        row = json.loads(records[0].getMessage().split(" ", 1)[1])
        assert row["forced"] is False
        assert row["U"] == 2
        assert row["required"] == 100
        assert row["remaining_before"] == 1000
        assert row["worker"] == [1, 0]
        assert row["request_id"] == "ready"
    assert not hasattr(router_module, "journal")


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel", [False, True], ids=["recover", "cancel"])
async def test_missing_initial_score_defers_without_leaking_program(
    case, monkeypatch, cancel
):
    retry = asyncio.Event()

    async def wait_for_score(delay):
        await retry.wait()

    clock = SimpleNamespace(**vars(asyncio))
    clock.sleep = wait_for_score
    monkeypatch.setattr(native_router, "asyncio", clock)
    task = await case.start("new", 100)
    assert not task.done()
    assert "new" not in case.router._table.programs
    if cancel:
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, 1)
        assert "new" not in case.router._table.programs
    else:
        case.score("new", 0)
        case.save()
        retry.set()
        assert (await asyncio.wait_for(task, 1)).assigned_replica_hint == (1, 0)
    assert not case.router._admission_gates


@pytest.mark.asyncio
async def test_stale_scores_block_admission_then_fresh_scores_restore_rank(case):
    resident = case.router._table.begin_request("resident", 900)
    resident.assigned_replica = (1, 0)
    case.score("low", 0.2)
    low = await case.start("low", 300)
    case.score("high", 1.4)
    high = await case.start("high", 200)
    resident.token_total = 650
    case.save(age=3)
    await case.router._scheduler_tick()
    assert not high.done() and not low.done()
    assert not case.events
    case.save()
    await case.router._scheduler_tick()
    await asyncio.wait_for(high, 1)
    assert case.admitted == ["high"]
    assert case.events[0]["score"] == 1.4
    assert case.events[0]["forced"] is False
    assert not low.done()


@pytest.mark.asyncio
async def test_capacity_fit_admissions_follow_score_order(case, monkeypatch):
    monkeypatch.setenv("DURU_ADMISSION_ALPHA", "0.7")
    for program, score in zip(["middle", "low", "high"], [0.1, 0.01, 0.2]):
        case.put(program, 300, score)
    await case.resume()
    assert case.admitted == ["high", "middle", "low"]
    assert case.router._stat_forced_resumes == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("fragmented", [False, True])
async def test_backfill_skips_nonfits_without_overcommitting(
    case, monkeypatch, fragmented
):
    monkeypatch.setenv("DURU_ADMISSION_ALPHA", "0.7")
    if fragmented:
        case.capacity.workers = {(1, 0): 500, (2, 0): 500}
    for program, demand, score in [
        ("large", 1100, 0.6),
        ("medium", 800, 0.5),
        ("small", 300, 0.4),
        ("tiny", 150, 0.3),
    ]:
        case.put(program, demand, score)
    await case.resume()
    assert case.admitted == (["small", "tiny"] if fragmented else ["medium", "tiny"])
    assert "large" in case.router._table.paused
    assert all(
        case.used(key) <= capacity for key, capacity in case.capacity.workers.items()
    )
    for program in list(case.admitted):
        await case.router.end_program(program)
    case.capacity.workers = {(1, 0): 1500}
    await case.resume()
    assert (
        "large" in case.admitted
    )  # A previous nonfit remains eligible on a later pass.


@pytest.mark.asyncio
async def test_new_arrival_joins_existing_rank(case):
    case.put("older_high", 800, 4)
    case.score("new_low", 0.1)
    task = await case.start("new_low", 400)
    assert case.admitted == ["older_high"]
    assert not task.done()
    assert case.used((1, 0)) == 800


@pytest.mark.asyncio
async def test_capacity_pauses_only_at_safe_boundary_and_never_for_score_alone(case):
    case.score("busy", 0.001)
    case.save()
    await case.router.before_request("busy", 900)
    await case.router.after_request("busy", 900, 0)
    case.put("high", 400, 100)
    case.save()
    await case.router._scheduler_tick()
    busy = case.router._table.programs["busy"]
    assert busy.lifecycle == ProgramLifecycle.ACTIVE
    assert "high" in case.router._table.paused
    await case.router.before_request("busy", 900)
    case.capacity.workers = {(1, 0): 800}
    await case.router._scheduler_tick()
    assert busy.lifecycle == ProgramLifecycle.ACTIVE
    assert not busy.marked_for_pause
    await case.router.after_request("busy", 900, 0)
    await case.router._scheduler_tick()
    assert busy.lifecycle == ProgramLifecycle.PAUSED
    assert case.used((1, 0)) == 0


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "idle_score,pending_score,winner", [(0.7, 0.5, "idle"), (0.5, 0.7, "pending")]
)
async def test_idle_reservation_competes_by_measured_u(
    case, idle_score, pending_score, winner
):
    idle = case.put("idle", 600, idle_score)
    idle.status = ProgramStatus.ACTING
    case.scores["idle"].update(score=None, kind="idle_measured_only")
    case.put("pending", 600, pending_score)
    await case.resume()
    assert case.admitted == [winner]
    assert idle.status == ProgramStatus.ACTING
    assert case.used((1, 0)) == 600


@pytest.mark.parametrize(
    "first_admissions, expected",
    [
        ((True, True), ["first", "second"]),
        ((True, False), ["first"]),
        ((False, True), ["first"]),
        ((False, False), ["first"]),
    ],
)
@pytest.mark.asyncio
async def test_admission_pass_shares_ceiling(case, first_admissions, expected):
    case.router._cfg.pause_threshold, case.router._cfg.resume_hysteresis = 0.95, 0.1
    first = case.put("first", 600, 2)
    second = case.put("second", 300, 1)
    first._u_new_admission, second._u_new_admission = first_admissions

    await case.resume()

    assert case.admitted == expected
    assert case.used((1, 0)) == (900 if len(expected) == 2 else 600)


@pytest.mark.asyncio
@pytest.mark.parametrize("diagnostics", [False, True])
async def test_failed_first_admission_retains_full_capacity_eligibility(
    case, monkeypatch, diagnostics
):
    monkeypatch.setenv("DURU_DIAGNOSTICS", "1" if diagnostics else "0")
    case.router._cfg.pause_threshold, case.router._cfg.resume_hysteresis = 0.95, 0.1
    case.put("occupant", 200, 1)
    await case.resume()
    case.score("new", 2)
    waiting = await case.start("new", 900)
    newcomer = case.router._table.programs["new"]
    assert not waiting.done()
    assert newcomer.lifecycle == ProgramLifecycle.PAUSED
    assert newcomer._u_new_admission
    assert case.used((1, 0)) == 200

    await case.resume()
    assert newcomer._u_new_admission
    assert not waiting.done()

    await case.router.end_program("occupant")
    await case.resume()
    await asyncio.wait_for(waiting, 1)
    assert case.admitted == ["occupant", "new"]
    assert newcomer.lifecycle == ProgramLifecycle.ACTIVE
    assert not newcomer._u_new_admission
    assert case.used((1, 0)) == 900


@pytest.mark.asyncio
async def test_newcomer_preserves_resume_headroom_for_established_waiters(case):
    case.router._cfg.pause_threshold, case.router._cfg.resume_hysteresis = 0.95, 0.1
    new = case.put("new", 900, 0.1)
    new._u_new_admission = True
    await case.resume()
    assert case.admitted == ["new"]  # First admission can use the full budget.
    new.token_total = 800
    case.put("older", 100, 5)
    case.put("newcomer", 100, 0.1)._u_new_admission = True
    await case.resume()
    assert case.admitted == ["new"]
    new.token_total = 700
    await case.resume()
    assert case.admitted == ["new", "older"]


@pytest.mark.asyncio
async def test_missing_scores_and_nonpending_requests_cannot_force_admission(
    case, monkeypatch
):
    monkeypatch.setenv("DURU_ADMISSION_ALPHA", "0.7")
    case.put("missing", 100, 1)
    del case.scores["missing"]
    case.put("no_request", 1100, 9)
    del case.scores["no_request"]["request_id"]
    case.put("idle", 1100, 3).status = ProgramStatus.ACTING
    case.put("valid", 100, 0.01)
    await case.resume()
    assert case.admitted == ["valid"]


@pytest.mark.asyncio
@pytest.mark.parametrize("alpha", [None, "0.7"])
async def test_cancelled_waiter_cannot_reappear_or_force_admit(
    case, monkeypatch, alpha
):
    if alpha:
        monkeypatch.setenv("DURU_ADMISSION_ALPHA", alpha)
    case.score("cancelled", 0.1)
    task = await case.start("cancelled", 1100)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(task, 1)
    case.scores["cancelled"]["score"] = 3
    await case.resume()
    assert not case.events
    assert not case.router._table.programs
    assert not case.router._admission_gates
    assert case.router._stat_forced_resumes == 0
    assert case.used((1, 0)) == 0


@pytest.mark.asyncio
async def test_old_waiter_cannot_remove_recreated_program(case):
    case.score("reused", 0.1)
    task = await case.start("reused", 1100)
    await case.router.end_program("reused")
    replacement = case.router._table.begin_request("reused", 100)
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(task, 1)
    assert case.router._table.programs["reused"] is replacement
    assert replacement.token_total == 100


@pytest.mark.asyncio
async def test_resume_updates_weighted_reservations_between_policy_choices(case):
    case.router._cfg.acting_token_weight = 0.5
    case.router._cfg.buffer_per_program = 10
    case.put("idle", 600, 3).status = ProgramStatus.ACTING
    case.put("pending", 600, 2)
    case.put("tail", 100, 1)

    await case.resume()

    # Idle requires 610 to fit, but reserves 300 + 10 after admission. The
    # next choice must see that weighted reservation, then the pending 610.
    assert case.admitted == ["idle", "pending"]
    assert [event["remaining"] for event in case.events] == [1000, 690]
    assert case.used((1, 0)) == 920
    assert list(case.router._table.paused) == ["tail"]


@pytest.mark.asyncio
async def test_capacity_fit_placement_breaks_replica_ties_by_key(case):
    # Same worker, reverse DP-rank insertion order: fits prefer the smaller key.
    case.capacity.workers = {(7, 1): 1000, (7, 0): 1000}
    candidate = case.put("candidate", 100, 2)

    await case.resume()

    assert candidate.assigned_replica == (7, 0)
    assert case.events[0]["forced"] is False
    assert case.events[0]["remaining"] == 1000


@pytest.mark.asyncio
async def test_host_checks_fit_before_alpha_and_preserves_replica_ties(
    case, monkeypatch
):
    monkeypatch.setenv("DURU_ADMISSION_ALPHA", "0.7")
    case.capacity.workers = {(7, 1): 100, (7, 0): 100}
    program = case.put("high", 100, 0.7)

    # A fit must never consult the exception, even when score >= alpha.
    with patch.object(
        router_module.DuruResumePlan,
        "may_force_admit",
        side_effect=AssertionError("Capacity fit must precede alpha exception"),
    ):
        await case.resume()

    assert program.assigned_replica == (7, 0)
    assert case.events[0]["forced"] is False


@pytest.mark.parametrize("channel", [None, "", "   "])
def test_duru_requires_score_channel_before_startup(monkeypatch, channel):
    if channel is None:
        monkeypatch.delenv("DURU_SCORES", raising=False)
    else:
        monkeypatch.setenv("DURU_SCORES", channel)
    with pytest.raises(ValueError, match="DURU_SCORES.*IPC channel"):
        DuruScheduler(
            FakeCapacity(),
            ThunderAgentConfig(),
            reader=Reader(snapshot_source=lambda: {}),
        )
    with pytest.raises(ValueError, match="DURU_SCORES.*IPC channel"):
        Reader()


def test_policies_and_prepared_plans_keep_their_own_scores(monkeypatch, caplog):
    monkeypatch.setenv("DURU_SCORES", "test-channel")
    monkeypatch.setenv("DURU_DIAGNOSTICS", "1")
    caplog.set_level("INFO", logger="dynamo.duru_router.diagnostics")
    programs = [router_module.Program(pid) for pid in ("a", "b")]
    snapshots = [
        dict(
            run_id=str(index),
            observation_time=time.monotonic() - index * 0.1,
            programs={
                pid: dict(score=score, U=score, request_id=f"{index}-{pid}")
                for pid, score in zip(("a", "b"), scores)
            },
        )
        for index, scores in enumerate(((2, 1), (1, 2)))
    ]
    readers = [Reader(snapshot_source=lambda s=s: s) for s in snapshots]
    schedulers = [
        DuruScheduler(FakeCapacity(), ThunderAgentConfig(), reader=reader)
        for reader in readers
    ]
    first, second = [scheduler._policy for scheduler in schedulers]
    plan = first.prepare_resume(programs, ThunderAgentConfig())
    other = second.prepare_resume(programs, ThunderAgentConfig())
    assert [p.program_id for p in plan.programs] == ["a", "b"]
    assert [p.program_id for p in other.programs] == ["b", "a"]
    assert first.score_row("a")["score"] == 2
    assert second.score_row("a")["score"] == 1

    # A later decision must not change an already prepared plan's audit identity.
    readers[0].snapshot_source = lambda: snapshots[1]
    first.prepare_resume(programs, ThunderAgentConfig())
    plan.record_admission(programs[0], ResumeChoice((1, 0), 1000, 0.85), 0)
    row = json.loads(caplog.records[-1].getMessage().split(" ", 1)[1])
    assert row["run_id"] == "0"
    assert row["request_id"] == "0-a"
    assert row["observation_time"] == snapshots[0]["observation_time"]
    assert row["U"] == 2

    readers[0].snapshot_source = lambda: {}
    assert not first.admission_ready("a")
    assert second.admission_ready("a")


def test_idle_measurement_and_stable_ties():
    candidates = [
        Candidate("first", 1, score=0.2, arrival=2),
        Candidate("unknown", 1, score=9, known=False),
        Candidate("second", 1, score=0.2, arrival=2),
        Candidate("idle", 1, measured_u=0.5, idle=True),
    ]
    assert [c.identity for c in DuruPolicy().order(candidates)] == [
        "idle",
        "first",
        "second",
    ]
