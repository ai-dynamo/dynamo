# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Lockstep controller and rank tests over the CPU fake of DeepInfer."""

import shutil
import signal
import socket
import tempfile
import threading
import time
from typing import Any, Optional

import pytest
import zmq

from dynamo.deepinfer.controller import Controller
from dynamo.deepinfer.protocol import (
    AddReq,
    Cmd,
    ControlServer,
    EngineMeta,
    LocalRankHandle,
    Op,
    Output,
    RankFailed,
    RankHandle,
    Ready,
    Reply,
    ZmqRankHandle,
    decode_cmd,
    encode,
    routing_id,
    wait_all_ready,
)
from dynamo.deepinfer.rank import RankRunner, RankSpec
from dynamo.deepinfer.request import prompt_bytes

pytestmark = [pytest.mark.unit, pytest.mark.gpu_0, pytest.mark.pre_merge]


@pytest.fixture
def fake_env(monkeypatch):
    # Short path: ipc:// socket paths are limited to ~107 bytes.
    workdir = tempfile.mkdtemp(prefix="di-")
    monkeypatch.setenv("DEEPINFER_FAKE_DP_ENDPOINT", f"ipc://{workdir}/dp")
    monkeypatch.setenv("DEEPINFER_FAKE_DP_TIMEOUT_S", "10")
    monkeypatch.setenv("DEEPINFER_PREFILL_ANY_ATTEMPT_INTERVAL", "4")
    yield workdir
    shutil.rmtree(workdir, ignore_errors=True)


def spec(rank: int = 0, dp: int = 1, **kw: Any) -> RankSpec:
    engine_args = {
        "model": "fake",
        "chunked_prefill": True,
        "max_model_len": 2048,
        "data_parallel_size": dp,
        "engine_rank": rank,
        **kw.pop("engine_args", {}),
    }
    return RankSpec(
        rank=rank,
        dp_size=dp,
        engine_args=engine_args,
        backend="fake",
        **kw,
    )


def add(rid: str, n: int = 40, max_tokens: int = 4) -> AddReq:
    return AddReq(
        request_id=rid,
        token_ids=prompt_bytes([2 + (i * 7) % 900 for i in range(n)]),
        sampling={"max_tokens": max_tokens},
    )


class Collector:
    def __init__(self) -> None:
        self.outputs: dict[str, list[Output]] = {}
        self._cv = threading.Condition()

    def __call__(self, merged: dict[str, Output]) -> None:
        with self._cv:
            for rid, out in merged.items():
                self.outputs.setdefault(rid, []).append(out)
            self._cv.notify_all()

    def finished(self, rid: str) -> Optional[str]:
        outs = self.outputs.get(rid) or []
        return outs[-1].finish_reason if outs else None

    def tokens(self, rid: str) -> list[int]:
        return [t for out in self.outputs.get(rid, []) for t in out.token_ids]

    def wait(self, predicate, timeout: float = 10.0) -> None:
        with self._cv:
            assert self._cv.wait_for(predicate, timeout), self.outputs

    def wait_finished(self, *rids: str, timeout: float = 10.0) -> None:
        self.wait(lambda: all(self.finished(r) for r in rids), timeout)


class Recording(RankHandle):
    """Records every Cmd a rank receives and every Reply it sends."""

    def __init__(self, inner: RankHandle) -> None:
        self.inner = inner
        self.rank = inner.rank
        self.inline = inner.inline
        self.cmds: list[Cmd] = []
        self.replies: list[Reply] = []

    def wait_ready(self, timeout=None):
        return self.inner.wait_ready(timeout)

    def send(self, cmd: Cmd) -> None:
        self.cmds.append(cmd)
        self.inner.send(cmd)

    def recv(self, timeout=None) -> Reply:
        reply = self.inner.recv(timeout)
        self.replies.append(reply)
        return reply

    def close(self, timeout: float = 30.0) -> None:
        self.inner.close(timeout)


def controller_for(
    handles: list[RankHandle], **kw: Any
) -> tuple[Controller, list[str]]:
    fatal: list[str] = []
    meta = handles[0].wait_ready(30)
    return Controller(handles, meta, on_fatal=fatal.append, **kw), fatal


def test_single_rank_streams_requests_to_completion(fake_env):
    runner = RankRunner(spec())
    controller, fatal = controller_for([LocalRankHandle(runner)])
    out = Collector()
    controller.start(out)
    for i in range(3):
        controller.submit(add(f"r{i}", n=150 + i, max_tokens=3 + i))
    out.wait_finished("r0", "r1", "r2")
    for i in range(3):
        assert len(out.tokens(f"r{i}")) == 3 + i
        assert out.finished(f"r{i}") == "length"
        assert controller.owner(f"r{i}") is None
    controller.close()
    controller.close()  # idempotent
    assert runner.engine.closed and not fatal
    assert runner.cache.stats().used_blocks == 0


def test_threaded_ranks_route_and_agree_on_the_flags(fake_env):
    handles = [
        Recording(LocalRankHandle(RankRunner(spec(r, dp=2)), threaded=True))
        for r in range(2)
    ]
    controller, fatal = controller_for(handles)
    out = Collector()
    # Queue the first batch before starting so it lands in one iteration.
    controller.submit(add("forced", max_tokens=6), dp_rank=1)
    controller.submit(add("free0"))
    controller.submit(add("free1"))
    controller.start(out)
    out.wait_finished("forced", "free0", "free1")
    controller.submit(add("late", max_tokens=2))
    out.wait_finished("late")
    controller.close()
    assert not fatal

    first = {h.rank: h.cmds[0] for h in handles}
    # Least loaded, ties to the lower rank: forced -> 1, then 0, then 0 (1 vs 1).
    assert [a.request_id for a in first[1].adds] == ["forced"]
    assert [a.request_id for a in first[0].adds] == ["free0", "free1"]
    seqs = [[c.seq for c in h.cmds] for h in handles]
    assert seqs[0] == seqs[1] == list(range(len(seqs[0])))
    assert [c.sync for c in handles[0].cmds] == [c.sync for c in handles[1].cmds]
    # Adds make the next step an ANY vote point on every rank.
    assert first[0].sync.new_prompts and first[0].sync.unfinished
    assert handles[0].cmds[-1].op == handles[1].cmds[-1].op == Op.CLOSE
    # The same prompt yields the same tokens on any rank.
    assert out.tokens("free0") == out.tokens("free1")


def test_ranks_step_and_close_on_the_controllers_flags(fake_env):
    """Ranks take every iteration's flags from the Cmd instead of all-reducing
    has_unfinished: an idle rank steps with a busy one, both reach the same
    collectives, and both close in the CLOSE iteration."""
    runners = [RankRunner(spec(r, dp=2)) for r in range(2)]
    handles = [Recording(LocalRankHandle(r, threaded=True)) for r in runners]
    controller, fatal = controller_for(handles)
    out = Collector()
    controller.start(out)
    controller.submit(add("on1", max_tokens=5), dp_rank=1)
    out.wait_finished("on1")
    assert controller.owner("on1") is None  # Reply.finished
    controller.close()
    assert not fatal

    calls = [r.engine.collectives for r in runners]
    assert calls[0] == calls[1]
    assert calls[0]["has_unfinished"] == calls[0]["sync"] == 0
    assert calls[0]["lockstep"] == 1 and calls[0]["vote_any"] >= 1
    syncs = [[c.sync for c in h.cmds] for h in handles]
    assert syncs[0] == syncs[1]
    assert [s.new_prompts for s in syncs[0]].count(True) == 1
    assert [s.close for s in syncs[0]] == [False] * (len(syncs[0]) - 1) + [True]
    replies = [h.replies for h in handles]
    assert [r.u for r in replies[0]] == [r.u for r in replies[1]]
    assert any(r.u for r in replies[0])  # rank 0 stepped with no request
    for rank in replies:
        assert [r.closed for r in rank] == [False] * (len(rank) - 1) + [True]
    assert all(r.engine.closed for r in runners)


@pytest.mark.timeout(60)
def test_child_process_rank_runs_in_lockstep(fake_env):
    child = ZmqRankHandle.spawn(spec(1, dp=2), fake_env)
    local = LocalRankHandle(RankRunner(spec(0, dp=2)))
    controller, fatal = controller_for([local, child])
    assert child.wait_ready(30).num_gpu_blocks == local.wait_ready().num_gpu_blocks
    out = Collector()
    controller.start(out)
    controller.submit(add("on_child", max_tokens=5), dp_rank=1)
    controller.submit(add("on_local", max_tokens=5), dp_rank=0)
    out.wait_finished("on_child", "on_local", timeout=20)
    controller.close(timeout=20)
    assert not fatal
    assert out.tokens("on_child") == out.tokens("on_local")
    assert child.proc.returncode == 0


@pytest.mark.timeout(60)
def test_dead_rank_is_fatal(fake_env):
    child = ZmqRankHandle.spawn(spec(0), fake_env)
    controller, fatal = controller_for([child], idle_heartbeat_s=0.05)
    controller.start(Collector())
    child.proc.kill()
    deadline = time.monotonic() + 10
    while not fatal and time.monotonic() < deadline:
        time.sleep(0.02)
    assert fatal and "exited" in fatal[0]
    controller.close(timeout=5)


def test_failed_child_rank_reports_and_exits_at_once(fake_env):
    child = ZmqRankHandle.spawn(spec(0, cache_manager="bogus"), fake_env)
    # A child that never gets ready does not delay noticing the failed one.
    pending = _CannedHandle(1, u=False, ready=False)
    started = time.monotonic()
    with pytest.raises(RankFailed, match="unknown kv cache manager"):
        wait_all_ready([pending, child], timeout=30)
    assert time.monotonic() - started < 20
    # os._exit: no interpreter teardown, so no DeepInfer close() barrier.
    assert child.proc.wait(10) == 1
    child.close(0)


@pytest.mark.timeout(60)
def test_child_watch_exits_when_a_child_dies_during_startup(fake_env):
    from dynamo.deepinfer.engine import ChildWatch

    child = ZmqRankHandle.spawn(spec(0, cache_manager="bogus"), fake_env)
    fatal: list[str] = []
    with ChildWatch([child], on_fatal=fatal.append, interval_s=0.05):
        # Stands in for rank 0's constructor, blocked in a collective.
        deadline = time.monotonic() + 30
        while not fatal and time.monotonic() < deadline:
            time.sleep(0.05)
    assert fatal and "exited with code 1" in fatal[0]
    child.close(0)


@pytest.mark.timeout(60)
def test_child_rank_ignores_group_sigterm(fake_env):
    child = ZmqRankHandle.spawn(spec(0), fake_env)
    controller, fatal = controller_for([child])
    child.proc.send_signal(signal.SIGTERM)
    child.proc.send_signal(signal.SIGINT)
    time.sleep(0.3)
    assert child.proc.poll() is None
    controller.close(timeout=20)  # the coordinated CLOSE still works
    assert child.proc.returncode == 0 and not fatal


class _CannedHandle(RankHandle):
    def __init__(self, rank: int, u: bool, ready: bool = True) -> None:
        self.rank, self.u, self.ready = rank, u, ready
        self._seq = 0

    def poll_ready(self):
        return self.wait_ready() if self.ready else None

    def wait_ready(self, timeout=None):
        return EngineMeta(
            block_size=128,
            num_gpu_blocks=8,
            max_num_seqs=4,
            max_model_len=256,
            max_num_batched_tokens=256,
            prefill_any_attempt_interval=100,
        )

    def send(self, cmd):
        self._seq = cmd.seq

    def recv(self, timeout=None):
        return Reply(seq=self._seq, rank=self.rank, u=self.u)

    def close(self, timeout=30.0):
        pass


def test_ranks_disagreeing_on_unfinished_is_fatal():
    controller, fatal = controller_for(
        [_CannedHandle(0, u=True), _CannedHandle(1, u=False)], idle_heartbeat_s=0.01
    )
    controller.start(Collector())
    deadline = time.monotonic() + 5
    while not fatal and time.monotonic() < deadline:
        time.sleep(0.01)
    assert fatal and "disagree" in fatal[0]
    controller.close()


def test_close_before_start_closes_every_rank(fake_env):
    runners = [RankRunner(spec(r, dp=2)) for r in range(2)]
    handles = [LocalRankHandle(runners[0]), LocalRankHandle(runners[1], threaded=True)]
    controller, fatal = controller_for(handles)
    controller.close()
    controller.close()
    assert all(r.engine.closed for r in runners) and not fatal


@pytest.mark.timeout(60)
@pytest.mark.parametrize("started", [False, True], ids=["at-exit", "cleanup"])
def test_remote_rank_lost_during_close_fails_the_close(started):
    """Once CLOSE went out, a dropped connection no longer counts as a lost
    rank; the close timeout keeps a rank that dies before it replies from
    hanging the head (at interpreter exit nothing else bounds it)."""
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    server = ControlServer(port, [1], timeout_s=5)
    ctx = zmq.Context()
    dealer = ctx.socket(zmq.DEALER)
    dealer.setsockopt(zmq.ROUTING_ID, routing_id(1))
    dealer.setsockopt(zmq.LINGER, 0)
    try:
        dealer.connect(f"tcp://127.0.0.1:{port}")
        canned = _CannedHandle(0, u=False)
        dealer.send(encode(Ready(rank=1, meta=canned.wait_ready())))
        remote = server.handle(1)
        remote.wait_ready(10)
        controller, fatal = controller_for([canned, remote])

        def killed_before_its_reply() -> None:
            while (cmd := decode_cmd(dealer.recv())).op != Op.CLOSE:
                dealer.send(encode(Reply(seq=cmd.seq, rank=1)))
            dealer.close(linger=0)

        rank = threading.Thread(target=killed_before_its_reply)
        rank.start()
        if started:
            controller.start(Collector())
        began = time.monotonic()
        controller.close(timeout=2)
        assert time.monotonic() - began < 10
        rank.join(10)
        assert len(fatal) == 1 and fatal[0].startswith(
            ("rank 1: no reply within 2s", "CLOSE iteration did not finish")
        ), fatal
    finally:
        server.close()
        ctx.destroy(linger=0)


def test_queued_prompts_above_the_prefill_watermark_all_finish(fake_env, monkeypatch):
    # Two 3700-token prompts take 29 blocks each: 59 of 64 blocks (92%). Held
    # while waiting, they would keep DeepInfer's prefill gate (<= 90%) shut.
    monkeypatch.setenv("DEEPINFER_FAKE_NUM_GPU_BLOCKS", "64")
    runner = RankRunner(spec(engine_args={"max_model_len": 4096}))
    adds = [add("a", n=3700), add("b", n=3700)]
    seen = _run(
        runner,
        [Cmd(seq=0, adds=adds)],
        lambda seen: _done("a")(seen) and _done("b")(seen),
    )
    for rid in ("a", "b"):
        assert seen[rid][-1].finish_reason == "length"
        assert len([t for o in seen[rid] for t in o.token_ids]) == 4
    assert runner.cache.stats().used_blocks == 0
    runner.apply(Cmd(seq=10_000, op=Op.CLOSE))


def test_new_prompts_prefill_while_another_rank_is_above_the_watermark(
    fake_env, monkeypatch
):
    monkeypatch.setenv("DEEPINFER_FAKE_NUM_GPU_BLOCKS", "20")
    monkeypatch.setenv("DEEPINFER_PREFILL_ATTEMPT_INTERVAL", "8")
    handles = [
        LocalRankHandle(
            RankRunner(spec(r, dp=2, engine_args={"max_model_len": 4096})),
            threaded=True,
        )
        for r in range(2)
    ]
    controller, fatal = controller_for(handles)
    out = Collector()
    controller._emit = out  # iterations are driven by hand below
    # Rank 0 decodes a request holding 18 of 20 blocks (95%): it cannot prefill.
    controller.submit(add("big", n=2200, max_tokens=60), dp_rank=0)
    for _ in range(20):
        controller._iterate(0.0)
        if out.tokens("big"):
            break
    assert out.tokens("big")
    first_token_at = None
    for it in range(30):
        # New prompts reach both ranks every iteration.
        controller.submit(add(f"r0-{it}", n=100, max_tokens=2), dp_rank=0)
        controller.submit(add(f"r1-{it}", n=100, max_tokens=2), dp_rank=1)
        controller._iterate(0.0)
        if out.tokens("r1-0"):
            first_token_at = it
            break
    # An ALL vote every iteration would wait for "big" to finish (~60 steps).
    assert first_token_at is not None and first_token_at <= 2
    controller.close()
    assert not fatal


def test_abort_of_running_request_frees_its_kv(fake_env, monkeypatch):
    monkeypatch.setenv("DEEPINFER_FAKE_STEP_DELAY_S", "0.005")
    runner = RankRunner(spec())
    controller, fatal = controller_for([LocalRankHandle(runner)])
    out = Collector()
    controller.start(out)
    controller.submit(add("victim", max_tokens=10_000))
    out.wait(lambda: len(out.tokens("victim")) >= 2)
    controller.abort("victim")
    controller.submit(add("after", max_tokens=2))
    out.wait_finished("after")
    stopped_at = len(out.tokens("victim"))
    time.sleep(0.05)
    assert len(out.tokens("victim")) == stopped_at  # a running abort emits nothing
    assert out.finished("victim") is None
    assert runner.cache.stats().used_blocks == 0
    controller.close()
    assert not fatal


def test_abort_of_partially_prefilled_request_frees_its_kv(fake_env):
    runner = RankRunner(spec(engine_args={"max_num_batched_tokens": 128}))
    reply = runner.apply(Cmd(seq=0, adds=[add("long", n=600)]))
    seq = runner.engine.scheduler.waiting[0]
    assert seq.num_cached_tokens == 128 and runner.cache.stats().used_blocks > 0
    runner.apply(Cmd(seq=1, aborts=["long"]))
    reply = runner.apply(Cmd(seq=2))
    assert runner.cache.stats().used_blocks == 0
    assert not reply.u
    runner.apply(Cmd(seq=3, op=Op.CLOSE))


def _run(
    runner: RankRunner, cmds: list[Cmd], until, limit: int = 200
) -> dict[str, list[Output]]:
    """Apply ``cmds`` then empty iterations until ``until(outputs)`` holds."""
    seen: dict[str, list[Output]] = {}
    for i in range(limit):
        cmd = cmds[i] if i < len(cmds) else Cmd(seq=i)
        for out in runner.apply(cmd).outputs:
            seen.setdefault(out.request_id, []).append(out)
        if until(seen):
            break
    return seen


def _done(rid):
    return lambda seen: any(o.finish_reason for o in seen.get(rid, []))
