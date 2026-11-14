# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""DeepInfer engines that span nodes, over the CPU fake on localhost.

The head runs in this process (rank 0 inline, its other local ranks as
children over ipc://); every other node is a real ``python -m
dynamo.deepinfer.node`` subprocess whose ranks reach the head's controller
over TCP. The fake's DP collectives rendezvous at MASTER_ADDR:MASTER_PORT-49,
so a rank that got the wrong MASTER_* never joins them.
"""

import asyncio
import json
import os
import random
import signal
import socket
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import msgspec
import pytest
import zmq

import dynamo.deepinfer
from dynamo.common.backend.publisher import ZmqSource
from dynamo.deepinfer.args import apply_env_defaults, parse_args
from dynamo.deepinfer.controller import Controller
from dynamo.deepinfer.controller_process import ControllerProcess
from dynamo.deepinfer.engine import DeepInferEngine, _shutdown_ranks, launch_ranks
from dynamo.deepinfer.node import parse_node_args
from dynamo.deepinfer.options import DeepInferOptions
from dynamo.deepinfer.protocol import ControlServer, RankFailed, routing_id

pytestmark = [pytest.mark.unit, pytest.mark.gpu_0, pytest.mark.pre_merge]

SRC = str(Path(dynamo.deepinfer.__file__).resolve().parents[2])


@pytest.fixture
def mn_env(monkeypatch):
    # parse_args and the launcher write os.environ; keep it per test.
    monkeypatch.setattr(os, "environ", dict(os.environ))
    for key in (
        "DEEPINFER_FAKE_DP_ENDPOINT",
        "CUDA_VISIBLE_DEVICES",
        "MASTER_ADDR",
        "MASTER_PORT",
    ):
        os.environ.pop(key, None)
    os.environ["DEEPINFER_FAKE_DP_TIMEOUT_S"] = "10"
    os.environ["DEEPINFER_PREFILL_ANY_ATTEMPT_INTERVAL"] = "4"
    os.environ["PYTHONPATH"] = os.pathsep.join(
        p for p in (SRC, os.environ.get("PYTHONPATH")) if p
    )
    nodes: list[subprocess.Popen] = []
    yield nodes
    for proc in nodes:
        if proc.poll() is None:
            proc.kill()
            proc.wait()


def _free_block(n: int) -> int:
    """First of ``n`` consecutive TCP ports that are free on every interface."""
    for _ in range(200):
        base = random.randint(20000, 50000)
        socks: list[socket.socket] = []
        try:
            for port in range(base, base + n):
                sock = socket.socket()
                socks.append(sock)
                sock.bind(("", port))
            return base
        except OSError:
            continue
        finally:
            for sock in socks:
                sock.close()
    raise RuntimeError("no free port block")


@dataclass
class Topology:
    """One engine: ``dp`` ranks over ``nnodes`` nodes, all on localhost, with
    its own ports (DeepInfer's store and the fake's DP rendezvous, the
    control port, and each node's KV event block)."""

    dp: int = 4
    nnodes: int = 2

    def __post_init__(self) -> None:
        self.per_node = self.dp // self.nnodes
        self.base = _free_block(3 + self.nnodes * self.per_node)

    @property
    def master_port(self) -> int:
        return self.base + 50  # store at base, the fake's rendezvous at base + 1

    @property
    def control_port(self) -> int:
        return self.base + 2

    def kv_port(self, node: int) -> int:
        return self.base + 3 + node * self.per_node

    def argv(self, node: int, *extra: str) -> list[str]:
        return [
            "--model",
            "fake-model",
            "--deepinfer-backend",
            "fake",
            "--namespace",
            "di-multinode",
            "--dp-size",
            str(self.dp),
            "--nnodes",
            str(self.nnodes),
            "--node-rank",
            str(node),
            "--master-addr",
            "127.0.0.1",
            "--master-port",
            str(self.master_port),
            "--control-port",
            str(self.control_port),
            "--kv-events-port-base",
            str(self.kv_port(node)),
            "--max-model-len",
            "2048",
            "--node-timeout-s",
            "5",
            "--startup-timeout-s",
            "60",
            *extra,
        ]


def start_node(
    nodes: list,
    topo: Topology,
    node: int,
    gpus: str,
    *extra: str,
    env: Optional[dict[str, str]] = None,
) -> subprocess.Popen:
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": gpus, **(env or {})}
    proc = subprocess.Popen(
        [sys.executable, "-m", "dynamo.deepinfer.node", *topo.argv(node, *extra)],
        env=env,
    )
    nodes.append(proc)
    return proc


def start_head(
    topo: Topology, gpus: str, *extra: str, fatal: Optional[list[str]] = None
) -> DeepInferEngine:
    """``DeepInferEngine.from_args`` with ``on_fatal`` recorded instead of
    ``os._exit``."""
    on_fatal = (fatal if fatal is not None else []).append
    os.environ["CUDA_VISIBLE_DEVICES"] = gpus
    args = parse_args(topo.argv(0, *extra))
    apply_env_defaults(args)
    launch = launch_ranks(args, on_fatal=on_fatal)
    controller = Controller(
        launch.handles, launch.meta, engine_type=args.engine_type, on_fatal=on_fatal
    )
    return DeepInferEngine(args, controller, launch.meta, launch)


class Ctx:
    def __init__(self) -> None:
        self._id = uuid.uuid4().hex

    def id(self) -> str:
        return self._id

    def is_stopped(self) -> bool:
        return False


def request(
    rank: int, n: int = 40, max_tokens: int = 3, salt: int = 0
) -> dict[str, Any]:
    return {
        "token_ids": [2 + (i * 7 + salt) % 900 for i in range(n)],
        "stop_conditions": {"max_tokens": max_tokens},
        "routing": {"dp_rank": rank},
    }


async def tokens_of(engine: DeepInferEngine, req: dict) -> list[int]:
    chunks = [c async for c in engine.generate(req, Ctx())]
    assert chunks[-1]["finish_reason"] == "length"
    return [t for c in chunks for t in c["token_ids"]]


def _children(pid: int) -> list[int]:
    kids = []
    for entry in os.listdir("/proc"):
        if not entry.isdigit():
            continue
        try:
            with open(f"/proc/{entry}/stat") as f:
                stat = f.read()
        except OSError:
            continue
        if int(stat.rsplit(")", 1)[1].split()[1]) == pid:
            kids.append(int(entry))
    return kids


def rank_processes(parent: int) -> list[tuple[int, int, dict[str, str]]]:
    """(rank, pid, environment) of each rank_worker child of ``parent``."""
    found = []
    for pid in _children(parent):
        try:
            with open(f"/proc/{pid}/cmdline", "rb") as f:
                argv = f.read().split(b"\0")
            with open(f"/proc/{pid}/environ", "rb") as f:
                items = f.read().decode(errors="replace").split("\0")
        except OSError:
            continue
        if b"dynamo.deepinfer.rank_worker" not in argv:
            continue
        spec = json.loads(argv[argv.index(b"--spec") + 1])
        env = dict(i.split("=", 1) for i in items if "=" in i)
        found.append((spec["rank"], pid, env))
    return sorted(found, key=lambda item: item[0])


def wait_for(predicate, timeout: float = 20.0, what: str = "condition") -> None:
    deadline = time.monotonic() + timeout
    while not predicate():
        assert time.monotonic() < deadline, f"timed out waiting for {what}"
        time.sleep(0.05)


@pytest.mark.timeout(120)
async def test_two_node_engine_serves_aborts_and_closes(mn_env):
    os.environ["DEEPINFER_FAKE_STEP_DELAY_S"] = "0.005"
    topo = Topology(dp=4, nnodes=2)
    node = start_node(mn_env, topo, 1, "2,3")
    fatal: list[str] = []
    engine = start_head(topo, "0,1", fatal=fatal)
    try:
        config = await engine.start(worker_id=1)
        # One Dynamo worker for the whole engine.
        llm = config.llm
        assert (llm.data_parallel_size, llm.data_parallel_start_rank) == (4, 0)
        assert engine.component_metrics_dp_ranks() == [0, 1, 2, 3]

        # Every rank serves, and lockstep makes the same prompt give the same
        # tokens wherever it runs.
        results = await asyncio.gather(
            *(tokens_of(engine, request(r)) for r in range(4))
        )
        assert all(r == results[0] for r in results) and len(results[0]) == 3

        # Each rank got one GPU: its position in its node's CUDA_VISIBLE_DEVICES.
        assert os.environ["CUDA_VISIBLE_DEVICES"] == "0"  # rank 0, in this process
        ranks = rank_processes(os.getpid()) + rank_processes(node.pid)
        gpus = [(r, env["CUDA_VISIBLE_DEVICES"]) for r, _, env in ranks]
        assert gpus == [(1, "1"), (2, "2"), (3, "3")]
        for _, _, env in ranks:
            assert env["MASTER_ADDR"] == "127.0.0.1"
            assert env["MASTER_PORT"] == str(topo.master_port)

        # Abort a request running on the other node.
        ctx = Ctx()
        stream = engine.generate(request(3, max_tokens=100_000), ctx)
        first = await stream.__anext__()
        assert first["token_ids"]
        started = time.monotonic()
        await engine.abort(ctx)
        rest = [c async for c in stream]
        assert rest[-1]["finish_reason"] == "cancelled"
        assert time.monotonic() - started < 2
        assert len(await tokens_of(engine, request(3, salt=1))) == 3
        stats = engine.controller.stats()[3]
        assert stats.num_running == 0 and stats.kv_used_blocks == 0

        # scancel signals the whole job: the node's ranks wait for the head's
        # CLOSE instead of dying mid-iteration.
        node.send_signal(signal.SIGTERM)
        await asyncio.sleep(0.3)
        assert node.poll() is None
        assert len(await tokens_of(engine, request(2, salt=2))) == 3
    finally:
        await engine.cleanup()
        await engine.cleanup()  # idempotent
    assert node.wait(30) == 0
    assert not fatal
    assert not rank_processes(os.getpid())


@pytest.mark.timeout(120)
async def test_controller_process_drives_every_node(mn_env):
    """--no-rank0-in-process: the controller runs in a process of its own,
    with this node's ranks as its children and the other nodes' ranks on its
    --control-port; this process only serves requests."""
    os.environ["DEEPINFER_FAKE_STEP_DELAY_S"] = "0.005"
    topo = Topology(dp=4, nnodes=2)
    node = start_node(mn_env, topo, 1, "2,3", "--kv-cache-manager", "prefix")
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
    engine, _ = await DeepInferEngine.from_args(
        topo.argv(0, "--no-rank0-in-process", "--kv-cache-manager", "prefix")
    )
    controller = engine.controller
    assert isinstance(controller, ControllerProcess)
    try:
        await engine.start(worker_id=1)
        results = await asyncio.gather(
            *(tokens_of(engine, request(r)) for r in range(4))
        )
        assert all(r == results[0] for r in results) and len(results[0]) == 3
        ranks = rank_processes(controller.proc.pid) + rank_processes(node.pid)
        gpus = [(r, env["CUDA_VISIBLE_DEVICES"]) for r, _, env in ranks]
        assert gpus == [(0, "0"), (1, "1"), (2, "2"), (3, "3")]
        assert not rank_processes(os.getpid())
        sources = [s.endpoint for s in await engine.kv_event_sources()]
        node_kv = topo.kv_port(1)
        assert sources[2:] == [f"tcp://127.0.0.1:{p}" for p in (node_kv, node_kv + 1)]

        # Abort a request running on the other node.
        ctx = Ctx()
        stream = engine.generate(request(3, max_tokens=100_000), ctx)
        await stream.__anext__()
        await engine.abort(ctx)
        assert [c async for c in stream][-1]["finish_reason"] == "cancelled"
        # The abort reached the rank (stats come once a second).
        wait_for(lambda: controller.stats()[3].num_running == 0, what="the abort")
    finally:
        await engine.cleanup()
    assert controller.proc.returncode == 0  # a coordinated CLOSE
    assert node.wait(30) == 0


@pytest.mark.timeout(120)
async def test_kv_event_sources_point_at_each_node(mn_env):
    topo = Topology(dp=4, nnodes=2)
    start_node(mn_env, topo, 1, "0,1", "--kv-cache-manager", "prefix")
    engine = start_head(topo, "0,1", "--kv-cache-manager", "prefix")
    sub = zmq.Context.instance().socket(zmq.SUB)
    sub.setsockopt(zmq.SUBSCRIBE, b"")
    sub.setsockopt(zmq.LINGER, 0)
    try:
        await engine.start(worker_id=1)
        head_kv, node_kv = topo.kv_port(0), topo.kv_port(1)
        # Ranks on the head through loopback; the node's ranks at the address
        # they advertised (here the route to 127.0.0.1) and their own port block.
        assert await engine.kv_event_sources() == [
            ZmqSource(endpoint=f"tcp://127.0.0.1:{head_kv}", topic="", dp_rank=0),
            ZmqSource(endpoint=f"tcp://127.0.0.1:{head_kv + 1}", topic="", dp_rank=1),
            ZmqSource(endpoint=f"tcp://127.0.0.1:{node_kv}", topic="", dp_rank=2),
            ZmqSource(endpoint=f"tcp://127.0.0.1:{node_kv + 1}", topic="", dp_rank=3),
        ]
        # The remote endpoint really carries rank 3's events.
        sub.connect(f"tcp://127.0.0.1:{node_kv + 1}")
        for salt in range(20):  # PUB drops batches until the subscription is live
            await tokens_of(engine, request(3, n=600, max_tokens=1, salt=salt))
            if sub.poll(200):
                break
        frames = sub.recv_multipart()
        assert len(frames) == 3
        _, events, dp_rank = msgspec.msgpack.decode(frames[2])
        assert dp_rank == 3 and events
    finally:
        sub.close()
        await engine.cleanup()
    assert mn_env[0].wait(30) == 0


@pytest.mark.timeout(120)
async def test_lost_remote_rank_is_fatal_on_both_sides(mn_env):
    topo = Topology(dp=4, nnodes=2)
    node = start_node(mn_env, topo, 1, "0,1")
    fatal: list[str] = []
    engine = start_head(topo, "0,1", fatal=fatal)
    try:
        await engine.start(worker_id=1)
        assert len(await tokens_of(engine, request(2))) == 3
        [pid] = [pid for r, pid, _ in rank_processes(node.pid) if r == 3]
        os.kill(pid, signal.SIGKILL)
        # The node stops its other rank and exits non-zero for the scheduler.
        assert node.wait(20) == 1
        # The head's watch fires even while no request runs (an inline step
        # blocked in a collective could not notice).
        wait_for(lambda: fatal, what="the head to declare the node lost")
        assert "lost its connection" in fatal[0]
        with pytest.raises(RankFailed):
            engine._launch.server.recv(2, 0.1)
    finally:
        await asyncio.to_thread(engine.controller.close, 5.0)
        engine._launch.close()


@pytest.mark.timeout(120)
async def test_dead_child_of_a_single_node_engine_is_fatal(mn_env):
    """A child killed by a signal sends no error reply; the head notices at
    once, though rank 0's inline step may wait for it in a collective."""
    os.environ["DEEPINFER_FAKE_DP_TIMEOUT_S"] = "30"  # the collective's own timeout
    topo = Topology(dp=2, nnodes=1)
    fatal: list[str] = []
    engine = start_head(topo, "0,1", fatal=fatal)
    try:
        await engine.start(worker_id=1)
        assert len(await tokens_of(engine, request(1))) == 3
        [(rank, pid, _)] = rank_processes(os.getpid())
        # Stopped, the child misses the next heartbeat iteration, so the head
        # waits for it when it dies.
        os.kill(pid, signal.SIGSTOP)
        await asyncio.sleep(1.5)
        assert not fatal
        os.kill(pid, signal.SIGKILL)
        killed = time.monotonic()
        wait_for(lambda: fatal, what="the head to notice its dead child")
        assert time.monotonic() - killed < 10
        assert fatal[0] == f"rank {rank} exited with code {-signal.SIGKILL}"
    finally:
        await asyncio.to_thread(engine.controller.close, 5.0)
        engine._launch.close()


@pytest.mark.timeout(120)
def test_lost_head_stops_the_remote_node(mn_env):
    topo = Topology(dp=2, nnodes=2)
    node = start_node(mn_env, topo, 1, "0")
    fatal: list[str] = []
    engine = start_head(topo, "0", fatal=fatal)
    launch = engine._launch
    # The head goes away without a CLOSE iteration (here: its control socket).
    launch.watch.stop()
    launch.server.close()
    assert node.wait(20) == 1
    assert not fatal


@pytest.mark.timeout(120)
def test_ranks_on_every_node_close_at_exit_when_start_never_ran(mn_env):
    topo = Topology(dp=4, nnodes=2)
    node = start_node(mn_env, topo, 1, "0,1")
    engine = start_head(topo, "0,1")
    _shutdown_ranks(engine.controller, engine._launch)  # what interpreter exit runs
    assert node.wait(30) == 0  # a coordinated CLOSE, not a kill
    assert not os.path.exists(engine._launch.ipc_dir)
    assert not engine._launch.watch._thread.is_alive()


@pytest.mark.timeout(120)
def test_remote_rank_that_fails_to_start_fails_the_head(mn_env):
    topo = Topology(dp=4, nnodes=2)
    # The fake reads this when it builds the engine: only the node's ranks fail.
    broken = {"DEEPINFER_FAKE_NUM_GPU_BLOCKS": "not-a-number"}
    node = start_node(mn_env, topo, 1, "0,1", env=broken)
    with pytest.raises(RankFailed, match="rank [23] failed to start"):
        start_head(topo, "0,1")
    assert node.wait(20) == 1


@pytest.mark.timeout(120)
def test_node_launched_with_other_options_is_refused(mn_env):
    topo = Topology(dp=2, nnodes=2)
    node = start_node(mn_env, topo, 1, "0", "--max-num-seqs", "8")
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    args = parse_args(topo.argv(0))
    apply_env_defaults(args)
    fatal: list[str] = []
    with pytest.raises(RankFailed, match="other DeepInfer options"):
        launch_ranks(args, on_fatal=fatal.append)
    # The head closed its control port, so the node's rank exits.
    assert node.wait(20) == 1


@pytest.mark.timeout(120)
def test_node_with_other_lockstep_settings_is_refused(mn_env):
    """Each rank derives its prefill vote all-reduces from its own intervals:
    a node whose environment overrides --prefill-attempt-interval would run
    mismatched collectives, so it fails at startup instead."""
    topo = Topology(dp=2, nnodes=2)
    env = {"DEEPINFER_PREFILL_ATTEMPT_INTERVAL": "70"}  # the head sets 8
    node = start_node(mn_env, topo, 1, "0", env=env)
    with pytest.raises(
        RankFailed, match=r"DEEPINFER_PREFILL_ATTEMPT_INTERVAL='70' \(head: '8'\)"
    ):
        start_head(topo, "0")
    assert node.wait(20) == 1


@pytest.mark.timeout(120)
async def test_engine_derived_setting_the_head_never_set_is_not_compared(mn_env):
    """DeepInfer's EngineArgs sets DEEPINFER_ZERO_OVERHEAD_SCHEDULING on every
    rank from the engine type; a head that runs no rank never has it."""
    topo = Topology(dp=2, nnodes=2)
    env = {"DEEPINFER_ZERO_OVERHEAD_SCHEDULING": "1"}
    start_node(mn_env, topo, 1, "0", env=env)
    engine = start_head(topo, "0")
    await engine.cleanup()
    assert mn_env[0].wait(30) == 0


@pytest.mark.timeout(120)
def test_engine_derived_setting_the_head_set_is_compared(mn_env, monkeypatch):
    topo = Topology(dp=2, nnodes=2)
    node = start_node(
        mn_env, topo, 1, "0", env={"DEEPINFER_ZERO_OVERHEAD_SCHEDULING": "1"}
    )
    monkeypatch.setenv("DEEPINFER_ZERO_OVERHEAD_SCHEDULING", "0")
    with pytest.raises(
        RankFailed, match=r"DEEPINFER_ZERO_OVERHEAD_SCHEDULING='1' \(head: '0'\)"
    ):
        start_head(topo, "0")
    assert node.wait(20) == 1


@pytest.mark.timeout(120)
async def test_node_with_another_deepinfer_environment_is_reported(mn_env, caplog):
    topo = Topology(dp=2, nnodes=2)
    env = {"DEEPINFER_LOGGING_INTERVAL_STEP": "7"}  # not a lockstep setting
    start_node(mn_env, topo, 1, "0", env=env)
    engine = start_head(topo, "0")
    await engine.cleanup()
    assert mn_env[0].wait(30) == 0
    [warning] = [r for r in caplog.records if "another environment" in r.message]
    assert "DEEPINFER_LOGGING_INTERVAL_STEP='7' (head: None)" in warning.message


@pytest.mark.timeout(60)
def test_node_launcher_kills_its_ranks_on_sigterm(mn_env):
    topo = Topology(dp=4, nnodes=2)  # no head: the ranks wait for it
    node = start_node(mn_env, topo, 1, "0,1", "--shutdown-grace-s", "0.5")
    wait_for(lambda: len(rank_processes(node.pid)) == 2, what="the node's ranks")
    pids = [pid for _, pid, _ in rank_processes(node.pid)]
    node.send_signal(signal.SIGTERM)
    assert node.wait(10) == 128 + signal.SIGTERM
    for pid in pids:
        assert not os.path.exists(f"/proc/{pid}") or _is_zombie(pid)


def _is_zombie(pid: int) -> bool:
    try:
        with open(f"/proc/{pid}/stat") as f:
            return f.read().rsplit(")", 1)[1].split()[0] == "Z"
    except OSError:
        return True


@pytest.mark.timeout(180)
async def test_two_engines_on_one_host_do_not_collide(mn_env):
    """Like two DEP2 prefill workers sharing a node: each engine has its own
    MASTER_PORT, control port, KV event ports and GPUs."""
    engines: list[DeepInferEngine] = []
    nodes = []
    topos = [Topology(dp=4, nnodes=2), Topology(dp=4, nnodes=2)]
    gpus = ["0,1", "2,3"]
    for topo, g in zip(topos, gpus):
        nodes.append(start_node(mn_env, topo, 1, g, "--kv-cache-manager", "prefix"))
    fatal: list[str] = []
    try:
        for topo, g in zip(topos, gpus):
            engine = start_head(topo, g, "--kv-cache-manager", "prefix", fatal=fatal)
            engines.append(engine)
            await engine.start(worker_id=len(engines))
        results = await asyncio.gather(
            *(
                tokens_of(engine, request(r, max_tokens=4, salt=salt))
                for engine in engines
                for r in range(4)
                for salt in (0, 1)
            )
        )
        # The same prompt gives the same tokens on either engine and any rank.
        assert results[0::2] == [results[0]] * 8 and results[1::2] == [results[1]] * 8
        assert results[0] != results[1]
        # Each rank of each engine: (MASTER_PORT, rank, GPU). Rank 1 is the
        # head's child; ranks 2 and 3 run on the engine's other "node".
        placed = {
            (env["MASTER_PORT"], r, env["CUDA_VISIBLE_DEVICES"])
            for parent in (os.getpid(), *(node.pid for node in nodes))
            for r, _, env in rank_processes(parent)
        }
        a, b = (str(t.master_port) for t in topos)
        assert placed == {
            (a, 1, "1"),
            (a, 2, "0"),
            (a, 3, "1"),
            (b, 1, "3"),
            (b, 2, "2"),
            (b, 3, "3"),
        }
    finally:
        for engine in engines:
            await engine.cleanup()
    for node in nodes:
        assert node.wait(30) == 0
    assert not fatal


@pytest.mark.timeout(120)
async def test_two_single_node_engines_share_a_node(mn_env):
    """Two DEP2 prefill-style workers on GPUs 0,1 and 2,3 of one node."""
    topos = [Topology(dp=2, nnodes=1), Topology(dp=2, nnodes=1)]
    engines = []
    try:
        for topo, g in zip(topos, ["0,1", "2,3"]):
            engines.append(start_head(topo, g, "--kv-cache-manager", "prefix"))
            await engines[-1].start(worker_id=len(engines))
        results = await asyncio.gather(
            *(tokens_of(e, request(r)) for e in engines for r in range(2))
        )
        assert results == [results[0]] * 4
        placed = {
            (env["MASTER_PORT"], r, env["CUDA_VISIBLE_DEVICES"])
            for r, _, env in rank_processes(os.getpid())
        }
        a, b = (str(t.master_port) for t in topos)
        assert placed == {(a, 1, "1"), (b, 1, "3")}
        assert [s.endpoint for e in engines for s in await e.kv_event_sources()] == [
            f"tcp://127.0.0.1:{t.kv_port(0) + r}" for t in topos for r in range(2)
        ]
    finally:
        for engine in engines:
            await engine.cleanup()


def test_control_server_ignores_connections_that_are_not_ranks():
    port = _free_block(1)
    server = ControlServer(port, [1], timeout_s=5)
    try:
        # A port scanner or probe: TCP without the ZMTP handshake.
        raw = socket.create_connection(("127.0.0.1", port))
        raw.sendall(b"GET / HTTP/1.0\r\n\r\n")
        raw.close()
        for _ in range(4):
            assert server.watch(0.1) is None
        # A rank that connected and went away is lost.
        ctx = zmq.Context()
        dealer = ctx.socket(zmq.DEALER)
        dealer.setsockopt(zmq.ROUTING_ID, routing_id(1))
        dealer.connect(f"tcp://127.0.0.1:{port}")
        dealer.send(b"\x80")  # not a valid message; only the connection matters
        wait_for(lambda: server._sock.poll(100), what="the rank's message")
        dealer.close(linger=0)
        ctx.term()
        wait_for(lambda: server.watch(0.1) is not None, what="the lost rank")
        assert "lost its connection" in server.lost
    finally:
        server.close()
        server.close()


def test_multinode_options(mn_env):
    args = parse_args(
        [
            "--model",
            "m",
            "--dp-size",
            "8",
            "--nnodes",
            "2",
            "--master-addr",
            "10.0.0.1",
            "--kv-cache-manager",
            "prefix",
            "--kv-events-port-base",
            "7000",
        ]
    )
    assert (args.ranks_per_node, args.local_ranks) == (4, range(0, 4))
    assert (args.master_addr, args.master_port, args.control_port) == (
        "10.0.0.1",
        23333,
        23400,
    )
    assert args.engine_id() == "10.0.0.1-23333"
    rank5 = args.engine_args(5)
    assert rank5["engine_rank"] == 5 and rank5["data_parallel_size"] == 8
    assert rank5["expert_parallel_size"] == 8
    specs = args.rank_specs(args.engine_id())
    assert [s.kv_events_endpoint for s in specs[3:6]] == [
        "tcp://*:7003",
        "tcp://*:7000",
        "tcp://*:7001",
    ]
    env: dict[str, str] = {"MASTER_ADDR": "elsewhere"}
    apply_env_defaults(args, env)
    assert (env["MASTER_ADDR"], env["MASTER_PORT"]) == ("10.0.0.1", "23333")

    # A non-head node derives the same specs from the same command line (plus
    # head-only Dynamo flags, which it ignores), except its KV event ports.
    opts, grace, ignored = parse_node_args(
        [
            "--model",
            "m",
            "--dp-size",
            "8",
            "--nnodes",
            "2",
            "--node-rank",
            "1",
            "--master-addr",
            "10.0.0.1",
            "--kv-cache-manager",
            "prefix",
            "--kv-events-port-base",
            "9000",
            "--namespace",
            "ns",
            "--dyn-reasoning-parser",
            "deepseek_r1",
        ]
    )
    assert isinstance(opts, DeepInferOptions) and grace == 90.0
    assert ignored == ["--namespace", "ns", "--dyn-reasoning-parser", "deepseek_r1"]
    assert opts.local_ranks == range(4, 8)
    node_specs = opts.rank_specs(opts.engine_id(), opts.local_ranks)
    assert [s.digest() for s in node_specs] == [s.digest() for s in specs[4:]]
    assert node_specs[0].kv_events_endpoint == "tcp://*:9000"

    # MASTER_ADDR/MASTER_PORT from the environment when the flags are absent.
    os.environ.update(MASTER_ADDR="10.0.0.2", MASTER_PORT="24000")
    env_args = parse_args(["--model", "m", "--dp-size", "4", "--nnodes", "2"])
    assert (env_args.master_addr, env_args.master_port) == ("10.0.0.2", 24000)


@pytest.mark.parametrize(
    "argv, match",
    [
        (["--dp-size", "6", "--nnodes", "4", "--master-addr", "h"], "multiple"),
        (["--dp-size", "4", "--nnodes", "2"], "--master-addr"),
        (
            [
                "--dp-size",
                "4",
                "--nnodes",
                "2",
                "--node-rank",
                "2",
                "--master-addr",
                "h",
            ],
            "outside",
        ),
        (
            [
                "--dp-size",
                "4",
                "--nnodes",
                "2",
                "--node-rank",
                "1",
                "--master-addr",
                "h",
            ],
            "dynamo.deepinfer.node",
        ),
        (
            [
                *("--dp-size", "4", "--nnodes", "2", "--master-addr", "h"),
                *("--master-port", "23450", "--control-port", "23400"),
            ],
            "collide",
        ),
    ],
)
def test_multinode_options_are_validated(mn_env, argv, match):
    with pytest.raises(ValueError, match=match):
        parse_args(["--model", "m", *argv])


def test_node_launcher_refuses_the_head_rank(mn_env, capsys):
    with pytest.raises(SystemExit):
        parse_node_args(["--model", "m", "--dp-size", "4", "--nnodes", "2"])
    assert "--node-rank" in capsys.readouterr().err
