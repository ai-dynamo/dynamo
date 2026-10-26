# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""DeepInferEngine conformance-style tests over in-process fake ranks."""

import asyncio
import contextlib
import os
import shutil
import tempfile
import time
import uuid
from typing import Any, Optional

import msgspec
import pytest
import zmq

from dynamo.common.backend.publisher import ComponentSnapshot, ZmqSource
from dynamo.common.constants import DisaggregationMode
from dynamo.deepinfer.args import (
    DeepInferArgs,
    apply_env_defaults,
    deepinfer_env,
    parse_args,
)
from dynamo.deepinfer.controller import Controller, Submit
from dynamo.deepinfer.controller_process import ControllerProcess, Outputs, Started
from dynamo.deepinfer.engine import DeepInferEngine
from dynamo.deepinfer.protocol import AddReq, EngineMeta, LocalRankHandle, Output
from dynamo.deepinfer.rank import RankRunner
from dynamo.llm.exceptions import InvalidArgument

pytestmark = [pytest.mark.unit, pytest.mark.gpu_0, pytest.mark.pre_merge]


@pytest.fixture
def fake_env(monkeypatch):
    # parse_args and the launcher write os.environ; keep it per test.
    monkeypatch.setattr(os, "environ", dict(os.environ))
    workdir = tempfile.mkdtemp(prefix="di-")
    os.environ["DEEPINFER_FAKE_DP_ENDPOINT"] = f"ipc://{workdir}/dp"
    os.environ["DEEPINFER_FAKE_DP_TIMEOUT_S"] = "10"
    os.environ["DEEPINFER_PREFILL_ANY_ATTEMPT_INTERVAL"] = "4"
    os.environ.pop("CUDA_VISIBLE_DEVICES", None)
    yield workdir
    shutil.rmtree(workdir, ignore_errors=True)


def make_args(*extra: str, dp: int = 1, max_model_len: int = 2048) -> DeepInferArgs:
    return parse_args(
        [
            "--model",
            "fake-model",
            "--deepinfer-backend",
            "fake",
            "--dp-size",
            str(dp),
            "--max-model-len",
            str(max_model_len),
            *extra,
        ]
    )


def make_engine(*extra: str, dp: int = 1, max_model_len: int = 2048) -> DeepInferEngine:
    """Rank 0 inline (as in production), ranks 1.. on threads instead of processes."""
    args = make_args(*extra, dp=dp, max_model_len=max_model_len)
    specs = args.rank_specs(engine_id="test")
    handles = [LocalRankHandle(RankRunner(specs[0]))] + [
        LocalRankHandle(RankRunner(s), threaded=True) for s in specs[1:]
    ]
    meta = handles[0].wait_ready()
    controller = Controller(handles, meta, engine_type=args.engine_type)
    return DeepInferEngine(args, controller, meta)


@contextlib.asynccontextmanager
async def running(*extra: str, dp: int = 1, **kw: Any):
    engine = make_engine(*extra, dp=dp, **kw)
    await engine.start(worker_id=1)
    try:
        yield engine
    finally:
        await engine.cleanup()


class Ctx:
    def __init__(self, rid: Optional[str] = None) -> None:
        self._id = rid or uuid.uuid4().hex

    def id(self) -> str:
        return self._id

    def is_stopped(self) -> bool:
        return False


def request(n: int = 40, max_tokens: Optional[int] = 4, **kw: Any) -> dict[str, Any]:
    req: dict[str, Any] = {"token_ids": [2 + (i * 7) % 900 for i in range(n)], **kw}
    if max_tokens is not None:
        req.setdefault("stop_conditions", {})["max_tokens"] = max_tokens
    return req


async def collect(
    engine: DeepInferEngine, req: dict, ctx: Optional[Ctx] = None
) -> list[dict]:
    return [c async for c in engine.generate(req, ctx or Ctx())]


def assert_one_terminal(chunks: list[dict]) -> dict:
    assert all(c["index"] == 0 for c in chunks)
    assert [bool(c.get("finish_reason")) for c in chunks] == [False] * (
        len(chunks) - 1
    ) + [True]
    return chunks[-1]


async def test_generate_streams_tokens_with_one_terminal_chunk(fake_env):
    async with running() as engine:
        chunks = await collect(engine, request(n=150, max_tokens=6))
    last = assert_one_terminal(chunks)
    tokens = [t for c in chunks for t in c["token_ids"]]
    assert len(tokens) == 6
    assert last["finish_reason"] == "length"
    assert last["completion_usage"] == {
        "prompt_tokens": 150,
        "completion_tokens": 6,
        "total_tokens": 156,
        "prompt_tokens_details": {"cached_tokens": 0},
    }


async def test_concurrent_requests_are_independent(fake_env):
    async with running(dp=2) as engine:
        results = await asyncio.gather(
            *(collect(engine, request(n=30 + i, max_tokens=3 + i)) for i in range(6))
        )
    for i, chunks in enumerate(results):
        assert_one_terminal(chunks)
        assert sum(len(c["token_ids"]) for c in chunks) == 3 + i


async def test_max_tokens_defaults_to_remaining_context(fake_env):
    async with running(max_model_len=160) as engine:
        chunks = await collect(engine, request(n=150, max_tokens=None))
    assert sum(len(c["token_ids"]) for c in chunks) == 10
    assert chunks[-1]["finish_reason"] == "length"


async def test_eos_finishes_with_stop(fake_env):
    os.environ["DEEPINFER_FAKE_EOS_AFTER"] = "3"
    async with running() as engine:
        chunks = await collect(engine, request(max_tokens=50))
    assert [t for c in chunks for t in c["token_ids"]][-1] == 1
    assert chunks[-1]["finish_reason"] == "stop"
    assert chunks[-1]["completion_usage"]["completion_tokens"] == 3


async def test_logprobs_are_aligned_with_tokens(fake_env):
    async with running() as engine:
        chunks = await collect(
            engine, request(max_tokens=4, output_options={"logprobs": 2})
        )
    for chunk in chunks:
        if chunk["token_ids"]:
            assert len(chunk["log_probs"]) == len(chunk["token_ids"])
            assert [len(p) for p in chunk["top_logprobs"]] == [2] * len(
                chunk["token_ids"]
            )
            assert chunk["top_logprobs"][0][0]["rank"] == 1


async def test_abort_ends_stream_with_cancelled_within_2s(fake_env):
    os.environ["DEEPINFER_FAKE_STEP_DELAY_S"] = "0.005"
    async with running() as engine:
        ctx = Ctx()
        stream = engine.generate(request(max_tokens=100_000), ctx)
        first = await stream.__anext__()
        assert first["token_ids"] and "finish_reason" not in first
        started = time.monotonic()
        await engine.abort(ctx)
        rest = [c async for c in stream]
        assert time.monotonic() - started < 2.0
        last = assert_one_terminal([first, *rest])
        assert last["finish_reason"] == "cancelled"
        tokens = sum(len(c["token_ids"]) for c in [first, *rest])
        assert last["completion_usage"]["completion_tokens"] == tokens
        # The rank dropped the request and freed its KV.
        await asyncio.sleep(0.1)
        stats = engine.controller.stats()[0]
        assert stats.num_running == 0 and stats.kv_used_blocks == 0


async def test_abort_before_generate_starts_is_honored(fake_env):
    async with running() as engine:
        ctx = Ctx()
        await engine.abort(ctx)
        chunks = await collect(engine, request(), ctx)
    assert chunks == [
        {
            "token_ids": [],
            "index": 0,
            "finish_reason": "cancelled",
            "completion_usage": {
                "prompt_tokens": 40,
                "completion_tokens": 0,
                "total_tokens": 40,
            },
        }
    ]


async def test_dropped_stream_aborts_the_request(fake_env):
    os.environ["DEEPINFER_FAKE_STEP_DELAY_S"] = "0.005"
    async with running() as engine:
        stream = engine.generate(request(max_tokens=100_000), Ctx())
        await stream.__anext__()
        await stream.aclose()
        deadline = time.monotonic() + 2
        while engine.controller.stats()[0].num_running and time.monotonic() < deadline:
            await asyncio.sleep(0.01)
        assert engine.controller.stats()[0].num_running == 0


async def test_routing_dp_rank_is_honored(fake_env):
    # Slow enough that the request still runs when its first chunk arrives.
    os.environ["DEEPINFER_FAKE_STEP_DELAY_S"] = "0.005"
    async with running(dp=2) as engine:
        for rank in (1, 0, 1):
            ctx = Ctx()
            stream = engine.generate(
                request(max_tokens=20, routing={"dp_rank": rank}), ctx
            )
            await stream.__anext__()
            assert engine.controller.owner(ctx.id()) == rank
            async for _ in stream:
                pass
        # Out of range falls back to the least-loaded rank instead of failing.
        chunks = await collect(engine, request(max_tokens=2, routing={"dp_rank": 7}))
        assert chunks[-1]["finish_reason"] == "length"


async def test_unsupported_requests_are_rejected(fake_env):
    async with running() as engine:
        with pytest.raises(InvalidArgument, match="n=1"):
            await collect(engine, request(sampling_options={"n": 2}))
        with pytest.raises(InvalidArgument, match="max_model_len"):
            await collect(engine, request(n=3000))


async def test_bad_token_id_is_rejected_without_leaking_a_stream(fake_env):
    async with running() as engine:
        with pytest.raises(InvalidArgument):
            await collect(engine, {"token_ids": [2**31], "stop_conditions": {}})
        with pytest.raises(InvalidArgument):
            await collect(engine, request(routing={"dp_rank": "x"}))
        assert not engine._streams


async def test_registration_and_metrics_declarations(fake_env):
    engine = make_engine(dp=2)
    config = await engine.start(worker_id=7)
    try:
        llm = config.llm
        assert config.model == "fake-model" and config.served_model_name == "fake-model"
        assert (llm.data_parallel_size, llm.data_parallel_start_rank) == (2, 0)
        assert llm.kv_cache_block_size == 128
        assert llm.total_kv_blocks == engine.meta.num_gpu_blocks
        assert llm.context_length == 2048
        assert llm.max_num_seqs == 256
        assert engine.component_metrics_dp_ranks() == [0, 1]
        assert await engine.kv_event_sources() == []
        assert await engine.is_quiescent() is None
        payload = await engine.health_check_payload()
        assert payload["token_ids"] == [0]
    finally:
        await engine.cleanup()


async def test_kv_event_sources_one_per_rank_with_a_cache_manager(fake_env):
    args = make_args(
        "--kv-cache-manager", "prefix", "--kv-events-port-base", "6000", dp=2
    )
    engine = DeepInferEngine(
        args, Controller([_Idle(0), _Idle(1)], IDLE_META), IDLE_META
    )
    assert await engine.kv_event_sources() == [
        ZmqSource(endpoint="tcp://127.0.0.1:6000", topic="", dp_rank=0),
        ZmqSource(endpoint="tcp://127.0.0.1:6001", topic="", dp_rank=1),
    ]
    assert [s.kv_events_endpoint for s in args.rank_specs("e")] == [
        "tcp://*:6000",
        "tcp://*:6001",
    ]


async def test_snapshots_are_published_per_rank(fake_env):
    published: list[tuple[int, ComponentSnapshot]] = []

    class Publisher:
        def publish(self, rank: int, snap: ComponentSnapshot) -> None:
            published.append((rank, snap))

    async with running(dp=2) as engine:
        engine.attach_snapshot_publisher(Publisher())
        engine.controller.snapshot_interval_s = 0.0
        await collect(engine, request(max_tokens=3))
    assert {rank for rank, _ in published} == {0, 1}
    for rank, snap in published:
        assert (
            snap.dp_rank == rank and snap.kv_total_blocks == engine.meta.num_gpu_blocks
        )
        assert snap.kv_cache_hit_rate is None


async def test_cleanup_is_idempotent_and_safe_before_start(fake_env):
    engine = make_engine(dp=2)
    await engine.cleanup()
    await engine.cleanup()
    with pytest.raises(RuntimeError):
        await collect(engine, request())


async def test_cleanup_ends_inflight_streams(fake_env):
    os.environ["DEEPINFER_FAKE_STEP_DELAY_S"] = "0.005"
    engine = make_engine()
    await engine.start(worker_id=1)
    stream = engine.generate(request(max_tokens=100_000), Ctx())
    await stream.__anext__()
    await engine.cleanup()
    rest = [c async for c in stream]
    assert rest[-1]["finish_reason"] == "cancelled"


def test_parse_args_defaults(fake_env):
    args = make_args(dp=4)
    assert args.engine_type == "NORMAL" and args.kv_transfer == "none"
    assert (args.namespace, args.component, args.endpoint) == (
        "dynamo",
        "backend",
        "generate",
    )
    rank3 = args.engine_args(3)
    assert rank3["engine_rank"] == 3 and rank3["data_parallel_size"] == 4
    assert rank3["expert_parallel_size"] == 4 and rank3["chunked_prefill"] is True

    prefill = make_args(
        "--disaggregation-mode", "prefill", "--engine-arg", "max_num_seqs=8"
    )
    assert prefill.disaggregation_mode == DisaggregationMode.PREFILL
    assert (prefill.engine_type, prefill.kv_transfer, prefill.component) == (
        "PROMPT",
        "nixl",
        "prefill",
    )
    assert prefill.engine_args(0)["max_num_seqs"] == 8
    decode = make_args("--disaggregation-mode", "decode")
    assert decode.engine_args(0)["stop_preempted_request"] is True
    with pytest.raises(ValueError):
        make_args("--kv-transfer", "nixl")

    tier = make_args("--kv-cache-manager", "prefix", "--prefix-host-bytes", "80GiB")
    options = tier.rank_specs("e")[0].cache_options
    assert options["host_bytes"] == 80 << 30 and options["snapshot_device"] is None
    with pytest.raises(ValueError, match="--kv-cache-manager prefix"):
        make_args("--prefix-host-bytes", "1G")

    ssd = ["--prefix-host-bytes", "1G", "--prefix-ssd-dir", "/ssd"]
    tier = make_args("--kv-cache-manager", "prefix", "--prefix-ssd-bytes", "2T", *ssd)
    for rank, spec in enumerate(tier.rank_specs("e")):
        assert spec.cache_options["ssd_dir"] == "/ssd" and spec.rank == rank
        assert spec.cache_options["ssd_bytes"] == 2 << 40
        assert spec.cache_options["ssd_rank"] == rank
        # The ranks of a node split the directory's free space.
        assert spec.cache_options["ssd_share"] == 1 / tier.ranks_per_node
    assert (
        make_args("--kv-cache-manager", "prefix")
        .rank_specs("e")[0]
        .cache_options["ssd_bytes"]
        == 0
    )
    with pytest.raises(ValueError, match="ssd-bytes needs --kv-cache-manager prefix"):
        make_args("--prefix-ssd-bytes", "1G", "--prefix-ssd-dir", "/ssd")
    with pytest.raises(ValueError, match="needs --prefix-host-bytes"):
        make_args(
            "--kv-cache-manager",
            "prefix",
            "--prefix-ssd-bytes",
            "1G",
            "--prefix-ssd-dir",
            "/ssd",
        )
    with pytest.raises(ValueError, match="--prefix-ssd-dir needs --prefix-ssd-bytes"):
        make_args(
            "--kv-cache-manager",
            "prefix",
            "--prefix-host-bytes",
            "1G",
            "--prefix-ssd-dir",
            "/ssd",
        )
    with pytest.raises(ValueError, match="--prefix-ssd-bytes needs --prefix-ssd-dir"):
        make_args(
            "--kv-cache-manager",
            "prefix",
            "--prefix-host-bytes",
            "1G",
            "--prefix-ssd-bytes",
            "1G",
        )


def test_launch_env_defaults(fake_env):
    env = deepinfer_env(make_args("--kv-cache-manager", "prefix"))
    assert env["DEEPINFER_YOCO_PREFILL_TRUNCATION_MODE"] == "compact"
    assert env["DEEPINFER_YOCO_PREFILL_KEEP_TOKENS"] == "140"
    assert env["DEEPINFER_GLOO_TIMEOUT"] == "600"
    assert env["DEEPINFER_USE_MULTIMODAL_POOL"] == "0"
    assert "UCX_TLS" not in env and "EP_DISABLE_GIN" not in env
    assert "UCX_TLS" in deepinfer_env(make_args("--disaggregation-mode", "decode"))


@pytest.mark.parametrize("value", ["1", "true", "off", "no", "2"])
def test_prefix_cache_refuses_extra_state_as_deepinfer_reads_it(fake_env, value):
    args = make_args("--kv-cache-manager", "prefix")
    with pytest.raises(ValueError, match="EXTRA_STATE"):
        apply_env_defaults(args, {"DEEPINFER_PREFILL_STORE_EXTRA_STATE": value})


@pytest.mark.parametrize("value", ["0", "", "false", "False"])
def test_prefix_cache_accepts_extra_state_off(fake_env, value):
    args = make_args("--kv-cache-manager", "prefix")
    apply_env_defaults(args, {"DEEPINFER_PREFILL_STORE_EXTRA_STATE": value})


def test_nixl_uses_the_rail_nccl_is_pinned_to(fake_env):
    decode = make_args("--disaggregation-mode", "decode")
    env = {"NCCL_IB_HCA": "=mlx5_0"}
    apply_env_defaults(decode, env)
    assert env["UCX_NET_DEVICES"] == "mlx5_0:1"
    env = {"NCCL_IB_HCA": "=mlx5_0", "UCX_NET_DEVICES": "mlx5_2:1"}
    apply_env_defaults(decode, env)
    assert env["UCX_NET_DEVICES"] == "mlx5_2:1"  # the pod's choice wins
    for hca in ("^mlx5_1", "mlx5_0,mlx5_1"):
        env = {"NCCL_IB_HCA": hca}
        apply_env_defaults(decode, env)
        assert "UCX_NET_DEVICES" not in env
    env = {"NCCL_IB_HCA": "=mlx5_0"}
    apply_env_defaults(make_args(), env)  # aggregated: no NIXL
    assert "UCX_NET_DEVICES" not in env


@pytest.mark.timeout(120)
async def test_ranks_close_at_exit_when_start_never_ran(fake_env):
    engine, _ = await DeepInferEngine.from_args(
        ["--model", "fake-model", "--deepinfer-backend", "fake", "--dp-size", "2"]
    )
    ipc_dir = engine._launch.ipc_dir
    assert engine._finalizer is not None and engine._finalizer.alive
    engine._finalizer()  # what interpreter exit runs
    child = engine.controller._handles[1]
    assert child.proc.returncode == 0  # a coordinated CLOSE, not a kill
    assert not os.path.exists(ipc_dir)
    await engine.cleanup()  # still safe


@pytest.mark.timeout(120)
async def test_from_args_launches_child_ranks(fake_env):
    engine, worker_config = await DeepInferEngine.from_args(
        ["--model", "fake-model", "--deepinfer-backend", "fake", "--dp-size", "2"]
    )
    try:
        assert worker_config.model_name == "fake-model"
        assert os.environ["CUDA_VISIBLE_DEVICES"] == "0"  # rank 0 runs here
        assert os.environ["DEEPINFER_PREFILL_ATTEMPT_INTERVAL"] == "8"
        await engine.start(worker_id=1)
        results = await asyncio.gather(
            *(
                collect(engine, request(max_tokens=3, routing={"dp_rank": r}))
                for r in (0, 1)
            )
        )
        tokens = [[t for c in chunks for t in c["token_ids"]] for chunks in results]
        assert tokens[0] == tokens[1] and len(tokens[0]) == 3
    finally:
        await engine.cleanup()
    child = engine.controller._handles[1]
    assert child.proc.returncode == 0
    assert not engine._finalizer.alive  # cleanup() took over


CONTROLLER_PROCESS_ARGV = [
    *("--model", "fake-model", "--deepinfer-backend", "fake", "--dp-size", "2"),
    "--no-rank0-in-process",
]


@pytest.mark.timeout(60)
async def test_dead_controller_process_is_fatal(fake_env):
    apply_env_defaults(parse_args(CONTROLLER_PROCESS_ARGV))
    fatal: list[str] = []
    controller = ControllerProcess.spawn(CONTROLLER_PROCESS_ARGV, 60, fatal.append)
    controller.start(lambda outputs: None)
    ranks = [
        int(pid)
        for pid in os.listdir("/proc")
        if pid.isdigit() and _ppid(pid) == controller.proc.pid
    ]
    assert len(ranks) == 2
    controller.proc.kill()
    deadline = time.monotonic() + 10
    while not fatal or any(_ppid(str(pid)) for pid in ranks):
        assert time.monotonic() < deadline, (fatal, ranks)
        await asyncio.sleep(0.05)
    assert fatal == ["the DeepInfer controller exited with code -9"]
    controller.close()
    shutil.rmtree(controller.ipc_dir)


def test_link_sends_queued_items_between_outputs():
    """A head that lags the controller never finds the link empty; what it
    queued (here a submit) must still go out before the next output."""
    ctx = zmq.Context()
    head, controller = ctx.socket(zmq.PAIR), ctx.socket(zmq.PAIR)
    head.bind("inproc://link")
    controller.connect("inproc://link")
    meta = EngineMeta(
        block_size=128,
        num_gpu_blocks=8,
        max_num_seqs=4,
        max_model_len=256,
        max_num_batched_tokens=256,
        prefill_any_attempt_interval=100,
    )
    link = ControllerProcess(None, head, "", Started(meta, 1), print)
    seen: list[bool] = []
    # Whether the submit reached the controller by the time each output is
    # handed to the engine.
    link._emit = lambda merged: seen.append(bool(controller.poll(0)))
    link._thread = object()  # started: other threads queue for the link
    for i in range(3):
        rank = msgspec.msgpack.encode([Output(request_id=f"r{i}", token_ids=[i])])
        controller.send(msgspec.msgpack.encode(Outputs(0.0, [msgspec.Raw(rank)])))
    link.submit(AddReq(request_id="new", token_ids=b"\x02\x00\x00\x00"))
    link._receive()
    assert seen == [True, True, True]
    assert (
        msgspec.msgpack.decode(controller.recv(), type=Submit).add.request_id == "new"
    )
    os.close(link._wake)
    ctx.destroy(linger=0)


def _ppid(pid: str) -> Optional[int]:
    """Parent of a live process, None once it is gone (or a zombie)."""
    try:
        with open(f"/proc/{pid}/stat") as f:
            fields = f.read().rsplit(")", 1)[1].split()
    except OSError:
        return None
    return None if fields[0] == "Z" else int(fields[1])


IDLE_META = EngineMeta(
    block_size=128,
    num_gpu_blocks=8,
    max_num_seqs=4,
    max_model_len=256,
    max_num_batched_tokens=256,
    prefill_any_attempt_interval=100,
)


class _Idle(LocalRankHandle):
    """A rank handle that is never driven (for declaration-only tests)."""

    def __init__(self, rank: int) -> None:
        self.rank = rank
        self.inline = rank == 0


def test_evict_page_cache_advises_checkpoint_shards_only(tmp_path):
    from dynamo.deepinfer.di_engine import evict_page_cache

    (tmp_path / "model-00001-of-00002.safetensors").write_bytes(b"\0" * 4096)
    sub = tmp_path / "sub"
    sub.mkdir()
    (sub / "model-00002-of-00002.safetensors").write_bytes(b"\0" * 8192)
    (tmp_path / "config.json").write_text("{}")
    assert evict_page_cache(str(tmp_path)) == (2, 4096 + 8192)


def test_engram_allocation_retries_after_oom(tmp_path, monkeypatch):
    import sys
    import types

    from dynamo.deepinfer import di_engine

    calls = []

    class EngramRuntime:
        @staticmethod
        def allocate(embeddings):
            calls.append(embeddings)
            if len(calls) < 3:
                raise RuntimeError(
                    "CUDA driver exception: 2 (CUDA_ERROR_OUT_OF_MEMORY, out of memory)"
                )
            return "buffer"

    module = types.ModuleType("engram")
    module.EngramRuntime = EngramRuntime
    monkeypatch.setitem(
        sys.modules, "deepinfer.model_executor.models.v4_modules.engram", module
    )
    monkeypatch.setattr(di_engine, "ENGRAM_ALLOC_RETRY_S", 0.0)
    di_engine.evict_page_cache_before_engram(str(tmp_path))
    assert EngramRuntime.allocate(["e"]) == "buffer"
    assert len(calls) == 3
