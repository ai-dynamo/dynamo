# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Prefill -> decode handoff over the CPU fake with LocalKVTransfer (and NIXL on host memory)."""

import asyncio
import contextlib
import copy
import os
import shutil
import tempfile
import time
import uuid
from typing import Any, Callable, Optional

import pytest

from dynamo.deepinfer import di_engine, disagg
from dynamo.deepinfer.args import parse_args
from dynamo.deepinfer.controller import Controller
from dynamo.deepinfer.disagg import (
    HANDOFF_KEY,
    DeepInferHandoff,
    handoff_from_prefill_result,
)
from dynamo.deepinfer.engine import DeepInferEngine
from dynamo.deepinfer.protocol import AddReq, Cmd, LocalRankHandle, Op, Output
from dynamo.deepinfer.rank import RankRunner, RankSpec
from dynamo.deepinfer.request import prompt_bytes
from dynamo.llm.exceptions import InvalidArgument

pytestmark = [pytest.mark.unit, pytest.mark.gpu_0, pytest.mark.pre_merge]

PROMPT = [2 + (i * 7) % 900 for i in range(300)]


@pytest.fixture
def fake_env(monkeypatch):
    monkeypatch.setattr(os, "environ", dict(os.environ))
    # Short path: ipc:// socket paths are limited to ~107 bytes.
    workdir = tempfile.mkdtemp(prefix="di-")
    os.environ["DEEPINFER_FAKE_DP_ENDPOINT"] = f"ipc://{workdir}/dp"
    os.environ["DEEPINFER_FAKE_DP_TIMEOUT_S"] = "10"
    os.environ["DEEPINFER_PREFILL_ANY_ATTEMPT_INTERVAL"] = "4"
    os.environ.pop("CUDA_VISIBLE_DEVICES", None)
    runners: list[RankRunner] = []
    yield runners
    for i, runner in enumerate(runners):
        runner.apply(Cmd(seq=1_000_000 + i, op=Op.CLOSE))
    shutil.rmtree(workdir, ignore_errors=True)


# -- rank level -----------------------------------------------------------------


def runner(
    runners: list[RankRunner], engine_type: str, kv_transfer: str = "local", **kw: Any
) -> RankRunner:
    engine_args = {"model": "fake", "chunked_prefill": True, "max_model_len": 2048}
    if engine_type == "COMPLETION":
        engine_args["stop_preempted_request"] = True
    r = RankRunner(
        RankSpec(
            rank=0,
            dp_size=1,
            engine_args=engine_args,
            engine_type=engine_type,
            backend="fake",
            engine_id=engine_type,
            kv_transfer=kv_transfer,
            **kw,
        )
    )
    runners.append(r)
    return r


def add(rid: str, max_tokens: int = 6, **kw: Any) -> AddReq:
    return AddReq(
        request_id=rid,
        token_ids=prompt_bytes(PROMPT),
        sampling={"max_tokens": max_tokens},
        **kw,
    )


def run(
    r: RankRunner,
    cmds: list[Cmd],
    until: Callable[[dict[str, list[Output]]], bool],
    timeout: float = 10.0,
) -> dict[str, list[Output]]:
    """Apply ``cmds``, then empty iterations until ``until(outputs)`` holds."""
    seen: dict[str, list[Output]] = {}
    deadline = time.monotonic() + timeout
    i = 0
    while True:
        cmd = cmds[i] if i < len(cmds) else Cmd(seq=10_000 + i)
        for out in r.apply(cmd).outputs:
            seen.setdefault(out.request_id, []).append(out)
        i += 1
        if i >= len(cmds) and until(seen):
            return seen
        assert time.monotonic() < deadline, seen
        if i > 50:
            time.sleep(0.001)  # asynchronous (NIXL) transfers


def finished(rid: str) -> Callable[[dict[str, list[Output]]], bool]:
    return lambda seen: any(o.finish_reason for o in seen.get(rid, []))


def tokens(outs: list[Output]) -> list[int]:
    return [t for o in outs for t in o.token_ids]


def prefill_handoff(prefill: RankRunner, rid: str) -> dict[str, Any]:
    seen = run(
        prefill,
        [Cmd(seq=0, adds=[add(rid, max_tokens=1, hold_kv=True)])],
        finished(rid),
    )
    terminal = seen[rid][-1]
    assert terminal.finish_reason == "length", terminal
    return handoff_from_prefill_result(
        {"disaggregated_params": terminal.disaggregated_params}
    )


def expected_tokens(runners: list[RankRunner], max_tokens: int = 6) -> list[int]:
    agg = runner(runners, "NORMAL", kv_transfer="none")
    seen = run(agg, [Cmd(seq=0, adds=[add("r", max_tokens)])], finished("r"))
    return tokens(seen["r"])


def settle(r: RankRunner, predicate: Callable[[], bool]) -> None:
    run(r, [], lambda _: predicate())


def test_hold_ids_are_unique_per_hold(fake_env):
    prefill = runner(fake_env, "PROMPT")
    first, second = prefill_handoff(prefill, "r"), prefill_handoff(prefill, "r")
    assert first["request_id"] != second["request_id"]
    assert all(h["request_id"].startswith("r#") for h in (first, second))
    assert prefill.prefill.num_held == 2
    # Releasing one hold leaves the other.
    decode = runner(fake_env, "COMPLETION")
    decode.engine.dynamo_transfer.release_remote(DeepInferHandoff.from_dict(first))
    prefill.apply(Cmd(seq=50))
    assert prefill.prefill.num_held == 1


def test_lease_expiry_frees_held_kv_and_decode_refuses_the_handoff(fake_env):
    prefill = runner(fake_env, "PROMPT", kv_lease_s=0.05)
    decode = runner(fake_env, "COMPLETION")
    handoff = prefill_handoff(prefill, "r")
    assert prefill.prefill.num_held == 1 and prefill.cache.stats().used_blocks == 3

    time.sleep(0.1)
    prefill.apply(Cmd(seq=50))
    assert prefill.prefill.num_held == 0 and prefill.cache.stats().used_blocks == 0

    seen = run(decode, [Cmd(seq=0, adds=[add("r", handoff=handoff)])], finished("r"))
    assert [(o.token_ids, o.finish_reason) for o in seen["r"]] == [([], "error")]
    assert decode.cache.stats().used_blocks == 0 and decode.decode.num_pending == 0


def test_abort_mid_transfer_frees_both_sides(fake_env):
    prefill = runner(fake_env, "PROMPT")
    decode = runner(fake_env, "COMPLETION", transfer_options={"delay_polls": 2})
    handoff = prefill_handoff(prefill, "r")

    assert decode.apply(Cmd(seq=0, adds=[add("r", handoff=handoff)])).outputs == []
    assert decode.decode.num_pending == 1 and decode.cache.stats().used_blocks == 3
    seen = run(
        decode, [Cmd(seq=1, aborts=["r"])], lambda _: decode.decode.num_pending == 0
    )
    assert "r" not in seen
    assert decode.cache.stats().used_blocks == 0
    assert not decode.engine.has_unfinished_requests()
    # The read completed, so its notification released the prefill copy.
    prefill.apply(Cmd(seq=50))
    assert prefill.prefill.num_held == 0 and prefill.cache.stats().used_blocks == 0


def test_abort_while_waiting_for_kv_space_releases_the_prefill_copy(
    fake_env, monkeypatch
):
    prefill = runner(fake_env, "PROMPT")
    first, second = prefill_handoff(prefill, "a"), prefill_handoff(prefill, "b")
    monkeypatch.setitem(os.environ, "DEEPINFER_FAKE_NUM_GPU_BLOCKS", "6")
    decode = runner(fake_env, "COMPLETION")  # 5 usable blocks: one 300-token request

    adds = [add("a", max_tokens=50, handoff=first), add("b", handoff=second)]
    outs = decode.apply(Cmd(seq=0, adds=adds)).outputs
    assert decode.decode.num_pending == 1  # "b" waits for KV space
    prefill.apply(Cmd(seq=50))
    assert prefill.prefill.num_held == 1  # "a" was read and released

    outs += decode.apply(Cmd(seq=1, aborts=["b"])).outputs
    assert decode.decode.num_pending == 0
    prefill.apply(Cmd(seq=51))
    assert prefill.prefill.num_held == 0 and prefill.cache.stats().used_blocks == 0
    seen = run(decode, [], finished("a"))
    assert all(o.request_id == "a" for o in outs)
    assert len(tokens(outs + seen["a"])) == 50
    assert seen["a"][-1].finish_reason == "length"


def test_layout_mismatch_is_an_error_and_releases_the_prefill_copy(fake_env):
    prefill = runner(fake_env, "PROMPT")
    decode = runner(fake_env, "COMPLETION")
    handoff = prefill_handoff(prefill, "r")
    bad = {**handoff, "layout_hash": "0" * 16}

    seen = run(decode, [Cmd(seq=0, adds=[add("r", handoff=bad)])], finished("r"))
    assert seen["r"][-1].finish_reason == "error"
    assert decode.cache.stats().used_blocks == 0
    prefill.apply(Cmd(seq=50))
    assert prefill.prefill.num_held == 0


def test_layout_counts_an_unset_setting_as_deepinfer_default(fake_env):
    """A prefill role that leaves DEEPINFER_USE_NEXTN_DECODE unset and a decode
    role that spells out DeepInfer's default (5) share one KV layout."""
    os.environ.pop("DEEPINFER_USE_NEXTN_DECODE", None)
    prefill = runner(fake_env, "PROMPT")
    os.environ["DEEPINFER_USE_NEXTN_DECODE"] = "5"
    decode = runner(fake_env, "COMPLETION")
    handoff = prefill_handoff(prefill, "r")

    seen = run(decode, [Cmd(seq=0, adds=[add("r", handoff=handoff)])], finished("r"))
    assert seen["r"][-1].finish_reason == "length"
    assert len(tokens(seen["r"])) == 6
    # Another value is another layout.
    os.environ["DEEPINFER_USE_NEXTN_DECODE"] = "3"
    other = di_engine.kv_layout(decode.engine, decode.api.envs)
    assert other.layout_hash != handoff["layout_hash"]


@pytest.mark.parametrize(
    "field, value",
    [("num_tokens", 299), ("block_ids", [1, 2]), ("first_token", 1 << 20)],
)
def test_handoff_that_does_not_fit_the_prompt_is_an_error(fake_env, field, value):
    prefill = runner(fake_env, "PROMPT")
    decode = runner(fake_env, "COMPLETION")
    handoff = {**prefill_handoff(prefill, "r"), field: value}

    seen = run(decode, [Cmd(seq=0, adds=[add("r", handoff=handoff)])], finished("r"))
    assert [(o.token_ids, o.finish_reason) for o in seen["r"]] == [([], "error")]
    assert decode.cache.stats().used_blocks == 0
    prefill.apply(Cmd(seq=50))
    assert prefill.prefill.num_held == 0


def test_rejected_decode_add_releases_the_prefill_copy(fake_env):
    prefill = runner(fake_env, "PROMPT")
    decode = runner(fake_env, "COMPLETION")
    handoff = prefill_handoff(prefill, "r")
    out_of_vocab = AddReq(
        request_id="r",
        token_ids=prompt_bytes([5000] + PROMPT[1:]),
        sampling={"max_tokens": 6},
        handoff=handoff,
    )
    seen = run(decode, [Cmd(seq=0, adds=[out_of_vocab])], finished("r"))
    assert seen["r"][-1].finish_reason == "error"
    prefill.apply(Cmd(seq=50))
    assert prefill.prefill.num_held == 0


def test_decode_reads_leave_headroom_for_running_requests(fake_env, monkeypatch):
    prefill = runner(fake_env, "PROMPT")
    handoffs = [prefill_handoff(prefill, rid) for rid in "abc"]
    # 9 usable blocks, 3 per request: a third read would leave none free, and
    # DeepInfer would then stop a running request to make room.
    monkeypatch.setitem(os.environ, "DEEPINFER_FAKE_NUM_GPU_BLOCKS", "10")
    decode = runner(fake_env, "COMPLETION")
    adds = [add(rid, max_tokens=50, handoff=h) for rid, h in zip("abc", handoffs)]
    seen = run(
        decode,
        [Cmd(seq=0, adds=adds)],
        lambda seen: all(finished(rid)(seen) for rid in "abc"),
    )
    for rid in "abc":
        assert seen[rid][-1].finish_reason == "length"
        assert len(tokens(seen[rid])) == 50
    assert decode.cache.stats().used_blocks == 0


def test_read_finishing_after_the_lease_is_discarded(fake_env, monkeypatch):
    monkeypatch.setattr(disagg, "READ_MARGIN_S", 0.0)
    prefill = runner(fake_env, "PROMPT", kv_lease_s=disagg.EXPIRY_SKEW_S + 0.5)
    decode = runner(fake_env, "COMPLETION", transfer_options={"delay_polls": 1})
    handoff = prefill_handoff(prefill, "r")

    decode.apply(Cmd(seq=0, adds=[add("r", handoff=handoff)]))
    assert decode.decode.num_pending == 1  # the read is in flight
    time.sleep(0.6)  # the read completes too close to the lease deadline
    seen = run(decode, [], finished("r"))
    assert [(o.token_ids, o.finish_reason) for o in seen["r"]] == [([], "error")]
    assert decode.cache.stats().used_blocks == 0


def test_transfer_failure_finishes_with_error_and_frees_decode_kv(fake_env):
    prefill = runner(fake_env, "PROMPT")
    decode = runner(fake_env, "COMPLETION", transfer_options={"delay_polls": 1})
    handoff = prefill_handoff(prefill, "r")

    decode.apply(Cmd(seq=0, adds=[add("r", handoff=handoff)]))
    prefill.engine.dynamo_transfer.shutdown()  # the peer goes away mid-read
    seen = run(decode, [], finished("r"))
    assert [(o.token_ids, o.finish_reason) for o in seen["r"]] == [([], "error")]
    assert decode.cache.stats().used_blocks == 0


def test_decode_with_max_tokens_one_returns_the_handoff_token(fake_env):
    expected = expected_tokens(fake_env, max_tokens=1)
    prefill = runner(fake_env, "PROMPT")
    decode = runner(fake_env, "COMPLETION")
    handoff = prefill_handoff(prefill, "r")

    seen = run(
        decode,
        [Cmd(seq=0, adds=[add("r", max_tokens=1, handoff=handoff)])],
        finished("r"),
    )
    assert [(o.token_ids, o.finish_reason) for o in seen["r"]] == [(expected, "length")]
    assert decode.cache.stats().used_blocks == 0
    prefill.apply(Cmd(seq=50))
    assert prefill.prefill.num_held == 0


def test_eos_as_first_token_is_terminal_without_a_hold(fake_env, monkeypatch):
    monkeypatch.setitem(os.environ, "DEEPINFER_FAKE_EOS_AFTER", "1")
    prefill = runner(fake_env, "PROMPT")
    seen = run(
        prefill,
        [Cmd(seq=0, adds=[add("r", max_tokens=1, hold_kv=True)])],
        finished("r"),
    )
    terminal = seen["r"][-1]
    assert (terminal.finish_reason, terminal.disaggregated_params) == ("stop", None)
    assert prefill.prefill.num_held == 0 and prefill.cache.stats().used_blocks == 0


def _nixl_or_skip() -> None:
    nixl = pytest.importorskip("dynamo.deepinfer.kv_transfer.nixl")
    try:
        nixl.NixlKVTransfer(rank=0, engine_id="probe").shutdown()
    except Exception as e:
        pytest.skip(f"NIXL/UCX cannot initialize here: {e}")


def test_nixl_handoff_reproduces_aggregated_tokens(fake_env):
    _nixl_or_skip()
    expected = expected_tokens(fake_env)
    prefill = runner(fake_env, "PROMPT", kv_transfer="nixl")
    decode = runner(fake_env, "COMPLETION", kv_transfer="nixl")
    handoff = prefill_handoff(prefill, "r")

    seen = run(decode, [Cmd(seq=0, adds=[add("r", handoff=handoff)])], finished("r"))
    assert tokens(seen["r"]) == expected
    settle(prefill, lambda: prefill.prefill.num_held == 0)
    assert prefill.cache.stats().used_blocks == 0
    settle(decode, lambda: decode.cache.stats().used_blocks == 0)


# -- engine level -----------------------------------------------------------------


def make_engine(mode: str, *extra: str) -> DeepInferEngine:
    args = parse_args(
        [
            "--model",
            "fake-model",
            "--deepinfer-backend",
            "fake",
            "--dp-size",
            "1",
            "--max-model-len",
            "2048",
            "--disaggregation-mode",
            mode,
            *extra,
        ]
    )
    handle = LocalRankHandle(RankRunner(args.rank_specs(engine_id=mode)[0]))
    meta = handle.wait_ready()
    controller = Controller(
        [handle], meta, engine_type=args.engine_type, idle_heartbeat_s=0.02
    )
    return DeepInferEngine(args, controller, meta)


@contextlib.asynccontextmanager
async def running(mode: str, *extra: str):
    engine = make_engine(mode, *extra)
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


async def collect(engine: DeepInferEngine, req: dict[str, Any]) -> list[dict]:
    return [c async for c in engine.generate(req, Ctx())]


async def eventually(predicate: Callable[[], Any], timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    while not await predicate():
        assert time.monotonic() < deadline
        await asyncio.sleep(0.01)


def usage(prompt: int, completion: int, cached: int = 0) -> dict[str, Any]:
    return {
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": prompt + completion,
        "prompt_tokens_details": {"cached_tokens": cached},
    }


async def test_engine_prefill_then_decode_matches_aggregated(fake_env):
    request = {"token_ids": PROMPT, "stop_conditions": {"max_tokens": 6}}
    async with running("agg") as agg:
        expected = [
            t
            for c in await collect(agg, copy.deepcopy(request))
            for t in c["token_ids"]
        ]

    async with running("prefill", "--kv-transfer", "local") as prefill, running(
        "decode", "--kv-transfer", "local"
    ) as decode:
        (chunk,) = await collect(prefill, copy.deepcopy(request))
        assert chunk["token_ids"] == expected[:1]
        assert chunk["finish_reason"] == "length"
        assert chunk["completion_usage"] == usage(len(PROMPT), 1)
        assert set(chunk["disaggregated_params"]) == {HANDOFF_KEY}

        # What the PrefillRouter sends to decode: the original request plus the
        # result, including the prefill's prompt_tokens_details (admission.rs).
        decode_request = {
            **copy.deepcopy(request),
            "prefill_result": {
                "disaggregated_params": chunk["disaggregated_params"],
                "prompt_tokens_details": chunk["completion_usage"][
                    "prompt_tokens_details"
                ],
            },
        }
        chunks = await collect(decode, decode_request)
        assert [t for c in chunks for t in c["token_ids"]] == expected
        assert chunks[-1]["finish_reason"] == "length"
        assert chunks[-1]["completion_usage"] == usage(len(PROMPT), 6)

        await eventually(prefill.is_quiescent)
        stats = prefill.controller.stats()[0]
        assert (stats.num_held, stats.kv_used_blocks) == (0, 0)

        async def decode_freed() -> bool:
            return decode.controller.stats()[0].kv_used_blocks == 0

        await eventually(decode_freed)


def held(engine: DeepInferEngine) -> int:
    return engine.controller.stats()[0].num_held


def spy_releases(engine: DeepInferEngine) -> list[dict[str, Any]]:
    released: list[dict[str, Any]] = []
    release = engine.controller.release

    def spy(handoff: dict[str, Any], rank: int) -> None:
        released.append(handoff)
        release(handoff, rank)

    engine.controller.release = spy  # type: ignore[method-assign]
    return released


async def test_prefill_handoff_nobody_received_is_released(fake_env):
    async with running("prefill", "--kv-transfer", "local") as prefill:
        released = spy_releases(prefill)
        # The stream is gone when the handoff is dispatched (a cancel raced the
        # final prefill step): nothing can deliver it to a decode peer.
        prefill.controller.submit(
            AddReq("r", prompt_bytes(PROMPT), {"max_tokens": 1}, hold_kv=True)
        )

        async def released_and_freed() -> bool:
            return bool(released) and held(prefill) == 0

        await eventually(released_and_freed)
        assert released[0]["request_id"].startswith("r#")


async def test_prefill_handoff_behind_a_cancel_is_released(fake_env):
    os.environ["DEEPINFER_FAKE_STEP_DELAY_S"] = "0.01"
    async with running("prefill", "--kv-transfer", "local") as prefill:
        released = spy_releases(prefill)
        ctx = Ctx()
        stream = prefill.generate(
            {"token_ids": PROMPT, "stop_conditions": {"max_tokens": 1}}, ctx
        )
        first = asyncio.ensure_future(stream.__anext__())
        await asyncio.sleep(0)
        rid = ctx.id()
        handoff = {"request_id": f"{rid}#9", "dp_rank": 0}
        await prefill.abort(ctx)
        # The handoff lands after the cancel, as when abort() beats _dispatch.
        prefill._streams[rid].put_nowait(
            Output(
                request_id=rid,
                token_ids=[5],
                finish_reason="length",
                disaggregated_params={HANDOFF_KEY: handoff},
            )
        )
        assert (await first)["finish_reason"] == "cancelled"
        await stream.aclose()
        assert released == [handoff]


async def test_decode_request_rejected_after_parsing_releases_the_handoff(fake_env):
    request = {"token_ids": PROMPT, "stop_conditions": {"max_tokens": 6}}
    async with running("prefill", "--kv-transfer", "local") as prefill, running(
        "decode", "--kv-transfer", "local"
    ) as decode:
        (chunk,) = await collect(prefill, copy.deepcopy(request))
        assert held(prefill) == 1
        bad = {
            **copy.deepcopy(request),
            "sampling_options": {"n": 2},
            "prefill_result": {"disaggregated_params": chunk["disaggregated_params"]},
        }
        with pytest.raises(InvalidArgument, match="n=1"):
            await collect(decode, bad)

        async def freed() -> bool:
            return held(prefill) == 0

        await eventually(freed)


async def test_prefix_hit_is_reported_as_cached_prompt_tokens():
    request = {"token_ids": PROMPT, "stop_conditions": {"max_tokens": 4}}
    async with running("agg", "--kv-cache-manager", "prefix") as engine:
        cold = (await collect(engine, copy.deepcopy(request)))[-1]["completion_usage"]
        warm = (await collect(engine, copy.deepcopy(request)))[-1]["completion_usage"]
    assert cold["prompt_tokens_details"] == {"cached_tokens": 0}
    # 300 prompt tokens, KEEP=140: the deepest 128-aligned snapshot is at 128.
    assert warm["prompt_tokens_details"] == {"cached_tokens": 128}
    assert warm["prompt_tokens"] == len(PROMPT)
