# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest

from dynamo.deepinfer.protocol import (
    AddReq,
    Cmd,
    EngineMeta,
    LocalRankHandle,
    Op,
    Output,
    RankFailed,
    RankStats,
    RawReply,
    Ready,
    Reply,
    Sync,
    decode_cmd,
    decode_outputs,
    decode_up,
    decode_up_raw,
    encode,
    num_outputs,
)
from dynamo.deepinfer.request import prompt_array, prompt_bytes

pytestmark = [pytest.mark.unit, pytest.mark.gpu_0, pytest.mark.pre_merge]

META = EngineMeta(
    block_size=128,
    num_gpu_blocks=64,
    max_num_seqs=8,
    max_model_len=4096,
    max_num_batched_tokens=4096,
    prefill_any_attempt_interval=100,
)


def test_cmd_roundtrip_keeps_prompt_and_sampling():
    cmd = Cmd(
        seq=7,
        adds=[
            AddReq(
                request_id="a",
                token_ids=prompt_bytes([5, 6, 70000]),
                sampling={"max_tokens": 3, "temperature": 0.5},
            ),
            AddReq(request_id="b", token_ids=prompt_bytes([1])),
        ],
        aborts=["c"],
        sync=Sync(unfinished=True, new_prompts=True),
    )
    back = decode_cmd(encode(cmd))
    assert back == cmd
    assert prompt_array(back.adds[0].token_ids).tolist() == [5, 6, 70000]
    assert back.adds[1].sampling == {}
    assert back.op == Op.STEP


def test_up_messages_decode_by_tag():
    reply = Reply(
        seq=3,
        rank=1,
        u=True,
        outputs=[
            Output(
                request_id="a",
                token_ids=[4, 5],
                logprobs=[-0.1, -0.2],
                top_token_ids=[[4, 9], [5, 8]],
                top_logprobs=[[-0.1, -2.0], [-0.2, -3.0]],
            ),
            Output(request_id="b", finish_reason="cancelled"),
        ],
        stats=RankStats(num_running=2, kv_used_blocks=4, kv_total_blocks=64),
    )
    ready = Ready(rank=1, meta=META)
    assert decode_up(encode(reply)) == reply
    assert decode_up(encode(ready)) == ready
    assert decode_up(encode(Reply(seq=-1, rank=2, error="boom"))).error == "boom"

    # A controller process reads the header and leaves the outputs encoded.
    assert RawReply.__struct_fields__ == Reply.__struct_fields__
    raw = decode_up_raw(encode(reply))
    assert isinstance(raw, RawReply) and (raw.seq, raw.u, raw.stats) == (
        3,
        True,
        reply.stats,
    )
    assert (
        num_outputs(raw.outputs) == 2 and decode_outputs(raw.outputs) == reply.outputs
    )
    assert num_outputs(decode_up_raw(encode(Reply(seq=4, rank=1))).outputs) == 0
    many = Reply(seq=5, rank=1, outputs=[Output(request_id=str(i)) for i in range(20)])
    assert num_outputs(decode_up_raw(encode(many)).outputs) == 20


class _StubRunner:
    def __init__(self, rank: int = 0, fail: bool = False) -> None:
        self.spec = SimpleNamespace(rank=rank)
        self.meta = META
        self.fail = fail
        self.applied: list[int] = []

    def apply(self, cmd: Cmd) -> Reply:
        if self.fail:
            raise RuntimeError("engine exploded")
        self.applied.append(cmd.seq)
        return Reply(seq=cmd.seq, rank=self.spec.rank, u=bool(cmd.adds))


def test_inline_handle_applies_on_recv():
    runner = _StubRunner()
    handle = LocalRankHandle(runner)
    assert handle.inline and handle.wait_ready() is META
    handle.send(Cmd(seq=0))
    assert runner.applied == []  # nothing runs until the controller receives
    assert handle.recv().seq == 0
    assert runner.applied == [0]
    with pytest.raises(RankFailed):
        handle.recv()


def test_handle_reports_runner_errors_as_replies():
    handle = LocalRankHandle(_StubRunner(fail=True))
    handle.send(Cmd(seq=4))
    reply = handle.recv()
    assert reply.seq == 4 and "engine exploded" in reply.error


def test_threaded_handle_runs_commands_in_order():
    runner = _StubRunner(rank=1)
    handle = LocalRankHandle(runner, threaded=True)
    assert not handle.inline and handle.rank == 1
    for seq in range(3):
        handle.send(Cmd(seq=seq))
    assert [handle.recv(timeout=5).seq for _ in range(3)] == [0, 1, 2]
    with pytest.raises(RankFailed):
        handle.recv(timeout=0.01)
    handle.close()
    assert runner.applied == [0, 1, 2]
