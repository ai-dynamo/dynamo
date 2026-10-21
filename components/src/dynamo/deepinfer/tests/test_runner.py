# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Runner contract tests; the native engine is replaced by a scripted fake."""

import queue

import pytest

from dynamo.deepinfer.runner import EngineRunner, RunnerClosed, TokenOutput
from dynamo.deepinfer.tests.fake_native import FakeEngine

pytestmark = [pytest.mark.unit, pytest.mark.gpu_0, pytest.mark.pre_merge]

TIMEOUT = 5


@pytest.fixture
def engine():
    return FakeEngine()


@pytest.fixture
def runner(engine):
    runner = EngineRunner(engine, engine.clone)
    runner.start()
    yield runner
    runner.stop(TIMEOUT)


def collect(runner, request_id, max_tokens):
    """Submit one request and gather everything its sink receives."""
    received: queue.Queue = queue.Queue()
    runner.submit(request_id, [1, 2, 3], max_tokens, received.put)
    return received


def drain_until_finished(received):
    tokens = []
    while True:
        item = received.get(timeout=TIMEOUT)
        if isinstance(item, BaseException):
            raise item
        tokens.extend(item)
        if item[-1].finished:
            return tokens


def test_streams_snapshots_until_finished(runner):
    tokens = drain_until_finished(collect(runner, "a", 3))

    assert [t.token_id for t in tokens] == [100, 101, 102]
    assert [t.finished for t in tokens] == [False, False, True]
    assert tokens[-1].finish_reason == "length"
    assert tokens[0] == TokenOutput(
        request_id="a",
        token_id=100,
        finished=False,
        finish_reason="",
        prompt_len=3,
        logprob=-0.5,
    )


def test_snapshots_survive_buffer_reuse(runner, engine):
    tokens = drain_until_finished(collect(runner, "a", 4))

    # The fake overwrites the same slot every step; had the runner kept
    # references instead of copies, every token would read 103.
    assert [t.token_id for t in tokens] == [100, 101, 102, 103]
    assert engine.slots["a"].token_id == 103


def test_concurrent_requests_do_not_cross_streams(runner):
    first = collect(runner, "a", 2)
    second = collect(runner, "b", 4)

    assert [t.request_id for t in drain_until_finished(first)] == ["a"] * 2
    assert [t.request_id for t in drain_until_finished(second)] == ["b"] * 4


def test_all_native_calls_happen_on_the_runner_thread(runner, engine):
    drain_until_finished(collect(runner, "a", 2))

    assert engine.step_threads == {"deepinfer-engine"}


def test_prompt_token_ids_and_params_reach_the_engine(runner, engine):
    drain_until_finished(collect(runner, "a", 1))

    assert engine.added == [("a", 1, [1, 2, 3])]


def test_rejected_request_fails_only_itself(runner, engine):
    engine.reject.add("bad")
    bad = collect(runner, "bad", 2)
    good = collect(runner, "good", 2)

    assert isinstance(bad.get(timeout=TIMEOUT), ValueError)
    assert len(drain_until_finished(good)) == 2


def test_abort_stops_delivery_and_reaches_the_engine(runner, engine):
    received = collect(runner, "a", 10**9)
    assert received.get(timeout=TIMEOUT)

    runner.abort("a")
    runner.stop(TIMEOUT)

    assert engine.aborted == ["a"]
    # Anything after the abort is dropped; the sink only ever sees the stop.
    leftovers = []
    while not received.empty():
        leftovers.append(received.get_nowait())
    assert all(isinstance(item, (list, RunnerClosed)) for item in leftovers), leftovers


def test_abort_of_unknown_request_is_a_noop(runner, engine):
    runner.abort("nope")
    drain_until_finished(collect(runner, "a", 1))

    assert engine.aborted == []


def test_step_failure_fails_every_waiter_and_closes_the_runner(runner, engine):
    engine.fail_on_step = 2
    first = collect(runner, "a", 10**9)
    second = collect(runner, "b", 10**9)

    for received in (first, second):
        item = received.get(timeout=TIMEOUT)
        while not isinstance(item, BaseException):
            item = received.get(timeout=TIMEOUT)
        assert isinstance(item, RuntimeError)
    assert engine.closed.wait(TIMEOUT)
    assert isinstance(runner.failure, RuntimeError)
    with pytest.raises(RunnerClosed, match="cuda error"):
        runner.submit("c", [1], 1, lambda _: None)


def test_stop_fails_pending_requests_and_closes_the_engine(runner, engine):
    received = collect(runner, "a", 10**9)
    assert received.get(timeout=TIMEOUT)

    assert runner.stop(TIMEOUT)

    assert engine.closed.is_set()
    item = received.get(timeout=TIMEOUT)
    while not isinstance(item, BaseException):
        item = received.get(timeout=TIMEOUT)
    assert isinstance(item, RunnerClosed)


def test_stop_is_idempotent_and_blocks_new_work(runner):
    assert runner.stop(TIMEOUT)
    assert runner.stop(TIMEOUT)

    with pytest.raises(RunnerClosed):
        runner.submit("a", [1], 1, lambda _: None)
    runner.abort("a")
