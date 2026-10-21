# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Runner contract tests; the native engine is replaced by a scripted fake."""

import asyncio
import threading

import pytest

from dynamo.deepinfer.runner import (
    EngineRunner,
    RequestRejected,
    RunnerClosed,
    TokenOutput,
)
from dynamo.deepinfer.tests.fake_native import FakeEngine

pytestmark = [pytest.mark.unit, pytest.mark.gpu_0, pytest.mark.pre_merge]

TIMEOUT = 5
THREAD_PREFIX = "deepinfer-engine"


@pytest.fixture
def engine():
    return FakeEngine()


@pytest.fixture
async def runner(engine):
    runner = EngineRunner()
    await runner.start(lambda: (engine, engine.clone))
    yield runner
    await runner.stop(TIMEOUT)


async def wait_until(predicate, timeout=TIMEOUT):
    deadline = asyncio.get_running_loop().time() + timeout
    while not predicate():
        assert asyncio.get_running_loop().time() < deadline, "timed out"
        await asyncio.sleep(0.005)


def submit(runner, request_id, max_tokens):
    return runner.submit(request_id, [1, 2, 3], max_tokens)


async def next_item(results):
    return await asyncio.wait_for(results.get(), TIMEOUT)


async def drain_until_finished(results):
    tokens = []
    while True:
        item = await next_item(results)
        if isinstance(item, BaseException):
            raise item
        tokens.extend(item)
        if item[-1].finished:
            return tokens


async def drain_until_error(results):
    item = await next_item(results)
    while not isinstance(item, BaseException):
        item = await next_item(results)
    return item


async def test_streams_snapshots_until_finished(runner):
    tokens = await drain_until_finished(submit(runner, "a", 3))

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


async def test_snapshots_survive_buffer_reuse(runner, engine):
    tokens = await drain_until_finished(submit(runner, "a", 4))

    # The fake overwrites the same slot every step; had the runner kept
    # references instead of copies, every token would read 103.
    assert [t.token_id for t in tokens] == [100, 101, 102, 103]
    assert engine.slots["a"].token_id == 103


async def test_concurrent_requests_do_not_cross_streams(runner):
    first = submit(runner, "a", 2)
    second = submit(runner, "b", 4)

    assert [t.request_id for t in await drain_until_finished(first)] == ["a"] * 2
    assert [t.request_id for t in await drain_until_finished(second)] == ["b"] * 4


async def test_every_native_call_runs_on_one_worker_thread(engine):
    construct_threads = set()

    def factory():
        construct_threads.add(threading.current_thread().name)
        return engine, engine.clone

    runner = EngineRunner()
    await runner.start(factory)
    await drain_until_finished(submit(runner, "a", 2))
    submit(runner, "b", 10**9)
    runner.abort("b")
    await wait_until(lambda: engine.aborted)
    await runner.stop(TIMEOUT)

    threads = (
        construct_threads
        | engine.add_threads
        | engine.step_threads
        | engine.abort_threads
        | engine.close_threads
    )
    assert len(threads) == 1
    assert next(iter(threads)).startswith(THREAD_PREFIX)
    assert threading.current_thread().name not in threads


async def test_prompt_token_ids_and_params_reach_the_engine(runner, engine):
    await drain_until_finished(submit(runner, "a", 1))

    assert engine.added == [("a", 1, [1, 2, 3])]


async def test_rejected_request_fails_only_itself(runner, engine):
    engine.reject.add("bad")
    bad = submit(runner, "bad", 2)
    good = submit(runner, "good", 2)

    assert isinstance(await drain_until_error(bad), ValueError)
    assert len(await drain_until_finished(good)) == 2


async def test_request_deepinfer_drops_is_rejected_not_left_hanging(runner, engine):
    engine.drop.add("bad")
    bad = submit(runner, "bad", 2)
    good = submit(runner, "good", 2)

    error = await asyncio.wait_for(bad.get(), TIMEOUT)

    assert isinstance(error, RequestRejected)
    assert len(await drain_until_finished(good)) == 2


async def test_duplicate_request_id_is_rejected(runner):
    submit(runner, "a", 10**9)

    with pytest.raises(ValueError, match="duplicate"):
        submit(runner, "a", 1)


async def test_abort_stops_delivery_and_reaches_the_engine(runner, engine):
    results = submit(runner, "a", 10**9)
    assert await next_item(results)

    runner.abort("a")
    await wait_until(lambda: engine.aborted)

    assert engine.aborted == ["a"]
    # Batches already queued may still be read; nothing new arrives afterwards.
    while not results.empty():
        assert isinstance(results.get_nowait(), list)
    await asyncio.sleep(0.05)
    assert results.empty()


async def test_abort_of_unknown_request_is_a_noop(runner, engine):
    runner.abort("nope")
    await drain_until_finished(submit(runner, "a", 1))

    assert engine.aborted == []


async def test_abort_right_after_submit_leaves_no_orphan(runner, engine):
    # A caller cancelled before its add ran: the abort is queued behind the add
    # on the single worker, so the request is added and then freed.
    submit(runner, "a", 10**9)
    runner.abort("a")

    await wait_until(lambda: engine.aborted)

    assert [added[0] for added in engine.added] == ["a"]
    assert engine.aborted == ["a"]
    assert not engine.pending


async def test_step_failure_fails_every_waiter_and_closes_the_runner(runner, engine):
    engine.fail_on_step = 2
    first = submit(runner, "a", 10**9)
    second = submit(runner, "b", 10**9)

    for results in (first, second):
        assert isinstance(await drain_until_error(results), RuntimeError)
    await wait_until(engine.closed.is_set)
    assert isinstance(runner.failure, RuntimeError)
    with pytest.raises(RunnerClosed, match="cuda error"):
        submit(runner, "c", 1)


async def test_stop_fails_pending_requests_and_closes_the_engine(runner, engine):
    results = submit(runner, "a", 10**9)
    assert await next_item(results)

    assert await runner.stop(TIMEOUT)

    assert engine.closed.is_set()
    assert isinstance(await drain_until_error(results), RunnerClosed)


async def test_stop_waits_for_the_inflight_step_and_delivers_it(runner, engine):
    engine.step_gate = threading.Event()
    results = submit(runner, "a", 1)
    await asyncio.get_running_loop().run_in_executor(
        None, engine.step_started.wait, TIMEOUT
    )

    stopping = asyncio.create_task(runner.stop(TIMEOUT))
    await asyncio.sleep(0.05)
    assert not stopping.done()  # the native step cannot be interrupted

    engine.step_gate.set()
    assert await stopping

    # The step's tokens were consumed from the engine; they must not be lost.
    tokens = await drain_until_finished(results)
    assert [t.token_id for t in tokens] == [100]
    assert engine.closed.is_set()


async def test_stop_is_idempotent_and_blocks_new_work(runner):
    assert await runner.stop(TIMEOUT)
    assert await runner.stop(TIMEOUT)

    with pytest.raises(RunnerClosed):
        submit(runner, "a", 1)
    runner.abort("a")


async def test_stop_reports_a_step_that_never_returns(runner, engine):
    engine.step_gate = threading.Event()
    submit(runner, "a", 10**9)
    await asyncio.get_running_loop().run_in_executor(
        None, engine.step_started.wait, TIMEOUT
    )

    assert not await runner.stop(0.05)

    engine.step_gate.set()  # let the worker thread finish
    assert await runner.stop(TIMEOUT)


async def test_failed_start_closes_the_runner():
    def factory():
        raise RuntimeError("out of memory")

    runner = EngineRunner()
    with pytest.raises(RuntimeError, match="out of memory"):
        await runner.start(factory)

    with pytest.raises(RunnerClosed):
        runner.submit("a", [1], 1)
    assert await runner.stop(TIMEOUT)


async def test_idle_runner_does_not_spin(runner, engine):
    await drain_until_finished(submit(runner, "a", 2))
    await asyncio.sleep(0.05)  # let the trailing flush settle
    steps, flushes = engine.steps, engine.flushes

    await asyncio.sleep(0.2)

    assert (engine.steps, engine.flushes) == (steps, flushes)


async def test_request_the_engine_swallowed_does_not_busy_loop(runner, engine):
    engine.swallow_requests = True
    submit(runner, "lost", 5)

    await asyncio.sleep(0.2)

    assert engine.flushes < 60  # backed off instead of spinning
    runner.abort("lost")
