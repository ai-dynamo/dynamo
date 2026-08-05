# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The structural claims of ``ASYNCIO_GIL_PATH.md``, asserted.

These are the claims that do not depend on hardware, thread count or
calibration -- the ones that follow from the code shape and must therefore
hold in the simulation exactly as they do on the real worker:

1. a batch of N responses costs ONE event-loop ready-deque entry, not N
   (the "Correction: responses are batched" section),
2. ingress lands on that SAME deque, so egress cost converts into admission
   latency (the "Ingress" structural difference),
3. the pull path adds one ready-deque entry per RESPONSE and push one per
   REQUEST (the reason push removes 2 of 3 GIL acquisitions),
4. push never falls back to yielding, which would reintroduce exactly what it
   removes,
5. the loop-load arithmetic the aggregates table is built from.

Timing-dependent claims (absolute GIL wait, the pull/push latency gap) are NOT
asserted here: they need the decode worker's ~45 GIL-capable threads, which a
test process does not have. See ``dynamo_sim/gil_noise.py``.
"""

from __future__ import annotations

import asyncio

import pytest

from egress_experiments.costs import Costs
from egress_experiments.fake_trtllm.aqueue import AsyncQueue, SyncQueue
from egress_experiments.fake_trtllm.engine import BatchConfig, ConstantIteration
from egress_experiments.harness import SimConfig, run_simulation

pytestmark = [
    pytest.mark.unit,
    pytest.mark.pre_merge,
    pytest.mark.gpu_0,
    pytest.mark.none,
]

#: Small and cost-free: these assertions are about counts, not microseconds.
_FREE = Costs().with_scale(0.0)


def _config(egress: str, **overrides) -> SimConfig:
    base = dict(
        egress=egress,
        requests=16,
        max_tokens=4,
        isl=8,
        batch=BatchConfig(total=8),
        iteration=ConstantIteration(4.0),
        costs=_FREE,
        lag_ms=1.0,
    )
    base.update(overrides)
    return SimConfig(**base)


# --------------------------------------------------------------------------
# 1. N responses -> ONE ready-deque entry
# --------------------------------------------------------------------------


def test_notify_many_costs_one_deque_entry_for_the_whole_batch():
    """The 132:1 ratio, isolated from any timing.

    ``proxy.py`` puts each response into its own per-request AsyncQueue with
    ``put_nowait`` -- which never touches the loop's ready deque -- and then
    issues a SINGLE ``notify_many`` per IPC batch.
    """

    async def main():
        loop = asyncio.get_running_loop()
        enqueues = []

        original = loop.call_soon_threadsafe

        def counting(callback, *args, **kwargs):
            enqueues.append(getattr(callback, "__name__", "?"))
            return original(callback, *args, **kwargs)

        loop.call_soon_threadsafe = counting  # type: ignore[method-assign]
        try:
            queues = [AsyncQueue() for _ in range(132)]
            sync_queues = []
            for queue in queues:
                queue.sync_q.bind_loop(loop)
                sync_queues.append(queue.sync_q)

            # The per-response work the dispatch thread does: 132 appends.
            for index, sync_queue in enumerate(sync_queues):
                sync_queue.put_nowait(index)
            assert enqueues == [], "put_nowait must not touch the loop"

            # The per-BATCH work: one call.
            SyncQueue.notify_many(loop, sync_queues)
            await asyncio.sleep(0)
            await asyncio.sleep(0)

            assert len(enqueues) == 1, (
                f"132 responses cost {len(enqueues)} deque entries; the whole "
                "argument requires exactly 1"
            )
            # And every consumer was actually woken.
            assert all(not q.empty() for q in queues)
        finally:
            del loop.call_soon_threadsafe  # type: ignore[attr-defined]

    asyncio.run(main())


def test_run_produces_many_responses_per_deque_entry():
    """End to end: the ratio is > 1 and equals responses / notify_many calls."""
    result = run_simulation(_config("push"))
    assert result.notify_many_calls > 0
    assert result.responses_dispatched == result.responses
    assert result.responses_per_deque_entry == pytest.approx(
        result.responses_dispatched / result.notify_many_calls
    )
    assert result.responses_per_deque_entry > 1.0


# --------------------------------------------------------------------------
# 2 & 3. pull adds an entry per response; push adds one per request
# --------------------------------------------------------------------------


def test_pull_hands_off_to_the_loop_once_per_response():
    """``demand_driven_python_stream`` advances ``__anext__`` per response.

    One hand-off per response, plus one final advance that raises
    StopAsyncIteration.
    """
    cfg = _config("pull")
    result = run_simulation(cfg)

    expected = cfg.requests * (cfg.max_tokens + 1)
    assert result.loop_handoffs == expected
    assert result.responses == cfg.requests * cfg.max_tokens
    # More cross-thread enqueues than responses: the pull path's cost.
    assert result.probe["enqueues"] > result.responses


def test_push_hands_off_to_the_loop_once_per_request():
    """``PythonPushEngine`` advances the generator ONCE per request."""
    cfg = _config("push")
    result = run_simulation(cfg)

    assert result.loop_handoffs == cfg.requests
    assert result.responses == cfg.requests * cfg.max_tokens
    # Strictly fewer cross-thread enqueues than responses -- the property the
    # push path exists to create.
    assert result.probe["enqueues"] < result.responses


def test_push_never_takes_the_yield_fallback():
    """A reachable ``yield`` in ``drive_push_egress_stream`` would push every
    response back onto the pull path via ``pybridge.push_forward_yield``,
    reintroducing the per-response GIL acquisition."""
    result = run_simulation(_config("push"))
    assert result.fallback_yields == 0


def test_push_removes_deque_entries_without_losing_responses():
    """Same responses delivered, far fewer hand-offs."""
    pull = run_simulation(_config("pull"))
    push = run_simulation(_config("push"))

    assert pull.responses == push.responses
    assert push.loop_handoffs < pull.loop_handoffs
    assert push.probe["enqueues"] < pull.probe["enqueues"]


def test_ingress_and_egress_share_one_deque():
    """Both request admission and response notification cross into the SAME
    loop through ``call_soon_threadsafe``.

    That sharing is the mechanism by which egress cost becomes admission
    latency: "``call_soon_threadsafe`` enqueues the new request onto the one
    asyncio deque that the response stream is draining through".
    """
    result = run_simulation(_config("push"))
    labels = set(result.probe["callbacks"])

    assert "response-notify (per IPC batch)" in labels, labels
    assert "push pump (per request)" in labels, labels
    # One probe, installed on one loop: both labels appearing means both
    # producers targeted that loop.
    assert result.probe["armed_callback_probe"] is True


def test_the_loop_and_the_dispatch_thread_are_different_threads():
    result = run_simulation(_config("push"))
    assert result.threads["dispatch"] == "proxy_dispatch_result_thread"
    assert result.threads["loop"] != result.threads["dispatch"]


# --------------------------------------------------------------------------
# 5. the aggregates arithmetic
# --------------------------------------------------------------------------


def test_loop_load_is_cost_times_demand():
    """The table's ``loop load`` row: demand / capacity, capacity = 1e6 / cost."""
    result = run_simulation(_config("push", costs=Costs()))
    expected = result.response_demand_per_s * result.loop_us_per_response / 1e6
    assert result.loop_load == pytest.approx(expected, rel=1e-9)
    assert result.loop_capacity_per_s == pytest.approx(
        1e6 / result.loop_us_per_response, rel=1e-9
    )


def test_default_costs_reproduce_the_published_stage_totals():
    """85.34 us across 3 stages, 44.0x trtllm-serve's 1.94 us."""
    costs = Costs()
    assert costs.loop_us_per_response_push == pytest.approx(85.34, abs=0.01)
    assert costs.loop_us_per_response_pull == pytest.approx(74.62, abs=0.01)

    result = run_simulation(_config("push", costs=costs))
    assert result.loop_us_per_response == pytest.approx(85.34, abs=0.01)
    assert result.serve_ratio == pytest.approx(44.0, abs=0.1)
    assert result.loop_capacity_per_s == pytest.approx(11718, rel=0.001)


def test_structural_results_do_not_depend_on_the_calibration():
    """Scaling every stage cost must not change any count.

    If it did, the conclusions would be an artefact of the calibration rather
    than of the code shape.
    """
    cheap = run_simulation(_config("push", costs=Costs().with_scale(0.0)))
    dear = run_simulation(_config("push", costs=Costs().with_scale(2.0)))

    assert cheap.loop_handoffs == dear.loop_handoffs
    assert cheap.responses == dear.responses
    assert cheap.fallback_yields == dear.fallback_yields == 0
    assert dear.loop_us_per_response == pytest.approx(
        2 * cheap.loop_us_per_response if cheap.loop_us_per_response else 170.68
    )


def test_no_errors_on_either_path():
    for egress in ("pull", "push"):
        result = run_simulation(_config(egress))
        assert result.errors == []
        assert result.requests_completed == result.config.requests


# --------------------------------------------------------------------------
# The spawn_blocking GIL acquisitions (the ingress/egress thread topology)
# --------------------------------------------------------------------------


def test_both_paths_cross_into_python_via_spawn_blocking_once_per_request():
    """``engine.rs:85`` -- invoke_generator is spawn_blocking + with_gil.

    ``push_egress.rs:475`` calls the same helper, so push pays it too. What
    push does not pay is a SECOND acquisition per response.
    """
    cfg = _config("push")
    result = run_simulation(cfg)
    assert result.blocking_gil_acquisitions == cfg.requests


def test_pull_adds_a_blocking_gil_acquisition_per_response():
    """``push_egress.rs:15`` -- pybridge.decode_response takes the GIL again on
    a spawn_blocking thread to depythonize each yielded object."""
    cfg = _config("pull")
    result = run_simulation(cfg)
    # invoke_generator once per request, decode_response once per response.
    assert result.blocking_gil_acquisitions == cfg.requests + result.responses


def test_push_removes_the_per_response_off_loop_acquisitions():
    """The mechanism the whole change exists for, stated as a ratio."""
    pull = run_simulation(_config("pull"))
    push = run_simulation(_config("push"))

    assert pull.responses == push.responses
    per_response_pull = pull.blocking_gil_acquisitions / pull.responses
    per_response_push = push.blocking_gil_acquisitions / push.responses
    assert per_response_pull > 1.0
    assert per_response_push < 0.5
    assert per_response_pull > per_response_push * 4


def test_blocking_acquisitions_really_happen_off_the_loop():
    """Not a bookkeeping counter: the GIL is genuinely taken cross-thread.

    ``Driver.spawn_blocking`` runs on a ThreadPoolExecutor, so the work lands
    on a thread that is neither the event loop nor the tokio stand-in.
    """
    import asyncio
    import threading

    from egress_experiments.dynamo_sim.rust_bridge import TokioRuntime

    tokio = TokioRuntime(blocking_threads=4)
    tokio.start()
    try:
        seen: list[str] = []

        async def probe() -> None:
            inner = asyncio.get_running_loop()
            for _ in range(8):
                await inner.run_in_executor(
                    tokio.blocking, lambda: seen.append(threading.current_thread().name)
                )

        tokio.submit(probe()).result(timeout=10)
        assert seen
        assert all(name.startswith("tokio-blocking") for name in seen), seen
        assert threading.current_thread().name not in seen
    finally:
        tokio.stop()
