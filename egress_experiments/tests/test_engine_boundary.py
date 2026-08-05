# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The mocked worker/engine boundary.

The goal these tests encode: the dynamo TRT-LLM worker calls
``generate_async(...)``, the mocked engine eats it, and responses come back
through ``handle_response`` on the IPC (zmq) path.

Each test pins one property of that boundary that the rest of the simulation
depends on being true.
"""

from __future__ import annotations

import asyncio
import threading
import time

import pytest

from egress_experiments.costs import Costs
from egress_experiments.dynamo_sim.worker import SamplingParams, TrtllmWorkerHandler
from egress_experiments.fake_trtllm.engine import ConstantIteration, EngineConfig
from egress_experiments.fake_trtllm.llm import FakeLLM
from egress_experiments.fake_trtllm.result import GenerationResult

pytestmark = [
    pytest.mark.unit,
    pytest.mark.pre_merge,
    pytest.mark.gpu_0,
    pytest.mark.none,
]

_perf = time.perf_counter_ns

# Costs zeroed: these tests are about structure, not timing. The default
# 85 us/response would only make them slower.
_FREE = Costs().with_scale(0.0)


def _fast_engine(**kwargs) -> EngineConfig:
    iteration_ms = kwargs.pop("iteration_ms", 5.0)
    return EngineConfig(iteration=ConstantIteration(iteration_ms), **kwargs)


def test_generate_async_returns_immediately_and_the_engine_eats_it():
    """The worker hands the request over and gets a handle back, not tokens."""

    async def main():
        loop = asyncio.get_running_loop()
        llm = FakeLLM(_fast_engine(max_tokens=3), costs=_FREE)
        llm.start(loop)
        try:
            started = _perf()
            # Exactly the call handler_base._generate_locally_impl makes.
            result = llm.generate_async(
                inputs=[1, 2, 3],
                sampling_params=SamplingParams(max_tokens=3),
                disaggregated_params=None,
                streaming=True,
                trace_headers=None,
                scheduling_params=None,
                priority=0.5,
                cache_salt=None,
            )
            handed_over_us = (_perf() - started) / 1000

            assert isinstance(result, GenerationResult)
            # Submitted, not generated: no tokens exist yet and the call
            # returned in far less than one engine iteration (5 ms).
            assert result.outputs[0].token_ids == []
            assert not result.finished
            assert handed_over_us < 5000

            chunks = [res async for res in result]
            assert len(chunks) == 3
            assert len(result.outputs[0].token_ids) == 3
            assert result.finished
            assert result.outputs[0].finish_reason == "length"
        finally:
            llm.shutdown()

    asyncio.run(main())


def test_handle_response_runs_on_the_loop_not_the_dispatch_thread():
    """The diagram puts handle_response in the "ON THE ASYNCIO LOOP" box.

    That follows from ``result.py``: ``__anext__`` -> ``_aresult_step`` ->
    ``aqueue.get()`` then ``_handle_response``. The dispatch thread only does
    ``put_nowait``. If this ever inverted, the whole cost argument would move.
    """

    async def main():
        loop = asyncio.get_running_loop()
        llm = FakeLLM(_fast_engine(max_tokens=4), costs=_FREE)
        llm.start(loop)
        try:
            result = llm.generate_async(
                inputs=[1], sampling_params=SamplingParams(max_tokens=4)
            )
            async for _ in result:
                pass

            loop_thread = threading.current_thread().name
            assert result.handle_response_threads, "handle_response never ran"
            assert set(result.handle_response_threads) == {loop_thread}
            assert llm.dispatch_thread_name == "proxy_dispatch_result_thread"
            assert loop_thread != llm.dispatch_thread_name
        finally:
            llm.shutdown()

    asyncio.run(main())


def test_responses_cross_a_real_ipc_boundary():
    """Responses are produced in another process and deserialised off-loop.

    A same-process shortcut would let the engine's Python contend for the app
    interpreter's GIL, which is exactly the distortion this simulation must
    not have.
    """

    async def main():
        loop = asyncio.get_running_loop()
        llm = FakeLLM(_fast_engine(max_tokens=2), costs=_FREE)
        llm.start(loop)
        try:
            engine_pid = llm._engine.proc.pid
            assert engine_pid != 0
            import os

            assert engine_pid != os.getpid()

            result = llm.generate_async(
                inputs=[1], sampling_params=SamplingParams(max_tokens=2)
            )
            async for _ in result:
                pass

            assert llm.ipc_messages >= 2
            assert llm.responses_dispatched == 2
        finally:
            llm.shutdown()

    asyncio.run(main())


def test_result_registry_is_popped_on_the_final_response():
    """``proxy.py:565`` pops ``_results`` when ``is_final`` arrives.

    Without this the registry grows without bound and late responses raise
    KeyError instead of being dropped.
    """

    async def main():
        loop = asyncio.get_running_loop()
        llm = FakeLLM(_fast_engine(max_tokens=2), costs=_FREE)
        llm.start(loop)
        try:
            result = llm.generate_async(
                inputs=[1], sampling_params=SamplingParams(max_tokens=2)
            )
            assert result.client_id in llm._results
            async for _ in result:
                pass
            # Popped by the dispatch thread when it saw is_final.
            assert result.client_id not in llm._results
        finally:
            llm.shutdown()

    asyncio.run(main())


def test_worker_drives_the_boundary_end_to_end():
    """The handler's own ``async for`` over the GenerationResult.

    This is the boundary as ``handler_base`` sees it: cumulative token_ids in,
    per-chunk deltas out, ``completion_usage`` on the final chunk.
    """

    async def main():
        loop = asyncio.get_running_loop()
        llm = FakeLLM(_fast_engine(max_tokens=5), costs=_FREE)
        llm.start(loop)
        try:
            handler = TrtllmWorkerHandler(llm, costs=_FREE)
            request = {"id": "req-0", "token_ids": [7, 8, 9], "max_tokens": 5}

            class _Ctx:
                def id(self):
                    return "req-0"

            chunks = [c async for c in handler.generate(request, _Ctx())]

            assert len(chunks) == 5
            # Every chunk is a DELTA of exactly one token, not the cumulative
            # list the engine actually streams.
            assert all(len(c["token_ids"]) == 1 for c in chunks)
            assert all(c["index"] == 0 for c in chunks)
            assert "completion_usage" not in chunks[0]

            final = chunks[-1]
            assert final["finish_reason"] == "length"
            assert final["completion_usage"] == {
                "prompt_tokens": 3,
                "completion_tokens": 5,
                "total_tokens": 8,
                "prompt_tokens_details": None,
            }
        finally:
            llm.shutdown()

    asyncio.run(main())


def test_abort_surface_exists_for_the_cancellation_monitor():
    """``_cancellation_monitor`` calls ``generation_result.abort()``."""

    async def main():
        loop = asyncio.get_running_loop()
        llm = FakeLLM(_fast_engine(max_tokens=8), costs=_FREE)
        llm.start(loop)
        try:
            result = llm.generate_async(
                inputs=[1], sampling_params=SamplingParams(max_tokens=8)
            )
            assert not result.aborted
            result.abort()
            assert result.aborted
        finally:
            llm.shutdown()

    asyncio.run(main())
