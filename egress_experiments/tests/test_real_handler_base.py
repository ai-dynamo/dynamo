# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Drive the **real** ``AggregatedHandler`` with the mocked engine.

The rest of this package models the worker's response loop. This module skips
the model and plugs :class:`~egress_experiments.fake_trtllm.llm.FakeLLM`
straight into the shipped handler: ``config.engine.llm`` is the mock, so
``handler_base._generate_locally_impl`` runs unmodified and calls
``self.engine.llm.generate_async(...)`` for real.

That is the goal statement taken literally -- the dynamo TRT-LLM worker with
the TRT-LLM engine stubbed out at the worker/engine boundary, responses coming
back through ``handle_response`` on the IPC path.

Requires the container: ``handler_base`` imports ``torch``,
``tensorrt_llm.*`` and ``dynamo._core``. Skipped everywhere else, which is why
the model-based tests in the sibling modules exist.

    pytest egress_experiments/tests/test_real_handler_base.py -m "unit and trtllm"
"""

from __future__ import annotations

import asyncio
import importlib.util

import pytest

pytestmark = [
    pytest.mark.unit,
    pytest.mark.pre_merge,
    pytest.mark.gpu_0,
    pytest.mark.trtllm,
]

if importlib.util.find_spec("tensorrt_llm") is None:
    pytest.skip("tensorrt_llm not installed", allow_module_level=True)
if importlib.util.find_spec("dynamo") is None:  # pragma: no cover
    pytest.skip("dynamo not installed", allow_module_level=True)

from tensorrt_llm.llmapi.llm import SamplingParams  # noqa: E402

from dynamo.trtllm.constants import DisaggregationMode  # noqa: E402
from dynamo.trtllm.request_handlers.aggregated_handler import (  # noqa: E402
    AggregatedHandler,
)
from dynamo.trtllm.request_handlers.handler_base import (  # noqa: E402
    RequestHandlerConfig,
)
from egress_experiments.costs import Costs  # noqa: E402
from egress_experiments.dynamo_sim.rust_bridge import FakeContext  # noqa: E402
from egress_experiments.fake_trtllm.engine import (  # noqa: E402
    ConstantIteration,
    EngineConfig,
)
from egress_experiments.fake_trtllm.llm import FakeLLM  # noqa: E402

_FREE = Costs().with_scale(0.0)


class _EngineShim:
    """``TensorRTLLMEngine`` reduced to what the handler touches."""

    def __init__(self, llm: FakeLLM) -> None:
        self.llm = llm

    async def cleanup(self) -> None:  # only reached on a fatal error
        self.llm.shutdown()


def _handler(llm: FakeLLM) -> AggregatedHandler:
    return AggregatedHandler(
        RequestHandlerConfig(
            engine=_EngineShim(llm),
            default_sampling_params=SamplingParams(),
            publisher=None,
            disaggregation_mode=DisaggregationMode.AGGREGATED,
        )
    )


def _request(request_id: str = "req-0", max_tokens: int = 4) -> dict:
    return {
        "id": request_id,
        "token_ids": [11, 12, 13],
        "stop_conditions": {"max_tokens": max_tokens},
        "sampling_options": {},
        "stream": True,
    }


def test_real_handler_streams_deltas_from_the_mocked_engine():
    """``_generate_locally_impl`` runs unmodified against the mock."""

    async def main():
        loop = asyncio.get_running_loop()
        llm = FakeLLM(
            EngineConfig(iteration=ConstantIteration(4.0), max_tokens=4), costs=_FREE
        )
        llm.start(loop)
        try:
            handler = _handler(llm)
            chunks = [
                c async for c in handler.generate(_request(), FakeContext("req-0"))
            ]

            assert len(chunks) == 4
            # Per-choice cursor turned the engine's cumulative token_ids into
            # one-token deltas.
            assert all(len(c["token_ids"]) == 1 for c in chunks)
            assert chunks[-1]["finish_reason"] == "length"
            assert chunks[-1]["completion_usage"]["prompt_tokens"] == 3
            assert chunks[-1]["completion_usage"]["completion_tokens"] == 4
        finally:
            llm.shutdown()

    asyncio.run(main())


def test_real_handler_pushes_when_given_a_response_sender():
    """The real ``push_egress_capable`` on the real handler, push path."""

    class Sender:
        def __init__(self):
            self.calls = []
            self.items = []

        def send(self, obj):
            self.calls.append("send")
            self.items.append(obj)

        def close(self):
            self.calls.append("close")

        def close_with_error(self, message):
            self.calls.append("close_with_error")

    async def main():
        loop = asyncio.get_running_loop()
        llm = FakeLLM(
            EngineConfig(iteration=ConstantIteration(4.0), max_tokens=3), costs=_FREE
        )
        llm.start(loop)
        try:
            handler = _handler(llm)
            sender = Sender()
            stream = handler.generate(
                _request(max_tokens=3), FakeContext("req-0"), response_sender=sender
            )
            # Rust advances this ONCE per request.
            advances = 0
            anext = stream.__anext__
            while True:
                try:
                    await anext()
                except StopAsyncIteration:
                    break
                advances += 1

            assert advances == 0, "push mode must yield nothing"
            assert sender.calls == ["send"] * 3 + ["close"]
            assert all(len(i["token_ids"]) == 1 for i in sender.items)
        finally:
            llm.shutdown()

    asyncio.run(main())


def test_real_handler_handle_response_runs_on_the_loop():
    """Same claim as the model-based test, now through the shipped handler."""

    async def main():
        import threading

        loop = asyncio.get_running_loop()
        llm = FakeLLM(
            EngineConfig(iteration=ConstantIteration(4.0), max_tokens=3), costs=_FREE
        )
        llm.start(loop)

        # The handler drops its GenerationResult once the stream ends and the
        # proxy pops it from _results on is_final, so capture it on the way out.
        captured = []
        original = llm.generate_async

        def capturing(*args, **kwargs):
            result = original(*args, **kwargs)
            captured.append(result)
            return result

        llm.generate_async = capturing  # type: ignore[method-assign]

        try:
            handler = _handler(llm)
            async for _ in handler.generate(
                _request(max_tokens=3), FakeContext("req-0")
            ):
                pass

            assert llm.responses_dispatched == 3
            assert len(captured) == 1
            threads = set(captured[0].handle_response_threads)
            assert threads == {threading.current_thread().name}
            assert llm.dispatch_thread_name == "proxy_dispatch_result_thread"
            assert llm.dispatch_thread_name not in threads
        finally:
            llm.shutdown()

    asyncio.run(main())
