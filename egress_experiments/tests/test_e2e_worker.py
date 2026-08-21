# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import signal
import sys
import threading
from types import SimpleNamespace

import pytest

import egress_experiments.e2e_worker as e2e_worker
from egress_experiments.costs import Costs
from egress_experiments.dynamo_sim.probes import LoopProbe
from egress_experiments.e2e_worker import (
    RuntimeRustEgressWorkerHandler,
    RuntimeTrtllmWorkerHandler,
    StatsSampler,
    _interrupt_for_shutdown,
    _validate_push_runtime,
    _validate_rust_runtime,
    adapt_runtime_request,
    parse_args,
    response_path_costs,
)
from egress_experiments.fake_trtllm.engine import (
    BatchConfig,
    ConstantIteration,
    EngineConfig,
)
from egress_experiments.fake_trtllm.llm import FakeLLM
from egress_experiments.fake_trtllm.llm import _build_native_event
from egress_experiments.fake_trtllm.result import Response, ResultPayload

pytestmark = [
    pytest.mark.unit,
    pytest.mark.pre_merge,
    pytest.mark.gpu_0,
    pytest.mark.none,
]


def test_adapt_runtime_request_maps_frontend_fields_without_mutating_input():
    request = {
        "token_ids": [1, 2, 3],
        "stop_conditions": {"max_tokens": 7},
        "sampling_options": {"n": 2},
    }

    adapted = adapt_runtime_request(request, request_id="req-7")

    assert adapted == {
        **request,
        "id": "req-7",
        "max_tokens": 7,
        "n": 2,
    }
    assert request == {
        "token_ids": [1, 2, 3],
        "stop_conditions": {"max_tokens": 7},
        "sampling_options": {"n": 2},
    }


def test_adapt_runtime_request_defaults_nullable_frontend_fields():
    adapted = adapt_runtime_request(
        {
            "token_ids": [1],
            "stop_conditions": {"max_tokens": None},
            "sampling_options": {"n": None},
        },
        request_id="req-nullable",
        default_max_tokens=19,
    )

    assert adapted["max_tokens"] == 19
    assert adapted["n"] == 1


@pytest.mark.parametrize("value", ["-1", "nan", "inf"])
def test_parse_args_rejects_invalid_cost_scale(value):
    with pytest.raises(SystemExit):
        parse_args(["--response-cost-scale", value])


@pytest.mark.parametrize("value", ["0", "nan", "inf"])
def test_parse_args_rejects_invalid_positive_float(value):
    with pytest.raises(SystemExit):
        parse_args(["--iteration-ms", value])


def test_parse_args_selects_python_or_rust_response_path():
    assert parse_args([]).response_path == "python"
    assert parse_args(["--response-path", "rust"]).response_path == "rust"


def test_response_path_costs_do_not_change_request_path_work():
    baseline = Costs()
    control = response_path_costs(0.0)

    assert control.handle_response_us == 0.0
    assert control.build_response_us == 0.0
    assert control.push_send_us == baseline.push_send_us
    assert control.prepare_request_us == baseline.prepare_request_us
    assert control.engine_submit_us == baseline.engine_submit_us


def test_loop_probe_reset_starts_a_fresh_measurement_window():
    probe = LoopProbe()
    probe.lag.add(1_000_000, probe.cap)
    probe.callbacks["response"].add(2_000_000, probe.cap)
    probe.enqueues = 3

    probe.reset()

    assert probe.report()["lag"]["n"] == 0
    assert probe.report()["callbacks"] == {}
    assert probe.report()["enqueues"] == 0


def test_native_event_rejects_a_stale_engine_generation():
    result = SimpleNamespace(
        response_request_key=SimpleNamespace(generation=2), response_sequence=0
    )
    response = Response(
        client_id=7,
        generation=1,
        result=ResultPayload(new_token_ids=[[11]]),
    )

    assert _build_native_event(result, response) is None
    assert result.response_sequence == 0


def test_validate_push_runtime_requires_real_decorator(monkeypatch):
    monkeypatch.setattr(e2e_worker, "USING_REAL_PUSH_EGRESS", False)

    with pytest.raises(RuntimeError, match="real push_egress_capable"):
        _validate_push_runtime()


def test_validate_rust_runtime_requires_native_processor_binding():
    class CoreWithoutNativeProcessor:
        pass

    with pytest.raises(RuntimeError, match="NativeResponseEgress"):
        _validate_rust_runtime(CoreWithoutNativeProcessor)


def test_sigterm_interrupts_worker_for_graceful_cleanup():
    with pytest.raises(KeyboardInterrupt):
        _interrupt_for_shutdown(signal.SIGTERM, None)


def test_main_treats_sigterm_interrupt_as_clean_shutdown(monkeypatch):
    def run(coroutine):
        coroutine.close()
        raise KeyboardInterrupt

    monkeypatch.setitem(sys.modules, "uvloop", SimpleNamespace(run=run))
    monkeypatch.setattr(signal, "signal", lambda *_args: signal.SIG_DFL)

    e2e_worker.main([])


def test_stats_sampler_reports_rates_batching_and_backlog():
    sampler = StatsSampler(start_ns=1_000_000_000)

    first = sampler.sample(
        now_ns=2_000_000_000,
        responses_dispatched=120,
        responses_yielded=100,
        notify_many_calls=4,
        ipc_batch_sizes=[20, 30, 30, 40],
    )
    second = sampler.sample(
        now_ns=3_000_000_000,
        responses_dispatched=320,
        responses_yielded=280,
        notify_many_calls=9,
        ipc_batch_sizes=[20, 30, 30, 40, 40, 40, 40, 40, 40],
    )

    assert first["response_rate"] == pytest.approx(120.0)
    assert first["yield_rate"] == pytest.approx(100.0)
    assert first["loop_backlog"] == 20
    assert first["responses_per_notify"] == pytest.approx(30.0)
    assert second["response_rate"] == pytest.approx(200.0)
    assert second["yield_rate"] == pytest.approx(180.0)
    assert second["loop_backlog"] == 40
    assert second["mean_ipc_batch"] == pytest.approx(40.0)


def test_runtime_handler_runs_response_work_on_one_loop_and_preserves_output():
    async def run() -> tuple[list[dict], FakeLLM, RuntimeTrtllmWorkerHandler]:
        loop = asyncio.get_running_loop()
        engine_config = EngineConfig(
            batch=BatchConfig(total=2),
            iteration=ConstantIteration(1.0),
            max_tokens=2,
        )
        llm = FakeLLM(engine_config, costs=Costs().with_scale(0.0))
        llm.start(loop)
        handler = RuntimeTrtllmWorkerHandler(llm, costs=Costs().with_scale(0.0))
        try:
            stream = handler.generate(
                {
                    "token_ids": [11, 12],
                    "stop_conditions": {"max_tokens": 2},
                    "sampling_options": {"n": 1},
                },
                context=None,
            )
            chunks = [chunk async for chunk in stream]
            return chunks, llm, handler
        finally:
            llm.shutdown()

    chunks, llm, handler = asyncio.run(run())

    assert len(chunks) == 2
    assert all(len(chunk["token_ids"]) == 1 for chunk in chunks)
    assert chunks[-1]["finish_reason"] == "length"
    assert chunks[-1]["completion_usage"] == {
        "prompt_tokens": 2,
        "completion_tokens": 2,
        "total_tokens": 4,
        "prompt_tokens_details": None,
    }
    assert handler.responses_yielded == llm.responses_dispatched == 2
    assert llm.notify_many_calls == 2
    result_threads = {
        thread_name
        for result in handler.completed_results
        for thread_name in result.handle_response_threads
    }
    assert result_threads == {threading.current_thread().name}


def test_rust_handler_processes_batches_off_loop_and_wakes_once_per_request():
    class RecordingProcessor:
        def __init__(self):
            self.registrations = []
            self.batches = []
            self.frames_sent = 0
            self.responses_processed = 0

        def register(
            self,
            client_id,
            prompt_tokens,
            num_choices,
            response_sender,
        ):
            key = SimpleNamespace(client_id=client_id, generation=1)
            self.registrations.append(
                (
                    client_id,
                    prompt_tokens,
                    num_choices,
                    response_sender,
                )
            )
            return key

        def process_batch(self, responses):
            self.batches.append(responses)
            self.responses_processed += len(responses)
            self.frames_sent += sum(len(response["outputs"]) for response in responses)
            return [
                SimpleNamespace(
                    client_id=response["client_id"],
                    generation=response["generation"],
                )
                for response in responses
                if response["is_final"] or response.get("error_msg")
            ]

        def cancel(self, _request_key):
            return False

    async def run():
        loop = asyncio.get_running_loop()
        engine_config = EngineConfig(
            batch=BatchConfig(total=1),
            iteration=ConstantIteration(1.0),
            max_tokens=3,
        )
        llm = FakeLLM(engine_config, costs=Costs().with_scale(0.0))
        processor = RecordingProcessor()
        handler = RuntimeRustEgressWorkerHandler(
            llm, processor=processor, costs=Costs().with_scale(0.0)
        )
        sender = object()
        llm.start(loop)
        try:
            chunks = [
                chunk
                async for chunk in handler.generate(
                    {
                        "token_ids": [11, 12],
                        "stop_conditions": {"max_tokens": 3},
                        "sampling_options": {"n": 1},
                    },
                    context=None,
                    response_sender=sender,
                )
            ]
            return chunks, llm, handler, processor, sender
        finally:
            llm.shutdown()

    chunks, llm, handler, processor, sender = asyncio.run(run())

    assert chunks == []
    assert len(processor.registrations) == 1
    assert processor.registrations[0][3] is sender
    assert [event["sequence"] for batch in processor.batches for event in batch] == [
        0,
        1,
        2,
    ]
    assert all(
        event["generation"] == 1
        for batch in processor.batches
        for event in batch
    )
    assert processor.responses_processed == llm.responses_dispatched == 3
    assert handler.responses_yielded == processor.frames_sent == 3
    assert llm.notify_many_calls == 0
    assert llm.native_completion_notify_calls == 1
    result_threads = {
        thread_name
        for result in handler.completed_results
        for thread_name in result.handle_response_threads
    }
    assert result_threads == set()


def test_rust_handler_propagates_native_batch_failure_without_hanging():
    class FailingProcessor:
        frames_sent = 0

        def register(self, client_id, *_args):
            return SimpleNamespace(client_id=client_id, generation=1)

        def process_batch(self, _responses):
            raise ValueError("malformed native response")

        def cancel(self, _request_key):
            return True

    class RecordingSender:
        def __init__(self):
            self.errors = []

        def close_with_error(self, message):
            self.errors.append(message)

    async def run():
        loop = asyncio.get_running_loop()
        llm = FakeLLM(
            EngineConfig(
                batch=BatchConfig(total=1),
                iteration=ConstantIteration(1.0),
                max_tokens=1,
            ),
            costs=Costs().with_scale(0.0),
        )
        processor = FailingProcessor()
        handler = RuntimeRustEgressWorkerHandler(llm, processor=processor)
        sender = RecordingSender()
        llm.start(loop)
        try:
            stream = handler.generate(
                {
                    "token_ids": [11],
                    "stop_conditions": {"max_tokens": 1},
                    "sampling_options": {"n": 1},
                },
                context=None,
                response_sender=sender,
            )
            with pytest.raises(RuntimeError, match="malformed native response"):
                await asyncio.wait_for(anext(stream), timeout=1.0)
            return sender
        finally:
            llm.shutdown()

    sender = asyncio.run(run())
    assert sender.errors == ["native response processing failed: malformed native response"]


def test_native_shutdown_resolves_inflight_registration():
    class RecordingProcessor:
        def __init__(self):
            self.cancelled = []

        def register(self, client_id, *_args):
            return SimpleNamespace(client_id=client_id, generation=1)

        def cancel(self, request_key):
            self.cancelled.append(request_key)
            return True

    class RecordingSender:
        def __init__(self):
            self.errors = []

        def close_with_error(self, message):
            self.errors.append(message)

    async def run():
        loop = asyncio.get_running_loop()
        llm = FakeLLM(
            EngineConfig(
                batch=BatchConfig(total=1),
                iteration=ConstantIteration(100.0),
                max_tokens=100,
            ),
            costs=Costs().with_scale(0.0),
        )
        processor = RecordingProcessor()
        sender = RecordingSender()
        llm.start(loop)
        result = llm.generate_async(
            sampling_params=SimpleNamespace(max_tokens=100, n=1),
            response_processor=processor,
            response_sender=sender,
        )
        llm.shutdown()
        with pytest.raises(RuntimeError, match="shut down"):
            await asyncio.wait_for(result.wait_native(), timeout=0.1)
        return llm, processor, sender

    llm, processor, sender = asyncio.run(run())
    assert len(processor.cancelled) == 1
    assert sender.errors == ["native response processing stopped: worker shut down"]
    assert llm._results == {}


def test_rust_handler_cancellation_removes_native_request_state():
    class RecordingProcessor:
        def __init__(self):
            self.registered = []
            self.cancelled = []
            self.frames_sent = 0

        def register(
            self,
            client_id,
            prompt_tokens,
            num_choices,
            response_sender,
        ):
            key = SimpleNamespace(client_id=client_id, generation=1)
            self.registered.append(key)
            return key

        def process_batch(self, _responses):
            return []

        def cancel(self, request_key):
            self.cancelled.append(request_key)
            return True

    async def run():
        loop = asyncio.get_running_loop()
        llm = FakeLLM(
            EngineConfig(
                batch=BatchConfig(total=1),
                iteration=ConstantIteration(50.0),
                max_tokens=100,
            ),
            costs=Costs().with_scale(0.0),
        )
        processor = RecordingProcessor()
        handler = RuntimeRustEgressWorkerHandler(llm, processor=processor)
        llm.start(loop)
        try:
            stream = handler.generate(
                {
                    "token_ids": [11],
                    "stop_conditions": {"max_tokens": 100},
                    "sampling_options": {"n": 1},
                },
                context=None,
                response_sender=object(),
            )
            pending = asyncio.create_task(anext(stream))
            while not processor.registered:
                await asyncio.sleep(0)
            pending.cancel()
            with pytest.raises(asyncio.CancelledError):
                await pending

            request_key = processor.registered[0]
            return request_key, processor.cancelled, dict(llm._results)
        finally:
            llm.shutdown()

    request_key, cancelled, remaining_results = asyncio.run(run())

    assert cancelled == [request_key]
    assert remaining_results == {}
