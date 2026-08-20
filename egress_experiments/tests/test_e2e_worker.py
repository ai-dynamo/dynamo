# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import signal
import threading

import pytest

import egress_experiments.e2e_worker as e2e_worker
from egress_experiments.costs import Costs
from egress_experiments.e2e_worker import (
    RuntimeTrtllmWorkerHandler,
    StatsSampler,
    _interrupt_for_shutdown,
    _validate_push_runtime,
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


def test_response_path_costs_do_not_change_request_path_work():
    baseline = Costs()
    control = response_path_costs(0.0)

    assert control.handle_response_us == 0.0
    assert control.build_response_us == 0.0
    assert control.push_send_us == baseline.push_send_us
    assert control.prepare_request_us == baseline.prepare_request_us
    assert control.engine_submit_us == baseline.engine_submit_us


def test_validate_push_runtime_requires_push_mode(monkeypatch):
    monkeypatch.delenv("DYN_TRTLLM_PUSH_EGRESS", raising=False)

    with pytest.raises(RuntimeError, match="must be 1"):
        _validate_push_runtime()


def test_validate_push_runtime_requires_real_decorator(monkeypatch):
    monkeypatch.setenv("DYN_TRTLLM_PUSH_EGRESS", "1")
    monkeypatch.setattr(e2e_worker, "USING_REAL_PUSH_EGRESS", False)

    with pytest.raises(RuntimeError, match="real push_egress_capable"):
        _validate_push_runtime()


def test_sigterm_interrupts_worker_for_graceful_cleanup():
    with pytest.raises(KeyboardInterrupt):
        _interrupt_for_shutdown(signal.SIGTERM, None)


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
