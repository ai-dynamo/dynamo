# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the experimental synchronous in-process vLLM facade."""

import asyncio
import threading
from concurrent.futures import Future
from types import SimpleNamespace
from typing import Any

import pytest
from vllm.sampling_params import SamplingParams

usage_lib = pytest.importorskip("vllm.usage.usage_lib")
UsageContext = usage_lib.UsageContext

from dynamo.vllm import sync_inproc_engine as sync_engine  # noqa: E402

pytestmark = [
    pytest.mark.unit,
    pytest.mark.vllm,
    pytest.mark.pre_merge,
    pytest.mark.gpu_0,
]


class FakeRequestOutput(sync_engine.RequestOutput):
    def __init__(self, request_id: str, finished: bool) -> None:
        self.request_id = request_id
        self.finished = finished
        self.outputs = []

    def add(self, other: Any, aggregate: bool = False) -> None:
        del aggregate
        self.finished = other.finished


class FakeRenderer:
    def __init__(self, owner: "FakeLLMEngine") -> None:
        self.owner = owner

    def shutdown(self) -> None:
        self.owner.method_threads.append(threading.get_ident())

    def clear_mm_cache(self) -> None:
        self.owner.method_threads.append(threading.get_ident())


class FakeEngineCore:
    def __init__(self, owner: "FakeLLMEngine") -> None:
        self.owner = owner
        self.batch_queue = None

    def get_kv_cache_group_metadata(self) -> list[dict[str, Any]]:
        self.owner.method_threads.append(threading.get_ident())
        return [{"kind": "full_attention", "block_size": 16}]

    def pause_scheduler(self, *args, **kwargs) -> Future[None]:
        del args, kwargs
        self.owner.method_threads.append(threading.get_ident())
        result: Future[None] = Future()
        result.set_result(None)
        return result

    def resume_scheduler(self) -> None:
        self.owner.method_threads.append(threading.get_ident())

    def collective_rpc(self, *args):
        del args
        self.owner.method_threads.append(threading.get_ident())
        return []


class FakeEngineCoreClient:
    def __init__(self, owner: "FakeLLMEngine") -> None:
        self.owner = owner
        self.engine_core = FakeEngineCore(owner)

    def shutdown(self) -> None:
        self.owner.method_threads.append(threading.get_ident())


class FakeLLMEngine:
    latest: "FakeLLMEngine | None" = None
    block_first_step = False
    fail_step = False
    outputless_steps = 0
    steps_per_request = 1

    def __init__(self, **kwargs) -> None:
        del kwargs
        type(self).latest = self
        self.constructor_thread = threading.get_ident()
        self.method_threads: list[int] = []
        self.active: dict[str, int] = {}
        self.aborted: list[str] = []
        self.step_calls = 0
        self.active_counts: list[int] = []
        self.first_step_entered = threading.Event()
        self.release_first_step = threading.Event()
        self.tokenizer = SimpleNamespace(tokenizer=SimpleNamespace(bos_token_id=1))
        self.renderer = FakeRenderer(self)
        self.engine_core = FakeEngineCoreClient(self)

    def has_unfinished_requests(self) -> bool:
        return bool(self.active)

    def add_request(
        self,
        request_id: str,
        prompt: Any,
        params: Any,
        **kwargs: Any,
    ) -> str:
        del prompt, params, kwargs
        self.method_threads.append(threading.get_ident())
        self.active[request_id] = type(self).steps_per_request
        # vLLM 0.24 returns a wave-suffixed EngineCore ID while public
        # RequestOutputs and abort_request() retain the caller's request ID.
        return f"{request_id}-internal"

    def abort_request(self, request_ids: list[str]) -> None:
        self.method_threads.append(threading.get_ident())
        for request_id in request_ids:
            self.active.pop(request_id, None)
            self.aborted.append(request_id)

    def step(self) -> list[FakeRequestOutput]:
        self.method_threads.append(threading.get_ident())
        self.step_calls += 1
        self.active_counts.append(len(self.active))
        if type(self).block_first_step and self.step_calls == 1:
            self.first_step_entered.set()
            self.release_first_step.wait(timeout=2)
        if type(self).fail_step:
            raise RuntimeError("step failed")
        if type(self).outputless_steps > 0:
            type(self).outputless_steps -= 1
            return []

        outputs = []
        for request_id in list(self.active):
            remaining = self.active[request_id] - 1
            finished = remaining == 0
            outputs.append(FakeRequestOutput(request_id, finished))
            if finished:
                self.active.pop(request_id)
            else:
                self.active[request_id] = remaining
        return outputs

    def do_log_stats(self) -> None:
        self.method_threads.append(threading.get_ident())

    def reset_prefix_cache(self, *args: Any) -> bool:
        del args
        self.method_threads.append(threading.get_ident())
        return True

    def start_profile(self, profile_prefix: str | None = None) -> None:
        del profile_prefix
        self.method_threads.append(threading.get_ident())

    def stop_profile(self) -> None:
        self.method_threads.append(threading.get_ident())

    def sleep(self, level: int = 1, mode: str = "abort") -> None:
        del level, mode
        self.method_threads.append(threading.get_ident())

    def wake_up(self, tags: list[str] | None = None) -> None:
        del tags
        self.method_threads.append(threading.get_ident())


@pytest.fixture(autouse=True)
def fake_vllm(monkeypatch: pytest.MonkeyPatch) -> None:
    FakeLLMEngine.latest = None
    FakeLLMEngine.block_first_step = False
    FakeLLMEngine.fail_step = False
    FakeLLMEngine.outputless_steps = 0
    FakeLLMEngine.steps_per_request = 1
    monkeypatch.setattr(sync_engine, "LLMEngine", FakeLLMEngine)
    monkeypatch.setattr(sync_engine, "RequestOutput", FakeRequestOutput)
    monkeypatch.setattr(sync_engine.Executor, "get_class", lambda config: object)
    monkeypatch.setattr(sync_engine, "shutdown_prometheus", lambda: None)


def make_client() -> sync_engine.SyncInprocEngineClient:
    config = SimpleNamespace(model_config=SimpleNamespace())
    return sync_engine.SyncInprocEngineClient.from_vllm_config(
        config,
        usage_context=UsageContext.ENGINE_CONTEXT,
        disable_log_stats=True,
    )


async def collect(
    client: sync_engine.SyncInprocEngineClient,
    request_id: str,
) -> list[sync_engine.RequestOutput]:
    params = SamplingParams(max_tokens=2)
    return [
        output
        async for output in client.generate(
            {"prompt_token_ids": [1]},
            params,
            request_id,
        )
    ]


@pytest.mark.asyncio
async def test_outputless_step_keeps_engine_running() -> None:
    FakeLLMEngine.outputless_steps = 1
    client = make_client()
    try:
        outputs = await asyncio.wait_for(collect(client, "request-1"), timeout=2)
        assert outputs[-1].finished
        assert FakeLLMEngine.latest.step_calls == 2
    finally:
        client.shutdown()


@pytest.mark.asyncio
async def test_new_request_joins_between_steps() -> None:
    FakeLLMEngine.block_first_step = True
    FakeLLMEngine.steps_per_request = 2
    client = make_client()
    try:
        first = asyncio.create_task(collect(client, "request-1"))
        await asyncio.to_thread(FakeLLMEngine.latest.first_step_entered.wait, 2)
        second = asyncio.create_task(collect(client, "request-2"))
        FakeLLMEngine.latest.release_first_step.set()
        await asyncio.wait_for(asyncio.gather(first, second), timeout=2)
        assert 2 in FakeLLMEngine.latest.active_counts
    finally:
        client.shutdown()


@pytest.mark.asyncio
async def test_cancellation_aborts_active_request() -> None:
    FakeLLMEngine.block_first_step = True
    FakeLLMEngine.steps_per_request = 2
    client = make_client()
    task = asyncio.create_task(collect(client, "request-1"))
    await asyncio.to_thread(FakeLLMEngine.latest.first_step_entered.wait, 2)
    task.cancel()
    FakeLLMEngine.latest.release_first_step.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert "request-1" in FakeLLMEngine.latest.aborted
    client.shutdown()


@pytest.mark.asyncio
async def test_cancellation_before_add_is_admitted_then_aborted() -> None:
    FakeLLMEngine.block_first_step = True
    FakeLLMEngine.steps_per_request = 2
    client = make_client()
    try:
        first = asyncio.create_task(collect(client, "request-1"))
        await asyncio.to_thread(FakeLLMEngine.latest.first_step_entered.wait, 2)

        pending = asyncio.create_task(collect(client, "request-2"))
        for _ in range(100):
            await asyncio.sleep(0)
            if client._commands.qsize() == 1:
                break
        assert client._commands.qsize() == 1
        pending.cancel()
        for _ in range(100):
            await asyncio.sleep(0)
            if client._commands.qsize() == 2:
                break
        assert client._commands.qsize() == 2
        FakeLLMEngine.latest.release_first_step.set()

        with pytest.raises(asyncio.CancelledError):
            await pending
        await asyncio.wait_for(first, timeout=2)
        assert "request-2" in FakeLLMEngine.latest.aborted
    finally:
        client.shutdown()


@pytest.mark.asyncio
async def test_driver_failure_reaches_waiter() -> None:
    FakeLLMEngine.fail_step = True
    client = make_client()
    with pytest.raises(RuntimeError, match="step failed"):
        await asyncio.wait_for(collect(client, "request-1"), timeout=2)
    client.shutdown()


@pytest.mark.asyncio
async def test_admin_and_shutdown_stay_on_driver_thread() -> None:
    client = make_client()
    engine = FakeLLMEngine.latest
    await client.check_health()
    await client.reset_prefix_cache()
    await client.sleep(mode="keep")
    await client.pause_generation(mode="keep")
    await client.resume_generation()
    await client.collective_rpc("ping")
    metadata = await client.engine_core.call_utility_async(
        "get_kv_cache_group_metadata"
    )
    assert metadata[0]["block_size"] == 16
    client.shutdown()
    assert engine.method_threads
    assert set(engine.method_threads) == {engine.constructor_thread}


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("tensor_parallel_size", 2, "tensor parallelism"),
        ("pipeline_parallel_size", 2, "pipeline parallelism"),
        ("data_parallel_size", 2, "data parallelism"),
    ],
)
def test_validate_sync_inproc_rejects_parallelism(
    field: str,
    value: int,
    message: str,
) -> None:
    parallel = SimpleNamespace(
        tensor_parallel_size=1,
        pipeline_parallel_size=1,
        data_parallel_size=1,
    )
    setattr(parallel, field, value)
    engine_args = SimpleNamespace(enable_lora=False)
    config = SimpleNamespace(parallel_config=parallel)
    with pytest.raises(ValueError, match=message):
        sync_engine.validate_sync_inproc_config(engine_args, config)


def test_validate_sync_inproc_rejects_lora() -> None:
    parallel = SimpleNamespace(
        tensor_parallel_size=1,
        pipeline_parallel_size=1,
        data_parallel_size=1,
    )
    engine_args = SimpleNamespace(enable_lora=True)
    config = SimpleNamespace(parallel_config=parallel)
    with pytest.raises(ValueError, match="LoRA"):
        sync_engine.validate_sync_inproc_config(engine_args, config)
