# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU preflight ownership, event-loop isolation, and cancellation settlement."""

import asyncio
import threading
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from tokenizers import Tokenizer as RustTokenizer
from tokenizers.models import WordLevel
from transformers import PreTrainedTokenizerFast

from dynamo.common.constants import DisaggregationMode
from dynamo.llm import ModelInput, ModelType, WorkerType
from dynamo.llm.exceptions import EngineShutdown
from dynamo.vllm import decision_native_errors as native_errors
from dynamo.vllm import decision_native_lifecycle as native_lifecycle
from dynamo.vllm import decision_native_registry as native_registry
from dynamo.vllm import decision_native_wire as native_wire
from dynamo.vllm.pooling_handlers import ClassifyWorkerHandler
from dynamo.vllm.tests.decision_head_test_support import (
    DecisionContext,
    Engine,
    NativeApi,
    adapter,
    monitor,
    wire,
)
from dynamo.vllm.tests.decision_read_contract_support import adapter as read_adapter
from dynamo.vllm.tests.decision_read_contract_support import wire as read_wire
from dynamo.vllm.tests.decision_read_test_support import Engine as ReadEngine

pytest_plugins = ["dynamo.vllm.tests.decision_pinned_fixtures"]

pytestmark = [
    pytest.mark.unit,
    pytest.mark.pre_merge,
    pytest.mark.vllm,
    pytest.mark.core,
    pytest.mark.gpu_0,
    pytest.mark.timeout(30),
]


@pytest.fixture
def native(monkeypatch):
    monkeypatch.setattr(native_lifecycle, "load_native", lambda: NativeApi())


@pytest.mark.asyncio
async def test_preflight_owns_an_independent_real_tokenizer(native):
    original = PreTrainedTokenizerFast(
        tokenizer_object=RustTokenizer(
            WordLevel({"hello": 0, "[UNK]": 1}, unk_token="[UNK]")
        )
    )
    engine = Engine()
    engine.renderer.get_tokenizer = lambda: original
    instance = await adapter(engine)
    private = instance.preflight_tokenizer
    assert private is not original
    assert private.backend_tokenizer is not original.backend_tokenizer
    assert private.encode("hello", add_special_tokens=False) == original.encode(
        "hello", add_special_tokens=False
    )
    private.add_tokens(["private"])
    assert original.convert_tokens_to_ids("private") is None
    original.encode = Mock(side_effect=AssertionError("ordinary tokenizer used"))
    await instance._prepare(wire())
    original.encode.assert_not_called()


@pytest.mark.asyncio
async def test_preflight_rejects_a_copied_wrapper_with_shared_backend(native):
    class AliasedBackend:
        def __init__(self, backend):
            self.backend_tokenizer = backend

        def __deepcopy__(self, memo):
            return AliasedBackend(self.backend_tokenizer)

    engine = Engine()
    tokenizer = AliasedBackend(object())
    engine.renderer.get_tokenizer = lambda: tokenizer
    with pytest.raises(native_errors.NativeRuntimeError, match="private tokenizer"):
        await adapter(engine)


@pytest.mark.asyncio
async def test_head_preflight_keeps_other_coroutines_runnable(native, monkeypatch):
    engine = Engine()
    instance = await adapter(engine)
    entered, release = threading.Event(), threading.Event()
    original = instance._prepare_text

    def blocking(request):
        entered.set()
        assert release.wait(3)
        return original(request)

    monkeypatch.setattr(instance, "_prepare_text", blocking)
    timer = threading.Timer(2, release.set)
    timer.start()
    task = asyncio.create_task(
        instance.execute(
            wire(), DecisionContext(), request_id="offloop", cancelled=lambda: False
        )
    )
    try:
        assert await asyncio.to_thread(entered.wait, 1)
        assert not release.is_set(), "preflight blocked event-loop progress"
        assert not task.done()
    finally:
        release.set()
        timer.cancel()
        await task


@pytest.mark.asyncio
async def test_started_preflight_drains_repeated_cancellation_before_unlock(native):
    instance = await adapter(Engine())
    entered, release = threading.Event(), threading.Event()
    second_entered = threading.Event()

    def blocked():
        entered.set()
        assert release.wait(3)

    first = asyncio.create_task(instance.run_preflight(blocked))
    assert await asyncio.to_thread(entered.wait, 1)
    first.cancel()
    await asyncio.sleep(0)
    first.cancel()
    second = asyncio.create_task(instance.run_preflight(second_entered.set))
    try:
        await asyncio.sleep(0)
        assert not first.done()
        assert not second_entered.is_set()
    finally:
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await first
        await second
    assert second_entered.is_set()


@pytest.mark.asyncio
async def test_queued_preflight_cancels_without_borrowing_the_guard(native):
    instance = await adapter(Engine())
    operation = Mock()
    await instance.preflight_lock.acquire()
    task = asyncio.create_task(instance.run_preflight(operation))
    await asyncio.sleep(0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    operation.assert_not_called()
    assert instance.preflight_lock.locked()
    instance.preflight_lock.release()


@pytest.mark.asyncio
@pytest.mark.parametrize("cancelled", [False, True])
async def test_failed_preflight_releases_its_guard_after_settlement(native, cancelled):
    instance = await adapter(Engine())
    entered, release = threading.Event(), threading.Event()

    def failure():
        entered.set()
        assert release.wait(3)
        raise ValueError("tokenizer failed")

    task = asyncio.create_task(instance.run_preflight(failure))
    assert await asyncio.to_thread(entered.wait, 1)
    if cancelled:
        task.cancel()
        await asyncio.sleep(0)
    release.set()
    expected = asyncio.CancelledError if cancelled else ValueError
    with pytest.raises(expected):
        await task
    assert not instance.preflight_lock.locked()


@pytest.mark.asyncio
async def test_context_stop_during_preflight_never_admits_inference(
    native, monkeypatch
):
    engine = Engine()
    instance = await adapter(engine)
    entered, release = threading.Event(), threading.Event()
    original = instance._prepare_text

    def blocking(request):
        entered.set()
        assert release.wait(3)
        return original(request)

    monkeypatch.setattr(instance, "_prepare_text", blocking)
    signal = asyncio.get_running_loop().create_future()
    context = SimpleNamespace(async_killed_or_stopped=lambda: signal)
    stopped = False
    task = asyncio.create_task(
        instance.execute(
            wire(), context, request_id="cancel-preflight", cancelled=lambda: stopped
        )
    )
    assert await asyncio.to_thread(entered.wait, 1)
    stopped = True
    signal.set_result(None)
    await asyncio.sleep(0)
    assert not task.done()
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert engine.admissions == []
    assert engine.calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["uncopyable", "alias"])
async def test_failed_private_tokenizer_clone_disables_only_optional_surface(
    native, failure
):
    attempts = []

    class Uncopyable:
        def __deepcopy__(self, memo):
            attempts.append(True)
            if failure == "uncopyable":
                raise TypeError("cannot isolate tokenizer")
            return self

    engine = Engine()
    engine.renderer.get_tokenizer = lambda: Uncopyable()
    config = SimpleNamespace(
        disaggregation_mode=DisaggregationMode.AGGREGATED,
        enable_rl=False,
        route_to_encoder=False,
        custom_encoder_class=None,
        enable_multimodal=False,
        realtime=False,
    )
    surface = await native_registry.initialize_native_decisions(
        "head",
        engine,
        engine.model_config,
        config,
        abort_monitor=monitor,
    )
    assert surface.spec is None
    assert surface.failed_profile is None
    runtime = SimpleNamespace(set_engine_specific=Mock())
    assert not native_registry.publish_native_decision_capability(
        runtime,
        surface.publication,
        ModelInput.Text,
        ModelType.Classify | ModelType.Pooling,
        WorkerType.Aggregated,
    )
    runtime.set_engine_specific.assert_not_called()
    assert surface.health_payload("ordinary") is None
    assert surface.maybe_handle({"input": "ordinary"}, object()) is None
    assert attempts == [True]
    handler = object.__new__(ClassifyWorkerHandler)
    handler.decision = surface
    handler.model_config = engine.model_config
    handler._default_pooling_task = "classify"
    handler._encode_batch = AsyncMock(
        return_value=[
            SimpleNamespace(
                outputs=SimpleNamespace(data=[0.7, 0.3]), prompt_token_ids=[1]
            )
        ]
    )
    result = [
        row
        async for row in handler.generate(
            {"model": "ordinary", "input": "ordinary"},
            SimpleNamespace(id=lambda: "ordinary"),
        )
    ]
    assert result[0]["data"][0]["probs"] == [0.7, 0.3]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "failure", [asyncio.CancelledError(), EngineShutdown("stopped")]
)
async def test_private_tokenizer_clone_preserves_cancellation_and_shutdown(
    native, failure
):
    class Interrupted:
        def __deepcopy__(self, memo):
            raise failure

    engine = Engine()
    engine.renderer.get_tokenizer = lambda: Interrupted()
    config = SimpleNamespace(
        disaggregation_mode=DisaggregationMode.AGGREGATED,
        enable_rl=False,
        route_to_encoder=False,
        custom_encoder_class=None,
        enable_multimodal=False,
        realtime=False,
    )
    with pytest.raises(type(failure)) as caught:
        await native_registry.initialize_native_decisions(
            "head", engine, engine.model_config, config, abort_monitor=monitor
        )
    assert caught.value is failure
    assert engine.calls == engine.aborted == []


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["head", "read"])
async def test_operator_derived_tokenization_budget_rejects_before_admission(
    native_reads, monkeypatch, kind
):
    if kind == "head":
        monkeypatch.setattr(native_lifecycle, "load_native", lambda: NativeApi())
        engine = Engine()
        instance = await adapter(engine)
        request = wire()
    else:
        engine = ReadEngine()
        instance = await read_adapter(engine)
        request = read_wire()
    payload = request[f"decision_{kind}"]
    payload["input"] = "x" * 2048
    payload["max_input_tokens"] = 32
    payload["max_tokenization_work"] = 32 * (instance.spec.max_candidates + 2)
    with pytest.raises(native_errors.NativeValidationError) as raised:
        await instance.execute(
            request, DecisionContext(), request_id="budget", cancelled=lambda: False
        )
    chain = []
    cause = raised.value
    while cause is not None:
        chain.append(str(cause))
        cause = cause.__cause__
    assert any("tokenization work budget" in text for text in chain)
    assert engine.calls == []
    assert engine.admissions == []


@pytest.mark.asyncio
async def test_read_labels_use_private_tokenizer_off_loop_but_render_stays_async(
    native_reads, monkeypatch
):
    engine = ReadEngine()
    instance = await read_adapter(engine)
    main = threading.get_ident()
    labels, renders = [], []
    original = native_wire.BudgetTokenizer.encode

    def encoded(budget, text, **kwargs):
        labels.append((threading.get_ident(), budget.tokenizer))
        return original(budget, text, **kwargs)

    render = instance.strategy._render

    async def rendered(*args, **kwargs):
        renders.append(threading.get_ident())
        return await render(*args, **kwargs)

    monkeypatch.setattr(native_wire.BudgetTokenizer, "encode", encoded)
    monkeypatch.setattr(instance.strategy, "_render", rendered)
    await instance._prepare(read_wire())
    assert labels and all(thread != main for thread, _ in labels)
    assert all(tokenizer is instance.preflight_tokenizer for _, tokenizer in labels)
    assert instance.preflight_tokenizer is not engine.renderer.get_tokenizer()
    assert renders and all(thread == main for thread in renders)
    assert engine.calls == []
