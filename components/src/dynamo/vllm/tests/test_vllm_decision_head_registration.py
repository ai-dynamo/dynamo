# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Native surfaces are initialized and qualified before worker registration."""

import asyncio
import logging
from argparse import Namespace
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from dynamo.common.constants import DisaggregationMode
from dynamo.common.engine_monitor import ENGINE_HEALTH_CHECK_TIMEOUT
from dynamo.llm import ModelInput, ModelType, WorkerType
from dynamo.llm.exceptions import EngineShutdown, WorkerOverloaded
from dynamo.vllm import decision_heads as heads
from dynamo.vllm import decision_native_errors as native_errors
from dynamo.vllm import decision_native_lifecycle as native_lifecycle
from dynamo.vllm import decision_native_loader as native_loader
from dynamo.vllm import decision_native_registry as native_registry
from dynamo.vllm import decision_surface as surfaces
from dynamo.vllm.args import Config
from dynamo.vllm.health_check import VllmEmbeddingHealthCheckPayload
from dynamo.vllm.pooling_handlers import ClassifyWorkerHandler, EmbeddingWorkerHandler
from dynamo.vllm.tests.decision_head_test_support import Engine, NativeApi, monitor
from dynamo.vllm.tests.decision_pinned_source import load_pinned_engine_errors

pytestmark = [
    pytest.mark.unit,
    pytest.mark.pre_merge,
    pytest.mark.vllm,
    pytest.mark.core,
    pytest.mark.gpu_0,
]


@pytest.fixture
def config():
    return Config.from_cli_args(
        Namespace(
            disaggregation_mode=DisaggregationMode.AGGREGATED,
            enable_rl=False,
            route_to_encoder=False,
            custom_encoder_class=None,
            enable_multimodal=False,
            realtime=False,
        )
    )


@pytest.fixture
def handler(monkeypatch, config):
    def base(self, *, engine, config, **kwargs):
        self.engine_client, self.config = engine, config

    monkeypatch.setattr(EmbeddingWorkerHandler, "__init__", base)
    monkeypatch.setattr(native_lifecycle, "load_native", lambda: NativeApi())
    monkeypatch.delenv("DYN_HEALTH_CHECK_PAYLOAD", raising=False)
    instance = ClassifyWorkerHandler(None, Engine(), config)
    instance.model_config = instance.engine_client.model_config
    instance._abort_monitor = monitor
    return instance


async def initialize(handler):
    handler.decision = await native_registry.initialize_native_decisions(
        "head",
        handler.engine_client,
        handler.model_config,
        handler.config,
        abort_monitor=handler._abort_monitor,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "profile,architecture",
    [
        (native_registry.D3_PROFILE, "D3ForSequenceClassification"),
        (native_registry.CLEF_FLASH_PROFILE, "ClefFlashForJointSchema"),
        (native_registry.CLEF_OMNI_PROFILE, "ClefOmniForJointSchema"),
    ],
)
async def test_ready_head_exposes_typed_initialized_profile(
    handler, profile, architecture
):
    handler.model_config.architecture = architecture
    if profile != native_registry.D3_PROFILE:
        handler.engine_client.values = [0.7, 0.3]
    assert handler.decision.publication is None
    await initialize(handler)
    assert handler.decision.spec.name == profile
    assert handler.decision.failed_profile is None
    assert handler.decision.spec.max_candidates == (
        3 if profile == native_registry.D3_PROFILE else 26
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("architecture", ["OrdinarySequenceClassification", None])
async def test_unknown_architecture_does_not_probe_optional_runtime(
    monkeypatch, handler, architecture
):
    handler.model_config.architecture = architecture
    create = AsyncMock()
    monkeypatch.setattr(heads.HeadExecutionAdapter, "create", create)
    await initialize(handler)
    create.assert_not_awaited()
    assert handler.decision.publication is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "field,value",
    [
        ("disaggregation_mode", "prefill"),
        ("enable_rl", True),
        ("route_to_encoder", True),
        ("custom_encoder_class", "encoder.Custom"),
        ("enable_multimodal", True),
        ("realtime", True),
    ],
)
async def test_unsupported_worker_configuration_disables_optional_head(
    monkeypatch, handler, field, value
):
    setattr(handler.config, field, value)
    create = AsyncMock()
    monkeypatch.setattr(heads.HeadExecutionAdapter, "create", create)
    await initialize(handler)
    create.assert_not_awaited()
    assert handler.decision.publication is None


@pytest.mark.asyncio
async def test_ineligible_worker_disables_optional_head_without_exception_log(
    handler, caplog
):
    handler.config.enable_rl = True
    with caplog.at_level(logging.INFO):
        await initialize(handler)
    assert handler.decision.publication is None
    records = [r for r in caplog.records if "native" in r.getMessage()]
    assert records and all(r.exc_info is None for r in records)


@pytest.mark.asyncio
async def test_head_liveness_uses_adapter_without_pooling_callback(handler):
    surface = await native_registry.initialize_native_decisions(
        "head",
        handler.engine_client,
        handler.model_config,
        handler.config,
        abort_monitor=monitor,
    )
    assert surface.spec is not None
    assert surface.health_payload("model") is not None
    payload = VllmEmbeddingHealthCheckPayload(
        "model", default_payload=surface.health_payload("model")
    ).to_dict()
    assert payload["native_pooling_health"] is True
    assert len(handler.engine_client.calls) == 2


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "field,value", [("runner_type", "generate"), ("is_encoder_decoder", True)]
)
async def test_unsupported_model_configuration_disables_optional_head(
    handler, field, value
):
    setattr(handler.model_config, field, value)
    await initialize(handler)
    assert handler.decision.spec is None
    assert handler.decision.failed_profile is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "failure",
    [native_errors.NativeRuntimeError("source unavailable"), ValueError("adapter bug")],
)
async def test_initialization_failure_preserves_ordinary_classification(
    monkeypatch, handler, failure, caplog
):
    monkeypatch.setattr(
        heads.HeadExecutionAdapter, "create", AsyncMock(side_effect=failure)
    )
    with caplog.at_level(logging.INFO):
        await initialize(handler)
    assert handler.decision.spec is None
    assert handler.decision.failed_profile == (
        None
        if isinstance(failure, native_errors.NativeRuntimeError)
        else native_registry.PROFILES[native_registry.D3_PROFILE]
    )

    assert str(failure) in caplog.text
    handler._encode_batch = AsyncMock(
        return_value=[
            SimpleNamespace(
                outputs=SimpleNamespace(data=[0.7, 0.3]), prompt_token_ids=[1]
            )
        ]
    )
    response = [
        row
        async for row in handler.generate(
            {"input": "probe"}, SimpleNamespace(id=lambda: "ordinary")
        )
    ]
    assert response[0]["data"][0]["probs"] == [0.7, 0.3]


@pytest.mark.asyncio
async def test_initialization_cancellation_propagates(monkeypatch, handler):
    monkeypatch.setattr(
        heads.HeadExecutionAdapter,
        "create",
        AsyncMock(side_effect=asyncio.CancelledError),
    )
    with pytest.raises(asyncio.CancelledError):
        await initialize(handler)
    assert handler.decision.publication is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "failure",
    [
        native_errors.NativeValidationError("canary exceeds context"),
        native_errors.NativeRuntimeError("canary contract changed"),
        native_errors.NativeExecutionError("canary result invalid"),
        ValueError("canary bug"),
    ],
)
async def test_invalid_canary_disables_only_optional_surface(
    monkeypatch, handler, failure, caplog
):
    adapter = await heads.HeadExecutionAdapter.create(
        handler.engine_client,
        native_registry.PROFILES[native_registry.D3_PROFILE],
        abort_monitor=monitor,
    )
    monkeypatch.setattr(adapter, "_prepare", AsyncMock(side_effect=failure))
    monkeypatch.setattr(
        heads.HeadExecutionAdapter, "create", AsyncMock(return_value=adapter)
    )
    await initialize(handler)
    assert handler.decision.spec is None
    assert (
        handler.decision.failed_profile
        == native_registry.PROFILES[native_registry.D3_PROFILE]
    )
    assert handler.decision.health_payload("model") is None
    assert str(failure) in caplog.text


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "fault", [None, "nonfinite", "nonvector", "wrong-width", "unknown-cache"]
)
@pytest.mark.parametrize(
    "profile,architecture,task",
    [
        (native_registry.D3_PROFILE, "D3ForSequenceClassification", "classify"),
        (
            native_registry.CLEF_FLASH_PROFILE,
            "ClefFlashForJointSchema",
            "token_classify",
        ),
        (native_registry.CLEF_OMNI_PROFILE, "ClefOmniForJointSchema", "token_classify"),
    ],
)
async def test_pooling_liveness_is_independent_of_decision_result_validation(
    handler, profile, architecture, task, fault
):
    handler.model_config.architecture = architecture
    if task == "token_classify":
        handler.engine_client.values = [0.7, 0.3]
    await initialize(handler)
    surface = handler.decision
    execute = surface.adapter.execute
    surface.adapter.execute = AsyncMock(side_effect=AssertionError("not health"))
    decode = Mock(wraps=surface.adapter._decode)
    surface.adapter._decode = decode
    if fault == "nonfinite":
        handler.engine_client.values[0] = float("nan")
    elif fault == "wrong-width":
        handler.engine_client.values = []
    elif fault == "nonvector":

        def nonvector(output):
            output.outputs.data.ndim = 2
            return output

        handler.engine_client.mutate_output = nonvector
    elif fault == "unknown-cache":

        def unknown_cache(output):
            output.num_cached_tokens = None
            return output

        handler.engine_client.mutate_output = unknown_cache
    payload = VllmEmbeddingHealthCheckPayload(
        "model", default_payload=surface.health_payload("model")
    ).to_dict()
    context = surfaces.Context()
    response = [row async for row in handler.generate(payload, context)]
    assert response == [{"status": "ok"}]
    decode.assert_not_called()
    surface.adapter.execute.assert_not_awaited()
    params = handler.engine_client.calls[-1]["pooling_params"]
    assert params.task == task
    if task == "token_classify":
        assert params.extra_kwargs["joint_schema"]["option_counts"] == [2]
    if fault is not None:
        canary, _ = await surface.adapter.prepare_canary()
        with pytest.raises(native_errors.NativeExecutionError) as failure:
            await execute(
                canary,
                surfaces.Context(),
                request_id="strict-after-health",
                cancelled=lambda: False,
            )
        assert failure.value.code == 502
        decode.assert_called_once()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "profile,architecture",
    [
        (native_registry.D3_PROFILE, "D3ForSequenceClassification"),
        (native_registry.CLEF_FLASH_PROFILE, "ClefFlashForJointSchema"),
        (native_registry.CLEF_OMNI_PROFILE, "ClefOmniForJointSchema"),
    ],
)
async def test_failed_decision_canary_retains_only_model_valid_engine_liveness(
    monkeypatch, handler, profile, architecture
):
    handler.model_config.architecture = architecture
    if profile != native_registry.D3_PROFILE:
        handler.engine_client.values = [0.7, 0.3]
    adapter = await heads.HeadExecutionAdapter.create(
        handler.engine_client,
        native_registry.PROFILES[profile],
        abort_monitor=monitor,
    )
    decode = Mock(
        side_effect=native_errors.NativeExecutionError("invalid canary scores")
    )
    monkeypatch.setattr(adapter, "_decode", decode)
    monkeypatch.setattr(
        heads.HeadExecutionAdapter, "create", AsyncMock(return_value=adapter)
    )
    await initialize(handler)
    assert handler.decision.spec is None
    assert handler.decision.failed_profile == native_registry.PROFILES[profile]
    assert handler.decision.adapter is None
    decode.assert_called_once()
    assert handler.engine_client.active == 0
    payload = VllmEmbeddingHealthCheckPayload(
        "model", default_payload=handler.decision.health_payload("model")
    ).to_dict()
    assert payload["native_pooling_health"] is True
    context = surfaces.Context()
    assert [row async for row in handler.generate(payload, context)] == [
        {"status": "ok"}
    ]
    decode.assert_called_once()
    assert len(handler.engine_client.calls) == 3
    handler._encode_batch = AsyncMock(
        return_value=[
            SimpleNamespace(
                outputs=SimpleNamespace(data=[0.7, 0.3]), prompt_token_ids=[1]
            )
        ]
    )
    ordinary = [
        row
        async for row in handler.generate(
            {"input": "ordinary classify"}, SimpleNamespace(id=lambda: "ordinary")
        )
    ]
    assert ordinary[0]["data"][0]["probs"] == [0.7, 0.3]


@pytest.mark.asyncio
async def test_initial_liveness_timeout_drains_owned_engine_child(monkeypatch, handler):
    monkeypatch.setenv("DYN_ENGINE_HEALTH_CHECK_TIMEOUT", "0.01")
    handler.engine_client.block = asyncio.Event()
    await initialize(handler)
    assert handler.decision.spec is None
    assert handler.decision.failed_profile is None
    assert handler.decision.health_payload("model") is None
    assert handler.engine_client.active == 0
    assert len(handler.engine_client.aborted) == 1


@pytest.mark.asyncio
async def test_decision_canary_cancellation_drains_and_propagates(handler):
    handler.engine_client.block = asyncio.Event()
    pending = asyncio.create_task(initialize(handler))
    await handler.engine_client.started.wait()
    pending.cancel()
    with pytest.raises(asyncio.CancelledError):
        await pending
    assert handler.engine_client.active == 0
    assert len(handler.engine_client.aborted) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("configured", ["0", "inf", "nan"])
async def test_decision_canary_keeps_a_finite_timeout(monkeypatch, handler, configured):
    monkeypatch.setenv("DYN_ENGINE_HEALTH_CHECK_TIMEOUT", configured)
    wait_for = AsyncMock(wraps=asyncio.wait_for)
    monkeypatch.setattr(surfaces, "asyncio", SimpleNamespace(wait_for=wait_for))
    await initialize(handler)
    assert handler.decision.spec is not None
    assert wait_for.call_args.kwargs["timeout"] == ENGINE_HEALTH_CHECK_TIMEOUT


@pytest.mark.asyncio
@pytest.mark.parametrize("dead", [False, True])
async def test_decision_canary_engine_faults_preserve_shutdown_after_cleanup(
    monkeypatch, handler, dead
):
    errors = load_pinned_engine_errors()
    api = NativeApi()
    api.engine_generate_error = errors.EngineGenerateError
    api.engine_dead_error = errors.EngineDeadError
    monkeypatch.setattr(native_lifecycle, "load_native", lambda: api)
    cause = (api.engine_dead_error if dead else api.engine_generate_error)()

    async def fail(**kwargs):
        handler.engine_client.calls.append(kwargs)
        raise cause
        yield

    monkeypatch.setattr(handler.engine_client, "encode", fail)

    def shutdown():
        assert handler.engine_client.active == 0
        assert len(handler.engine_client.aborted) == 1

    callback = Mock(side_effect=shutdown)
    expected = pytest.raises(EngineShutdown) if dead else nullcontext()
    with expected as failure:
        surface = await native_registry.initialize_native_decisions(
            "head",
            handler.engine_client,
            handler.model_config,
            handler.config,
            abort_monitor=monitor,
            on_engine_dead=callback,
        )
    if dead:
        assert failure.value.__cause__ is cause
    else:
        assert surface.spec is None
        assert surface.failed_profile is None
        assert surface.health_payload("model") is None
    assert len(handler.engine_client.aborted) == 1
    assert callback.call_count == int(dead)


@pytest.mark.asyncio
@pytest.mark.parametrize("ineligible", ["version", "scheduler", "prefix-cache"])
async def test_pre_engine_ineligibility_never_publishes_a_failure(
    monkeypatch, handler, ineligible
):
    if ineligible == "version":
        monkeypatch.setattr(native_loader, "version", lambda _: "0.32.0")
        monkeypatch.setattr(native_lifecycle, "load_native", native_loader.load_native)
    elif ineligible == "scheduler":
        handler.engine_client.vllm_config.scheduler_config.scheduler_cls = (
            "dynamo.vllm.scheduler.InstrumentedScheduler"
        )
    else:
        handler.engine_client.vllm_config.cache_config.enable_prefix_caching = True
    await initialize(handler)
    surface = handler.decision
    assert surface.spec is None and surface.failed_profile is None
    assert handler.engine_client.calls == []
    runtime = SimpleNamespace(set_engine_specific=Mock())
    assert not native_registry.publish_native_decision_capability(
        runtime,
        surface.publication,
        ModelInput.Text,
        ModelType.Classify | ModelType.Pooling,
        WorkerType.Aggregated,
    )
    runtime.set_engine_specific.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("fault", ["generate", "retryable", "overload"])
@pytest.mark.parametrize("recovers", [True, False])
async def test_transient_canary_has_two_attempts_in_one_overall_deadline(
    monkeypatch, handler, fault, recovers
):
    errors = load_pinned_engine_errors()
    api = NativeApi()
    api.engine_generate_error = errors.EngineGenerateError
    api.retryable_request_error = errors.RetryableRequestError
    api.overload_errors = (errors.QueueOverflowError,)
    api.overload_error = WorkerOverloaded
    monkeypatch.setattr(native_lifecycle, "load_native", lambda: api)
    error = {
        "generate": errors.EngineGenerateError,
        "retryable": errors.RetryableRequestError,
        "overload": errors.QueueOverflowError,
    }[fault]
    original = handler.engine_client.encode
    attempts, closed = [], []

    async def encode(**kwargs):
        attempts.append(kwargs["request_id"])
        try:
            if len(attempts) > 1 and (len(attempts) == 2 or not recovers):
                raise error()
            async for output in original(**kwargs):
                yield output
        finally:
            closed.append(kwargs["request_id"])

    monkeypatch.setattr(handler.engine_client, "encode", encode)
    wait_for = AsyncMock(wraps=asyncio.wait_for)
    monkeypatch.setattr(surfaces, "asyncio", SimpleNamespace(wait_for=wait_for))
    await initialize(handler)
    assert (handler.decision.spec is not None) == recovers
    assert handler.decision.failed_profile is None
    assert handler.decision.health_payload("model") is not None
    assert len(attempts) == len(set(attempts)) == len(closed) == 3
    assert handler.engine_client.active == 0
    assert len(handler.engine_client.aborted) == (1 if recovers else 2)
    wait_for.assert_awaited_once()
    assert wait_for.call_args.kwargs["timeout"] == ENGINE_HEALTH_CHECK_TIMEOUT


@pytest.mark.asyncio
@pytest.mark.parametrize("second_fault", ["contract", "timeout"])
async def test_canary_retry_preserves_contract_and_outer_timeout_classification(
    monkeypatch, handler, second_fault
):
    errors = load_pinned_engine_errors()
    api = NativeApi()
    api.retryable_request_error = errors.RetryableRequestError
    monkeypatch.setattr(native_lifecycle, "load_native", lambda: api)
    if second_fault == "timeout":
        monkeypatch.setenv("DYN_ENGINE_HEALTH_CHECK_TIMEOUT", "0.01")
    else:
        handler.engine_client.values = []
    original = handler.engine_client.encode
    attempts = []

    async def encode(**kwargs):
        attempts.append(kwargs["request_id"])
        if len(attempts) == 2:
            raise errors.RetryableRequestError()
        if len(attempts) == 3 and second_fault == "timeout":
            handler.engine_client.block = asyncio.Event()
        async for output in original(**kwargs):
            yield output

    monkeypatch.setattr(handler.engine_client, "encode", encode)
    await initialize(handler)
    assert handler.decision.spec is None
    assert handler.decision.failed_profile == (
        native_registry.PROFILES[native_registry.D3_PROFILE]
        if second_fault == "contract"
        else None
    )
    assert handler.decision.pooling_probe is not None
    assert len(attempts) == 3
    assert len(handler.engine_client.aborted) == 2
    assert handler.engine_client.active == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["head", "read"])
@pytest.mark.parametrize("fault", ["dead", "retryable"])
async def test_cold_supported_tasks_errors_preserve_engine_classification(
    monkeypatch, handler, kind, fault
):
    errors = load_pinned_engine_errors()
    api = NativeApi()
    api.engine_dead_error = errors.EngineDeadError
    api.retryable_request_error = errors.RetryableRequestError
    monkeypatch.setattr(native_lifecycle, "load_native", lambda: api)
    profile = native_registry.PROFILES[
        native_registry.D3_PROFILE if kind == "head" else native_registry.LIQUID_PROFILE
    ]
    handler.model_config.architecture = profile.architecture
    cause = (
        errors.EngineDeadError if fault == "dead" else errors.RetryableRequestError
    )()
    supported = AsyncMock(side_effect=cause)
    monkeypatch.setattr(handler.engine_client, "get_supported_tasks", supported)
    callback = Mock()
    expected = pytest.raises(EngineShutdown) if fault == "dead" else nullcontext()
    with expected as failure:
        surface = await native_registry.initialize_native_decisions(
            kind,
            handler.engine_client,
            handler.model_config,
            handler.config,
            abort_monitor=monitor,
            on_engine_dead=callback,
        )
    if fault == "dead":
        assert failure.value.__cause__ is cause
    else:
        assert surface.spec is None and surface.failed_profile is None
    supported.assert_awaited_once()
    assert callback.call_count == int(fault == "dead")
    assert handler.engine_client.calls == handler.engine_client.aborted == []


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["head", "read"])
async def test_optional_startup_deadline_bounds_cold_supported_tasks(
    monkeypatch, handler, kind
):
    monkeypatch.setenv("DYN_ENGINE_HEALTH_CHECK_TIMEOUT", "0.01")
    handler.model_config.architecture = native_registry.PROFILES[
        native_registry.D3_PROFILE if kind == "head" else native_registry.LIQUID_PROFILE
    ].architecture
    drained = asyncio.Event()

    async def blocked():
        try:
            await asyncio.Event().wait()
        finally:
            drained.set()

    monkeypatch.setattr(handler.engine_client, "get_supported_tasks", blocked)
    surface = await asyncio.wait_for(
        native_registry.initialize_native_decisions(
            kind,
            handler.engine_client,
            handler.model_config,
            handler.config,
            abort_monitor=monitor,
        ),
        timeout=1,
    )
    assert surface.spec is None and surface.failed_profile is None
    assert surface.pooling_probe is None
    assert drained.is_set()
    assert handler.engine_client.calls == handler.engine_client.aborted == []


@pytest.mark.asyncio
async def test_cold_query_and_canary_share_one_remaining_deadline(monkeypatch, handler):
    monkeypatch.setenv("DYN_ENGINE_HEALTH_CHECK_TIMEOUT", "0.2")
    original_tasks = handler.engine_client.get_supported_tasks
    entered = asyncio.Event()
    remaining = []
    deadline = None

    async def timed(operation, *, timeout):
        nonlocal deadline
        assert timeout == 0.2
        deadline = asyncio.get_running_loop().time() + timeout
        entered.set()
        return await asyncio.wait_for(operation, timeout=timeout)

    async def tasks():
        assert (
            entered.is_set()
        ), "cold utility query must be inside the startup deadline"
        await asyncio.sleep(0.01)
        remaining.append(deadline - asyncio.get_running_loop().time())
        return await original_tasks()

    wait_for = AsyncMock(side_effect=timed)
    monkeypatch.setattr(surfaces, "asyncio", SimpleNamespace(wait_for=wait_for))
    monkeypatch.setattr(handler.engine_client, "get_supported_tasks", tasks)
    handler.engine_client.block = asyncio.Event()
    await initialize(handler)
    wait_for.assert_awaited_once()
    assert len(remaining) == 1 and 0 < remaining[0] < 0.2
    assert handler.decision.spec is None and handler.decision.failed_profile is None
    assert handler.decision.pooling_probe is None
    assert handler.engine_client.active == 0
    assert len(handler.engine_client.aborted) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "extra", [{"decision_head": {}}, {"native_pooling_health": True}]
)
async def test_public_classify_controls_cannot_invoke_native_surface(handler, extra):
    await initialize(handler)
    handler.decision.adapter.execute = AsyncMock(
        side_effect=AssertionError("not native")
    )
    handler._encode_batch = AsyncMock(
        return_value=[
            SimpleNamespace(
                outputs=SimpleNamespace(data=[0.25, 0.75]),
                prompt_token_ids=[1, 2],
            )
        ]
    )
    handler._id2label = lambda: {0: "no", 1: "yes"}
    context = SimpleNamespace(id=lambda: "ordinary", metadata={})
    request = {"model": "test", "input": "ordinary classify", **extra}
    rows = [row async for row in handler.generate(request, context)]
    assert rows[0]["data"][0]["probs"] == [0.25, 0.75]
    handler.decision.adapter.execute.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("fault", ["generation", "validation", "engine-generate"])
async def test_rejected_native_probe_does_not_poison_ordinary_classification(
    monkeypatch, handler, fault
):
    errors = load_pinned_engine_errors()
    api = NativeApi()
    api.validation_error = errors.VLLMValidationError
    api.engine_generate_error = errors.EngineGenerateError
    monkeypatch.setattr(native_lifecycle, "load_native", lambda: api)
    cause = {
        "generation": errors.GenerationError(),
        "validation": errors.VLLMValidationError("prepared read rejected"),
        "engine-generate": errors.EngineGenerateError(),
    }[fault]

    async def encode(**kwargs):
        handler.engine_client.calls.append(kwargs)
        if getattr(kwargs["pooling_params"], "use_activation", None) is False:
            raise cause
        yield SimpleNamespace(
            outputs=SimpleNamespace(data=[0.7, 0.3]),
            prompt_token_ids=[1],
            finished=True,
        )

    monkeypatch.setattr(handler.engine_client, "encode", encode)
    await initialize(handler)
    surface = handler.decision
    assert surface.spec is None and surface.health_payload("model") is None
    assert (surface.failed_profile is None) == (fault == "engine-generate")
    assert len(handler.engine_client.calls) == len(handler.engine_client.aborted) == 1
    payload = VllmEmbeddingHealthCheckPayload(
        "model", default_payload=surface.health_payload("model")
    ).to_dict()
    assert "native_pooling_health" not in payload
    rows = [
        row
        async for row in handler.generate(
            {"input": "ordinary classify"}, SimpleNamespace(id=lambda: "ordinary")
        )
    ]
    assert rows[0]["data"][0]["probs"] == [0.7, 0.3]


@pytest.mark.asyncio
@pytest.mark.parametrize("stage", ["create", "prepare", "empty-preparation"])
async def test_runtime_ineligibility_is_limited_to_adapter_construction(
    monkeypatch, handler, stage, caplog
):
    adapter = await heads.HeadExecutionAdapter.create(
        handler.engine_client,
        native_registry.PROFILES[native_registry.D3_PROFILE],
        abort_monitor=monitor,
    )
    failure = native_errors.NativeRuntimeError("stage-specific runtime failure")
    create = AsyncMock(
        return_value=adapter, side_effect=failure if stage == "create" else None
    )
    monkeypatch.setattr(heads.HeadExecutionAdapter, "create", create)
    if stage == "prepare":
        monkeypatch.setattr(adapter, "_prepare", AsyncMock(side_effect=failure))
    elif stage == "empty-preparation":
        monkeypatch.setattr(
            adapter, "_prepare", AsyncMock(return_value=SimpleNamespace(reads=()))
        )
    await initialize(handler)
    assert (handler.decision.failed_profile is None) == (stage == "create")
    assert handler.decision.spec is handler.decision.pooling_probe is None
    assert handler.engine_client.calls == []
    if stage == "empty-preparation":
        assert isinstance(
            caplog.records[-1].exc_info[1], native_errors.NativeExecutionError
        )
