# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared runtime policy rejects the same unsafe metadata across all executors."""

import copy
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from dynamo.common.constants import DisaggregationMode
from dynamo.llm import ModelInput, WorkerType
from dynamo.vllm import decision_execution as causal
from dynamo.vllm import decision_native_errors as native_errors
from dynamo.vllm import decision_native_lifecycle as native_lifecycle
from dynamo.vllm import decision_native_registry as native_registry
from dynamo.vllm import decision_runtime_policy as runtime_policy
from dynamo.vllm.decision_heads import HeadExecutionAdapter
from dynamo.vllm.decision_reads import ReadExecutionAdapter
from dynamo.vllm.tests.decision_head_test_support import Engine as HeadEngine
from dynamo.vllm.tests.decision_head_test_support import NativeApi, monitor
from dynamo.vllm.tests.decision_read_test_support import Engine as ReadEngine

pytest_plugins = ["dynamo.vllm.tests.decision_pinned_fixtures"]

pytestmark = [
    pytest.mark.unit,
    pytest.mark.pre_merge,
    pytest.mark.vllm,
    pytest.mark.core,
    pytest.mark.gpu_0,
]


@pytest.mark.asyncio
@pytest.mark.parametrize("async_mode", [False, True])
@pytest.mark.parametrize(
    "scheduler",
    [
        None,
        "vllm.v1.core.sched.scheduler.Scheduler",
        "vllm.v1.core.sched.async_scheduler.AsyncScheduler",
        "custom.Scheduler",
    ],
)
async def test_pinned_scheduler_policy_is_shared_by_all_executors(
    monkeypatch, native_reads, scheduler, async_mode
):
    monkeypatch.setattr(causal, "version", lambda name: "0.33.0")
    monkeypatch.setattr(causal, "_decision_api_available", lambda: True)
    monkeypatch.setattr(causal, "qualified_cache_runtime", lambda _: True)
    read_api = native_lifecycle.load_native()
    head, read = HeadEngine(), ReadEngine()
    config = copy.deepcopy(read.vllm_config)
    config.model_config.architecture = "Qwen3ForCausalLM"
    config.scheduler_config.max_num_seqs = 1
    expected_name = (
        "vllm.v1.core.sched.async_scheduler.AsyncScheduler"
        if async_mode
        else "vllm.v1.core.sched.scheduler.Scheduler"
    )
    expected = scheduler is None or scheduler == expected_name
    for runtime in (config, head.vllm_config, read.vllm_config):
        runtime.scheduler_config.scheduler_cls = scheduler
        runtime.scheduler_config.async_scheduling = async_mode
        assert (
            runtime_policy.qualified_runtime(
                runtime,
                runner_type=runtime.model_config.runner_type,
                diffusion=False,
                min_model_len=1,
            )
            is expected
        )
    writes = SimpleNamespace(set_engine_specific=Mock())
    worker = SimpleNamespace(
        disaggregation_mode=DisaggregationMode.AGGREGATED,
        enable_rl=False,
        route_to_encoder=False,
        custom_encoder_class=None,
        enable_multimodal=False,
        realtime=False,
    )
    assert (
        causal.publish_vllm_decision_capability(
            writes,
            ModelInput.Tokens,
            WorkerType.Aggregated,
            config,
            worker,
            native_generate_supported=True,
        )
        is expected
    )
    assert writes.set_engine_specific.called is expected
    for adapter, engine, profile, api in (
        (HeadExecutionAdapter, head, native_registry.D3_PROFILE, NativeApi()),
        (ReadExecutionAdapter, read, native_registry.LIQUID_PROFILE, read_api),
    ):
        monkeypatch.setattr(native_lifecycle, "load_native", lambda: api)
        if expected:
            assert await adapter.create(
                engine, native_registry.PROFILES[profile], abort_monitor=monitor
            )
        else:
            with pytest.raises(native_errors.NativeRuntimeError, match="incompatible"):
                await adapter.create(
                    engine, native_registry.PROFILES[profile], abort_monitor=monitor
                )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "path,value",
    [
        ("model_config.model_impl", "transformers"),
        ("model_config.model_class_overrides", {"architecture": "custom"}),
        ("model_config.convert_type", "embed"),
        ("model_config.is_encoder_decoder", True),
        ("model_config.is_diffusion", True),
        ("model_config.max_model_len", 0),
        ("model_config.max_model_len", True),
        ("parallel_config.data_parallel_size", 2),
        ("parallel_config.data_parallel_size", True),
        ("lora_config", object()),
        ("speculative_config", object()),
        ("kv_transfer_config", object()),
        ("ec_transfer_config", object()),
    ],
)
async def test_common_runtime_mutations_disable_causal_heads_and_reads(
    monkeypatch, path, value
):
    monkeypatch.setattr(causal, "version", lambda name: "0.32.0")
    monkeypatch.setattr(causal, "_decision_api_available", lambda: True)
    monkeypatch.setattr(causal, "qualified_cache_runtime", lambda _: True)
    monkeypatch.setattr(native_lifecycle, "load_native", lambda: NativeApi())
    head, read = HeadEngine(), ReadEngine()
    config = copy.deepcopy(read.vllm_config)
    config.model_config.architecture = "Qwen3ForCausalLM"
    config.scheduler_config.max_num_seqs = 1
    for runtime in (config, head.vllm_config, read.vllm_config):
        target, _, field = path.rpartition(".")
        setattr(getattr(runtime, target) if target else runtime, field, value)
        assert not runtime_policy.qualified_runtime(
            runtime,
            runner_type=runtime.model_config.runner_type,
            diffusion=False,
            min_model_len=1,
        )
    writes = SimpleNamespace(set_engine_specific=Mock())
    worker = SimpleNamespace(
        disaggregation_mode=DisaggregationMode.AGGREGATED,
        enable_rl=False,
        route_to_encoder=False,
        custom_encoder_class=None,
        enable_multimodal=False,
        realtime=False,
    )
    assert not causal.publish_vllm_decision_capability(
        writes,
        ModelInput.Tokens,
        WorkerType.Aggregated,
        config,
        worker,
        native_generate_supported=True,
    )
    writes.set_engine_specific.assert_not_called()
    for adapter, engine, profile in (
        (HeadExecutionAdapter, head, native_registry.D3_PROFILE),
        (ReadExecutionAdapter, read, native_registry.LIQUID_PROFILE),
    ):
        with pytest.raises(native_errors.NativeRuntimeError, match="incompatible"):
            await adapter.create(
                engine, native_registry.PROFILES[profile], abort_monitor=monitor
            )


@pytest.mark.parametrize(
    "field",
    [
        "model_impl",
        "model_class_overrides",
        "convert_type",
        "is_diffusion",
        "is_encoder_decoder",
        "max_model_len",
        "runner_type",
    ],
)
def test_missing_model_policy_metadata_fails_closed(field):
    runtime = HeadEngine().vllm_config
    delattr(runtime.model_config, field)
    assert not runtime_policy.qualified_runtime(
        runtime, runner_type="pooling", diffusion=False, min_model_len=1
    )
