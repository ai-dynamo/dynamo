# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

pytest.importorskip("vllm.lora.request")

from dynamo.common.lora.manager import LoRAInfo  # noqa: E402
from dynamo.vllm import handlers  # noqa: E402
from dynamo.vllm.tests.test_vllm_legacy_lora import _make_prefill_handler  # noqa: E402

pytestmark = [
    pytest.mark.unit,
    pytest.mark.vllm,
    pytest.mark.core,
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
    pytest.mark.timeout(10),
]


@pytest.mark.asyncio
@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("max_cpu_loras", [None, 2])
async def test_standard_handler_preserves_administrative_registration_capacity(
    monkeypatch, enabled, max_cpu_loras
):
    monkeypatch.delenv("DYN_LORA_RUNTIME_LOAD_ENABLED", raising=False)
    config = handlers.Config()
    config.model = "base"
    config.engine_args = SimpleNamespace(
        model="base",
        load_format="auto",
        trust_remote_code=False,
        enable_lora=enabled,
        max_loras=1,
        max_cpu_loras=max_cpu_loras,
    )
    engine = SimpleNamespace(
        vllm_config=SimpleNamespace(
            parallel_config=SimpleNamespace(
                data_parallel_external_lb=True, data_parallel_rank=0
            )
        )
    )
    monkeypatch.setattr(handlers, "VllmEngineMonitor", Mock())
    monkeypatch.setattr(handlers, "VllmMultimodalRequestProcessor", Mock())

    handler = handlers.DecodeWorkerHandler(
        runtime=Mock(), config=config, engine=engine, default_sampling_params=None
    )

    assert handler._lora_capacity is None
    if enabled:
        handler._lora_state.loaded_loras = {
            "first": LoRAInfo(id=1, path="/cache/first"),
            "second": LoRAInfo(id=2, path="/cache/second"),
        }
        reserved_id = await handler._reserve_admin_lora_slot(
            "third", handler._lora_capacity
        )
        assert reserved_id is not None
        assert reserved_id not in (1, 2)
        assert handler._lora_state.loaded_loras["third"].id == -1


@pytest.mark.asyncio
@pytest.mark.parametrize("quarantined", [False, True])
async def test_admin_name_cannot_reclaim_quarantined_rollback_id(
    monkeypatch, quarantined
):
    handler = _make_prefill_handler()
    if quarantined:
        handler._lora_state.rollback_reserved_ids.add(123)
        handler._lora_state.uncertain_engine_lora_ids.add(123)
    monkeypatch.setattr(handlers, "lora_name_to_id", lambda _name: 123)

    reserved = await handler._reserve_admin_lora_slot("<rollback-reserved>", None)

    assert reserved == (124 if quarantined else 123)
    if quarantined:
        assert 123 in handler._lora_state.rollback_reserved_ids
        assert 123 in handler._lora_state.uncertain_engine_lora_ids


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "generation_error", [None, RuntimeError("engine failed"), GeneratorExit()]
)
@pytest.mark.parametrize("cleanup_fails", [False, True])
async def test_cleanup_error_preserves_generation_failure(
    generation_error, cleanup_fails, caplog
):
    handler = _make_prefill_handler()
    handler._lora_state.loaded_loras["adapterA"] = LoRAInfo(
        id=123, path="/cache/adapter"
    )
    cleanup_error = ValueError("private-provider-detail")

    class FailingGenerator:
        def __aiter__(self):
            return self

        async def __anext__(self):
            if generation_error is not None:
                raise generation_error
            raise StopAsyncIteration

        async def aclose(self):
            if cleanup_fails:
                raise cleanup_error

    admission = handler._generate_with_lora_admission_lock(
        handler._resolve_lora_request("adapterA"), lambda _: FailingGenerator()
    )
    expected = generation_error or (
        cleanup_error if cleanup_fails else StopAsyncIteration()
    )
    with pytest.raises(type(expected)) as error:
        await anext(admission)
    if not isinstance(expected, StopAsyncIteration):
        assert error.value is expected
    assert handler._lora_state.active_requests == {}
    assert "private-provider-detail" not in caplog.text
