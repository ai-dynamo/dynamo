# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: E402 - optional vLLM dependency is checked before Dynamo imports

import pytest

pytest.importorskip("vllm.lora.request")

from dynamo.common.lora.runtime import ResolvedLoRA
from dynamo.vllm import handlers as handlers_mod
from dynamo.vllm.tests.test_vllm_runtime_lora import _coordinator, _request, _snapshot

pytestmark = [
    pytest.mark.unit,
    pytest.mark.vllm,
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
]


@pytest.mark.parametrize(
    "handler_type",
    [handlers_mod.DecodeWorkerHandler, handlers_mod.PrefillWorkerHandler],
)
@pytest.mark.parametrize("same_adapter", [True, False])
@pytest.mark.asyncio
async def test_reused_client_id_has_independent_handler_admission_owners(
    monkeypatch, tmp_path, handler_type, same_adapter
):
    snapshot = _snapshot(tmp_path)
    contexts = []

    async def resolve(**kwargs):
        contexts.append(kwargs["context"])
        return ResolvedLoRA(snapshot, "digest:immutable")

    state_handler, coordinator = _coordinator(monkeypatch, tmp_path, resolve)
    handler = handler_type.__new__(handler_type)
    handler.__dict__.update(vars(state_handler))
    handler._runtime_lora_coordinator = coordinator
    request = _request("wandb-artifact:///entity/project/adapter:v1")
    first_stack, first, first_runtime = await handler._prepare_lora_admission(
        request, "reused-client-id"
    )
    second_stack, second, second_runtime = await handler._prepare_lora_admission(
        request if same_adapter else _request("wandb-artifact:///other/adapter:v1"),
        "reused-client-id",
    )
    try:
        assert first_runtime and second_runtime
        expected = (
            {first.lora_name: 2}
            if same_adapter
            else {first.lora_name: 1, second.lora_name: 1}
        )
        assert state_handler._lora_state.runtime_pending_leases == expected
        assert first_stack.admission_id != second_stack.admission_id
        assert all(context.request_id == "reused-client-id" for context in contexts)
        await first_stack.aclose()
        assert state_handler._lora_state.runtime_pending_leases == {second.lora_name: 1}

        async def generate(admitted):
            assert admitted.lora_name == second.lora_name
            assert state_handler._lora_state.runtime_active_leases == {
                second.lora_name: 1
            }
            yield "token"

        output = [
            token
            async for token in handler._generate_with_lora_admission_lock(
                second,
                generate,
                request_id="reused-client-id",
                admission_id=second_stack.admission_id,
                runtime_lora=True,
            )
        ]
        assert output == ["token"]
        assert not state_handler._lora_state.runtime_active_leases
        assert not state_handler._lora_state.runtime_pending_admissions
    finally:
        await first_stack.aclose()
        await second_stack.aclose()
        await coordinator.close()
