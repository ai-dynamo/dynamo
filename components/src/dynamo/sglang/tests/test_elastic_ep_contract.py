# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from dynamo.sglang.elastic_ep import scale_external_elastic_ep

pytestmark = [
    pytest.mark.unit,
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
    pytest.mark.sglang,
    pytest.mark.core,
]


@pytest.fixture
def manager(monkeypatch):
    module = ModuleType("sglang.srt.managers.io_struct")
    module.ScaleElasticEPReqInput = SimpleNamespace
    monkeypatch.setitem(sys.modules, module.__name__, module)
    return SimpleNamespace(
        scale_elastic_ep=AsyncMock(
            return_value=SimpleNamespace(
                success=True,
                message="accepted",
                old_ep_size=8,
                new_ep_size=12,
                pending_ep_size=12,
                operation_id="grow-8-12",
                instance_id="runtime-1",
                scale_phase="joining",
                terminal=False,
                effective_ep_size=8,
                conflict=False,
            )
        )
    )


@pytest.mark.asyncio
async def test_native_growth_correlation(manager):
    response = await scale_external_elastic_ep(
        manager,
        {
            "new_ep_size": 12,
            "operation_id": "grow-8-12",
            "expected_instance_id": "runtime-1",
            "expected_joining_allocation_ids": ["pod-2"],
        },
    )
    assert vars(manager.scale_elastic_ep.await_args.args[0]) == {
        "new_ep_size": 12,
        "operation_id": "grow-8-12",
        "expected_instance_id": "runtime-1",
        "expected_joining_allocation_ids": ["pod-2"],
    }
    assert response["operation_id"] == "grow-8-12"
    assert response["instance_id"] == "runtime-1"
    assert response["pending_ep_size"] == 12
    assert response["terminal"] is False


@pytest.mark.asyncio
async def test_unsuccessful_pending_native_result_is_not_a_terminal_rejection(manager):
    manager.scale_elastic_ep.return_value.success = False
    response = await scale_external_elastic_ep(
        manager, {"new_ep_size": 12, "operation_id": "grow-8-12"}
    )
    assert response["status"] == "error"
    assert response["pending_ep_size"] == 12
    assert response["terminal"] is False


@pytest.mark.asyncio
async def test_timeout_propagates_without_fabricating_a_rejection(manager):
    manager.scale_elastic_ep.side_effect = TimeoutError("ambiguous submission")
    with pytest.raises(TimeoutError, match="ambiguous submission"):
        await scale_external_elastic_ep(
            manager, {"new_ep_size": 12, "operation_id": "grow-8-12"}
        )


@pytest.mark.parametrize(
    "body",
    [
        [],
        {"new_ep_size": True},
        {"new_ep_size": 0},
        {"new_ep_size": 12, "operation_id": ""},
        {"new_ep_size": 12, "expected_joining_allocation_ids": [""]},
    ],
)
@pytest.mark.asyncio
async def test_invalid_native_request_has_no_engine_effect(manager, body):
    response = await scale_external_elastic_ep(manager, body)
    assert response["status"] == "error"
    manager.scale_elastic_ep.assert_not_awaited()
